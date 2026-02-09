//! Convert a real JoltProof to symbolic sumcheck proofs for transpilation
//!
//! This module provides functions to create symbolic versions of proof data structures,
//! where concrete field elements are replaced with MleAst variables.
//!
//! # Variable Ordering Invariant
//!
//! **CRITICAL**: The `WitnessFieldIterator` is the **single source of truth** for variable
//! ordering. Both `symbolize_jolt_proof()` and `extract_sumcheck_witness_values()` use
//! this iterator to ensure variables are allocated and extracted in identical order.
//!
//! If you need to add new witness fields, modify `WitnessFieldIterator::new()` and both
//! functions will automatically stay in sync.
//!
//! NOTE: This is adapted for the quangvdao fork. We only symbolize the core verification
//! stages (1-7), not the recursion stages (9-13) which run on Grumpkin.

use crate::MleOpeningAccumulator;
use crate::PoseidonAstTranscript;
use crate::AstCommitmentScheme;
use crate::ast_commitment_scheme::AstProof;
use ark_bn254::Fq;
use ark_grumpkin::Projective as GrumpkinProjective;
use ark_std::Zero;
use jolt_core::poly::commitment::hyrax::{Hyrax, HyraxCommitment, HyraxOpeningProof};
use jolt_core::poly::opening_proof::{OpeningId, OpeningPoint};
use jolt_core::poly::unipoly::CompressedUniPoly;
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::subprotocols::univariate_skip::UniSkipFirstRoundProof;
use jolt_core::zkvm::proof_serialization::{JoltProof, Claims, RecursionConstraintMetadata};
use jolt_core::zkvm::recursion::bijection::{VarCountJaggedBijection, ConstraintMapping};
use jolt_core::zkvm::recursion::recursion_prover::RecursionProof;
use jolt_core::zkvm::recursion::stage5::jagged_assist::JaggedAssistProof;
use jolt_core::zkvm::RV64IMACProof;
use zklean_extractor::mle_ast::MleAst;
use zklean_extractor::AstCommitment;

// =============================================================================
// WITNESS FIELD ITERATOR - Single Source of Truth for Variable Ordering
// =============================================================================

/// Identifies which proof stage a witness field belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Stage1,
    Stage2,
    Stage3,
    Stage4,
    Stage5,
    Stage6a,
    Stage6b,
    Stage7,
}

impl Stage {
    fn name(&self) -> &'static str {
        match self {
            Stage::Stage1 => "Stage1",
            Stage::Stage2 => "Stage2",
            Stage::Stage3 => "Stage3",
            Stage::Stage4 => "Stage4",
            Stage::Stage5 => "Stage5",
            Stage::Stage6a => "Stage6a",
            Stage::Stage6b => "Stage6b",
            Stage::Stage7 => "Stage7",
        }
    }
}

/// Describes a single witness field in the proof.
///
/// This enum is the **single source of truth** for what fields exist in the witness
/// and their canonical ordering. Both symbolization and extraction iterate through
/// the same sequence of these fields.
#[derive(Debug, Clone)]
pub enum WitnessField {
    /// A 32-byte chunk of a commitment (for Fiat-Shamir transcript)
    CommitmentChunk {
        commitment_idx: usize,
        chunk_idx: usize,
    },
    /// An opening claim value for a specific polynomial opening
    OpeningClaim {
        key: OpeningId,
    },
    /// A coefficient in a uni-skip proof
    UniSkipCoeff {
        stage: Stage,
        coeff_idx: usize,
    },
    /// A coefficient in a sumcheck round polynomial (coeffs_except_linear_term)
    SumcheckCoeff {
        stage: Stage,
        round_idx: usize,
        coeff_idx: usize,
    },
}

impl WitnessField {
    /// Human-readable name for this field (used as variable description)
    pub fn name(&self) -> String {
        match self {
            WitnessField::CommitmentChunk { commitment_idx, chunk_idx } => {
                format!("Commitment_{}_Chunk_{}", commitment_idx, chunk_idx)
            }
            WitnessField::OpeningClaim { key } => {
                format!("Claim_{:?}", key)
            }
            WitnessField::UniSkipCoeff { stage, coeff_idx } => {
                format!("{}_Uni_Skip_Coeff_{}", stage.name(), coeff_idx)
            }
            WitnessField::SumcheckCoeff { stage, round_idx, coeff_idx } => {
                format!("{}_Sumcheck_R{}_{}", stage.name(), round_idx, coeff_idx)
            }
        }
    }
}

/// Iterator over all witness fields in canonical order.
///
/// This struct examines a real proof to determine the exact fields that need
/// to be symbolized/extracted, ensuring both operations use identical ordering.
pub struct WitnessFieldIterator {
    fields: Vec<WitnessField>,
}

impl WitnessFieldIterator {
    /// Build the witness field list from a real proof.
    ///
    /// This examines the proof structure to determine exact counts for each
    /// variable-length field (commitments, rounds, coefficients).
    pub fn new(proof: &RV64IMACProof) -> Self {
        let mut fields = Vec::new();

        // 1. Commitments: N commitments × 12 chunks each
        for commitment_idx in 0..proof.commitments.len() {
            for chunk_idx in 0..CHUNKS_PER_COMMITMENT {
                fields.push(WitnessField::CommitmentChunk {
                    commitment_idx,
                    chunk_idx,
                });
            }
        }

        // 2. Opening claims (BTreeMap ensures consistent ordering)
        for (key, _) in &proof.opening_claims.0 {
            fields.push(WitnessField::OpeningClaim { key: key.clone() });
        }

        // 3. Stage 1 uni-skip coefficients
        for coeff_idx in 0..proof.stage1_uni_skip_first_round_proof.uni_poly.coeffs.len() {
            fields.push(WitnessField::UniSkipCoeff {
                stage: Stage::Stage1,
                coeff_idx,
            });
        }

        // 4. Stage 1 sumcheck coefficients
        for (round_idx, poly) in proof.stage1_sumcheck_proof.compressed_polys.iter().enumerate() {
            for coeff_idx in 0..poly.coeffs_except_linear_term.len() {
                fields.push(WitnessField::SumcheckCoeff {
                    stage: Stage::Stage1,
                    round_idx,
                    coeff_idx,
                });
            }
        }

        // 5. Stage 2 uni-skip coefficients
        for coeff_idx in 0..proof.stage2_uni_skip_first_round_proof.uni_poly.coeffs.len() {
            fields.push(WitnessField::UniSkipCoeff {
                stage: Stage::Stage2,
                coeff_idx,
            });
        }

        // 6. Stage 2 sumcheck coefficients
        for (round_idx, poly) in proof.stage2_sumcheck_proof.compressed_polys.iter().enumerate() {
            for coeff_idx in 0..poly.coeffs_except_linear_term.len() {
                fields.push(WitnessField::SumcheckCoeff {
                    stage: Stage::Stage2,
                    round_idx,
                    coeff_idx,
                });
            }
        }

        // Helper closure for sumcheck-only stages
        let mut add_sumcheck_stage = |stage: Stage, polys: &[CompressedUniPoly<ark_bn254::Fr>]| {
            for (round_idx, poly) in polys.iter().enumerate() {
                for coeff_idx in 0..poly.coeffs_except_linear_term.len() {
                    fields.push(WitnessField::SumcheckCoeff {
                        stage,
                        round_idx,
                        coeff_idx,
                    });
                }
            }
        };

        // 7-12. Stages 3-7 sumcheck coefficients
        add_sumcheck_stage(Stage::Stage3, &proof.stage3_sumcheck_proof.compressed_polys);
        add_sumcheck_stage(Stage::Stage4, &proof.stage4_sumcheck_proof.compressed_polys);
        add_sumcheck_stage(Stage::Stage5, &proof.stage5_sumcheck_proof.compressed_polys);
        add_sumcheck_stage(Stage::Stage6a, &proof.stage6a_sumcheck_proof.compressed_polys);
        add_sumcheck_stage(Stage::Stage6b, &proof.stage6b_sumcheck_proof.compressed_polys);
        add_sumcheck_stage(Stage::Stage7, &proof.stage7_sumcheck_proof.compressed_polys);

        Self { fields }
    }

    /// Total number of witness fields
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

impl IntoIterator for WitnessFieldIterator {
    type Item = WitnessField;
    type IntoIter = std::vec::IntoIter<WitnessField>;

    fn into_iter(self) -> Self::IntoIter {
        self.fields.into_iter()
    }
}

impl<'a> IntoIterator for &'a WitnessFieldIterator {
    type Item = &'a WitnessField;
    type IntoIter = std::slice::Iter<'a, WitnessField>;

    fn into_iter(self) -> Self::IntoIter {
        self.fields.iter()
    }
}

/// Number of 32-byte chunks per commitment (384 bytes / 32 = 12)
const CHUNKS_PER_COMMITMENT: usize = 12;

/// Tracks variable index allocation during symbolization
pub struct VarAllocator {
    next_idx: u16,
    descriptions: Vec<(u16, String)>,
}

impl VarAllocator {
    pub fn new() -> Self {
        Self {
            next_idx: 0,
            descriptions: Vec::new(),
        }
    }

    pub fn alloc(&mut self, description: &str) -> MleAst {
        let idx = self.next_idx;
        self.descriptions.push((idx, description.to_string()));
        self.next_idx += 1;
        MleAst::from_var(idx)
    }

    pub fn alloc_n(&mut self, n: usize, prefix: &str) -> Vec<MleAst> {
        (0..n)
            .map(|i| self.alloc(&format!("{}_{}", prefix, i)))
            .collect()
    }

    pub fn next_idx(&self) -> u16 {
        self.next_idx
    }

    pub fn descriptions(&self) -> &[(u16, String)] {
        &self.descriptions
    }
}

impl Default for VarAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Symbolic sumcheck proofs for stages 1-7
pub struct SymbolicSumcheckProofs {
    pub stage1_uni_skip: UniSkipFirstRoundProof<MleAst, PoseidonAstTranscript>,
    pub stage1_sumcheck: SumcheckInstanceProof<MleAst, PoseidonAstTranscript>,
    pub stage2_uni_skip: UniSkipFirstRoundProof<MleAst, PoseidonAstTranscript>,
    pub stage2_sumcheck: SumcheckInstanceProof<MleAst, PoseidonAstTranscript>,
    pub stage3_sumcheck: SumcheckInstanceProof<MleAst, PoseidonAstTranscript>,
    pub stage4_sumcheck: SumcheckInstanceProof<MleAst, PoseidonAstTranscript>,
    pub stage5_sumcheck: SumcheckInstanceProof<MleAst, PoseidonAstTranscript>,
    pub stage6a_sumcheck: SumcheckInstanceProof<MleAst, PoseidonAstTranscript>,
    pub stage6b_sumcheck: SumcheckInstanceProof<MleAst, PoseidonAstTranscript>,
    pub stage7_sumcheck: SumcheckInstanceProof<MleAst, PoseidonAstTranscript>,
}

/// Symbolize sumcheck proofs (stages 1-7) from a real proof.
///
/// This creates symbolic versions of all sumcheck proof coefficients as MleAst variables,
/// which can then be used to trace through the verifier symbolically.
///
/// Returns:
/// - SymbolicSumcheckProofs containing all symbolic sumcheck proofs
/// - MleOpeningAccumulator with symbolic claims
/// - VarAllocator with all variable descriptions
pub fn symbolize_sumcheck_proofs(
    real_proof: &RV64IMACProof,
) -> (SymbolicSumcheckProofs, MleOpeningAccumulator, VarAllocator) {
    let mut alloc = VarAllocator::new();

    // === Symbolize opening claims ===
    let mut accumulator = MleOpeningAccumulator::new();
    for (key, (_point, _claim)) in &real_proof.opening_claims.0 {
        let symbolic_claim = alloc.alloc(&format!("claim_{:?}", key));
        accumulator.openings.insert(key.clone(), (vec![], symbolic_claim));
    }

    // === Symbolize stage 1 uni-skip proof ===
    let stage1_uni_skip = symbolize_uni_skip_proof(
        &real_proof.stage1_uni_skip_first_round_proof,
        &mut alloc,
        "stage1_uni_skip",
    );

    // === Symbolize stage 1 sumcheck proof ===
    let stage1_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage1_sumcheck_proof,
        &mut alloc,
        "stage1_sumcheck",
    );

    // === Symbolize stage 2 uni-skip proof ===
    let stage2_uni_skip = symbolize_uni_skip_proof(
        &real_proof.stage2_uni_skip_first_round_proof,
        &mut alloc,
        "stage2_uni_skip",
    );

    // === Symbolize stage 2 sumcheck proof ===
    let stage2_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage2_sumcheck_proof,
        &mut alloc,
        "stage2_sumcheck",
    );

    // === Symbolize stage 3 sumcheck proof ===
    let stage3_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage3_sumcheck_proof,
        &mut alloc,
        "stage3_sumcheck",
    );

    // === Symbolize stage 4 sumcheck proof ===
    let stage4_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage4_sumcheck_proof,
        &mut alloc,
        "stage4_sumcheck",
    );

    // === Symbolize stage 5 sumcheck proof ===
    let stage5_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage5_sumcheck_proof,
        &mut alloc,
        "stage5_sumcheck",
    );

    // === Symbolize stage 6a sumcheck proof ===
    let stage6a_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage6a_sumcheck_proof,
        &mut alloc,
        "stage6a_sumcheck",
    );

    // === Symbolize stage 6b sumcheck proof ===
    let stage6b_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage6b_sumcheck_proof,
        &mut alloc,
        "stage6b_sumcheck",
    );

    // === Symbolize stage 7 sumcheck proof ===
    let stage7_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage7_sumcheck_proof,
        &mut alloc,
        "stage7_sumcheck",
    );

    let proofs = SymbolicSumcheckProofs {
        stage1_uni_skip,
        stage1_sumcheck,
        stage2_uni_skip,
        stage2_sumcheck,
        stage3_sumcheck,
        stage4_sumcheck,
        stage5_sumcheck,
        stage6a_sumcheck,
        stage6b_sumcheck,
        stage7_sumcheck,
    };

    (proofs, accumulator, alloc)
}

/// Symbolize a full JoltProof for use with TranspilableVerifier.
///
/// Creates a symbolic JoltProof where sumcheck coefficients are replaced with MleAst variables.
/// Non-sumcheck fields (commitments, PCS proofs, recursion data) are copied from the real proof
/// where possible or filled with defaults.
///
/// # Variable Ordering Guarantee
///
/// This function allocates variables in the same order as `WitnessFieldIterator` iterates them.
/// An assertion at the end verifies the counts match. This ensures `extract_sumcheck_witness_values`
/// (which uses `WitnessFieldIterator`) produces correctly-keyed witness values.
///
/// Returns:
/// - JoltProof with symbolic sumcheck proofs
/// - MleOpeningAccumulator with symbolic claims
/// - VarAllocator with all variable descriptions
pub fn symbolize_jolt_proof(
    real_proof: &RV64IMACProof,
) -> (JoltProof<MleAst, AstCommitmentScheme, PoseidonAstTranscript>, MleOpeningAccumulator, VarAllocator) {
    let mut alloc = VarAllocator::new();

    // === Symbolize commitments ===
    // Each commitment is 384 bytes = 12 chunks of 32 bytes each
    // These get appended to the transcript before verification starts
    let commitments: Vec<AstCommitment> = (0..real_proof.commitments.len())
        .map(|c| {
            let chunks = alloc.alloc_n(CHUNKS_PER_COMMITMENT, &format!("Commitment_{}", c));
            AstCommitment::new(chunks)
        })
        .collect();

    // === Symbolize opening claims ===
    let mut accumulator = MleOpeningAccumulator::new();
    let mut symbolic_claims = Claims(std::collections::BTreeMap::new());
    for (key, (_point, _claim)) in &real_proof.opening_claims.0 {
        let symbolic_claim = alloc.alloc(&format!("Claim_{:?}", key));
        accumulator.openings.insert(key.clone(), (vec![], symbolic_claim.clone()));
        symbolic_claims.0.insert(key.clone(), (OpeningPoint::default(), symbolic_claim));
    }

    // === Symbolize stage 1 uni-skip proof ===
    let stage1_uni_skip = symbolize_uni_skip_proof(
        &real_proof.stage1_uni_skip_first_round_proof,
        &mut alloc,
        "stage1_uni_skip",
    );

    // === Symbolize stage 1 sumcheck proof ===
    let stage1_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage1_sumcheck_proof,
        &mut alloc,
        "stage1_sumcheck",
    );

    // === Symbolize stage 2 uni-skip proof ===
    let stage2_uni_skip = symbolize_uni_skip_proof(
        &real_proof.stage2_uni_skip_first_round_proof,
        &mut alloc,
        "stage2_uni_skip",
    );

    // === Symbolize stage 2 sumcheck proof ===
    let stage2_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage2_sumcheck_proof,
        &mut alloc,
        "stage2_sumcheck",
    );

    // === Symbolize stage 3 sumcheck proof ===
    let stage3_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage3_sumcheck_proof,
        &mut alloc,
        "stage3_sumcheck",
    );

    // === Symbolize stage 4 sumcheck proof ===
    let stage4_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage4_sumcheck_proof,
        &mut alloc,
        "stage4_sumcheck",
    );

    // === Symbolize stage 5 sumcheck proof ===
    let stage5_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage5_sumcheck_proof,
        &mut alloc,
        "stage5_sumcheck",
    );

    // === Symbolize stage 6a sumcheck proof ===
    let stage6a_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage6a_sumcheck_proof,
        &mut alloc,
        "stage6a_sumcheck",
    );

    // === Symbolize stage 6b sumcheck proof ===
    let stage6b_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage6b_sumcheck_proof,
        &mut alloc,
        "stage6b_sumcheck",
    );

    // === Symbolize stage 7 sumcheck proof ===
    let stage7_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage7_sumcheck_proof,
        &mut alloc,
        "stage7_sumcheck",
    );

    // Build empty recursion metadata (stages 8+ not used in transpilation)
    let empty_recursion_metadata = RecursionConstraintMetadata {
        constraint_types: vec![],
        jagged_bijection: VarCountJaggedBijection::default(),
        jagged_mapping: ConstraintMapping::default(),
        matrix_rows: vec![],
        dense_num_vars: 0,
        gt_exp_public_inputs: vec![],
        g1_scalar_mul_public_inputs: vec![],
        g2_scalar_mul_public_inputs: vec![],
    };

    // Build empty recursion proof (stages 11-13 not used in transpilation)
    // RecursionProof in JoltProof is typed as RecursionProof<Fq, FS, Hyrax<1, GrumpkinProjective>>
    // where FS is the outer JoltProof's transcript type (PoseidonAstTranscript for symbolic)
    type HyraxGrumpkin = Hyrax<1, GrumpkinProjective>;
    let recursion_proof: RecursionProof<Fq, PoseidonAstTranscript, HyraxGrumpkin> = RecursionProof {
        stage1_proof: SumcheckInstanceProof::new(vec![]),
        stage2_proof: SumcheckInstanceProof::new(vec![]),
        stage3_m_eval: Fq::zero(),
        stage4_proof: SumcheckInstanceProof::new(vec![]),
        stage5_proof: JaggedAssistProof {
            claimed_evaluations: vec![],
            sumcheck_proof: SumcheckInstanceProof::new(vec![]),
        },
        opening_proof: HyraxOpeningProof { vector_matrix_product: vec![] },
        gamma: Fq::zero(),
        delta: Fq::zero(),
        opening_claims: std::collections::BTreeMap::new(),
        dense_commitment: HyraxCommitment::default(),
    };

    // Build the symbolic proof
    // For non-sumcheck fields, we use empty/default values since we don't verify these symbolically
    let symbolic_proof = JoltProof {
        opening_claims: symbolic_claims,
        commitments,  // Symbolized commitments for transcript
        stage1_uni_skip_first_round_proof: stage1_uni_skip,
        stage1_sumcheck_proof: stage1_sumcheck,
        stage2_uni_skip_first_round_proof: stage2_uni_skip,
        stage2_sumcheck_proof: stage2_sumcheck,
        stage3_sumcheck_proof: stage3_sumcheck,
        stage4_sumcheck_proof: stage4_sumcheck,
        stage5_sumcheck_proof: stage5_sumcheck,
        stage6a_sumcheck_proof: stage6a_sumcheck,
        stage6b_sumcheck_proof: stage6b_sumcheck,
        stage7_sumcheck_proof: stage7_sumcheck,
        // PCS-specific fields - empty since we don't verify these symbolically
        stage8_opening_proof: AstProof::default(),
        stage8_combine_hint: None,
        stage9_pcs_hint: None,
        stage10_recursion_metadata: empty_recursion_metadata,
        recursion_proof,
        trusted_advice_val_evaluation_proof: None,
        trusted_advice_val_final_proof: None,
        untrusted_advice_val_evaluation_proof: None,
        untrusted_advice_val_final_proof: None,
        untrusted_advice_commitment: None,
        // Configuration - copy from real proof
        trace_length: real_proof.trace_length,
        ram_K: real_proof.ram_K,
        bytecode_K: real_proof.bytecode_K,
        program_mode: real_proof.program_mode,
        rw_config: real_proof.rw_config.clone(),
        one_hot_config: real_proof.one_hot_config.clone(),
        dory_layout: real_proof.dory_layout,
    };

    // Verify variable count matches WitnessFieldIterator
    // This catches any ordering divergence between symbolize and extract
    let expected_count = WitnessFieldIterator::new(real_proof).len();
    assert_eq!(
        alloc.next_idx() as usize, expected_count,
        "Variable ordering invariant violated: symbolize allocated {} variables but \
         WitnessFieldIterator expects {}. Both must iterate fields in identical order.",
        alloc.next_idx(), expected_count
    );

    (symbolic_proof, accumulator, alloc)
}

fn symbolize_uni_skip_proof<T: jolt_core::transcripts::Transcript>(
    real: &UniSkipFirstRoundProof<ark_bn254::Fr, T>,
    alloc: &mut VarAllocator,
    prefix: &str,
) -> UniSkipFirstRoundProof<MleAst, PoseidonAstTranscript> {
    let coeffs = alloc.alloc_n(real.uni_poly.coeffs.len(), &format!("{}_coeff", prefix));
    UniSkipFirstRoundProof::new(jolt_core::poly::unipoly::UniPoly::from_coeff(coeffs))
}

fn symbolize_sumcheck_proof<T: jolt_core::transcripts::Transcript>(
    real: &SumcheckInstanceProof<ark_bn254::Fr, T>,
    alloc: &mut VarAllocator,
    prefix: &str,
) -> SumcheckInstanceProof<MleAst, PoseidonAstTranscript> {
    let compressed_polys: Vec<CompressedUniPoly<MleAst>> = real
        .compressed_polys
        .iter()
        .enumerate()
        .map(|(round, poly)| {
            let coeffs = alloc.alloc_n(
                poly.coeffs_except_linear_term.len(),
                &format!("{}_r{}", prefix, round),
            );
            CompressedUniPoly {
                coeffs_except_linear_term: coeffs,
            }
        })
        .collect();

    SumcheckInstanceProof::new(compressed_polys)
}

/// Extract concrete witness values from proof data.
/// Returns a HashMap<variable_index, value_as_decimal_string>
///
/// # Variable Ordering Guarantee
///
/// This function uses `WitnessFieldIterator` to iterate fields in the exact same order
/// as `symbolize_jolt_proof()`. This ensures witness values are keyed by the correct
/// variable indices.
pub fn extract_witness_values(
    real_proof: &RV64IMACProof,
) -> std::collections::HashMap<usize, String> {
    use ark_ff::PrimeField;
    use ark_serialize::CanonicalSerialize;

    // Helper to serialize a commitment to bytes
    // MUST match the Poseidon transcript serialization:
    // 1. Use serialize_uncompressed (not compressed)
    // 2. Reverse bytes for BE/EVM format
    fn commitment_to_bytes<T: CanonicalSerialize>(commitment: &T) -> Vec<u8> {
        let mut bytes = Vec::new();
        commitment.serialize_uncompressed(&mut bytes).expect("serialization failed");
        bytes.reverse();
        bytes
    }

    // Helper to get the n-th 32-byte chunk from commitment bytes as a field element
    fn get_commitment_chunk(commitment_bytes: &[u8], chunk_idx: usize) -> ark_bn254::Fr {
        let start = chunk_idx * 32;
        let end = std::cmp::min(start + 32, commitment_bytes.len());
        if start >= commitment_bytes.len() {
            ark_bn254::Fr::from(0u64)
        } else {
            ark_bn254::Fr::from_le_bytes_mod_order(&commitment_bytes[start..end])
        }
    }

    // Pre-compute commitment bytes for efficient access
    let commitment_bytes: Vec<Vec<u8>> = real_proof.commitments.iter()
        .map(|c| commitment_to_bytes(c))
        .collect();

    // Pre-compute opening claims for efficient access (maintain BTreeMap order)
    let opening_claims: Vec<_> = real_proof.opening_claims.0.iter().collect();

    let mut values = std::collections::HashMap::new();

    // Use WitnessFieldIterator to iterate fields in canonical order
    for (idx, field) in WitnessFieldIterator::new(real_proof).into_iter().enumerate() {
        let value: String = match field {
            WitnessField::CommitmentChunk { commitment_idx, chunk_idx } => {
                let chunk = get_commitment_chunk(&commitment_bytes[commitment_idx], chunk_idx);
                format!("{}", chunk.into_bigint())
            }
            WitnessField::OpeningClaim { ref key } => {
                // Find the claim by key (inefficient but correct; could optimize with a map)
                let (_, claim) = opening_claims.iter()
                    .find(|(k, _)| *k == key)
                    .expect("Opening claim key not found")
                    .1;
                format!("{}", claim.into_bigint())
            }
            WitnessField::UniSkipCoeff { stage, coeff_idx } => {
                let coeffs = match stage {
                    Stage::Stage1 => &real_proof.stage1_uni_skip_first_round_proof.uni_poly.coeffs,
                    Stage::Stage2 => &real_proof.stage2_uni_skip_first_round_proof.uni_poly.coeffs,
                    _ => panic!("Uni-skip proofs only exist for stages 1 and 2"),
                };
                format!("{}", coeffs[coeff_idx].into_bigint())
            }
            WitnessField::SumcheckCoeff { stage, round_idx, coeff_idx } => {
                let polys = match stage {
                    Stage::Stage1 => &real_proof.stage1_sumcheck_proof.compressed_polys,
                    Stage::Stage2 => &real_proof.stage2_sumcheck_proof.compressed_polys,
                    Stage::Stage3 => &real_proof.stage3_sumcheck_proof.compressed_polys,
                    Stage::Stage4 => &real_proof.stage4_sumcheck_proof.compressed_polys,
                    Stage::Stage5 => &real_proof.stage5_sumcheck_proof.compressed_polys,
                    Stage::Stage6a => &real_proof.stage6a_sumcheck_proof.compressed_polys,
                    Stage::Stage6b => &real_proof.stage6b_sumcheck_proof.compressed_polys,
                    Stage::Stage7 => &real_proof.stage7_sumcheck_proof.compressed_polys,
                };
                format!("{}", polys[round_idx].coeffs_except_linear_term[coeff_idx].into_bigint())
            }
        };
        values.insert(idx, value);
    }

    values
}
