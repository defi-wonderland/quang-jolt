//! Convert a real JoltProof to symbolic sumcheck proofs for transpilation
//!
//! This module provides functions to create symbolic versions of proof data structures,
//! where concrete field elements are replaced with MleAst variables.
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
use jolt_core::poly::opening_proof::OpeningPoint;
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

/// Extract concrete witness values from proof data
/// Returns a HashMap<variable_index, value_as_decimal_string>
///
/// The indices match exactly what symbolize_jolt_proof allocates.
pub fn extract_sumcheck_witness_values(
    real_proof: &RV64IMACProof,
) -> std::collections::HashMap<usize, String> {
    use ark_ff::PrimeField;
    use ark_serialize::CanonicalSerialize;
    let mut values: std::collections::HashMap<usize, String> = std::collections::HashMap::new();
    let mut idx: usize = 0;

    // Helper to convert bytes to field element chunks
    fn bytes_to_chunks(bytes: &[u8]) -> Vec<ark_bn254::Fr> {
        let num_chunks = 12; // Always 12 chunks per commitment
        (0..num_chunks)
            .map(|i| {
                let start = i * 32;
                let end = std::cmp::min(start + 32, bytes.len());
                if start >= bytes.len() {
                    ark_bn254::Fr::from(0u64)
                } else {
                    ark_bn254::Fr::from_le_bytes_mod_order(&bytes[start..end])
                }
            })
            .collect()
    }

    // Helper to serialize a commitment to bytes
    // MUST match the Poseidon transcript serialization:
    // 1. Use serialize_uncompressed (not compressed)
    // 2. Reverse bytes for BE/EVM format
    fn commitment_to_bytes<T: CanonicalSerialize>(commitment: &T) -> Vec<u8> {
        let mut bytes = Vec::new();
        commitment.serialize_uncompressed(&mut bytes).expect("serialization failed");
        // Reverse bytes to match Poseidon transcript format (BE for EVM compatibility)
        bytes.reverse();
        bytes
    }

    // === Commitments (N commitments × 12 chunks each) ===
    for commitment in &real_proof.commitments {
        let chunks = bytes_to_chunks(&commitment_to_bytes(commitment));
        for chunk in chunks {
            values.insert(idx, format!("{}", chunk.into_bigint()));
            idx += 1;
        }
    }

    // === Opening claims ===
    for (_key, (_point, claim)) in &real_proof.opening_claims.0 {
        values.insert(idx, format!("{}", claim.into_bigint()));
        idx += 1;
    }

    // === Stage 1 uni-skip proof ===
    for coeff in &real_proof.stage1_uni_skip_first_round_proof.uni_poly.coeffs {
        values.insert(idx, format!("{}", coeff.into_bigint()));
        idx += 1;
    }

    // === Stage 1 sumcheck proof ===
    for poly in &real_proof.stage1_sumcheck_proof.compressed_polys {
        for coeff in &poly.coeffs_except_linear_term {
            values.insert(idx, format!("{}", coeff.into_bigint()));
            idx += 1;
        }
    }

    // === Stage 2 uni-skip proof ===
    for coeff in &real_proof.stage2_uni_skip_first_round_proof.uni_poly.coeffs {
        values.insert(idx, format!("{}", coeff.into_bigint()));
        idx += 1;
    }

    // === Stage 2 sumcheck proof ===
    for poly in &real_proof.stage2_sumcheck_proof.compressed_polys {
        for coeff in &poly.coeffs_except_linear_term {
            values.insert(idx, format!("{}", coeff.into_bigint()));
            idx += 1;
        }
    }

    // === Stage 3 sumcheck proof ===
    for poly in &real_proof.stage3_sumcheck_proof.compressed_polys {
        for coeff in &poly.coeffs_except_linear_term {
            values.insert(idx, format!("{}", coeff.into_bigint()));
            idx += 1;
        }
    }

    // === Stage 4 sumcheck proof ===
    for poly in &real_proof.stage4_sumcheck_proof.compressed_polys {
        for coeff in &poly.coeffs_except_linear_term {
            values.insert(idx, format!("{}", coeff.into_bigint()));
            idx += 1;
        }
    }

    // === Stage 5 sumcheck proof ===
    for poly in &real_proof.stage5_sumcheck_proof.compressed_polys {
        for coeff in &poly.coeffs_except_linear_term {
            values.insert(idx, format!("{}", coeff.into_bigint()));
            idx += 1;
        }
    }

    // === Stage 6a sumcheck proof ===
    for poly in &real_proof.stage6a_sumcheck_proof.compressed_polys {
        for coeff in &poly.coeffs_except_linear_term {
            values.insert(idx, format!("{}", coeff.into_bigint()));
            idx += 1;
        }
    }

    // === Stage 6b sumcheck proof ===
    for poly in &real_proof.stage6b_sumcheck_proof.compressed_polys {
        for coeff in &poly.coeffs_except_linear_term {
            values.insert(idx, format!("{}", coeff.into_bigint()));
            idx += 1;
        }
    }

    // === Stage 7 sumcheck proof ===
    for poly in &real_proof.stage7_sumcheck_proof.compressed_polys {
        for coeff in &poly.coeffs_except_linear_term {
            values.insert(idx, format!("{}", coeff.into_bigint()));
            idx += 1;
        }
    }

    values
}
