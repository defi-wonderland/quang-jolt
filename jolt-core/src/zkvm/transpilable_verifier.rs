//! Transpilable verifier for Jolt proof verification.
//!
//! This module provides a generic verifier that can be used with different accumulator types:
//! - `VerifierOpeningAccumulator<F>` for concrete verification
//! - `MleOpeningAccumulator` for symbolic transpilation to Gnark
//!
//! The verifier is generic over the accumulator type `A: OpeningAccumulator<F>`, which allows
//! it to be used for both purposes without code duplication.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use ark_bn254::Fq;
use ark_grumpkin::Projective as GrumpkinProjective;

use crate::poly::commitment::commitment_scheme::{CommitmentScheme, RecursionExt};
use crate::poly::commitment::hyrax::{Hyrax, PedersenGenerators};
use crate::poly::opening_proof::{
    compute_advice_lagrange_factor, DoryOpeningState, OpeningAccumulator, OpeningId, OpeningPoint,
    PolynomialId, SumcheckId, VerifierOpeningAccumulator,
};
use crate::zkvm::recursion::recursion_prover::RecursionProof;
use crate::zkvm::recursion::recursion_verifier::{RecursionVerifier, RecursionVerifierInput};
use crate::zkvm::recursion::MAX_RECURSION_DENSE_NUM_VARS;
use crate::subprotocols::sumcheck::BatchedSumcheck;
use crate::zkvm::bytecode::chunks::total_lanes;
use crate::zkvm::bytecode::read_raf_checking::{
    BytecodeReadRafAddressSumcheckVerifier, BytecodeReadRafCycleSumcheckVerifier,
    BytecodeReadRafSumcheckParams,
};
use crate::zkvm::claim_reductions::{
    hamming_weight::HammingWeightClaimReductionVerifier,
    increments::IncClaimReductionSumcheckVerifier, AdviceKind,
    InstructionLookupsClaimReductionSumcheckVerifier, RamRaClaimReductionSumcheckVerifier,
    RegistersClaimReductionSumcheckVerifier,
};
use crate::zkvm::config::{OneHotParams, ProgramMode};
use crate::zkvm::program::VerifierProgram;
use crate::zkvm::ram::val_final::ValFinalSumcheckVerifier;
use crate::zkvm::witness::{all_committed_polynomials, CommittedPolynomial};
use crate::zkvm::Serializable;
use crate::zkvm::{
    fiat_shamir_preamble,
    instruction_lookups::{
        ra_virtual::RaSumcheckVerifier as LookupsRaSumcheckVerifier,
        read_raf_checking::InstructionReadRafSumcheckVerifier,
    },
    proof_serialization::JoltProof,
    r1cs::key::UniformSpartanKey,
    ram::{
        hamming_booleanity::HammingBooleanitySumcheckVerifier,
        output_check::OutputSumcheckVerifier,
        ra_virtual::RamRaVirtualSumcheckVerifier,
        raf_evaluation::RafEvaluationSumcheckVerifier as RamRafEvaluationSumcheckVerifier,
        read_write_checking::RamReadWriteCheckingVerifier,
        val_evaluation::ValEvaluationSumcheckVerifier as RamValEvaluationSumcheckVerifier,
        verifier_accumulate_advice,
    },
    registers::{
        read_write_checking::RegistersReadWriteCheckingVerifier,
        val_evaluation::ValEvaluationSumcheckVerifier as RegistersValEvaluationSumcheckVerifier,
    },
    spartan::{
        instruction_input::InstructionInputSumcheckVerifier, outer::OuterRemainingSumcheckVerifier,
        product::ProductVirtualRemainderVerifier, shift::ShiftSumcheckVerifier,
        verify_stage1_uni_skip, verify_stage2_uni_skip,
    },
    ProverDebugInfo,
};
use crate::{
    field::JoltField,
    pprof_scope,
    subprotocols::{
        booleanity::{
            BooleanityAddressSumcheckVerifier, BooleanityCycleSumcheckVerifier,
            BooleanitySumcheckParams,
        },
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};
use anyhow::Context;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::MemoryLayout;
use itertools::Itertools;
use jolt_platform::{end_cycle_tracking, start_cycle_tracking};
use tracer::JoltDevice;

// Cycle-marker labels must be static strings: the tracer keys markers by the guest string pointer.
const CYCLE_VERIFY_STAGE8: &str = "jolt_verify_stage8";
const CYCLE_VERIFY_STAGE8_DORY_PCS: &str = "jolt_verify_stage8_dory_pcs";
const CYCLE_VERIFY_STAGE8_RECURSION: &str = "jolt_verify_stage8_recursion";

struct CycleMarkerGuard(&'static str);
impl CycleMarkerGuard {
    #[inline(always)]
    fn new(label: &'static str) -> Self {
        start_cycle_tracking(label);
        Self(label)
    }
}
impl Drop for CycleMarkerGuard {
    #[inline(always)]
    fn drop(&mut self) {
        end_cycle_tracking(self.0);
    }
}

/// Transpilable verifier that is generic over the opening accumulator type.
///
/// This allows the same verification logic to be used for:
/// - Concrete verification with `VerifierOpeningAccumulator<F>`
/// - Symbolic transpilation with `MleOpeningAccumulator`
pub struct TranspilableVerifier<
    'a,
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + RecursionExt<F>,
    ProofTranscript: Transcript,
    A: OpeningAccumulator<F> = VerifierOpeningAccumulator<F>,
> {
    pub trusted_advice_commitment: Option<PCS::Commitment>,
    pub program_io: JoltDevice,
    pub proof: JoltProof<F, PCS, ProofTranscript>,
    pub preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
    pub transcript: ProofTranscript,
    pub opening_accumulator: A,
    pub spartan_key: UniformSpartanKey<F>,
    pub one_hot_params: OneHotParams,
    /// Symbolic recursion proof for stage 8 verification.
    /// None for real verification (uses self.proof.recursion_proof with Hyrax).
    /// Some for symbolic transpilation (uses MleAst version).
    pub symbolic_recursion_proof: Option<RecursionProof<F, ProofTranscript, PCS>>,
}

// =============================================================================
// Constructor for real verification (VerifierOpeningAccumulator)
// =============================================================================

impl<
        'a,
        F: JoltField,
        PCS: CommitmentScheme<Field = F> + RecursionExt<F>,
        ProofTranscript: Transcript,
    > TranspilableVerifier<'a, F, PCS, ProofTranscript, VerifierOpeningAccumulator<F>>
{
    /// Create a new TranspilableVerifier with concrete VerifierOpeningAccumulator.
    pub fn new(
        preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
        proof: JoltProof<F, PCS, ProofTranscript>,
        mut program_io: JoltDevice,
        trusted_advice_commitment: Option<PCS::Commitment>,
        _debug_info: Option<ProverDebugInfo<F, ProofTranscript, PCS>>,
    ) -> Result<Self, ProofVerifyError> {
        // Memory layout checks
        if program_io.memory_layout != preprocessing.memory_layout {
            return Err(ProofVerifyError::MemoryLayoutMismatch);
        }
        if program_io.inputs.len() > preprocessing.memory_layout.max_input_size as usize {
            return Err(ProofVerifyError::InputTooLarge);
        }
        if program_io.outputs.len() > preprocessing.memory_layout.max_output_size as usize {
            return Err(ProofVerifyError::OutputTooLarge);
        }

        // Truncate trailing zeros on device outputs
        program_io.outputs.truncate(
            program_io
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        let mut opening_accumulator = VerifierOpeningAccumulator::new(proof.trace_length.log_2());
        // Populate claims in the verifier accumulator
        for (key, (_, claim)) in &proof.opening_claims.0 {
            opening_accumulator
                .openings
                .insert(*key, (OpeningPoint::default(), *claim));
        }

        #[cfg(test)]
        let mut transcript = ProofTranscript::new(b"Jolt");
        #[cfg(not(test))]
        let transcript = ProofTranscript::new(b"Jolt");

        #[cfg(test)]
        {
            if let Some(debug_info) = _debug_info {
                transcript.compare_to(debug_info.transcript);
                opening_accumulator.compare_to(debug_info.opening_accumulator);
            }
        }

        let spartan_key = UniformSpartanKey::new(proof.trace_length.next_power_of_two());
        let one_hot_params =
            OneHotParams::from_config(&proof.one_hot_config, proof.bytecode_K, proof.ram_K);

        Ok(Self {
            trusted_advice_commitment,
            program_io,
            proof,
            preprocessing,
            transcript,
            opening_accumulator,
            spartan_key,
            one_hot_params,
            symbolic_recursion_proof: None,
        })
    }
}

// =============================================================================
// Generic verification methods (work with any OpeningAccumulator)
// =============================================================================

impl<
        'a,
        F: JoltField,
        PCS: CommitmentScheme<Field = F> + RecursionExt<F>,
        ProofTranscript: Transcript,
        A: OpeningAccumulator<F>,
    > TranspilableVerifier<'a, F, PCS, ProofTranscript, A>
{
    /// Create a TranspilableVerifier with a pre-configured opening accumulator.
    ///
    /// This constructor is used for symbolic transpilation where the accumulator
    /// is already populated with MleAst claims (or similar symbolic values).
    pub fn new_with_accumulator(
        preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
        proof: JoltProof<F, PCS, ProofTranscript>,
        program_io: JoltDevice,
        trusted_advice_commitment: Option<PCS::Commitment>,
        transcript: ProofTranscript,
        opening_accumulator: A,
        symbolic_recursion_proof: Option<RecursionProof<F, ProofTranscript, PCS>>,
    ) -> Self {
        let spartan_key = UniformSpartanKey::new(proof.trace_length.next_power_of_two());
        let one_hot_params =
            OneHotParams::from_config(&proof.one_hot_config, proof.bytecode_K, proof.ram_K);

        Self {
            trusted_advice_commitment,
            program_io,
            proof,
            preprocessing,
            transcript,
            opening_accumulator,
            spartan_key,
            one_hot_params,
            symbolic_recursion_proof,
        }
    }

    /// Verify the Jolt proof (stages 1-8 + recursion sumchecks).
    ///
    /// Stages 1-7 run all sumcheck stages. Stage 8 (verify_stage8_wo_pcs) runs
    /// claim collection, transcript bridge, and recursion sumchecks when
    /// symbolic_recursion_proof is present. PCS openings (Dory + Hyrax) are
    /// handled natively in Go.
    #[tracing::instrument(skip_all)]
    pub fn verify(mut self) -> Result<(), anyhow::Error>
    where
        F: allocative::Allocative,
        A: Default,
    {
        let _pprof_verify = pprof_scope!("verify");

        fiat_shamir_preamble(
            &self.program_io,
            self.proof.ram_K,
            self.proof.trace_length,
            &mut self.transcript,
        );

        // Append commitments to transcript
        for commitment in &self.proof.commitments {
            self.transcript.append_serializable(commitment);
        }
        // Append untrusted advice commitment to transcript
        if let Some(ref untrusted_advice_commitment) = self.proof.untrusted_advice_commitment {
            self.transcript
                .append_serializable(untrusted_advice_commitment);
        }
        // Append trusted advice commitment to transcript
        if let Some(ref trusted_advice_commitment) = self.trusted_advice_commitment {
            self.transcript
                .append_serializable(trusted_advice_commitment);
        }

        self.verify_stage1()?;
        self.verify_stage2()?;
        self.verify_stage3()?;
        self.verify_stage4()?;
        self.verify_stage5()?;
        let (bytecode_read_raf_params, booleanity_params) = self.verify_stage6a()?;
        self.verify_stage6b(bytecode_read_raf_params, booleanity_params)?;
        self.verify_stage7()?;
        // self.verify_stage8_with_recursion()?;  // PCS-specific — handled natively in Go
        if self.symbolic_recursion_proof.is_some() {
            self.verify_stage8_wo_pcs()?;
        }
        // self.verify_pcs();  // PCS openings handled natively in Go

        Ok(())
    }

    fn verify_stage1(&mut self) -> Result<(), anyhow::Error> {
        let uni_skip_params = verify_stage1_uni_skip(
            &self.proof.stage1_uni_skip_first_round_proof,
            &self.spartan_key,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 1 univariate skip first round")?;

        let spartan_outer_remaining = OuterRemainingSumcheckVerifier::new(
            self.spartan_key,
            self.proof.trace_length,
            uni_skip_params,
            &self.opening_accumulator,
        );

        let _r_stage1 = BatchedSumcheck::verify(
            &self.proof.stage1_sumcheck_proof,
            vec![&spartan_outer_remaining],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 1")?;

        Ok(())
    }

    fn verify_stage2(&mut self) -> Result<(), anyhow::Error> {
        let uni_skip_params = verify_stage2_uni_skip(
            &self.proof.stage2_uni_skip_first_round_proof,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 2 univariate skip first round")?;

        let spartan_product_virtual_remainder = ProductVirtualRemainderVerifier::new(
            self.proof.trace_length,
            uni_skip_params,
            &self.opening_accumulator,
        );
        let ram_raf_evaluation = RamRafEvaluationSumcheckVerifier::new(
            &self.program_io.memory_layout,
            &self.one_hot_params,
            &self.opening_accumulator,
        );
        let ram_read_write_checking = RamReadWriteCheckingVerifier::new(
            &self.opening_accumulator,
            &mut self.transcript,
            &self.one_hot_params,
            self.proof.trace_length,
            &self.proof.rw_config,
        );
        let ram_output_check =
            OutputSumcheckVerifier::new(self.proof.ram_K, &self.program_io, &mut self.transcript);
        let instruction_claim_reduction = InstructionLookupsClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let _r_stage2 = BatchedSumcheck::verify(
            &self.proof.stage2_sumcheck_proof,
            vec![
                &spartan_product_virtual_remainder
                    as &dyn SumcheckInstanceVerifier<F, ProofTranscript, A>,
                &ram_raf_evaluation,
                &ram_read_write_checking,
                &ram_output_check,
                &instruction_claim_reduction,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 2")?;

        Ok(())
    }

    fn verify_stage3(&mut self) -> Result<(), anyhow::Error> {
        let spartan_shift = ShiftSumcheckVerifier::new(
            self.proof.trace_length.log_2(),
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let spartan_instruction_input =
            InstructionInputSumcheckVerifier::new(&self.opening_accumulator, &mut self.transcript);
        let spartan_registers_claim_reduction = RegistersClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let _r_stage3 = BatchedSumcheck::verify(
            &self.proof.stage3_sumcheck_proof,
            vec![
                &spartan_shift as &dyn SumcheckInstanceVerifier<F, ProofTranscript, A>,
                &spartan_instruction_input,
                &spartan_registers_claim_reduction,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 3")?;

        Ok(())
    }

    fn verify_stage4(&mut self) -> Result<(), anyhow::Error> {
        verifier_accumulate_advice::<F, A>(
            self.proof.ram_K,
            &self.program_io,
            self.proof.untrusted_advice_commitment.is_some(),
            self.trusted_advice_commitment.is_some(),
            &mut self.opening_accumulator,
            &mut self.transcript,
            self.proof
                .rw_config
                .needs_single_advice_opening(self.proof.trace_length.log_2()),
        );

        let registers_read_write_checking = RegistersReadWriteCheckingVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
            &self.proof.rw_config,
        );

        // Get the program image words from the preprocessing
        let program_image_words = self.preprocessing.program.program_image_words();

        let ram_val_evaluation = RamValEvaluationSumcheckVerifier::new(
            &self.preprocessing.shared.program_meta,
            program_image_words,
            &self.program_io,
            self.proof.trace_length,
            self.proof.ram_K,
            self.proof.program_mode,
            &self.opening_accumulator,
        );
        let ram_val_final = ValFinalSumcheckVerifier::new(
            &self.preprocessing.shared.program_meta,
            program_image_words,
            &self.program_io,
            self.proof.trace_length,
            self.proof.ram_K,
            self.proof.program_mode,
            &self.opening_accumulator,
            &self.proof.rw_config,
        );

        let _r_stage4 = BatchedSumcheck::verify(
            &self.proof.stage4_sumcheck_proof,
            vec![
                &registers_read_write_checking
                    as &dyn SumcheckInstanceVerifier<F, ProofTranscript, A>,
                &ram_val_evaluation,
                &ram_val_final,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 4")?;

        Ok(())
    }

    fn verify_stage5(&mut self) -> Result<(), anyhow::Error> {
        let n_cycle_vars = self.proof.trace_length.log_2();
        let registers_val_evaluation =
            RegistersValEvaluationSumcheckVerifier::new(&self.opening_accumulator);
        let ram_ra_reduction = RamRaClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let lookups_read_raf = InstructionReadRafSumcheckVerifier::new(
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let _r_stage5 = BatchedSumcheck::verify(
            &self.proof.stage5_sumcheck_proof,
            vec![
                &registers_val_evaluation as &dyn SumcheckInstanceVerifier<F, ProofTranscript, A>,
                &ram_ra_reduction,
                &lookups_read_raf,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 5")?;

        Ok(())
    }

    fn verify_stage6a(
        &mut self,
    ) -> Result<(BytecodeReadRafSumcheckParams<F>, BooleanitySumcheckParams<F>), anyhow::Error> {
        let n_cycle_vars = self.proof.trace_length.log_2();
        let program_preprocessing = match self.proof.program_mode {
            ProgramMode::Committed => {
                // Ensure we have committed program commitments for committed mode.
                let _ = self.preprocessing.program.as_committed()?;
                None
            }
            ProgramMode::Full => self.preprocessing.program.full().map(|p| p.as_ref()),
        };
        let bytecode_read_raf = BytecodeReadRafAddressSumcheckVerifier::new(
            program_preprocessing,
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
            self.proof.program_mode,
        )?;
        let booleanity_params = BooleanitySumcheckParams::new(
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let booleanity = BooleanityAddressSumcheckVerifier::new(booleanity_params);

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript, A>> =
            vec![&bytecode_read_raf, &booleanity];

        let _r_stage6a = BatchedSumcheck::verify(
            &self.proof.stage6a_sumcheck_proof,
            instances,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 6a")?;
        Ok((bytecode_read_raf.into_params(), booleanity.into_params()))
    }

    fn verify_stage6b(
        &mut self,
        bytecode_read_raf_params: BytecodeReadRafSumcheckParams<F>,
        booleanity_params: BooleanitySumcheckParams<F>,
    ) -> Result<(), anyhow::Error> {
        let booleanity = BooleanityCycleSumcheckVerifier::new(booleanity_params);
        let ram_hamming_booleanity =
            HammingBooleanitySumcheckVerifier::new(&self.opening_accumulator);
        let ram_ra_virtual = RamRaVirtualSumcheckVerifier::new(
            self.proof.trace_length,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let lookups_ra_virtual = LookupsRaSumcheckVerifier::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let inc_reduction = IncClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let bytecode_read_raf = BytecodeReadRafCycleSumcheckVerifier::new(bytecode_read_raf_params);

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript, A>> = vec![
            &bytecode_read_raf,
            &ram_hamming_booleanity,
            &booleanity,
            &ram_ra_virtual,
            &lookups_ra_virtual,
            &inc_reduction,
        ];

        let _r_stage6b = BatchedSumcheck::verify(
            &self.proof.stage6b_sumcheck_proof,
            instances,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 6b")?;

        Ok(())
    }

    fn verify_stage7(&mut self) -> Result<(), anyhow::Error> {
        let hw_verifier = HammingWeightClaimReductionVerifier::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript, A>> =
            vec![&hw_verifier];

        let _r_stage7 = BatchedSumcheck::verify(
            &self.proof.stage7_sumcheck_proof,
            instances,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 7")?;

        Ok(())
    }

    /// Stage 8 without PCS: claim collection + transcript bridge + recursion sumchecks.
    ///
    /// This is the transpilable portion of stage 8. It:
    /// 1. Collects all opening claims from the accumulator
    /// 2. Feeds claims into the transcript and samples gamma powers
    /// 3. Samples gamma/delta challenges (after Dory IPA boundary)
    /// 4. Appends the dense commitment to the transcript
    /// 5. Runs recursion sumcheck verification (stages 1-5 of recursion SNARK)
    ///
    /// PCS operations (Dory verify_with_hint + Hyrax opening) are handled natively in Go.
    #[tracing::instrument(skip_all, name = "verify_stage8_wo_pcs")]
    fn verify_stage8_wo_pcs(&mut self) -> Result<(), anyhow::Error>
    where
        F: allocative::Allocative,
        A: Default,
    {
        let symbolic_proof = self
            .symbolic_recursion_proof
            .as_ref()
            .expect("verify_stage8_wo_pcs requires symbolic_recursion_proof");

        // === 1. CLAIM COLLECTION (from verify_stage8_with_pcs_hint) ===
        let claims = {
            let (opening_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(0),
                SumcheckId::HammingWeightClaimReduction,
            );
            let log_k_chunk = self.one_hot_params.log_k_chunk;
            let r_address_stage7 = &opening_point.r[..log_k_chunk];

            let mut polynomial_claims = Vec::new();

            // Dense polynomials: RamInc and RdInc (from IncClaimReduction in Stage 6)
            let (_, ram_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamInc,
                SumcheckId::IncClaimReduction,
            );
            let (_, rd_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RdInc,
                SumcheckId::IncClaimReduction,
            );

            let lagrange_factor: F = r_address_stage7.iter().map(|r| F::one() - *r).product();
            polynomial_claims.push((CommittedPolynomial::RamInc, ram_inc_claim * lagrange_factor));
            polynomial_claims.push((CommittedPolynomial::RdInc, rd_inc_claim * lagrange_factor));

            // Sparse polynomials: all RA polys (from HammingWeightClaimReduction)
            for i in 0..self.one_hot_params.instruction_d {
                let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::InstructionRa(i),
                    SumcheckId::HammingWeightClaimReduction,
                );
                polynomial_claims.push((CommittedPolynomial::InstructionRa(i), claim));
            }
            for i in 0..self.one_hot_params.bytecode_d {
                let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::BytecodeRa(i),
                    SumcheckId::HammingWeightClaimReduction,
                );
                polynomial_claims.push((CommittedPolynomial::BytecodeRa(i), claim));
            }
            for i in 0..self.one_hot_params.ram_d {
                let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::RamRa(i),
                    SumcheckId::HammingWeightClaimReduction,
                );
                polynomial_claims.push((CommittedPolynomial::RamRa(i), claim));
            }

            // Advice polynomials
            if let Some((advice_point, advice_claim)) = self
                .opening_accumulator
                .get_advice_opening(AdviceKind::Trusted, SumcheckId::AdviceClaimReduction)
            {
                let lagrange_factor =
                    compute_advice_lagrange_factor::<F>(&opening_point.r, &advice_point.r);
                polynomial_claims.push((
                    CommittedPolynomial::TrustedAdvice,
                    advice_claim * lagrange_factor,
                ));
            }

            if let Some((advice_point, advice_claim)) = self
                .opening_accumulator
                .get_advice_opening(AdviceKind::Untrusted, SumcheckId::AdviceClaimReduction)
            {
                let lagrange_factor =
                    compute_advice_lagrange_factor::<F>(&opening_point.r, &advice_point.r);
                polynomial_claims.push((
                    CommittedPolynomial::UntrustedAdvice,
                    advice_claim * lagrange_factor,
                ));
            }

            // Bytecode chunk polynomials
            if self.proof.program_mode == ProgramMode::Committed {
                let (bytecode_point, _) =
                    self.opening_accumulator.get_committed_polynomial_opening(
                        CommittedPolynomial::BytecodeChunk(0),
                        SumcheckId::BytecodeClaimReduction,
                    );
                let lagrange_factor =
                    compute_advice_lagrange_factor::<F>(&opening_point.r, &bytecode_point.r);

                let num_chunks = total_lanes().div_ceil(self.one_hot_params.k_chunk);
                for i in 0..num_chunks {
                    let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                        CommittedPolynomial::BytecodeChunk(i),
                        SumcheckId::BytecodeClaimReduction,
                    );
                    polynomial_claims.push((
                        CommittedPolynomial::BytecodeChunk(i),
                        claim * lagrange_factor,
                    ));
                }
            }

            // Program-image polynomial
            if self.proof.program_mode == ProgramMode::Committed {
                let (prog_point, prog_claim) =
                    self.opening_accumulator.get_committed_polynomial_opening(
                        CommittedPolynomial::ProgramImageInit,
                        SumcheckId::ProgramImageClaimReduction,
                    );
                let lagrange_factor =
                    compute_advice_lagrange_factor::<F>(&opening_point.r, &prog_point.r);
                polynomial_claims.push((
                    CommittedPolynomial::ProgramImageInit,
                    prog_claim * lagrange_factor,
                ));
            }

            let claims: Vec<F> = polynomial_claims.iter().map(|(_, c)| *c).collect();
            claims
        };

        // === 2. TRANSCRIPT: append claims + sample gamma powers ===
        self.transcript.append_scalars(&claims);
        self.transcript.debug_state("after_append_claims");
        let _gamma_powers: Vec<F> = self.transcript.challenge_scalar_powers(claims.len());
        self.transcript.debug_state("after_gamma_powers");

        // === 2.5. DORY IPA TRANSCRIPT REPLAY ===
        // Replay Dory IPA bytes through the Poseidon sponge via thread-local tunneling.
        // The ops were pre-loaded by main.rs before verify(). Each Absorb op sets
        // pending u16 var indices that PoseidonAstTranscript::append_bytes consumes.
        if let Some(mut ops) = crate::zkvm::dory_replay::take_dory_replay_ops() {
            while let Some(op) = ops.pop_front() {
                match op {
                    crate::zkvm::dory_replay::DoryReplayOp::Absorb { byte_size, var_indices } => {
                        crate::zkvm::dory_replay::set_pending_dory_absorb_indices(var_indices);
                        self.transcript.append_bytes(&vec![0u8; byte_size]);
                    }
                    crate::zkvm::dory_replay::DoryReplayOp::Squeeze => {
                        let _: F = self.transcript.challenge_scalar();
                    }
                }
            }
        }

        self.transcript.debug_state("after_dory_replay");

        // === 3. gamma/delta challenges ===
        let _gamma: F = self.transcript.challenge_scalar();
        let _delta: F = self.transcript.challenge_scalar();
        self.transcript.debug_state("after_gamma_delta");

        // === 4. Dense commitment ===
        // Use pre-serialized bytes from thread-local (set by main.rs) if available,
        // because the symbolic AstCommitment::default() doesn't contain the real bytes.
        // Format matches PoseidonTranscript::append_serializable: uncompressed + reversed.
        if let Some(dense_bytes) = crate::zkvm::dory_replay::take_dense_commitment_bytes() {
            self.transcript.append_bytes(&dense_bytes);
        } else {
            self.transcript
                .append_serializable(&self.proof.recursion_proof.dense_commitment);
        }

        self.transcript.debug_state("after_dense_commitment");

        // === 5. Build RecursionVerifier from metadata ===
        let metadata = &self.proof.stage10_recursion_metadata;
        let verifier_input = {
            let constraint_types = metadata.constraint_types.clone();
            let num_constraints = constraint_types.len();
            let num_constraints_padded = num_constraints.next_power_of_two();

            use crate::zkvm::recursion::constraints_sys::PolyType;
            let num_rows_unpadded = PolyType::NUM_TYPES * num_constraints_padded;
            let num_s_vars = (num_rows_unpadded as f64).log2().ceil() as usize;
            let num_constraint_vars = 11;
            let num_vars = num_s_vars + num_constraint_vars;

            RecursionVerifierInput {
                constraint_types,
                num_vars,
                num_constraint_vars,
                num_s_vars,
                num_constraints,
                num_constraints_padded,
                jagged_bijection: metadata.jagged_bijection.clone(),
                jagged_mapping: metadata.jagged_mapping.clone(),
                matrix_rows: metadata.matrix_rows.clone(),
                gt_exp_public_inputs: metadata.gt_exp_public_inputs.clone(),
                g1_scalar_mul_public_inputs: metadata.g1_scalar_mul_public_inputs.clone(),
                g2_scalar_mul_public_inputs: metadata.g2_scalar_mul_public_inputs.clone(),
            }
        };

        let recursion_verifier = RecursionVerifier::<Fq>::new(verifier_input);

        // === 6. RECURSION SUMCHECKS ===
        self.transcript.debug_state("before_recursion_sumchecks");
        let mut recursion_accumulator = A::default();
        recursion_verifier
            .verify_sumchecks::<F, ProofTranscript, A>(
                &symbolic_proof.stage1_proof,
                &symbolic_proof.stage2_proof,
                symbolic_proof.stage3_m_eval,
                &symbolic_proof.stage4_proof,
                &symbolic_proof.stage5_proof,
                &mut self.transcript,
                &mut recursion_accumulator,
            )
            .map_err(|e| anyhow::anyhow!("Recursion sumchecks failed: {e:?}"))?;

        Ok(())
    }

    /// Stage 8: PCS batch opening verification using a recursion hint (when supported by the PCS).
    #[allow(dead_code)]
    #[tracing::instrument(skip_all, name = "verify_stage8_with_pcs_hint")]
    fn verify_stage8_with_pcs_hint(
        &mut self,
        stage8_hint: &<PCS as RecursionExt<F>>::Hint,
    ) -> Result<(), anyhow::Error>
    where
        PCS: RecursionExt<F>,
    {
        // Get the unified opening point from HammingWeightClaimReduction
        // This contains (r_address_stage7 || r_cycle_stage6) in big-endian
        let (opening_point, polynomial_claims, claims) = {
            let _span = tracing::info_span!("stage8_collect_claims").entered();

            let (opening_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(0),
                SumcheckId::HammingWeightClaimReduction,
            );
            let log_k_chunk = self.one_hot_params.log_k_chunk;
            let r_address_stage7 = &opening_point.r[..log_k_chunk];

            // 1. Collect all (polynomial, claim) pairs
            let mut polynomial_claims = Vec::new();

            // Dense polynomials: RamInc and RdInc (from IncClaimReduction in Stage 6)
            let (_, ram_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamInc,
                SumcheckId::IncClaimReduction,
            );
            let (_, rd_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RdInc,
                SumcheckId::IncClaimReduction,
            );

            // Apply Lagrange factor for dense polys
            // Note: r_address is in big-endian, Lagrange factor uses ∏(1 - r_i)
            let lagrange_factor: F = r_address_stage7.iter().map(|r| F::one() - *r).product();

            polynomial_claims.push((CommittedPolynomial::RamInc, ram_inc_claim * lagrange_factor));
            polynomial_claims.push((CommittedPolynomial::RdInc, rd_inc_claim * lagrange_factor));

            // Sparse polynomials: all RA polys (from HammingWeightClaimReduction)
            for i in 0..self.one_hot_params.instruction_d {
                let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::InstructionRa(i),
                    SumcheckId::HammingWeightClaimReduction,
                );
                polynomial_claims.push((CommittedPolynomial::InstructionRa(i), claim));
            }
            for i in 0..self.one_hot_params.bytecode_d {
                let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::BytecodeRa(i),
                    SumcheckId::HammingWeightClaimReduction,
                );
                polynomial_claims.push((CommittedPolynomial::BytecodeRa(i), claim));
            }
            for i in 0..self.one_hot_params.ram_d {
                let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::RamRa(i),
                    SumcheckId::HammingWeightClaimReduction,
                );
                polynomial_claims.push((CommittedPolynomial::RamRa(i), claim));
            }

            // Advice polynomials (if present): fold into the Stage 8 batch via a Lagrange embedding
            // so the verifier samples the same gamma powers as the prover.
            if let Some((advice_point, advice_claim)) = self
                .opening_accumulator
                .get_advice_opening(AdviceKind::Trusted, SumcheckId::AdviceClaimReduction)
            {
                let lagrange_factor =
                    compute_advice_lagrange_factor::<F>(&opening_point.r, &advice_point.r);
                polynomial_claims.push((
                    CommittedPolynomial::TrustedAdvice,
                    advice_claim * lagrange_factor,
                ));
            }

            if let Some((advice_point, advice_claim)) = self
                .opening_accumulator
                .get_advice_opening(AdviceKind::Untrusted, SumcheckId::AdviceClaimReduction)
            {
                let lagrange_factor =
                    compute_advice_lagrange_factor::<F>(&opening_point.r, &advice_point.r);
                polynomial_claims.push((
                    CommittedPolynomial::UntrustedAdvice,
                    advice_claim * lagrange_factor,
                ));
            }

            // Bytecode chunk polynomials: committed in Bytecode context and embedded into the
            // main opening point by fixing the extra cycle variables to 0.
            if self.proof.program_mode == ProgramMode::Committed {
                let (bytecode_point, _) =
                    self.opening_accumulator.get_committed_polynomial_opening(
                        CommittedPolynomial::BytecodeChunk(0),
                        SumcheckId::BytecodeClaimReduction,
                    );
                #[cfg(test)]
                {
                    let log_t = opening_point.r.len() - log_k_chunk;
                    let log_k = bytecode_point.r.len() - log_k_chunk;
                    if log_k == log_t {
                        assert_eq!(
                            bytecode_point.r, opening_point.r,
                            "BytecodeChunk opening point must equal unified opening point when log_K == log_T"
                        );
                    }
                }
                let lagrange_factor =
                    compute_advice_lagrange_factor::<F>(&opening_point.r, &bytecode_point.r);

                let num_chunks = total_lanes().div_ceil(self.one_hot_params.k_chunk);
                for i in 0..num_chunks {
                    let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                        CommittedPolynomial::BytecodeChunk(i),
                        SumcheckId::BytecodeClaimReduction,
                    );
                    polynomial_claims.push((
                        CommittedPolynomial::BytecodeChunk(i),
                        claim * lagrange_factor,
                    ));
                }
            }

            // Program-image polynomial: opened by ProgramImageClaimReduction in Stage 6b.
            // Embed into the top-left block of the main matrix (same trick as advice).
            if self.proof.program_mode == ProgramMode::Committed {
                let (prog_point, prog_claim) =
                    self.opening_accumulator.get_committed_polynomial_opening(
                        CommittedPolynomial::ProgramImageInit,
                        SumcheckId::ProgramImageClaimReduction,
                    );
                let lagrange_factor =
                    compute_advice_lagrange_factor::<F>(&opening_point.r, &prog_point.r);
                polynomial_claims.push((
                    CommittedPolynomial::ProgramImageInit,
                    prog_claim * lagrange_factor,
                ));
            }

            let claims: Vec<F> = polynomial_claims.iter().map(|(_, c)| *c).collect();
            (opening_point, polynomial_claims, claims)
        };

        // 2. Sample gamma and compute powers for RLC
        let gamma_powers: Vec<F> = {
            let _span =
                tracing::info_span!("stage8_gamma_powers", num_claims = claims.len()).entered();
            self.transcript.append_scalars(&claims);
            self.transcript.challenge_scalar_powers(claims.len())
        };

        // Build state for computing joint commitment/claim
        let state = DoryOpeningState {
            opening_point: opening_point.r.clone(),
            gamma_powers: gamma_powers.clone(),
            polynomial_claims,
        };

        // Compute joint commitment: Σ γ_i · C_i
        // Use precomputed hint if available, otherwise compute directly
        let joint_commitment = {
            let _span = tracing::info_span!("stage8_joint_commitment").entered();
            if let Some(combine_hint) = &self.proof.stage8_combine_hint {
                // Use the precomputed hint (recursion-offloaded path)
                PCS::combine_with_hint_fq12(combine_hint)
            } else {
                // Build commitments map and compute directly
                let mut commitments_map = HashMap::new();
                for (polynomial, commitment) in all_committed_polynomials(&self.one_hot_params)
                    .into_iter()
                    .zip_eq(&self.proof.commitments)
                {
                    commitments_map.insert(polynomial, commitment.clone());
                }

                // Add advice commitments if they're part of the batch
                if let Some(ref commitment) = self.trusted_advice_commitment {
                    if state
                        .polynomial_claims
                        .iter()
                        .any(|(p, _)| *p == CommittedPolynomial::TrustedAdvice)
                    {
                        commitments_map
                            .insert(CommittedPolynomial::TrustedAdvice, commitment.clone());
                    }
                }
                if let Some(ref commitment) = self.proof.untrusted_advice_commitment {
                    if state
                        .polynomial_claims
                        .iter()
                        .any(|(p, _)| *p == CommittedPolynomial::UntrustedAdvice)
                    {
                        commitments_map
                            .insert(CommittedPolynomial::UntrustedAdvice, commitment.clone());
                    }
                }

                // Add program commitments in committed mode
                if self.proof.program_mode == ProgramMode::Committed {
                    if let Ok(committed) = self.preprocessing.program.as_committed() {
                        for (idx, commitment) in committed.bytecode_commitments.iter().enumerate() {
                            commitments_map
                                .entry(CommittedPolynomial::BytecodeChunk(idx))
                                .or_insert_with(|| commitment.clone());
                        }

                        // Add trusted program-image commitment if it's part of the batch
                        if state
                            .polynomial_claims
                            .iter()
                            .any(|(p, _)| *p == CommittedPolynomial::ProgramImageInit)
                        {
                            commitments_map.insert(
                                CommittedPolynomial::ProgramImageInit,
                                committed.program_image_commitment.clone(),
                            );
                        }
                    }
                }

                self.compute_joint_commitment(&mut commitments_map, &state)
            }
        };

        // Compute joint claim: Σ γ_i · claim_i
        let joint_claim: F = {
            let _span = tracing::info_span!("stage8_joint_claim").entered();
            gamma_powers
                .iter()
                .zip(claims.iter())
                .map(|(gamma, claim)| *gamma * claim)
                .sum()
        };

        // Verify opening using the hint-based PCS verifier.
        //
        // Important: This must remain transcript-compatible with the prover's `PCS::prove`
        // for end-to-end recursion correctness (gamma/delta are sampled from the same transcript).
        PCS::verify_with_hint(
            &self.proof.stage8_opening_proof,
            &self.preprocessing.generators,
            &mut self.transcript,
            &opening_point.r,
            &joint_claim,
            &joint_commitment,
            stage8_hint,
        )
        .context("Stage 8 (hint)")
    }

    /// Compute joint commitment for the batch opening.
    #[allow(dead_code)]
    fn compute_joint_commitment(
        &self,
        commitment_map: &mut HashMap<CommittedPolynomial, PCS::Commitment>,
        state: &DoryOpeningState<F>,
    ) -> PCS::Commitment {
        // Accumulate gamma coefficients per polynomial
        let mut rlc_map = HashMap::new();
        for (gamma, (poly, _claim)) in state
            .gamma_powers
            .iter()
            .zip(state.polynomial_claims.iter())
        {
            *rlc_map.entry(*poly).or_insert(F::zero()) += *gamma;
        }

        let (coeffs, commitments): (Vec<F>, Vec<PCS::Commitment>) = rlc_map
            .into_iter()
            .map(|(k, v)| (v, commitment_map.remove(&k).unwrap()))
            .unzip();

        PCS::combine_commitments(&commitments, &coeffs)
    }

    /// Verify Stage 8 with recursion proof.
    ///
    /// Not called from `verify()` — for Gnark transpilation, Stage 8 is handled
    /// natively in Go (combined_circuit.go + hyrax_verifier.go).
    /// Kept here as compilable reference of the full verification pipeline.
    #[allow(dead_code)]
    #[tracing::instrument(skip_all, name = "verify_stage8_with_recursion")]
    fn verify_stage8_with_recursion(&mut self) -> Result<(), anyhow::Error>
    where
        PCS: RecursionExt<F>,
        <PCS as RecursionExt<F>>::Hint: Clone,
    {
        let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE8);
        // 1. Verify Dory proof with hints
        let hint = {
            let _span = tracing::info_span!("stage8_clone_hint").entered();
            self.proof.stage9_pcs_hint.clone().ok_or_else(|| {
                anyhow::anyhow!("stage9_pcs_hint is required for recursion verification")
            })?
        };

        {
            let _span = tracing::info_span!("stage8_verify_dory_with_hint").entered();
            let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE8_DORY_PCS);
            self.verify_stage8_with_pcs_hint(&hint)?;
        }

        // 2. Extract data for RecursionVerifier
        let recursion_proof = &self.proof.recursion_proof;

        // Extract constraint counts from the opening claims in recursion_proof
        let (_num_gt_exp, _num_gt_mul, _num_g1_scalar_mul) = {
            let _span = tracing::info_span!(
                "stage8_count_constraint_types",
                num_claims = recursion_proof.opening_claims.len()
            )
            .entered();
            let mut num_gt_exp = 0;
            let mut num_gt_mul = 0;
            let mut num_g1_scalar_mul = 0;

            // Count constraint types based on the virtual polynomial types in opening claims
            use crate::zkvm::witness::{RecursionPoly, VirtualPolynomial};
            for key in recursion_proof.opening_claims.keys() {
                if let OpeningId::Polynomial(
                    PolynomialId::Virtual(VirtualPolynomial::Recursion(rec_poly)),
                    _,
                ) = key
                {
                    match rec_poly {
                        RecursionPoly::GtExp { instance, .. } => {
                            num_gt_exp = num_gt_exp.max(instance + 1);
                        }
                        RecursionPoly::GtMul { instance, .. } => {
                            num_gt_mul = num_gt_mul.max(instance + 1);
                        }
                        RecursionPoly::G1ScalarMul { instance, .. } => {
                            num_g1_scalar_mul = num_g1_scalar_mul.max(instance + 1);
                        }
                        _ => {} // G1Add, G2Add, G2ScalarMul - ignore for now
                    }
                }
            }
            (num_gt_exp, num_gt_mul, num_g1_scalar_mul)
        };

        // Use metadata from proof instead of placeholders
        let metadata = &self.proof.stage10_recursion_metadata;

        let verifier_input = {
            let _span = tracing::info_span!("stage8_build_verifier_input").entered();
            let constraint_types = metadata.constraint_types.clone();
            let num_constraints = constraint_types.len();
            let num_constraints_padded = num_constraints.next_power_of_two();

            // Calculate the constraint system parameters for the verifier input.
            //
            // IMPORTANT: These must match the recursion prover's matrix construction:
            // - `num_constraint_vars = 11` (uniform matrix compatible with packed GT exp)
            // - `num_rows_unpadded = PolyType::NUM_TYPES * num_constraints_padded`
            use crate::zkvm::recursion::constraints_sys::PolyType;
            let num_rows_unpadded = PolyType::NUM_TYPES * num_constraints_padded;
            let num_s_vars = (num_rows_unpadded as f64).log2().ceil() as usize;
            let num_constraint_vars = 11; // All constraints padded to 11 variables (zero padding)
            let num_vars = num_s_vars + num_constraint_vars;

            let jagged_bijection = metadata.jagged_bijection.clone();
            let jagged_mapping = metadata.jagged_mapping.clone();
            let matrix_rows = metadata.matrix_rows.clone();

            RecursionVerifierInput {
                constraint_types,
                num_vars,
                num_constraint_vars,
                num_s_vars,
                num_constraints,
                num_constraints_padded,
                jagged_bijection,
                jagged_mapping,
                matrix_rows,
                gt_exp_public_inputs: metadata.gt_exp_public_inputs.clone(),
                g1_scalar_mul_public_inputs: metadata.g1_scalar_mul_public_inputs.clone(),
                g2_scalar_mul_public_inputs: metadata.g2_scalar_mul_public_inputs.clone(),
            }
        };

        // 3. Verify recursion proof
        let recursion_verifier = {
            let _span = tracing::info_span!("stage8_create_recursion_verifier").entered();
            RecursionVerifier::<Fq>::new(verifier_input)
        };

        // Sample the same challenges from the main transcript that the prover did
        let _gamma: Fq = self.transcript.challenge_scalar();
        let _delta: Fq = self.transcript.challenge_scalar();

        type HyraxPCS = Hyrax<1, GrumpkinProjective>;

        // Use cached Hyrax setup from preprocessing (avoids 40ms regeneration)
        let hyrax_verifier_setup = &self.preprocessing.hyrax_recursion_setup;
        assert!(
            metadata.dense_num_vars <= MAX_RECURSION_DENSE_NUM_VARS,
            "dense_num_vars {} exceeds max {}",
            metadata.dense_num_vars,
            MAX_RECURSION_DENSE_NUM_VARS
        );

        // Add dense commitment to transcript (must match prover's order)
        self.transcript
            .append_serializable(&recursion_proof.dense_commitment);

        let verification_result = {
            let _span = tracing::info_span!("stage8_recursion_verifier_verify").entered();
            let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE8_RECURSION);
            recursion_verifier
                .verify::<ProofTranscript, HyraxPCS>(
                    recursion_proof,
                    &mut self.transcript,
                    &recursion_proof.dense_commitment,
                    hyrax_verifier_setup,
                )
                .map_err(|e| anyhow::anyhow!("Recursion verification failed: {e:?}"))?
        };

        if !verification_result {
            return Err(anyhow::anyhow!("Recursion proof verification failed"));
        }

        Ok(())
    }
}

// =============================================================================
// Preprocessing types
// =============================================================================

/// Shared preprocessing data between Full and Committed program modes.
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct SharedPreprocessing {
    pub program_meta: crate::zkvm::program::ProgramMetadata,
}

/// Verifier preprocessing data.
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::VerifierSetup,
    pub program: VerifierProgram<PCS>,
    pub shared: SharedPreprocessing,
    pub ram: crate::zkvm::ram::RAMPreprocessing,
    pub memory_layout: MemoryLayout,
    /// Cached Hyrax setup for recursion verification (avoids 40ms regeneration per verify)
    pub hyrax_recursion_setup: PedersenGenerators<GrumpkinProjective>,
}

impl<F, PCS> Serializable for JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
}

impl<F, PCS> JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub fn save_to_target_dir(&self, target_dir: &str) -> std::io::Result<()> {
        let filename = Path::new(target_dir).join("jolt_verifier_preprocessing.dat");
        let mut file = File::create(filename.as_path())?;
        let mut data = Vec::new();
        self.serialize_compressed(&mut data).unwrap();
        file.write_all(&data)?;
        Ok(())
    }

    pub fn read_from_target_dir(target_dir: &str) -> std::io::Result<Self> {
        let filename = Path::new(target_dir).join("jolt_verifier_preprocessing.dat");
        let mut file = File::open(filename.as_path())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(Self::deserialize_compressed(&*data).unwrap())
    }
}
