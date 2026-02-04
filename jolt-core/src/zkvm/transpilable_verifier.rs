//! Transpilable verifier for Jolt proof verification.
//!
//! This module provides a generic verifier that can be used with different accumulator types:
//! - `VerifierOpeningAccumulator<F>` for concrete verification
//! - `MleOpeningAccumulator` for symbolic transpilation to Gnark
//!
//! The verifier is generic over the accumulator type `A: OpeningAccumulator<F>`, which allows
//! it to be used for both purposes without code duplication.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::poly::commitment::commitment_scheme::{CommitmentScheme, RecursionExt};
use crate::subprotocols::sumcheck::BatchedSumcheck;
use crate::zkvm::bytecode::read_raf_checking::{
    BytecodeReadRafAddressSumcheckVerifier, BytecodeReadRafCycleSumcheckVerifier,
    BytecodeReadRafSumcheckParams,
};
use crate::zkvm::claim_reductions::RegistersClaimReductionSumcheckVerifier;
use crate::zkvm::config::{OneHotConfig, OneHotParams, ProgramMode};
use crate::zkvm::program::VerifierProgram;
use crate::zkvm::ram::val_final::ValFinalSumcheckVerifier;
use crate::zkvm::Serializable;
use crate::zkvm::{
    claim_reductions::{
        hamming_weight::HammingWeightClaimReductionVerifier,
        increments::IncClaimReductionSumcheckVerifier,
        InstructionLookupsClaimReductionSumcheckVerifier, RamRaClaimReductionSumcheckVerifier,
    },
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
    poly::opening_proof::{OpeningAccumulator, OpeningPoint, VerifierOpeningAccumulator},
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
use tracer::JoltDevice;

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
}

impl<'a, F: JoltField, PCS: CommitmentScheme<Field = F> + RecursionExt<F>, ProofTranscript: Transcript>
    TranspilableVerifier<'a, F, PCS, ProofTranscript, VerifierOpeningAccumulator<F>>
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
        })
    }
}

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
        }
    }

    /// Run verification stages 1-5.
    ///
    /// This verifies all stages that can be transpiled to Gnark.
    /// Stages 6, 7, and 8 are PCS-specific and are not included here.
    #[tracing::instrument(skip_all)]
    pub fn verify(mut self) -> Result<(), anyhow::Error> {
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
        // Note: Stages 6, 7, 8 use VerifierOpeningAccumulator-specific methods
        // and are not transpilable to Gnark. They handle PCS (Hyrax/Dory) verification.

        Ok(())
    }

    /// Run only Stage 1 verification for incremental debugging.
    ///
    /// This allows testing the transpiled circuit with just Stage 1
    /// before adding more stages.
    #[tracing::instrument(skip_all)]
    pub fn verify_stage1_only(mut self) -> Result<(), anyhow::Error> {
        let _pprof_verify = pprof_scope!("verify_stage1_only");

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
        // Stages 2-5 commented out for incremental debugging

        Ok(())
    }

    /// Run Stages 1-2 verification for incremental debugging.
    #[tracing::instrument(skip_all)]
    pub fn verify_stages_1_2(mut self) -> Result<(), anyhow::Error> {
        let _pprof_verify = pprof_scope!("verify_stages_1_2");

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

        Ok(())
    }

    /// Run Stages 1-3 verification for incremental debugging.
    #[tracing::instrument(skip_all)]
    pub fn verify_stages_1_3(mut self) -> Result<(), anyhow::Error> {
        let _pprof_verify = pprof_scope!("verify_stages_1_3");

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

        Ok(())
    }

    /// Run Stages 1-4 verification for incremental debugging.
    #[tracing::instrument(skip_all)]
    pub fn verify_stages_1_4(mut self) -> Result<(), anyhow::Error> {
        let _pprof_verify = pprof_scope!("verify_stages_1_4");

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

        Ok(())
    }

    /// Run Stages 1-5 verification (all sumcheck stages, excluding Hyrax opening).
    #[tracing::instrument(skip_all)]
    pub fn verify_stages_1_5(mut self) -> Result<(), anyhow::Error> {
        let _pprof_verify = pprof_scope!("verify_stages_1_5");

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

    /// Run Stages 1-6a verification (sumcheck stages, excluding Hyrax opening).
    #[tracing::instrument(skip_all)]
    pub fn verify_stages_1_6a(mut self) -> Result<(), anyhow::Error> {
        let _pprof_verify = pprof_scope!("verify_stages_1_6a");

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
        self.verify_stage6a()?;

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

    /// Run Stages 1-6b verification (sumcheck stages, excluding Hyrax opening).
    #[tracing::instrument(skip_all)]
    pub fn verify_stages_1_6b(mut self) -> Result<(), anyhow::Error> {
        let _pprof_verify = pprof_scope!("verify_stages_1_6b");

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

        Ok(())
    }

    fn verify_stage6b(
        &mut self,
        bytecode_read_raf_params: BytecodeReadRafSumcheckParams<F>,
        booleanity_params: BooleanitySumcheckParams<F>,
    ) -> Result<(), anyhow::Error> {
        // Initialize Stage 6b cycle verifiers
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

        // Note: In Full mode, we skip BytecodeClaimReduction, AdviceClaimReduction, and
        // ProgramImageClaimReduction which are only used in Committed mode.
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

    /// Run Stages 1-7 verification (all sumcheck stages, excluding Hyrax opening).
    #[tracing::instrument(skip_all)]
    pub fn verify_stages_1_7(mut self) -> Result<(), anyhow::Error> {
        let _pprof_verify = pprof_scope!("verify_stages_1_7");

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

        Ok(())
    }

    fn verify_stage7(&mut self) -> Result<(), anyhow::Error> {
        // Create verifier for HammingWeightClaimReduction
        let hw_verifier = HammingWeightClaimReductionVerifier::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // Note: In Full mode, we skip BytecodeClaimReduction and AdviceClaimReduction
        // which are only used in Committed mode.
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

    // =========================================================================
    // Stage 8 is NOT transpilable to Gnark (Dory/Hyrax opening verification).
    // For Gnark transpilation, this would be replaced by native Gnark operations.
    // =========================================================================
}

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
