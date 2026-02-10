//! Unified verifier for the recursion SNARK protocol
//!
//! This module provides a high-level verifier that verifies:
//! - Stage 1: Packed GT exp sumcheck
//! - Stage 2: Batched constraint sumchecks (shift + claim reduction + remaining constraints)
//! - Stage 3: Virtualization direct evaluation
//! - Stage 4: Jagged transform sumcheck
//! - Stage 5: Jagged assist sumcheck
//!
//! The verifier returns an opening accumulator for PCS verification.
//!
//! ## Transpilation Support
//!
//! The `verify_sumchecks_symbolic` method is fully generic over `F: JoltField` and
//! `A: OpeningAccumulator<F>`, allowing it to be called with either:
//! - `F = Fq, A = VerifierOpeningAccumulator<Fq>` for real verification
//! - `F = MleAst, A = MleOpeningAccumulator` for symbolic transpilation to gnark circuits
//!
//! For symbolic execution, curve constants (Fq12 values) are converted to the target
//! field type F using `convert_fq_to_field`.

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        opening_proof::{OpeningAccumulator, SumcheckId, VerifierOpeningAccumulator},
    },
    transcripts::Transcript,
    zkvm::witness::VirtualPolynomial,
};
use ark_bn254::Fq;
use ark_std::Zero;

use super::{
    bijection::{ConstraintMapping, VarCountJaggedBijection},
    constraints_sys::ConstraintType,
    curve::Bn254Recursion,
    recursion_prover::RecursionProof,
    stage1::gt_exp::{PackedGtExpParams, PackedGtExpPublicInputs, PackedGtExpVerifier},
    stage2::{
        g1_add::G1AddParams,
        g1_scalar_mul::G1ScalarMulPublicInputs,
        g2_add::G2AddParams,
        g2_scalar_mul::G2ScalarMulPublicInputs,
        gt_mul::{GtMulParams, GtMulVerifier, GtMulVerifierSpec},
        packed_gt_exp_reduction::{
            PackedGtExpClaimReductionParams, PackedGtExpClaimReductionVerifier,
        },
        shift_rho::{ShiftRhoParams, ShiftRhoVerifier},
        shift_scalar_mul::{
            g1_shift_params, g2_shift_params, ShiftG1ScalarMulVerifier, ShiftG2ScalarMulVerifier,
        },
    },
    stage3::virtualization::{
        extract_virtual_claims_from_accumulator, DirectEvaluationParams, DirectEvaluationVerifier,
    },
    stage4::jagged::{JaggedSumcheckParams, JaggedSumcheckVerifier},
    stage5::jagged_assist::JaggedAssistVerifier,
};
use crate::subprotocols::{sumcheck::BatchedSumcheck, sumcheck_verifier::SumcheckInstanceVerifier};

use jolt_platform::{end_cycle_tracking, start_cycle_tracking};

// Cycle-marker labels must be static strings: the tracer keys markers by the guest string pointer.
const CYCLE_RECURSION_STAGE1: &str = "jolt_recursion_stage1";
const CYCLE_RECURSION_STAGE2: &str = "jolt_recursion_stage2";
const CYCLE_RECURSION_STAGE3: &str = "jolt_recursion_stage3";
const CYCLE_RECURSION_STAGE4: &str = "jolt_recursion_stage4";
const CYCLE_RECURSION_STAGE5: &str = "jolt_recursion_stage5";
const CYCLE_RECURSION_PCS_OPENING: &str = "jolt_recursion_pcs_opening";

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

/// Input required by the verifier
#[derive(Clone, Debug)]
pub struct RecursionVerifierInput {
    /// Constraint types to verify
    pub constraint_types: Vec<ConstraintType>,
    /// Number of variables in the constraint system
    pub num_vars: usize,
    /// Number of constraint variables (x variables) in the matrix
    pub num_constraint_vars: usize,
    /// Number of s-variables for virtualization
    pub num_s_vars: usize,
    /// Total number of constraints
    pub num_constraints: usize,
    /// Padded number of constraints
    pub num_constraints_padded: usize,
    /// Jagged bijection for Stage 3
    pub jagged_bijection: VarCountJaggedBijection,
    /// Mapping for decoding polynomial indices to matrix rows
    pub jagged_mapping: ConstraintMapping,
    /// Precomputed matrix row indices for each polynomial index
    pub matrix_rows: Vec<usize>,
    /// Public inputs for packed GT exp (base Fq12 and scalar bits)
    pub gt_exp_public_inputs: Vec<PackedGtExpPublicInputs>,
    /// Public inputs for G1 scalar multiplication (scalar per G1ScalarMul constraint)
    pub g1_scalar_mul_public_inputs: Vec<G1ScalarMulPublicInputs>,
    /// Public inputs for G2 scalar multiplication (scalar per G2ScalarMul constraint)
    pub g2_scalar_mul_public_inputs: Vec<G2ScalarMulPublicInputs>,
}

/// Unified verifier for the recursion SNARK
pub struct RecursionVerifier<F: JoltField = Fq> {
    /// Input parameters for verification
    input: RecursionVerifierInput,
    /// Phantom data for the field type
    _phantom: std::marker::PhantomData<F>,
}

impl RecursionVerifier<Fq> {
    /// Create a new recursion verifier
    pub fn new(input: RecursionVerifierInput) -> Self {
        Self {
            input,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get a reference to the verifier input (useful for transpilation)
    pub fn input(&self) -> &RecursionVerifierInput {
        &self.input
    }

    /// Verify the full two-stage recursion proof and PCS opening
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify")]
    pub fn verify<T: Transcript, PCS: CommitmentScheme<Field = Fq>>(
        &self,
        proof: &RecursionProof<Fq, T, PCS>,
        transcript: &mut T,
        matrix_commitment: &PCS::Commitment,
        verifier_setup: &PCS::VerifierSetup,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        // Initialize opening accumulator
        let mut accumulator = VerifierOpeningAccumulator::<Fq>::new(self.input.num_vars);

        // Populate accumulator with opening claims from proof
        for (key, value) in &proof.opening_claims {
            accumulator.openings.insert(*key, value.clone());
        }

        // Run all sumcheck stages (generic over accumulator type)
        self.verify_sumchecks_generic(
            &proof.stage1_proof,
            &proof.stage2_proof,
            proof.stage3_m_eval,
            &proof.stage4_proof,
            &proof.stage5_proof,
            transcript,
            &mut accumulator,
        )?;

        // ============ PCS OPENING VERIFICATION ============
        let _cycle_pcs = CycleMarkerGuard::new(CYCLE_RECURSION_PCS_OPENING);
        tracing::info_span!("verify_recursion_pcs_opening").in_scope(|| {
            tracing::info!("Verifying PCS opening proof");
            // Verify opening proof using PCS
            accumulator.verify_single::<T, PCS>(
                &proof.opening_proof,
                matrix_commitment.clone(),
                verifier_setup,
                transcript,
            )
        })?;
        drop(_cycle_pcs);

        Ok(true)
    }

    /// Verify all sumcheck stages (1-5) without PCS opening verification.
    ///
    /// This method is generic over `A: OpeningAccumulator<Fq>`, enabling:
    /// - Real verification with `VerifierOpeningAccumulator<Fq>`
    /// - Symbolic transpilation with `MleOpeningAccumulator`
    ///
    /// For transpilation, this records all sumcheck verification constraints
    /// to the AST without performing PCS opening verification (which is done
    /// natively in gnark via Hyrax).
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_sumchecks_generic")]
    pub fn verify_sumchecks_generic<T: Transcript, A: OpeningAccumulator<Fq>>(
        &self,
        stage1_proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<Fq, T>,
        stage2_proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<Fq, T>,
        stage3_m_eval: Fq,
        stage4_proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<Fq, T>,
        stage5_proof: &super::stage5::jagged_assist::JaggedAssistProof<Fq, T>,
        transcript: &mut T,
        accumulator: &mut A,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Delegate to the fully generic symbolic method with F = Fq
        self.verify_sumchecks_symbolic::<Fq, T, A>(
            stage1_proof,
            stage2_proof,
            stage3_m_eval,
            stage4_proof,
            stage5_proof,
            transcript,
            accumulator,
        )
    }

    /// Verify all sumcheck stages (1-5) - fully generic over field type F.
    ///
    /// This method is generic over:
    /// - `F: JoltField` - the field type (Fq for real verification, MleAst for transpilation)
    /// - `T: Transcript` - the transcript type
    /// - `A: OpeningAccumulator<F>` - the accumulator type
    ///
    /// For symbolic transpilation with `F = MleAst`:
    /// - Sumcheck proofs contain symbolic MleAst elements
    /// - Curve constants (Fq12 values) are converted to F via `convert_fq_to_field`
    /// - The accumulator records all constraints for AST generation
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_sumchecks_symbolic")]
    pub fn verify_sumchecks_symbolic<F: JoltField + allocative::Allocative, T: Transcript, A: OpeningAccumulator<F>>(
        &self,
        stage1_proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<F, T>,
        stage2_proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<F, T>,
        stage3_m_eval: F,
        stage4_proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<F, T>,
        stage5_proof: &super::stage5::jagged_assist::JaggedAssistProof<F, T>,
        transcript: &mut T,
        accumulator: &mut A,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // ============ STAGE 1: Packed GT Exp ============
        let _cycle_stage1 = CycleMarkerGuard::new(CYCLE_RECURSION_STAGE1);
        let _r_stage1_packed = tracing::info_span!("verify_recursion_stage1").in_scope(|| {
            tracing::info!("Verifying Stage 1: Packed GT exp sumcheck");
            self.verify_stage1_symbolic::<F, T, A>(stage1_proof, transcript, accumulator)
        })?;
        drop(_cycle_stage1);

        // ============ STAGE 2: Batched Constraint Sumchecks ============
        let _cycle_stage2 = CycleMarkerGuard::new(CYCLE_RECURSION_STAGE2);
        let r_x = tracing::info_span!("verify_recursion_stage2").in_scope(|| {
            tracing::info!("Verifying Stage 2: Batched constraint sumchecks");
            self.verify_stage2_symbolic::<F, T, A>(stage2_proof, transcript, accumulator)
        })?;
        drop(_cycle_stage2);

        // Debug hook: allow stopping after Stage 2 to isolate failures.
        #[cfg(test)]
        if std::env::var("JOLT_RECURSION_STOP_AFTER_STAGE2").is_ok() {
            return Ok(());
        }

        // ============ STAGE 3: Virtualization Direct Evaluation ============
        let _cycle_stage3 = CycleMarkerGuard::new(CYCLE_RECURSION_STAGE3);
        let r_s = tracing::info_span!("verify_recursion_stage3").in_scope(|| {
            tracing::info!("Verifying Stage 3: Virtualization direct evaluation");
            self.verify_stage3_symbolic::<F, T, A>(transcript, accumulator, &r_x, stage3_m_eval)
        })?;
        drop(_cycle_stage3);

        // ============ STAGE 4: Jagged Transform Sumcheck ============
        let _cycle_stage4 = CycleMarkerGuard::new(CYCLE_RECURSION_STAGE4);
        let r_dense = tracing::info_span!("verify_recursion_stage4").in_scope(|| {
            tracing::info!("Verifying Stage 4: Jagged transform sumcheck");
            self.verify_stage4_symbolic::<F, T, A>(
                stage4_proof,
                stage5_proof,
                transcript,
                accumulator,
                &r_s,
            )
        })?;
        drop(_cycle_stage4);

        // ============ STAGE 5: Jagged Assist ============
        let _cycle_stage5 = CycleMarkerGuard::new(CYCLE_RECURSION_STAGE5);
        tracing::info_span!("verify_recursion_stage5").in_scope(|| {
            tracing::info!("Verifying Stage 5: Jagged assist");
            self.verify_stage5_symbolic::<F, T, A>(
                stage5_proof,
                transcript,
                accumulator,
                &r_dense,
                &r_x,
            )
        })?;
        drop(_cycle_stage5);

        Ok(())
    }

    /// Verify Stage 1: Packed GT exp sumcheck (generic over accumulator)
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage1_generic")]
    fn verify_stage1_generic<T: Transcript, A: OpeningAccumulator<Fq>>(
        &self,
        proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<Fq, T>,
        transcript: &mut T,
        accumulator: &mut A,
    ) -> Result<Vec<<Fq as crate::field::JoltField>::Challenge>, Box<dyn std::error::Error>> {
        self.verify_stage1_symbolic::<Fq, T, A>(proof, transcript, accumulator)
    }

    /// Verify Stage 1: Packed GT exp sumcheck - fully generic over F
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage1_symbolic")]
    fn verify_stage1_symbolic<F: JoltField, T: Transcript, A: OpeningAccumulator<F>>(
        &self,
        proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<F, T>,
        transcript: &mut T,
        accumulator: &mut A,
    ) -> Result<Vec<F::Challenge>, Box<dyn std::error::Error>> {
        if self.input.gt_exp_public_inputs.is_empty() {
            return Err("No PackedGtExp constraints to verify in Stage 1".into());
        }

        let params = PackedGtExpParams::new();
        // Use PackedGtExpVerifier<F> which is already generic over F
        let verifier: PackedGtExpVerifier<F> =
            PackedGtExpVerifier::new(params, self.input.gt_exp_public_inputs.clone(), transcript);

        let r_stage1 = BatchedSumcheck::verify(proof, vec![&verifier], accumulator, transcript)?;
        Ok(r_stage1)
    }

    /// Verify Stage 2: Batched constraint sumchecks - fully generic over F.
    ///
    /// NOTE: This is currently a work-in-progress. Some Stage 2 verifiers (GtMul, G1ScalarMul, G2ScalarMul, etc.)
    /// All Stage 2 verifiers are now generic over F and can be used for both
    /// concrete (Fq) and symbolic (MleAst) execution.
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage2_symbolic")]
    fn verify_stage2_symbolic<F: JoltField + allocative::Allocative, T: Transcript, A: OpeningAccumulator<F>>(
        &self,
        proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<F, T>,
        transcript: &mut T,
        accumulator: &mut A,
    ) -> Result<Vec<F::Challenge>, Box<dyn std::error::Error>> {
        let env_flag_default = |name: &str, default: bool| -> bool {
            std::env::var(name)
                .ok()
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(default)
        };
        let enable_shift_rho = env_flag_default("JOLT_RECURSION_ENABLE_SHIFT_RHO", true);
        let enable_claim_reduction = env_flag_default("JOLT_RECURSION_ENABLE_PGX_REDUCTION", true);
        let enable_shift_g1_scalar_mul =
            env_flag_default("JOLT_RECURSION_ENABLE_SHIFT_G1_SCALAR_MUL", true);
        let enable_shift_g2_scalar_mul =
            env_flag_default("JOLT_RECURSION_ENABLE_SHIFT_G2_SCALAR_MUL", true);

        let mut verifiers: Vec<Box<dyn SumcheckInstanceVerifier<F, T, A>>> = Vec::new();

        // Count constraints by type and collect per-type sequential indices
        let mut num_gt_exp = 0usize;
        let mut num_gt_mul = 0usize;
        let mut num_g1_scalar_mul = 0usize;
        let mut num_g2_scalar_mul = 0usize;
        let mut num_g1_add = 0usize;
        let mut num_g2_add = 0usize;

        let mut gt_mul_indices = Vec::new();
        let mut g1_scalar_mul_base_points = Vec::new();
        let mut g1_scalar_mul_indices = Vec::new();
        let mut g2_scalar_mul_base_points = Vec::new();
        let mut g2_scalar_mul_indices = Vec::new();
        let mut g1_add_indices = Vec::new();
        let mut g2_add_indices = Vec::new();

        for constraint in self.input.constraint_types.iter() {
            match constraint {
                ConstraintType::PackedGtExp => {
                    num_gt_exp += 1;
                }
                ConstraintType::GtMul => {
                    gt_mul_indices.push(num_gt_mul);
                    num_gt_mul += 1;
                }
                ConstraintType::G1ScalarMul { base_point } => {
                    g1_scalar_mul_base_points.push(*base_point);
                    g1_scalar_mul_indices.push(num_g1_scalar_mul);
                    num_g1_scalar_mul += 1;
                }
                ConstraintType::G2ScalarMul { base_point } => {
                    g2_scalar_mul_base_points.push(*base_point);
                    g2_scalar_mul_indices.push(num_g2_scalar_mul);
                    num_g2_scalar_mul += 1;
                }
                ConstraintType::G1Add => {
                    g1_add_indices.push(num_g1_add);
                    num_g1_add += 1;
                }
                ConstraintType::G2Add => {
                    g2_add_indices.push(num_g2_add);
                    num_g2_add += 1;
                }
            }
        }

        // Packed GT exp auxiliary subprotocols (shift rho + claim reduction)
        if num_gt_exp > 0 {
            let claim_indices: Vec<usize> = (0..num_gt_exp).collect();
            if enable_shift_rho {
                let shift_verifier = ShiftRhoVerifier::<F>::new(
                    ShiftRhoParams::new(num_gt_exp),
                    claim_indices.clone(),
                    transcript,
                );
                verifiers.push(Box::new(shift_verifier));
            }

            if enable_claim_reduction {
                let reduction_verifier = PackedGtExpClaimReductionVerifier::<F>::new(
                    PackedGtExpClaimReductionParams::new(2 * num_gt_exp),
                    claim_indices,
                    transcript,
                );
                verifiers.push(Box::new(reduction_verifier));
            }
        }

        // GT mul - now generic over F
        if num_gt_mul > 0 {
            use super::curve::RecursionCurve;
            let params = GtMulParams::new(num_gt_mul);
            let g_mle = Bn254Recursion::g_mle();
            let spec = GtMulVerifierSpec::<F>::new_from_fq(params, &g_mle);
            let verifier = GtMulVerifier::<F>::from_spec(spec, gt_mul_indices, transcript);
            verifiers.push(Box::new(verifier));
        }

        // G1 scalar mul shift - generic over F
        if enable_shift_g1_scalar_mul && num_g1_scalar_mul > 0 {
            let mut pairs: Vec<(VirtualPolynomial, VirtualPolynomial)> =
                Vec::with_capacity(num_g1_scalar_mul * 2);
            for i in 0..num_g1_scalar_mul {
                pairs.push((
                    VirtualPolynomial::g1_scalar_mul_xa(i),
                    VirtualPolynomial::g1_scalar_mul_xa_next(i),
                ));
                pairs.push((
                    VirtualPolynomial::g1_scalar_mul_ya(i),
                    VirtualPolynomial::g1_scalar_mul_ya_next(i),
                ));
            }
            verifiers.push(Box::new(ShiftG1ScalarMulVerifier::<F>::new(
                g1_shift_params(pairs.len()),
                pairs,
                transcript,
            )));
        }

        // G1 scalar mul - now generic over F
        if num_g1_scalar_mul > 0 {
            use super::stage2::g1_scalar_mul::{
                G1ScalarMulParams, G1ScalarMulVerifier, G1ScalarMulVerifierSpec,
            };

            let params = G1ScalarMulParams::new(num_g1_scalar_mul);
            debug_assert_eq!(
                self.input.g1_scalar_mul_public_inputs.len(),
                num_g1_scalar_mul,
                "RecursionVerifierInput.g1_scalar_mul_public_inputs must match number of G1ScalarMul constraints"
            );
            let spec = G1ScalarMulVerifierSpec::new(
                params,
                self.input.g1_scalar_mul_public_inputs.clone(),
                g1_scalar_mul_base_points,
            );
            let verifier = G1ScalarMulVerifier::<F>::from_spec(spec, g1_scalar_mul_indices, transcript);
            verifiers.push(Box::new(verifier));
        }

        // G2 scalar mul shift - generic over F
        if enable_shift_g2_scalar_mul && num_g2_scalar_mul > 0 {
            let mut pairs: Vec<(VirtualPolynomial, VirtualPolynomial)> =
                Vec::with_capacity(num_g2_scalar_mul * 4);
            for i in 0..num_g2_scalar_mul {
                pairs.push((
                    VirtualPolynomial::g2_scalar_mul_xa_c0(i),
                    VirtualPolynomial::g2_scalar_mul_xa_next_c0(i),
                ));
                pairs.push((
                    VirtualPolynomial::g2_scalar_mul_xa_c1(i),
                    VirtualPolynomial::g2_scalar_mul_xa_next_c1(i),
                ));
                pairs.push((
                    VirtualPolynomial::g2_scalar_mul_ya_c0(i),
                    VirtualPolynomial::g2_scalar_mul_ya_next_c0(i),
                ));
                pairs.push((
                    VirtualPolynomial::g2_scalar_mul_ya_c1(i),
                    VirtualPolynomial::g2_scalar_mul_ya_next_c1(i),
                ));
            }
            verifiers.push(Box::new(ShiftG2ScalarMulVerifier::<F>::new(
                g2_shift_params(pairs.len()),
                pairs,
                transcript,
            )));
        }

        // G2 scalar mul - now generic over F
        if num_g2_scalar_mul > 0 {
            use super::stage2::g2_scalar_mul::{
                G2ScalarMulParams, G2ScalarMulVerifier, G2ScalarMulVerifierSpec,
            };

            let params = G2ScalarMulParams::new(num_g2_scalar_mul);
            debug_assert_eq!(
                self.input.g2_scalar_mul_public_inputs.len(),
                num_g2_scalar_mul,
                "RecursionVerifierInput.g2_scalar_mul_public_inputs must match number of G2ScalarMul constraints"
            );
            let spec = G2ScalarMulVerifierSpec::new(
                params,
                self.input.g2_scalar_mul_public_inputs.clone(),
                g2_scalar_mul_base_points,
            );
            let verifier = G2ScalarMulVerifier::<F>::from_spec(spec, g2_scalar_mul_indices, transcript);
            verifiers.push(Box::new(verifier));
        }

        // G1 add - generic over F (via macro)
        if num_g1_add > 0 {
            use super::stage2::g1_add::{G1AddVerifier, G1AddVerifierSpec};
            let params = G1AddParams::new(num_g1_add);
            let spec = G1AddVerifierSpec::<F>::new(params);
            let verifier = G1AddVerifier::<F>::from_spec(spec, g1_add_indices, transcript);
            verifiers.push(Box::new(verifier));
        }

        // G2 add - generic over F (via macro)
        if num_g2_add > 0 {
            use super::stage2::g2_add::{G2AddVerifier, G2AddVerifierSpec};
            let params = G2AddParams::new(num_g2_add);
            let spec = G2AddVerifierSpec::<F>::new(params);
            let verifier = G2AddVerifier::<F>::from_spec(spec, g2_add_indices, transcript);
            verifiers.push(Box::new(verifier));
        }

        if verifiers.is_empty() {
            return Err("No constraints to verify in Stage 2".into());
        }

        let verifier_refs: Vec<&dyn SumcheckInstanceVerifier<F, T, A>> =
            verifiers.iter().map(|v| &**v).collect();
        let r_x = BatchedSumcheck::verify(proof, verifier_refs, accumulator, transcript)?;
        Ok(r_x)
    }

    /// Verify Stage 2: Batched constraint sumchecks (generic over accumulator).
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage2_generic")]
    fn verify_stage2_generic<T: Transcript, A: OpeningAccumulator<Fq>>(
        &self,
        proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<Fq, T>,
        transcript: &mut T,
        accumulator: &mut A,
    ) -> Result<Vec<<Fq as crate::field::JoltField>::Challenge>, Box<dyn std::error::Error>> {
        let env_flag_default = |name: &str, default: bool| -> bool {
            std::env::var(name)
                .ok()
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(default)
        };
        let enable_shift_rho = env_flag_default("JOLT_RECURSION_ENABLE_SHIFT_RHO", true);
        let enable_shift_g1_scalar_mul =
            env_flag_default("JOLT_RECURSION_ENABLE_SHIFT_G1_SCALAR_MUL", true);
        let enable_shift_g2_scalar_mul =
            env_flag_default("JOLT_RECURSION_ENABLE_SHIFT_G2_SCALAR_MUL", true);
        let enable_claim_reduction = env_flag_default("JOLT_RECURSION_ENABLE_PGX_REDUCTION", true);

        let mut verifiers: Vec<Box<dyn SumcheckInstanceVerifier<Fq, T, A>>> = Vec::new();

        // Count constraints by type and collect per-type sequential indices (matching the extractor).
        let mut num_gt_exp = 0usize;
        let mut num_gt_mul = 0usize;
        let mut num_g1_scalar_mul = 0usize;
        let mut num_g2_scalar_mul = 0usize;
        let mut num_g1_add = 0usize;
        let mut num_g2_add = 0usize;

        let mut gt_mul_indices = Vec::new();
        let mut g1_scalar_mul_base_points = Vec::new();
        let mut g1_scalar_mul_indices = Vec::new();
        let mut g2_scalar_mul_base_points = Vec::new();
        let mut g2_scalar_mul_indices = Vec::new();
        let mut g1_add_indices = Vec::new();
        let mut g2_add_indices = Vec::new();

        for constraint in self.input.constraint_types.iter() {
            match constraint {
                ConstraintType::PackedGtExp => {
                    num_gt_exp += 1;
                }
                ConstraintType::GtMul => {
                    gt_mul_indices.push(num_gt_mul);
                    num_gt_mul += 1;
                }
                ConstraintType::G1ScalarMul { base_point } => {
                    g1_scalar_mul_base_points.push(*base_point);
                    g1_scalar_mul_indices.push(num_g1_scalar_mul);
                    num_g1_scalar_mul += 1;
                }
                ConstraintType::G2ScalarMul { base_point } => {
                    g2_scalar_mul_base_points.push(*base_point);
                    g2_scalar_mul_indices.push(num_g2_scalar_mul);
                    num_g2_scalar_mul += 1;
                }
                ConstraintType::G1Add => {
                    g1_add_indices.push(num_g1_add);
                    num_g1_add += 1;
                }
                ConstraintType::G2Add => {
                    g2_add_indices.push(num_g2_add);
                    num_g2_add += 1;
                }
            }
        }

        // Packed GT exp auxiliary subprotocols (shift rho + claim reduction).
        if num_gt_exp > 0 {
            let claim_indices: Vec<usize> = (0..num_gt_exp).collect();
            if enable_shift_rho {
                let shift_verifier = ShiftRhoVerifier::<Fq>::new(
                    ShiftRhoParams::new(num_gt_exp),
                    claim_indices.clone(),
                    transcript,
                );
                verifiers.push(Box::new(shift_verifier));
            }

            if enable_claim_reduction {
                let reduction_verifier = PackedGtExpClaimReductionVerifier::<Fq>::new(
                    PackedGtExpClaimReductionParams::new(2 * num_gt_exp),
                    claim_indices,
                    transcript,
                );
                verifiers.push(Box::new(reduction_verifier));
            }
        }

        // GT mul
        if num_gt_mul > 0 {
            use super::curve::RecursionCurve;
            let params = GtMulParams::new(num_gt_mul);
            let g_mle = Bn254Recursion::g_mle();
            let spec = GtMulVerifierSpec::<Fq>::new_from_fq(params, &g_mle);
            let verifier = GtMulVerifier::<Fq>::from_spec(spec, gt_mul_indices, transcript);
            verifiers.push(Box::new(verifier));
        }

        // G1 scalar mul shift: link A and A_next (x/y)
        if enable_shift_g1_scalar_mul && num_g1_scalar_mul > 0 {
            let mut pairs: Vec<(VirtualPolynomial, VirtualPolynomial)> =
                Vec::with_capacity(num_g1_scalar_mul * 2);
            for i in 0..num_g1_scalar_mul {
                pairs.push((
                    VirtualPolynomial::g1_scalar_mul_xa(i),
                    VirtualPolynomial::g1_scalar_mul_xa_next(i),
                ));
                pairs.push((
                    VirtualPolynomial::g1_scalar_mul_ya(i),
                    VirtualPolynomial::g1_scalar_mul_ya_next(i),
                ));
            }
            verifiers.push(Box::new(ShiftG1ScalarMulVerifier::<Fq>::new(
                g1_shift_params(pairs.len()),
                pairs,
                transcript,
            )));
        }

        // G1 scalar mul
        if num_g1_scalar_mul > 0 {
            use super::stage2::g1_scalar_mul::{
                G1ScalarMulParams, G1ScalarMulVerifier, G1ScalarMulVerifierSpec,
            };

            let params = G1ScalarMulParams::new(num_g1_scalar_mul);
            debug_assert_eq!(
                self.input.g1_scalar_mul_public_inputs.len(),
                num_g1_scalar_mul,
                "RecursionVerifierInput.g1_scalar_mul_public_inputs must match number of G1ScalarMul constraints"
            );
            let spec = G1ScalarMulVerifierSpec::new(
                params,
                self.input.g1_scalar_mul_public_inputs.clone(),
                g1_scalar_mul_base_points,
            );
            let verifier = G1ScalarMulVerifier::from_spec(spec, g1_scalar_mul_indices, transcript);
            verifiers.push(Box::new(verifier));
        }

        // G2 scalar mul shift: link A and A_next (x/y, c0/c1)
        if enable_shift_g2_scalar_mul && num_g2_scalar_mul > 0 {
            let mut pairs: Vec<(VirtualPolynomial, VirtualPolynomial)> =
                Vec::with_capacity(num_g2_scalar_mul * 4);
            for i in 0..num_g2_scalar_mul {
                pairs.push((
                    VirtualPolynomial::g2_scalar_mul_xa_c0(i),
                    VirtualPolynomial::g2_scalar_mul_xa_next_c0(i),
                ));
                pairs.push((
                    VirtualPolynomial::g2_scalar_mul_xa_c1(i),
                    VirtualPolynomial::g2_scalar_mul_xa_next_c1(i),
                ));
                pairs.push((
                    VirtualPolynomial::g2_scalar_mul_ya_c0(i),
                    VirtualPolynomial::g2_scalar_mul_ya_next_c0(i),
                ));
                pairs.push((
                    VirtualPolynomial::g2_scalar_mul_ya_c1(i),
                    VirtualPolynomial::g2_scalar_mul_ya_next_c1(i),
                ));
            }
            verifiers.push(Box::new(ShiftG2ScalarMulVerifier::<Fq>::new(
                g2_shift_params(pairs.len()),
                pairs,
                transcript,
            )));
        }

        // G2 scalar mul
        if num_g2_scalar_mul > 0 {
            use super::stage2::g2_scalar_mul::{
                G2ScalarMulParams, G2ScalarMulVerifier, G2ScalarMulVerifierSpec,
            };

            let params = G2ScalarMulParams::new(num_g2_scalar_mul);
            debug_assert_eq!(
                self.input.g2_scalar_mul_public_inputs.len(),
                num_g2_scalar_mul,
                "RecursionVerifierInput.g2_scalar_mul_public_inputs must match number of G2ScalarMul constraints"
            );
            let spec = G2ScalarMulVerifierSpec::new(
                params,
                self.input.g2_scalar_mul_public_inputs.clone(),
                g2_scalar_mul_base_points,
            );
            let verifier = G2ScalarMulVerifier::from_spec(spec, g2_scalar_mul_indices, transcript);
            verifiers.push(Box::new(verifier));
        }

        // G1 add
        if num_g1_add > 0 {
            use super::stage2::g1_add::{G1AddVerifier, G1AddVerifierSpec};
            let params = G1AddParams::new(num_g1_add);
            let spec = G1AddVerifierSpec::new(params);
            let verifier = G1AddVerifier::from_spec(spec, g1_add_indices, transcript);
            verifiers.push(Box::new(verifier));
        }

        // G2 add
        if num_g2_add > 0 {
            use super::stage2::g2_add::{G2AddVerifier, G2AddVerifierSpec};
            let params = G2AddParams::new(num_g2_add);
            let spec = G2AddVerifierSpec::new(params);
            let verifier = G2AddVerifier::from_spec(spec, g2_add_indices, transcript);
            verifiers.push(Box::new(verifier));
        }

        // TODO: wiring/boundary constraints.

        if verifiers.is_empty() {
            return Err("No constraints to verify in Stage 2".into());
        }

        let verifier_refs: Vec<&dyn SumcheckInstanceVerifier<Fq, T, A>> =
            verifiers.iter().map(|v| &**v).collect();
        let r_x = BatchedSumcheck::verify(proof, verifier_refs, accumulator, transcript)?;
        Ok(r_x)
    }

    /// Verify Stage 3: Direct evaluation protocol - fully generic over F.
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage3_symbolic")]
    fn verify_stage3_symbolic<F: JoltField, T: Transcript, A: OpeningAccumulator<F>>(
        &self,
        transcript: &mut T,
        accumulator: &mut A,
        r_x: &[F::Challenge],
        stage3_m_eval: F,
    ) -> Result<Vec<F::Challenge>, Box<dyn std::error::Error>> {
        // Convert r_x challenges to field elements
        let r_x_f: Vec<F> = r_x.iter().map(|c| (*c).into()).collect();

        // For symbolic execution, we need to extract virtual claims generically.
        // The claims come from Stage 2 accumulated polynomial openings.
        // For now, we create placeholder claims that will be populated by the accumulator.
        let num_claims = self.input.num_constraints * 5; // Approximate: 5 poly types per constraint
        let virtual_claims: Vec<F> = vec![F::zero(); num_claims];

        let params = DirectEvaluationParams::new(
            self.input.num_s_vars,
            self.input.num_constraints,
            self.input.num_constraints_padded,
            self.input.num_constraint_vars,
        );

        // Convert r_x to challenges for DirectEvaluationVerifier
        let r_x_challenges: Vec<F::Challenge> = r_x.to_vec();

        let verifier = DirectEvaluationVerifier::<F>::new_generic(params, virtual_claims, r_x_challenges);
        let r_s = verifier
            .verify_generic(transcript, accumulator, stage3_m_eval)
            .map_err(Box::<dyn std::error::Error>::from)?;

        // r_s is already Vec<F::Challenge>, just reverse it
        let r_s_challenges: Vec<F::Challenge> = r_s.into_iter().rev().collect();

        Ok(r_s_challenges)
    }

    /// Verify Stage 3: Direct evaluation protocol (virtualization) - generic over accumulator.
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage3_generic")]
    fn verify_stage3_generic<T: Transcript, A: OpeningAccumulator<Fq>>(
        &self,
        transcript: &mut T,
        accumulator: &mut A,
        r_x: &[<Fq as crate::field::JoltField>::Challenge],
        stage3_m_eval: Fq,
    ) -> Result<Vec<<Fq as crate::field::JoltField>::Challenge>, Box<dyn std::error::Error>> {
        let r_x_fq: Vec<Fq> = r_x.iter().map(|c| (*c).into()).collect();

        let virtual_claims = extract_virtual_claims_from_accumulator(
            accumulator,
            &self.input.constraint_types,
            &self.input.gt_exp_public_inputs,
        );

        let params = DirectEvaluationParams::new(
            self.input.num_s_vars,
            self.input.num_constraints,
            self.input.num_constraints_padded,
            self.input.num_constraint_vars,
        );

        let verifier = DirectEvaluationVerifier::new(params, virtual_claims, r_x_fq);
        let m_eval_fq: Fq = stage3_m_eval;
        let r_s = verifier
            .verify_generic(transcript, accumulator, m_eval_fq)
            .map_err(Box::<dyn std::error::Error>::from)?;

        let r_s_challenges: Vec<<Fq as JoltField>::Challenge> =
            r_s.into_iter().rev().map(|f| f.into()).collect();

        Ok(r_s_challenges)
    }

    /// Verify Stage 4: Jagged transform sumcheck - fully generic over F.
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage4_symbolic")]
    fn verify_stage4_symbolic<F: JoltField, T: Transcript, A: OpeningAccumulator<F>>(
        &self,
        stage4_proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<F, T>,
        stage5_proof: &super::stage5::jagged_assist::JaggedAssistProof<F, T>,
        transcript: &mut T,
        accumulator: &mut A,
        r_s: &[F::Challenge],
    ) -> Result<Vec<F::Challenge>, Box<dyn std::error::Error>> {
        let (_, sparse_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::DorySparseConstraintMatrix,
            SumcheckId::RecursionVirtualization,
        );

        let r_s_final: Vec<F> = r_s
            .iter()
            .take(self.input.num_s_vars)
            .map(|c| (*c).into())
            .collect();

        // Get the actual dense_size from the jagged bijection
        // Using dense_size_value() which doesn't require a generic field parameter
        let dense_size = self.input.jagged_bijection.dense_size_value();
        let num_dense_vars = dense_size.next_power_of_two().trailing_zeros() as usize;

        let params = JaggedSumcheckParams::new(
            self.input.num_s_vars,
            self.input.num_constraint_vars,
            num_dense_vars,
        );

        // Convert per-polynomial claimed evals → per-row evals
        let num_rows = 1usize << self.input.num_s_vars;
        let mut claimed_evaluations_by_row = vec![F::zero(); num_rows];
        for (poly_idx, claimed_eval) in stage5_proof.claimed_evaluations.iter().enumerate() {
            let matrix_row = self.input.matrix_rows[poly_idx];
            if matrix_row < num_rows {
                claimed_evaluations_by_row[matrix_row] = claimed_evaluations_by_row[matrix_row] + *claimed_eval;
            }
        }

        let verifier = JaggedSumcheckVerifier::new(
            r_s_final,
            sparse_claim,
            params,
            claimed_evaluations_by_row,
        );

        let r_dense =
            BatchedSumcheck::verify(stage4_proof, vec![&verifier], accumulator, transcript)?;

        Ok(r_dense)
    }

    /// Verify Stage 4: Jagged transform sumcheck - generic over accumulator.
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage4_generic")]
    fn verify_stage4_generic<T: Transcript, A: OpeningAccumulator<Fq>>(
        &self,
        stage4_proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<Fq, T>,
        stage5_proof: &super::stage5::jagged_assist::JaggedAssistProof<Fq, T>,
        transcript: &mut T,
        accumulator: &mut A,
        r_s: &[<Fq as crate::field::JoltField>::Challenge],
    ) -> Result<Vec<<Fq as crate::field::JoltField>::Challenge>, Box<dyn std::error::Error>> {
        let (_, sparse_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::DorySparseConstraintMatrix,
            SumcheckId::RecursionVirtualization,
        );

        let r_s_final: Vec<Fq> = r_s
            .iter()
            .take(self.input.num_s_vars)
            .map(|c| (*c).into())
            .collect();

        let dense_size = <VarCountJaggedBijection as crate::zkvm::recursion::bijection::JaggedTransform<Fq>>::dense_size(&self.input.jagged_bijection);
        let num_dense_vars = dense_size.next_power_of_two().trailing_zeros() as usize;

        let params = JaggedSumcheckParams::new(
            self.input.num_s_vars,
            self.input.num_constraint_vars,
            num_dense_vars,
        );

        // Convert per-polynomial claimed evals → per-row evals.
        let num_rows = 1usize << self.input.num_s_vars;
        let mut claimed_evaluations_by_row = vec![Fq::zero(); num_rows];
        for (poly_idx, claimed_eval) in stage5_proof.claimed_evaluations.iter().enumerate() {
            let matrix_row = self.input.matrix_rows[poly_idx];
            if matrix_row < num_rows {
                claimed_evaluations_by_row[matrix_row] += *claimed_eval;
            }
        }

        let verifier = JaggedSumcheckVerifier::new(
            r_s_final,
            sparse_claim,
            params,
            claimed_evaluations_by_row,
        );

        let r_dense =
            BatchedSumcheck::verify(stage4_proof, vec![&verifier], accumulator, transcript)?;

        Ok(r_dense)
    }

    /// Verify Stage 5: Jagged assist - fully generic over F.
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage5_symbolic")]
    fn verify_stage5_symbolic<F: JoltField, T: Transcript, A: OpeningAccumulator<F>>(
        &self,
        stage5_proof: &super::stage5::jagged_assist::JaggedAssistProof<F, T>,
        transcript: &mut T,
        accumulator: &mut A,
        r_dense: &[F::Challenge],
        r_x: &[F::Challenge],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let r_dense_f: Vec<F> = r_dense.iter().map(|c| (*c).into()).collect();
        let r_x_prev: Vec<F> = r_x.iter().map(|c| (*c).into()).collect();

        // Get the actual dense_size from the jagged bijection
        let dense_size = self.input.jagged_bijection.dense_size_value();
        let num_dense_vars = dense_size.next_power_of_two().trailing_zeros() as usize;
        let num_bits = std::cmp::max(self.input.num_constraint_vars, num_dense_vars);

        let assist_verifier = JaggedAssistVerifier::<F, T>::new(
            stage5_proof.claimed_evaluations.clone(),
            r_x_prev,
            r_dense_f,
            &self.input.jagged_bijection,
            num_bits,
            transcript,
        );

        let _r_assist = BatchedSumcheck::verify(
            &stage5_proof.sumcheck_proof,
            vec![&assist_verifier],
            accumulator,
            transcript,
        )?;

        Ok(())
    }

    /// Verify Stage 5: Jagged assist (batch MLE verification) - generic over accumulator.
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage5_generic")]
    fn verify_stage5_generic<T: Transcript, A: OpeningAccumulator<Fq>>(
        &self,
        stage5_proof: &super::stage5::jagged_assist::JaggedAssistProof<Fq, T>,
        transcript: &mut T,
        accumulator: &mut A,
        r_dense: &[<Fq as crate::field::JoltField>::Challenge],
        r_x: &[<Fq as crate::field::JoltField>::Challenge],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let r_dense_fq: Vec<Fq> = r_dense.iter().map(|c| (*c).into()).collect();
        let r_x_prev: Vec<Fq> = r_x.iter().map(|c| (*c).into()).collect();

        let dense_size = <VarCountJaggedBijection as crate::zkvm::recursion::bijection::JaggedTransform<Fq>>::dense_size(&self.input.jagged_bijection);
        let num_dense_vars = dense_size.next_power_of_two().trailing_zeros() as usize;
        let num_bits = std::cmp::max(self.input.num_constraint_vars, num_dense_vars);

        let assist_verifier = JaggedAssistVerifier::<Fq, T>::new(
            stage5_proof.claimed_evaluations.clone(),
            r_x_prev,
            r_dense_fq,
            &self.input.jagged_bijection,
            num_bits,
            transcript,
        );

        let _r_assist = BatchedSumcheck::verify(
            &stage5_proof.sumcheck_proof,
            vec![&assist_verifier],
            accumulator,
            transcript,
        )?;

        Ok(())
    }
}
