//! GT multiplication sumcheck for proving GT multiplication constraints
//! Proves: 0 = Σ_x eq(eq_point, x) * Σ_i (instance_batch_coeff)^i * C_i(x)
//! Where C_i(x) = a_i(x) × b_i(x) - c_i(x) - Q_i(x) × g(x)
//!
//! This is a separate sumcheck protocol for GT multiplication constraints.
//! Output: Virtual polynomial claims for each polynomial in each constraint

use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial, multilinear_polynomial::MultilinearPolynomial,
        opening_proof::SumcheckId,
    },
};

use crate::zkvm::witness::{GtMulTerm, RecursionPoly, TermEnum, VirtualPolynomial};

use crate::subprotocols::constraint_list_sumcheck::{
    sequential_opening_specs, ConstraintListProver, ConstraintListProverSpec, ConstraintListSpec,
    ConstraintListVerifier, ConstraintListVerifierSpec, OpeningSpec,
};

use allocative::Allocative;

// ============================================================================
// Opening Specs - 4 polynomial types (lhs, rhs, result, quotient)
// ============================================================================

const GT_MUL_OPENING_SPECS: [OpeningSpec; 4] = sequential_opening_specs::<4>();

// ============================================================================
// Parameters and Witness Types
// ============================================================================

/// Individual polynomial data for a single GT mul constraint
#[derive(Clone, Allocative)]
pub struct GtMulConstraintPolynomials<F: JoltField> {
    pub lhs: Vec<F>,
    pub rhs: Vec<F>,
    pub result: Vec<F>,
    pub quotient: Vec<F>,
    pub constraint_index: usize,
}

/// Parameters for GT mul sumcheck
#[derive(Clone, Allocative)]
pub struct GtMulParams {
    /// Number of constraint variables (x) - fixed at 11 for uniform matrix
    pub num_constraint_vars: usize,

    /// Number of constraints
    pub num_constraints: usize,

    /// Sumcheck instance identifier
    pub sumcheck_id: SumcheckId,
}

impl GtMulParams {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            num_constraint_vars: 11, // 11 vars for uniform matrix (4 element + 7 padding)
            num_constraints,
            sumcheck_id: SumcheckId::GtMul,
        }
    }
}

// ============================================================================
// Prover Spec
// ============================================================================

/// Prover-side specification for GT mul constraints.
#[derive(Clone, Allocative)]
pub struct GtMulProverSpec<F: JoltField> {
    params: GtMulParams,
    polys_by_kind: Vec<Vec<MultilinearPolynomial<F>>>,
    shared_polys: Vec<MultilinearPolynomial<F>>,
}

impl<F: JoltField> GtMulProverSpec<F> {
    /// Create a new prover spec from constraint polynomials and the g polynomial.
    pub fn new(
        params: GtMulParams,
        constraint_polys: Vec<GtMulConstraintPolynomials<F>>,
        g_poly: DensePolynomial<F>,
    ) -> Self {
        let num_instances = constraint_polys.len();
        debug_assert_eq!(
            num_instances, params.num_constraints,
            "GtMulProverSpec: params.num_constraints must match constraint_polys length"
        );

        let mut lhs = Vec::with_capacity(num_instances);
        let mut rhs = Vec::with_capacity(num_instances);
        let mut result = Vec::with_capacity(num_instances);
        let mut quotient = Vec::with_capacity(num_instances);

        for poly in constraint_polys {
            lhs.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                poly.lhs,
            )));
            rhs.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                poly.rhs,
            )));
            result.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                poly.result,
            )));
            quotient.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                poly.quotient,
            )));
        }

        Self {
            params,
            polys_by_kind: vec![lhs, rhs, result, quotient],
            shared_polys: vec![MultilinearPolynomial::LargeScalars(g_poly)],
        }
    }
}

impl<F: JoltField + Allocative> ConstraintListSpec for GtMulProverSpec<F> {
    fn sumcheck_id(&self) -> SumcheckId {
        self.params.sumcheck_id
    }

    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars
    }

    fn num_instances(&self) -> usize {
        self.params.num_constraints
    }

    fn opening_specs(&self) -> &'static [OpeningSpec] {
        &GT_MUL_OPENING_SPECS
    }

    fn build_virtual_poly(&self, term_index: usize, instance: usize) -> VirtualPolynomial {
        VirtualPolynomial::Recursion(RecursionPoly::GtMul {
            term: GtMulTerm::from_index(term_index).expect("invalid GtMulTerm index"),
            instance,
        })
    }
}

impl<F: JoltField + Allocative> ConstraintListProverSpec<F, 3> for GtMulProverSpec<F> {
    fn polys_by_kind(&self) -> &[Vec<MultilinearPolynomial<F>>] {
        &self.polys_by_kind
    }

    fn polys_by_kind_mut(&mut self) -> &mut [Vec<MultilinearPolynomial<F>>] {
        &mut self.polys_by_kind
    }

    fn shared_polys(&self) -> &[MultilinearPolynomial<F>] {
        &self.shared_polys
    }

    fn shared_polys_mut(&mut self) -> &mut [MultilinearPolynomial<F>] {
        &mut self.shared_polys
    }

    fn eval_constraint(
        &self,
        _instance: usize,
        eval_index: usize,
        poly_evals: &[[F; 3]],
        _public_evals: &[[F; 3]],
        shared_evals: &[[F; 3]],
        _term_batch_coeff: Option<F>,
    ) -> F {
        let lhs = poly_evals[0][eval_index];
        let rhs = poly_evals[1][eval_index];
        let result = poly_evals[2][eval_index];
        let quotient = poly_evals[3][eval_index];
        let g = shared_evals[0][eval_index];
        lhs * rhs - result - quotient * g
    }
}

// ============================================================================
// Verifier Spec
// ============================================================================

use ark_bn254::Fq;
use super::super::convert_fq_to_field;

/// Verifier-side specification for GT mul constraints.
/// Generic over F to support both concrete (Fq) and symbolic (MleAst) execution.
#[derive(Clone)]
pub struct GtMulVerifierSpec<F: JoltField> {
    params: GtMulParams,
    /// Pre-converted g MLE values (padded to num_constraint_vars)
    g_mle: Vec<F>,
}

impl<F: JoltField> Allocative for GtMulVerifierSpec<F> {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl<F: JoltField> GtMulVerifierSpec<F> {
    /// Create a new verifier spec from Fq g_mle values.
    /// The g_mle values are converted to generic F and padded appropriately.
    pub fn new_from_fq(params: GtMulParams, g_mle_4var: &[Fq]) -> Self {
        use crate::zkvm::recursion::constraints_sys::DoryMatrixBuilder;

        let g_mle_padded = if params.num_constraint_vars == 11 {
            DoryMatrixBuilder::pad_4var_to_11var_zero_padding(g_mle_4var)
        } else if params.num_constraint_vars == 8 {
            DoryMatrixBuilder::pad_4var_to_8var_zero_padding(g_mle_4var)
        } else {
            g_mle_4var.to_vec()
        };

        let g_mle: Vec<F> = g_mle_padded
            .iter()
            .map(|fq| convert_fq_to_field::<F>(*fq))
            .collect();

        Self { params, g_mle }
    }

    /// Create a new verifier spec with already-converted g_mle values.
    pub fn new(params: GtMulParams, g_mle: Vec<F>) -> Self {
        Self { params, g_mle }
    }
}

impl<F: JoltField> ConstraintListSpec for GtMulVerifierSpec<F> {
    fn sumcheck_id(&self) -> SumcheckId {
        self.params.sumcheck_id
    }

    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars
    }

    fn num_instances(&self) -> usize {
        self.params.num_constraints
    }

    fn opening_specs(&self) -> &'static [OpeningSpec] {
        &GT_MUL_OPENING_SPECS
    }

    fn build_virtual_poly(&self, term_index: usize, instance: usize) -> VirtualPolynomial {
        VirtualPolynomial::Recursion(RecursionPoly::GtMul {
            term: GtMulTerm::from_index(term_index).expect("invalid GtMulTerm index"),
            instance,
        })
    }
}

impl<F: JoltField> ConstraintListVerifierSpec<F, 3> for GtMulVerifierSpec<F> {
    fn compute_shared_scalars(&self, eval_point: &[F]) -> Vec<F> {
        // Evaluate g polynomial at eval_point using pre-converted values
        let g_poly = MultilinearPolynomial::<F>::LargeScalars(DensePolynomial::new(self.g_mle.clone()));
        let g_eval = g_poly.evaluate_dot_product::<F>(eval_point);
        vec![g_eval]
    }

    fn eval_constraint_at_point(
        &self,
        _instance: usize,
        opened_claims: &[F],
        shared_scalars: &[F],
        _eval_point: &[F],
        _term_batch_coeff: Option<F>,
    ) -> F {
        let lhs = opened_claims[0];
        let rhs = opened_claims[1];
        let result = opened_claims[2];
        let quotient = opened_claims[3];
        let g_eval = shared_scalars[0];

        lhs * rhs - result - quotient * g_eval
    }
}

// ============================================================================
// Type Aliases (no wrapper structs needed!)
// ============================================================================

/// Prover for GT mul sumcheck.
///
/// This is a type alias - no manual trait delegation required.
pub type GtMulProver<F> = ConstraintListProver<F, GtMulProverSpec<F>, 3>;

/// Verifier for GT mul sumcheck.
///
/// This is a type alias - no manual trait delegation required.
pub type GtMulVerifier<F> = ConstraintListVerifier<F, GtMulVerifierSpec<F>, 3>;
