//! MLE Opening Accumulator for symbolic transpilation
//!
//! This module provides an `OpeningAccumulator` implementation that works with `MleAst`
//! instead of concrete field elements. This allows the verifier code to be transpiled
//! to Gnark circuits while keeping the code structure identical to the real verifier.
//!
//! ## How it works
//!
//! In the real verifier:
//! 1. Claims are loaded from the proof into `VerifierOpeningAccumulator`
//! 2. During verification, stages read claims via `get_virtual_polynomial_opening`
//! 3. Points are derived from transcript challenges and stored via `append_virtual`
//!
//! For transpilation:
//! 1. Claims are `MleAst` variables (inputs to the circuit)
//! 2. Points are `MleAst` derived from symbolic transcript challenges
//! 3. The accumulator stores `(Vec<MleAst>, MleAst)` instead of `(Vec<F::Challenge>, F)`

use ark_std::Zero;
use zklean_extractor::mle_ast::MleAst;
use jolt_core::poly::opening_proof::{
    OpeningAccumulator, OpeningId, OpeningPoint, Openings, SumcheckId, BIG_ENDIAN,
};
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::claim_reductions::AdviceKind;
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use std::collections::BTreeMap;

/// Opening accumulator for MleAst symbolic execution.
///
/// Stores polynomial opening claims as MleAst symbolic values,
/// allowing the verifier to be transpiled to a Gnark circuit.
#[derive(Clone, Debug)]
pub struct MleOpeningAccumulator {
    /// Map from opening ID to (point, claim)
    /// - point: Vec<MleAst> representing the evaluation point (challenges)
    /// - claim: MleAst representing the claimed evaluation
    pub openings: BTreeMap<OpeningId, (Vec<MleAst>, MleAst)>,
}

impl MleOpeningAccumulator {
    /// Create a new accumulator with no claims.
    pub fn new() -> Self {
        Self {
            openings: BTreeMap::new(),
        }
    }

    /// Create an accumulator pre-populated with claims from the proof.
    ///
    /// The claims are MleAst variables that will become circuit inputs.
    /// Points are initialized as empty and will be populated during verification.
    ///
    /// # Arguments
    /// * `claims` - Iterator of (OpeningId, MleAst) pairs representing the claims
    pub fn new_with_claims<I>(claims: I) -> Self
    where
        I: IntoIterator<Item = (OpeningId, MleAst)>,
    {
        let mut openings = BTreeMap::new();
        for (key, claim) in claims {
            // Point is initially empty, will be set via append_virtual
            openings.insert(key, (vec![], claim));
        }
        Self { openings }
    }

    /// Create an accumulator from the OpeningIds in a real proof.
    ///
    /// Each OpeningId gets assigned a unique MleAst variable starting from `start_var_idx`.
    /// Returns the accumulator and a vector of (var_idx, OpeningId) pairs for tracking inputs.
    ///
    /// # Arguments
    /// * `opening_ids` - Iterator of OpeningIds from a real proof
    /// * `start_var_idx` - Starting variable index for MleAst variables
    ///
    /// # Returns
    /// * `Self` - The populated accumulator
    /// * `Vec<(u16, OpeningId)>` - Mapping from variable index to OpeningId
    /// * `u16` - Next available variable index
    pub fn from_opening_ids<I>(opening_ids: I, start_var_idx: u16) -> (Self, Vec<(u16, OpeningId)>, u16)
    where
        I: IntoIterator<Item = OpeningId>,
    {
        let mut openings = BTreeMap::new();
        let mut var_mapping = Vec::new();
        let mut var_idx = start_var_idx;

        for key in opening_ids {
            let claim = MleAst::from_var(var_idx);
            var_mapping.push((var_idx, key.clone()));
            openings.insert(key, (vec![], claim));
            var_idx += 1;
        }

        (Self { openings }, var_mapping, var_idx)
    }

    /// Add a claim for untrusted advice at a given sumcheck.
    pub fn add_untrusted_advice_claim(&mut self, sumcheck: SumcheckId, claim: MleAst) {
        let key = OpeningId::UntrustedAdvice(sumcheck);
        self.openings.insert(key, (vec![], claim));
    }

    /// Add a claim for trusted advice at a given sumcheck.
    pub fn add_trusted_advice_claim(&mut self, sumcheck: SumcheckId, claim: MleAst) {
        let key = OpeningId::TrustedAdvice(sumcheck);
        self.openings.insert(key, (vec![], claim));
    }

    /// Add a claim for a virtual polynomial.
    ///
    /// This is called at initialization to register claims from the proof.
    pub fn add_virtual_claim(
        &mut self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
        claim: MleAst,
    ) {
        let key = OpeningId::virtual_poly(polynomial, sumcheck);
        self.openings.insert(key, (vec![], claim));
    }

    /// Add a claim for a committed polynomial.
    pub fn add_committed_claim(
        &mut self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
        claim: MleAst,
    ) {
        let key = OpeningId::committed(polynomial, sumcheck);
        self.openings.insert(key, (vec![], claim));
    }

    /// Update the opening point for a virtual polynomial.
    ///
    /// Called during verification when the point is derived from challenges.
    /// The claim should already exist (added at initialization).
    pub fn append_virtual(
        &mut self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
        point: Vec<MleAst>,
    ) {
        let key = OpeningId::virtual_poly(polynomial, sumcheck);
        if let Some((stored_point, _)) = self.openings.get_mut(&key) {
            *stored_point = point;
        } else {
            panic!(
                "MleOpeningAccumulator::append_virtual: no claim found for {:?} {:?}",
                polynomial, sumcheck
            );
        }
    }

    /// Update the opening point for a committed polynomial.
    pub fn append_committed(
        &mut self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
        point: Vec<MleAst>,
    ) {
        let key = OpeningId::committed(polynomial, sumcheck);
        if let Some((stored_point, _)) = self.openings.get_mut(&key) {
            *stored_point = point;
        } else {
            panic!(
                "MleOpeningAccumulator::append_committed: no claim found for {:?} {:?}",
                polynomial, sumcheck
            );
        }
    }

    /// Get all opening IDs in this accumulator.
    pub fn opening_ids(&self) -> impl Iterator<Item = &OpeningId> {
        self.openings.keys()
    }

    /// Get the number of openings.
    pub fn len(&self) -> usize {
        self.openings.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.openings.is_empty()
    }
}

impl Default for MleOpeningAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl OpeningAccumulator<MleAst> for MleOpeningAccumulator {
    fn get_virtual_polynomial_opening(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, MleAst>, MleAst) {
        let key = OpeningId::virtual_poly(polynomial, sumcheck);
        let (point, claim) = self
            .openings
            .get(&key)
            .unwrap_or_else(|| panic!("No opening found for {:?} {:?}", sumcheck, polynomial));

        // Convert Vec<MleAst> to OpeningPoint<BIG_ENDIAN, MleAst>
        // Note: MleAst::Challenge = MleAst, so this works directly
        let opening_point = OpeningPoint::new(point.clone());
        (opening_point, claim.clone())
    }

    fn get_committed_polynomial_opening(
        &self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, MleAst>, MleAst) {
        let key = OpeningId::committed(polynomial, sumcheck);
        let (point, claim) = self
            .openings
            .get(&key)
            .unwrap_or_else(|| panic!("No opening found for {:?} {:?}", sumcheck, polynomial));

        let opening_point = OpeningPoint::new(point.clone());
        (opening_point, claim.clone())
    }

    fn get_advice_opening(
        &self,
        kind: AdviceKind,
        sumcheck: SumcheckId,
    ) -> Option<(OpeningPoint<BIG_ENDIAN, MleAst>, MleAst)> {
        let key = match kind {
            AdviceKind::Trusted => OpeningId::TrustedAdvice(sumcheck),
            AdviceKind::Untrusted => OpeningId::UntrustedAdvice(sumcheck),
        };
        let (point, claim) = self.openings.get(&key)?;
        let opening_point = OpeningPoint::new(point.clone());
        Some((opening_point, claim.clone()))
    }

    fn append_virtual<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, MleAst>,
    ) {
        // Convert OpeningPoint to Vec<MleAst> for storage
        let point: Vec<MleAst> = opening_point.r;
        let key = OpeningId::virtual_poly(polynomial, sumcheck);
        if let Some((stored_point, claim)) = self.openings.get_mut(&key) {
            // CRITICAL: Append the claim to transcript, matching WonderJolt's
            // VerifierOpeningAccumulator::append_virtual behavior.
            // This is necessary for transcript consistency during transpilation.
            transcript.append_scalar(claim);
            *stored_point = point;
        } else {
            // For recursion stages, claims may not be pre-populated.
            // Insert a symbolic zero claim and append to transcript for consistency.
            // The actual claim value is derived from witness data during proving.
            let symbolic_claim = MleAst::zero();
            transcript.append_scalar(&symbolic_claim);
            self.openings.insert(key, (point, symbolic_claim));
        }
    }

    fn append_untrusted_advice<T: Transcript>(
        &mut self,
        _transcript: &mut T,
        sumcheck_id: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, MleAst>,
    ) {
        let point: Vec<MleAst> = opening_point.r;
        let key = OpeningId::UntrustedAdvice(sumcheck_id);
        if let Some((stored_point, _)) = self.openings.get_mut(&key) {
            *stored_point = point;
        } else {
            self.openings.insert(key, (point, MleAst::zero()));
        }
    }

    fn append_trusted_advice<T: Transcript>(
        &mut self,
        _transcript: &mut T,
        sumcheck_id: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, MleAst>,
    ) {
        let point: Vec<MleAst> = opening_point.r;
        let key = OpeningId::TrustedAdvice(sumcheck_id);
        if let Some((stored_point, _)) = self.openings.get_mut(&key) {
            *stored_point = point;
        } else {
            self.openings.insert(key, (point, MleAst::zero()));
        }
    }

    fn append_dense<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomial: CommittedPolynomial,
        sumcheck_id: SumcheckId,
        opening_point: Vec<MleAst>,  // Note: Vec<F::Challenge> = Vec<MleAst>
    ) {
        let point: Vec<MleAst> = opening_point;
        let key = OpeningId::committed(polynomial, sumcheck_id);

        // Get the claim and append to transcript (matching WonderJolt's VerifierOpeningAccumulator)
        if let Some((stored_point, claim)) = self.openings.get_mut(&key) {
            transcript.append_scalar(claim);
            *stored_point = point;
        } else {
            // If claim doesn't exist yet, insert with zero claim (shouldn't happen normally)
            self.openings.insert(key, (point, MleAst::zero()));
        }
    }

    fn append_sparse<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        opening_point: Vec<MleAst>,  // Note: Vec<F::Challenge> = Vec<MleAst>
    ) {
        for polynomial in polynomials.into_iter() {
            let key = OpeningId::committed(polynomial, sumcheck);

            // Get the claim and append to transcript (matching VerifierOpeningAccumulator)
            if let Some((stored_point, claim)) = self.openings.get_mut(&key) {
                transcript.append_scalar(claim);
                *stored_point = opening_point.clone();
            } else {
                // If claim doesn't exist yet, insert with zero claim
                // This can happen for sparse polynomials that aren't pre-populated
                self.openings.insert(key, (opening_point.clone(), MleAst::zero()));
            }
        }
    }

    fn populate_from_openings(&mut self, openings: &Openings<MleAst>) {
        for (key, (point, claim)) in openings {
            self.openings.insert(*key, (point.r.clone(), claim.clone()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_basic() {
        let mut acc = MleOpeningAccumulator::new();

        // Add a claim
        let claim = MleAst::from_var(0);
        acc.add_virtual_claim(VirtualPolynomial::Product, SumcheckId::SpartanOuter, claim.clone());

        // Update point
        let point = vec![MleAst::from_var(1), MleAst::from_var(2)];
        acc.append_virtual(VirtualPolynomial::Product, SumcheckId::SpartanOuter, point.clone());

        // Retrieve
        let (retrieved_point, retrieved_claim) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::Product,
            SumcheckId::SpartanOuter,
        );

        assert_eq!(retrieved_point.r.len(), 2);
        // Claims should match (MleAst comparison)
    }
}
