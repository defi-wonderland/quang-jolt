// Package jolt_verifier provides combined Jolt verification circuits.
//
// This file defines JoltCombinedCircuit which combines:
// - Stages 1-7: Transpiled sumcheck verification (~3.1M constraints)
// - Stage 8: Hyrax PCS opening verification (~3.5M constraints for sqrt(N)=1024)
//
// Total: ~6.6M constraints for full Jolt verification in Groth16.
package jolt_verifier

import (
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/algebra/native/sw_grumpkin"
	"github.com/consensys/gnark/std/math/emulated"
)

// JoltCombinedCircuit combines transpiled sumcheck stages (1-7) with native Hyrax verification (stage 8).
//
// Architecture:
// - JoltStagesCircuit: Contains all sumcheck verification constraints (transpiled from Rust)
// - Hyrax fields: Native Grumpkin MSM verification for polynomial commitment opening
//
// The two parts share evaluation claims: sumcheck stages produce claimed evaluations,
// Hyrax verification proves those evaluations are correct openings of committed polynomials.
//
// IMPORTANT: Identity point handling
// gnark's scalar multiplication doesn't correctly handle identity points (0, 0).
// Row commitments may contain identity points where the coefficient matrix row is all zeros.
// We filter these out: HyraxRowCommitments and HyraxL contain only non-identity entries.
// MSM #2 (Generators, U, R) uses the full HyraxSqrtN size since generators are never identity.
type JoltCombinedCircuit struct {
	// =========================================================================
	// Stages 1-7: Transpiled sumcheck verification
	// Embedded struct contains all sumcheck witness variables
	// =========================================================================
	JoltStagesCircuit

	// =========================================================================
	// Stage 8: Hyrax PCS opening verification
	// Uses native Grumpkin curve operations (BN254/Grumpkin 2-cycle)
	// =========================================================================

	// HyraxSqrtN is the full polynomial structure size (used for MSM #2: Generators × U)
	// For Jolt with ~2^22 dense polynomial, this is typically 2048
	HyraxSqrtN int

	// HyraxSqrtN1 is the filtered size for MSM #1 (non-identity row commitments)
	// This may be smaller than HyraxSqrtN due to identity point filtering
	HyraxSqrtN1 int

	// RowCommitments are the non-identity Grumpkin points from the Hyrax commitment
	// Size: HyraxSqrtN1 (filtered to remove identity points)
	HyraxRowCommitments []sw_grumpkin.G1Affine `gnark:",public"`

	// Generators are the SRS points G_0, ..., G_{sqrt(N)-1}
	// Size: HyraxSqrtN (full size, generators are never identity)
	HyraxGenerators []sw_grumpkin.G1Affine `gnark:",public"`

	// L is the eq vector for non-identity rows only
	// L[a] = eq(original_index[a], z_L) for non-identity row indices
	// Size: HyraxSqrtN1 (same as RowCommitments)
	HyraxL []sw_grumpkin.Scalar `gnark:",public"`

	// U is the prover's claimed projection vector u = L^T * F
	// where F is the coefficient matrix (sqrt(N) x sqrt(N))
	// Size: HyraxSqrtN (full size)
	HyraxU []sw_grumpkin.Scalar `gnark:",public"`

	// R is the eq vector for the "right" half of the evaluation point z
	// R[b] = eq(b, z_R) for b in {0,1}^{log(sqrt(N))}
	// Size: HyraxSqrtN (full size)
	HyraxR []frontend.Variable `gnark:",public"`

	// V is the claimed polynomial evaluation: f(z) = <u, R> = L^T * F * R
	// This should match the evaluation claim from the sumcheck stages
	HyraxV frontend.Variable `gnark:",public"`
}

// Define implements the combined circuit constraints.
func (c *JoltCombinedCircuit) Define(api frontend.API) error {
	// =========================================================================
	// Part 1: Stages 1-7 sumcheck verification (transpiled)
	// =========================================================================
	if err := c.JoltStagesCircuit.Define(api); err != nil {
		return err
	}

	// =========================================================================
	// Part 2: Stage 8 Hyrax PCS verification (native Grumpkin)
	// =========================================================================
	return c.defineHyraxVerification(api)
}

// defineHyraxVerification implements the Hyrax opening verification protocol.
//
// Protocol:
//  1. Verifier computes C' = sum(L[a] * RowCommitments[a]) -- MSM #1
//  2. Verifier computes Com(u) = sum(u[j] * Generators[j]) -- MSM #2
//  3. Verifier checks Com(u) == C'
//  4. Verifier checks <u, R> == v (dot product)
func (c *JoltCombinedCircuit) defineHyraxVerification(api frontend.API) error {
	// Skip Hyrax verification if no row commitments (allows testing stages-only)
	if len(c.HyraxRowCommitments) == 0 {
		return nil
	}

	// Initialize the Grumpkin curve gadget
	curve, err := sw_grumpkin.NewCurve(api)
	if err != nil {
		return err
	}

	// Initialize emulated scalar field for Grumpkin scalars
	scalarField, err := emulated.NewField[sw_grumpkin.ScalarField](api)
	if err != nil {
		return err
	}

	// Convert slices to pointer slices (required by MultiScalarMul API)
	rowPtrs := make([]*sw_grumpkin.G1Affine, len(c.HyraxRowCommitments))
	for i := range c.HyraxRowCommitments {
		rowPtrs[i] = &c.HyraxRowCommitments[i]
	}

	genPtrs := make([]*sw_grumpkin.G1Affine, len(c.HyraxGenerators))
	for i := range c.HyraxGenerators {
		genPtrs[i] = &c.HyraxGenerators[i]
	}

	lScalars := make([]*sw_grumpkin.Scalar, len(c.HyraxL))
	for i := range c.HyraxL {
		lScalars[i] = &c.HyraxL[i]
	}

	uScalars := make([]*sw_grumpkin.Scalar, len(c.HyraxU))
	for i := range c.HyraxU {
		uScalars[i] = &c.HyraxU[i]
	}

	// =========================================================================
	// MSM #1: Verifier derives target commitment
	// C' = sum_{a=0}^{sqrt(N)-1} L[a] * RowCommitments[a]
	// =========================================================================
	Cprime, err := curve.MultiScalarMul(rowPtrs, lScalars)
	if err != nil {
		return err
	}

	// =========================================================================
	// MSM #2: Commit to u, compare with C'
	// Com(u) = sum_{j=0}^{sqrt(N)-1} u[j] * Generators[j]
	// =========================================================================
	ComU, err := curve.MultiScalarMul(genPtrs, uScalars)
	if err != nil {
		return err
	}

	// Assert Com(u) == C'
	// This is the binding property: prover cannot fake u without breaking Pedersen
	curve.AssertIsEqual(ComU, Cprime)

	// =========================================================================
	// Dot product: <u, R> == v
	//
	// Field considerations:
	// - U elements are sw_grumpkin.Scalar (emulated Fq, Grumpkin scalar field)
	// - R elements are frontend.Variable (native Fr, BN254 scalar field)
	// - V is the claimed evaluation (native Fr)
	//
	// We perform the computation in the emulated Grumpkin scalar field
	// to properly handle full-size scalars.
	// =========================================================================

	// Initialize accumulator for dot product in emulated field
	dotProduct := scalarField.Zero()

	for i := range c.HyraxU {
		// Convert R[i] (native variable) to an emulated scalar
		// Decompose to 254 bits (BN254 scalar field size) and reconstruct
		rBits := api.ToBinary(c.HyraxR[i], 254)
		rEmulated := scalarField.FromBits(rBits...)

		// Compute u[i] * R[i] in emulated field
		term := scalarField.Mul(&c.HyraxU[i], rEmulated)

		// Accumulate
		dotProduct = scalarField.Add(dotProduct, term)
	}

	// Convert V to emulated and compare
	vBits := api.ToBinary(c.HyraxV, 254)
	vEmulated := scalarField.FromBits(vBits...)
	scalarField.AssertIsEqual(dotProduct, vEmulated)

	return nil
}
