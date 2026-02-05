// Package hyrax implements an isolated Hyrax polynomial commitment verifier circuit
// for benchmarking Grumpkin MSM constraint costs in gnark.
//
// This is a standalone experiment to validate feasibility of Hyrax verification
// inside a BN254 Groth16 circuit, leveraging the BN254/Grumpkin 2-cycle.
//
// Reference: docs/10_Hyrax_gnark_Implementation_Plan.md
package hyrax

import (
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/algebra/native/sw_grumpkin"
	"github.com/consensys/gnark/std/math/emulated"
)

// HyraxVerifierCircuit implements the Hyrax opening verification protocol.
//
// The Hyrax PCS uses a square-root structure: for a polynomial with N coefficients,
// the prover commits to sqrt(N) row commitments. Opening requires two MSMs of size sqrt(N).
//
// Protocol (from doc 09, Step 2):
//  1. Verifier computes C' = sum(L[a] * RowCommitments[a]) -- MSM #1
//  2. Verifier computes Com(u) = sum(u[j] * Generators[j]) -- MSM #2
//  3. Verifier checks Com(u) == C'
//  4. Verifier checks <u, R> == v (dot product)
//
// IMPORTANT: Identity point handling
// gnark's scalar multiplication doesn't correctly handle identity points (0, 0).
// Row commitments may contain identity points where the coefficient matrix row is all zeros.
// We filter these out: RowCommitments and L contain only non-identity entries.
// MSM #2 (Generators, U, R) uses the full sqrt(N) size since generators are never identity.
type HyraxVerifierCircuit struct {
	// SqrtN is the full polynomial structure size (used for MSM #2: Generators × U)
	SqrtN int

	// SqrtN1 is the filtered size for MSM #1 (non-identity row commitments)
	// This may be smaller than SqrtN due to identity point filtering
	SqrtN1 int

	// RowCommitments are the non-identity Grumpkin points from the Hyrax commitment
	// Size: SqrtN1 (filtered to remove identity points)
	RowCommitments []sw_grumpkin.G1Affine `gnark:",public"`

	// Generators are the SRS points G_0, ..., G_{sqrt(N)-1}
	// Size: SqrtN (full size, generators are never identity)
	Generators []sw_grumpkin.G1Affine `gnark:",public"`

	// L is the eq vector for non-identity rows only
	// L[a] = eq(original_index[a], z_L) for non-identity row indices
	// Size: SqrtN1 (same as RowCommitments)
	L []sw_grumpkin.Scalar `gnark:",public"`

	// U is the prover's claimed projection vector u = L^T * F
	// where F is the coefficient matrix (sqrt(N) x sqrt(N))
	// Size: SqrtN (full size)
	U []sw_grumpkin.Scalar `gnark:",public"`

	// R is the eq vector for the "right" half of the evaluation point z
	// R[b] = eq(b, z_R) for b in {0,1}^{log(sqrt(N))}
	// Size: SqrtN (full size)
	R []frontend.Variable `gnark:",public"`

	// V is the claimed polynomial evaluation: f(z) = <u, R> = L^T * F * R
	V frontend.Variable `gnark:",public"`
}

// Define implements the circuit constraints for Hyrax verification.
func (c *HyraxVerifierCircuit) Define(api frontend.API) error {
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
	rowPtrs := make([]*sw_grumpkin.G1Affine, len(c.RowCommitments))
	for i := range c.RowCommitments {
		rowPtrs[i] = &c.RowCommitments[i]
	}

	genPtrs := make([]*sw_grumpkin.G1Affine, len(c.Generators))
	for i := range c.Generators {
		genPtrs[i] = &c.Generators[i]
	}

	lScalars := make([]*sw_grumpkin.Scalar, len(c.L))
	for i := range c.L {
		lScalars[i] = &c.L[i]
	}

	uScalars := make([]*sw_grumpkin.Scalar, len(c.U))
	for i := range c.U {
		uScalars[i] = &c.U[i]
	}

	// =========================================================================
	// Step 2b: MSM #1 -- Verifier derives target commitment
	// C' = sum_{a=0}^{sqrt(N)-1} L[a] * RowCommitments[a]
	// =========================================================================
	Cprime, err := curve.MultiScalarMul(rowPtrs, lScalars)
	if err != nil {
		return err
	}

	// =========================================================================
	// Step 2d Check 1: MSM #2 -- Commit to u, compare with C'
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
	// Step 2d Check 2: Dot product -- <u, R> == v
	//
	// Field considerations:
	// - U elements are sw_grumpkin.Scalar (emulated Fq, Grumpkin scalar field)
	// - R elements are frontend.Variable (native Fr, BN254 scalar field)
	// - V is the claimed evaluation (native Fr)
	//
	// Both BN254 Fr and Grumpkin Fr are ~254-bit primes. For the dot product,
	// we perform the computation entirely in the emulated Grumpkin scalar field
	// to properly handle full-size scalars.
	//
	// To convert native variables to emulated elements, we decompose to bits
	// and reconstruct. This adds constraints but handles arbitrary field values.
	//
	// Note: Grumpkin scalar field uses 4 limbs × 64 bits = 256 bits total.
	// =========================================================================

	// Initialize accumulator for dot product in emulated field
	dotProduct := scalarField.Zero()

	for i := range c.U {
		// Convert R[i] (native variable) to an emulated scalar
		// Decompose to 254 bits (BN254 scalar field size) and reconstruct
		rBits := api.ToBinary(c.R[i], 254)
		rEmulated := scalarField.FromBits(rBits...)

		// Compute u[i] * R[i] in emulated field
		term := scalarField.Mul(&c.U[i], rEmulated)

		// Accumulate
		dotProduct = scalarField.Add(dotProduct, term)
	}

	// Convert V to emulated and compare
	vBits := api.ToBinary(c.V, 254)
	vEmulated := scalarField.FromBits(vBits...)
	scalarField.AssertIsEqual(dotProduct, vEmulated)

	return nil
}
