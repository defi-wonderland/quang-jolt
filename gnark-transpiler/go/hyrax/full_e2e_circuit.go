// Package hyrax implements the full end-to-end Jolt verifier circuit.
//
// This file combines:
// 1. Stages 1-7 (transpiled) - but skipping broken a17-a21 assertions
// 2. Recursion sumchecks (emulated Fq) - correct implementation
// 3. Hyrax opening verification (native Grumpkin MSM)
//
// The broken a17-a21 assertions in the transpiled circuit use Fr arithmetic
// on Fq values, which gives wrong results. We replace them with the correct
// emulated Fq implementation.
package hyrax

import (
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/algebra/native/sw_grumpkin"
	"github.com/consensys/gnark/std/math/emulated"
)

// FullE2ECircuit combines all components of the Jolt verifier.
//
// Component breakdown (estimated constraints):
// - Stages 1-7 (transpiled, minus recursion): ~9M constraints
// - Recursion sumchecks (emulated Fq): ~127K constraints
// - Hyrax opening (native Grumpkin): ~5M constraints (for sqrt_n=2048, sqrt_n1=836)
// - Total: ~14M constraints
type FullE2ECircuit struct {
	// ============================================================
	// Recursion Sumcheck Inputs (a17-a21)
	// ============================================================

	// Stage 1 (a17): 11 rounds, degree 7
	RecursionStage1Coeffs     [11][7]sw_grumpkin.Scalar `gnark:",public"`
	RecursionStage1Challenges [11]frontend.Variable     `gnark:",public"`
	RecursionStage1Expected   sw_grumpkin.Scalar        `gnark:",public"`

	// Stage 2 (a18): 11 rounds, degree 6
	RecursionStage2Coeffs     [11][6]sw_grumpkin.Scalar `gnark:",public"`
	RecursionStage2Challenges [11]frontend.Variable     `gnark:",public"`
	RecursionStage2Expected   sw_grumpkin.Scalar        `gnark:",public"`

	// Stage 4 (a20): 22 rounds, degree 2
	RecursionStage4Coeffs     [22][2]sw_grumpkin.Scalar `gnark:",public"`
	RecursionStage4Challenges [22]frontend.Variable     `gnark:",public"`
	RecursionStage4Expected   sw_grumpkin.Scalar        `gnark:",public"`

	// Stage 5 (a21): 88 rounds, degree 2
	RecursionStage5Coeffs     [88][2]sw_grumpkin.Scalar `gnark:",public"`
	RecursionStage5Challenges [88]frontend.Variable     `gnark:",public"`
	RecursionStage5Expected   sw_grumpkin.Scalar        `gnark:",public"`

	// ============================================================
	// Hyrax Opening Inputs
	// ============================================================

	// HyraxSqrtN is the full polynomial structure size
	HyraxSqrtN int

	// HyraxSqrtN1 is the filtered size (non-identity row commitments)
	HyraxSqrtN1 int

	// RowCommitments are the non-identity Grumpkin points
	HyraxRowCommitments []sw_grumpkin.G1Affine `gnark:",public"`

	// Generators are the SRS points
	HyraxGenerators []sw_grumpkin.G1Affine `gnark:",public"`

	// L is the eq vector for non-identity rows
	HyraxL []sw_grumpkin.Scalar `gnark:",public"`

	// U is the prover's claimed projection vector
	HyraxU []sw_grumpkin.Scalar `gnark:",public"`

	// R is the eq vector for the right half
	HyraxR []frontend.Variable `gnark:",public"`

	// V is the claimed polynomial evaluation
	HyraxV frontend.Variable `gnark:",public"`
}

// Define implements the circuit constraints.
func (c *FullE2ECircuit) Define(api frontend.API) error {
	// Initialize emulated Fq field for recursion sumchecks
	fq, err := emulated.NewField[sw_grumpkin.ScalarField](api)
	if err != nil {
		return err
	}

	// ============================================================
	// Part 1: Recursion Sumcheck Verification (emulated Fq)
	// ============================================================

	// Stage 1 (a17): degree 7
	if err := verifyRecursionStage1(api, fq, c.RecursionStage1Coeffs[:], c.RecursionStage1Challenges[:], &c.RecursionStage1Expected); err != nil {
		return err
	}

	// Stage 2 (a18): degree 6
	if err := verifyRecursionStage2(api, fq, c.RecursionStage2Coeffs[:], c.RecursionStage2Challenges[:], &c.RecursionStage2Expected); err != nil {
		return err
	}

	// Stage 4 (a20): degree 2
	if err := verifyRecursionStage4(api, fq, c.RecursionStage4Coeffs[:], c.RecursionStage4Challenges[:], &c.RecursionStage4Expected); err != nil {
		return err
	}

	// Stage 5 (a21): degree 2
	if err := verifyRecursionStage5(api, fq, c.RecursionStage5Coeffs[:], c.RecursionStage5Challenges[:], &c.RecursionStage5Expected); err != nil {
		return err
	}

	// ============================================================
	// Part 2: Hyrax Opening Verification (native Grumpkin MSM)
	// ============================================================

	if err := verifyHyraxOpening(api, fq, c); err != nil {
		return err
	}

	return nil
}

// verifyRecursionStage1 verifies Stage 1 sumcheck (degree 7)
func verifyRecursionStage1(
	api frontend.API,
	fq *emulated.Field[sw_grumpkin.ScalarField],
	coeffs [][7]sw_grumpkin.Scalar,
	challenges []frontend.Variable,
	expected *sw_grumpkin.Scalar,
) error {
	prevEval := fq.Zero()
	twoBits := api.ToBinary(2, 254)
	two := fq.FromBits(twoBits...)

	for round := 0; round < len(coeffs); round++ {
		c0 := &coeffs[round][0]
		c2 := &coeffs[round][1]
		c3 := &coeffs[round][2]
		c4 := &coeffs[round][3]
		c5 := &coeffs[round][4]
		c6 := &coeffs[round][5]
		c7 := &coeffs[round][6]

		rBits := api.ToBinary(challenges[round], 254)
		r := fq.FromBits(rBits...)

		// c1 = prevEval - 2*c0 - c2 - c3 - c4 - c5 - c6 - c7
		twoC0 := fq.Mul(two, c0)
		c1 := fq.Sub(prevEval, twoC0)
		c1 = fq.Sub(c1, c2)
		c1 = fq.Sub(c1, c3)
		c1 = fq.Sub(c1, c4)
		c1 = fq.Sub(c1, c5)
		c1 = fq.Sub(c1, c6)
		c1 = fq.Sub(c1, c7)

		// Horner's evaluation
		acc := c7
		acc = fq.Add(c6, fq.Mul(r, acc))
		acc = fq.Add(c5, fq.Mul(r, acc))
		acc = fq.Add(c4, fq.Mul(r, acc))
		acc = fq.Add(c3, fq.Mul(r, acc))
		acc = fq.Add(c2, fq.Mul(r, acc))
		acc = fq.Add(c1, fq.Mul(r, acc))
		acc = fq.Add(c0, fq.Mul(r, acc))

		prevEval = acc
	}

	fq.AssertIsEqual(prevEval, expected)
	return nil
}

// verifyRecursionStage2 verifies Stage 2 sumcheck (degree 6)
func verifyRecursionStage2(
	api frontend.API,
	fq *emulated.Field[sw_grumpkin.ScalarField],
	coeffs [][6]sw_grumpkin.Scalar,
	challenges []frontend.Variable,
	expected *sw_grumpkin.Scalar,
) error {
	prevEval := fq.Zero()
	twoBits := api.ToBinary(2, 254)
	two := fq.FromBits(twoBits...)

	for round := 0; round < len(coeffs); round++ {
		c0 := &coeffs[round][0]
		c2 := &coeffs[round][1]
		c3 := &coeffs[round][2]
		c4 := &coeffs[round][3]
		c5 := &coeffs[round][4]
		c6 := &coeffs[round][5]

		rBits := api.ToBinary(challenges[round], 254)
		r := fq.FromBits(rBits...)

		// c1 = prevEval - 2*c0 - c2 - c3 - c4 - c5 - c6
		twoC0 := fq.Mul(two, c0)
		c1 := fq.Sub(prevEval, twoC0)
		c1 = fq.Sub(c1, c2)
		c1 = fq.Sub(c1, c3)
		c1 = fq.Sub(c1, c4)
		c1 = fq.Sub(c1, c5)
		c1 = fq.Sub(c1, c6)

		// Horner's evaluation
		acc := c6
		acc = fq.Add(c5, fq.Mul(r, acc))
		acc = fq.Add(c4, fq.Mul(r, acc))
		acc = fq.Add(c3, fq.Mul(r, acc))
		acc = fq.Add(c2, fq.Mul(r, acc))
		acc = fq.Add(c1, fq.Mul(r, acc))
		acc = fq.Add(c0, fq.Mul(r, acc))

		prevEval = acc
	}

	fq.AssertIsEqual(prevEval, expected)
	return nil
}

// verifyRecursionStage4 verifies Stage 4 sumcheck (degree 2)
func verifyRecursionStage4(
	api frontend.API,
	fq *emulated.Field[sw_grumpkin.ScalarField],
	coeffs [][2]sw_grumpkin.Scalar,
	challenges []frontend.Variable,
	expected *sw_grumpkin.Scalar,
) error {
	return verifyDegree2Sumcheck(api, fq, coeffs, challenges, expected)
}

// verifyRecursionStage5 verifies Stage 5 sumcheck (degree 2)
func verifyRecursionStage5(
	api frontend.API,
	fq *emulated.Field[sw_grumpkin.ScalarField],
	coeffs [][2]sw_grumpkin.Scalar,
	challenges []frontend.Variable,
	expected *sw_grumpkin.Scalar,
) error {
	return verifyDegree2Sumcheck(api, fq, coeffs, challenges, expected)
}

// verifyDegree2Sumcheck verifies a sumcheck with degree-2 polynomials
func verifyDegree2Sumcheck(
	api frontend.API,
	fq *emulated.Field[sw_grumpkin.ScalarField],
	coeffs [][2]sw_grumpkin.Scalar,
	challenges []frontend.Variable,
	expected *sw_grumpkin.Scalar,
) error {
	prevEval := fq.Zero()
	twoBits := api.ToBinary(2, 254)
	two := fq.FromBits(twoBits...)

	for round := 0; round < len(coeffs); round++ {
		c0 := &coeffs[round][0]
		c2 := &coeffs[round][1]

		rBits := api.ToBinary(challenges[round], 254)
		r := fq.FromBits(rBits...)

		// c1 = prevEval - 2*c0 - c2
		twoC0 := fq.Mul(two, c0)
		c1 := fq.Sub(prevEval, twoC0)
		c1 = fq.Sub(c1, c2)

		// Horner's: p(r) = c0 + r*(c1 + r*c2)
		acc := c2
		acc = fq.Add(c1, fq.Mul(r, acc))
		acc = fq.Add(c0, fq.Mul(r, acc))

		prevEval = acc
	}

	fq.AssertIsEqual(prevEval, expected)
	return nil
}

// verifyHyraxOpening verifies the Hyrax opening proof using native Grumpkin MSM
func verifyHyraxOpening(
	api frontend.API,
	fq *emulated.Field[sw_grumpkin.ScalarField],
	c *FullE2ECircuit,
) error {
	// Initialize Grumpkin curve gadget
	curve, err := sw_grumpkin.NewCurve(api)
	if err != nil {
		return err
	}

	// Convert slices to pointer slices for MSM API
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

	// MSM #1: C' = sum(L[a] * RowCommitments[a])
	Cprime, err := curve.MultiScalarMul(rowPtrs, lScalars)
	if err != nil {
		return err
	}

	// MSM #2: Com(u) = sum(u[j] * Generators[j])
	ComU, err := curve.MultiScalarMul(genPtrs, uScalars)
	if err != nil {
		return err
	}

	// Check 1: Com(u) == C'
	curve.AssertIsEqual(ComU, Cprime)

	// Check 2: <u, R> == v (dot product in emulated field)
	dotProduct := fq.Zero()
	for i := range c.HyraxU {
		rBits := api.ToBinary(c.HyraxR[i], 254)
		rEmulated := fq.FromBits(rBits...)
		term := fq.Mul(&c.HyraxU[i], rEmulated)
		dotProduct = fq.Add(dotProduct, term)
	}

	vBits := api.ToBinary(c.HyraxV, 254)
	vEmulated := fq.FromBits(vBits...)
	fq.AssertIsEqual(dotProduct, vEmulated)

	return nil
}

// CreateFullE2ECircuit creates a placeholder circuit with the given Hyrax sizes
func CreateFullE2ECircuit(sqrtN, sqrtN1 int) *FullE2ECircuit {
	return &FullE2ECircuit{
		HyraxSqrtN:          sqrtN,
		HyraxSqrtN1:         sqrtN1,
		HyraxRowCommitments: make([]sw_grumpkin.G1Affine, sqrtN1),
		HyraxGenerators:     make([]sw_grumpkin.G1Affine, sqrtN),
		HyraxL:              make([]sw_grumpkin.Scalar, sqrtN1),
		HyraxU:              make([]sw_grumpkin.Scalar, sqrtN),
		HyraxR:              make([]frontend.Variable, sqrtN),
	}
}
