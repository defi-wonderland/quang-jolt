// Package hyrax implements the full recursion sumcheck verifier using emulated Fq arithmetic.
//
// This file contains a complete gnark circuit for verifying all recursion sumcheck stages
// (a17-a21) using proper Fq arithmetic instead of the broken Fr arithmetic in the transpiled code.
//
// Polynomial degrees per stage (from witness analysis):
// - Stage 1 (a17): 11 rounds, 7 coefficients [c0, c2-c7] → degree 7
// - Stage 2 (a18): 11 rounds, 6 coefficients [c0, c2-c6] → degree 6
// - Stage 4 (a20): 22 rounds, 2 coefficients [c0, c2] → degree 2
// - Stage 5 (a21): 88 rounds, 2 coefficients [c0, c2] → degree 2
package hyrax

import (
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/algebra/native/sw_grumpkin"
	"github.com/consensys/gnark/std/math/emulated"
)

// Constants for each stage
const (
	Stage1Rounds     = 11
	Stage1CoeffsNum  = 7 // c0, c2, c3, c4, c5, c6, c7 (degree 7)
	Stage2Rounds     = 11
	Stage2CoeffsNum  = 6 // c0, c2, c3, c4, c5, c6 (degree 6)
	Stage4Rounds     = 22
	Stage4CoeffsNum  = 2 // c0, c2 (degree 2)
	Stage5Rounds     = 88
	Stage5CoeffsNum  = 2 // c0, c2 (degree 2)
	CoeffsPerRound   = 7 // For backwards compat with Stage1OnlyCircuit
)

// ================================================
// Stage 1: Degree 7 polynomial (11 rounds)
// ================================================

// Stage1OnlyCircuit verifies Stage 1 sumcheck using emulated Fq
type Stage1OnlyCircuit struct {
	Coeffs     [Stage1Rounds][Stage1CoeffsNum]sw_grumpkin.Scalar `gnark:",public"`
	Challenges [Stage1Rounds]frontend.Variable                   `gnark:",public"`
	Expected   sw_grumpkin.Scalar                                `gnark:",public"`
}

func (c *Stage1OnlyCircuit) Define(api frontend.API) error {
	fq, err := emulated.NewField[sw_grumpkin.ScalarField](api)
	if err != nil {
		return err
	}
	return verifySumcheckDegree7(api, fq, c.Coeffs[:], c.Challenges[:], &c.Expected)
}

// verifySumcheckDegree7 verifies a sumcheck with degree-7 polynomials
// Coefficients: [c0, c2, c3, c4, c5, c6, c7] (c1 is reconstructed)
func verifySumcheckDegree7(
	api frontend.API,
	fq *emulated.Field[sw_grumpkin.ScalarField],
	coeffs [][7]sw_grumpkin.Scalar,
	challenges []frontend.Variable,
	expected *sw_grumpkin.Scalar,
) error {
	numRounds := len(coeffs)
	prevEval := fq.Zero()
	twoBits := api.ToBinary(2, 254)
	two := fq.FromBits(twoBits...)

	for round := 0; round < numRounds; round++ {
		c0 := &coeffs[round][0]
		c2 := &coeffs[round][1]
		c3 := &coeffs[round][2]
		c4 := &coeffs[round][3]
		c5 := &coeffs[round][4]
		c6 := &coeffs[round][5]
		c7 := &coeffs[round][6]

		rBits := api.ToBinary(challenges[round], 254)
		r := fq.FromBits(rBits...)

		// Reconstruct c1 = prevEval - 2*c0 - c2 - c3 - c4 - c5 - c6 - c7
		twoC0 := fq.Mul(two, c0)
		c1 := fq.Sub(prevEval, twoC0)
		c1 = fq.Sub(c1, c2)
		c1 = fq.Sub(c1, c3)
		c1 = fq.Sub(c1, c4)
		c1 = fq.Sub(c1, c5)
		c1 = fq.Sub(c1, c6)
		c1 = fq.Sub(c1, c7)

		// Horner's: p(r) = c0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*(c5 + r*(c6 + r*c7))))))
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

// ================================================
// Stage 2: Degree 6 polynomial (11 rounds)
// ================================================

// Stage2OnlyCircuit verifies Stage 2 sumcheck using emulated Fq
type Stage2OnlyCircuit struct {
	Coeffs     [Stage2Rounds][Stage2CoeffsNum]sw_grumpkin.Scalar `gnark:",public"`
	Challenges [Stage2Rounds]frontend.Variable                   `gnark:",public"`
	Expected   sw_grumpkin.Scalar                                `gnark:",public"`
}

func (c *Stage2OnlyCircuit) Define(api frontend.API) error {
	fq, err := emulated.NewField[sw_grumpkin.ScalarField](api)
	if err != nil {
		return err
	}
	return verifySumcheckDegree6(api, fq, c.Coeffs[:], c.Challenges[:], &c.Expected)
}

// verifySumcheckDegree6 verifies a sumcheck with degree-6 polynomials
// Coefficients: [c0, c2, c3, c4, c5, c6] (c1 is reconstructed)
func verifySumcheckDegree6(
	api frontend.API,
	fq *emulated.Field[sw_grumpkin.ScalarField],
	coeffs [][6]sw_grumpkin.Scalar,
	challenges []frontend.Variable,
	expected *sw_grumpkin.Scalar,
) error {
	numRounds := len(coeffs)
	prevEval := fq.Zero()
	twoBits := api.ToBinary(2, 254)
	two := fq.FromBits(twoBits...)

	for round := 0; round < numRounds; round++ {
		c0 := &coeffs[round][0]
		c2 := &coeffs[round][1]
		c3 := &coeffs[round][2]
		c4 := &coeffs[round][3]
		c5 := &coeffs[round][4]
		c6 := &coeffs[round][5]

		rBits := api.ToBinary(challenges[round], 254)
		r := fq.FromBits(rBits...)

		// Reconstruct c1 = prevEval - 2*c0 - c2 - c3 - c4 - c5 - c6
		twoC0 := fq.Mul(two, c0)
		c1 := fq.Sub(prevEval, twoC0)
		c1 = fq.Sub(c1, c2)
		c1 = fq.Sub(c1, c3)
		c1 = fq.Sub(c1, c4)
		c1 = fq.Sub(c1, c5)
		c1 = fq.Sub(c1, c6)

		// Horner's: p(r) = c0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*(c5 + r*c6)))))
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

// ================================================
// Stages 4 & 5: Degree 2 polynomial (22 and 88 rounds)
// ================================================

// Stage4OnlyCircuit verifies Stage 4 sumcheck using emulated Fq
type Stage4OnlyCircuit struct {
	Coeffs     [Stage4Rounds][Stage4CoeffsNum]sw_grumpkin.Scalar `gnark:",public"`
	Challenges [Stage4Rounds]frontend.Variable                   `gnark:",public"`
	Expected   sw_grumpkin.Scalar                                `gnark:",public"`
}

func (c *Stage4OnlyCircuit) Define(api frontend.API) error {
	fq, err := emulated.NewField[sw_grumpkin.ScalarField](api)
	if err != nil {
		return err
	}
	return verifySumcheckDegree2(api, fq, c.Coeffs[:], c.Challenges[:], &c.Expected)
}

// Stage5OnlyCircuit verifies Stage 5 sumcheck using emulated Fq
type Stage5OnlyCircuit struct {
	Coeffs     [Stage5Rounds][Stage5CoeffsNum]sw_grumpkin.Scalar `gnark:",public"`
	Challenges [Stage5Rounds]frontend.Variable                   `gnark:",public"`
	Expected   sw_grumpkin.Scalar                                `gnark:",public"`
}

func (c *Stage5OnlyCircuit) Define(api frontend.API) error {
	fq, err := emulated.NewField[sw_grumpkin.ScalarField](api)
	if err != nil {
		return err
	}
	return verifySumcheckDegree2(api, fq, c.Coeffs[:], c.Challenges[:], &c.Expected)
}

// verifySumcheckDegree2 verifies a sumcheck with degree-2 polynomials
// Coefficients: [c0, c2] (c1 is reconstructed)
func verifySumcheckDegree2(
	api frontend.API,
	fq *emulated.Field[sw_grumpkin.ScalarField],
	coeffs [][2]sw_grumpkin.Scalar,
	challenges []frontend.Variable,
	expected *sw_grumpkin.Scalar,
) error {
	numRounds := len(coeffs)
	prevEval := fq.Zero()
	twoBits := api.ToBinary(2, 254)
	two := fq.FromBits(twoBits...)

	for round := 0; round < numRounds; round++ {
		c0 := &coeffs[round][0]
		c2 := &coeffs[round][1]

		rBits := api.ToBinary(challenges[round], 254)
		r := fq.FromBits(rBits...)

		// Reconstruct c1 = prevEval - 2*c0 - c2
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

// ================================================
// Combined circuit for all stages
// ================================================

// AllRecursionStagesCircuit verifies all recursion sumcheck stages
type AllRecursionStagesCircuit struct {
	// Stage 1: 11 rounds, degree 7
	Stage1Coeffs     [Stage1Rounds][Stage1CoeffsNum]sw_grumpkin.Scalar `gnark:",public"`
	Stage1Challenges [Stage1Rounds]frontend.Variable                   `gnark:",public"`
	Stage1Expected   sw_grumpkin.Scalar                                `gnark:",public"`

	// Stage 2: 11 rounds, degree 6
	Stage2Coeffs     [Stage2Rounds][Stage2CoeffsNum]sw_grumpkin.Scalar `gnark:",public"`
	Stage2Challenges [Stage2Rounds]frontend.Variable                   `gnark:",public"`
	Stage2Expected   sw_grumpkin.Scalar                                `gnark:",public"`

	// Stage 4: 22 rounds, degree 2
	Stage4Coeffs     [Stage4Rounds][Stage4CoeffsNum]sw_grumpkin.Scalar `gnark:",public"`
	Stage4Challenges [Stage4Rounds]frontend.Variable                   `gnark:",public"`
	Stage4Expected   sw_grumpkin.Scalar                                `gnark:",public"`

	// Stage 5: 88 rounds, degree 2
	Stage5Coeffs     [Stage5Rounds][Stage5CoeffsNum]sw_grumpkin.Scalar `gnark:",public"`
	Stage5Challenges [Stage5Rounds]frontend.Variable                   `gnark:",public"`
	Stage5Expected   sw_grumpkin.Scalar                                `gnark:",public"`
}

func (c *AllRecursionStagesCircuit) Define(api frontend.API) error {
	fq, err := emulated.NewField[sw_grumpkin.ScalarField](api)
	if err != nil {
		return err
	}

	// Stage 1: degree 7
	if err := verifySumcheckDegree7(api, fq, c.Stage1Coeffs[:], c.Stage1Challenges[:], &c.Stage1Expected); err != nil {
		return err
	}

	// Stage 2: degree 6
	if err := verifySumcheckDegree6(api, fq, c.Stage2Coeffs[:], c.Stage2Challenges[:], &c.Stage2Expected); err != nil {
		return err
	}

	// Stage 4: degree 2
	if err := verifySumcheckDegree2(api, fq, c.Stage4Coeffs[:], c.Stage4Challenges[:], &c.Stage4Expected); err != nil {
		return err
	}

	// Stage 5: degree 2
	if err := verifySumcheckDegree2(api, fq, c.Stage5Coeffs[:], c.Stage5Challenges[:], &c.Stage5Expected); err != nil {
		return err
	}

	return nil
}

// Legacy type aliases for backwards compatibility
type FullRecursionCircuit = AllRecursionStagesCircuit
type AllSumcheckStagesCircuit = AllRecursionStagesCircuit

// verifySumcheckStage is a legacy wrapper for degree-7 verification
func verifySumcheckStage(
	api frontend.API,
	fq *emulated.Field[sw_grumpkin.ScalarField],
	coeffs [][CoeffsPerRound]sw_grumpkin.Scalar,
	challenges []frontend.Variable,
	expected *sw_grumpkin.Scalar,
) error {
	return verifySumcheckDegree7(api, fq, coeffs, challenges, expected)
}

// verifySumcheckStageFixed is a legacy wrapper
func verifySumcheckStageFixed(
	api frontend.API,
	fq *emulated.Field[sw_grumpkin.ScalarField],
	coeffs [][7]sw_grumpkin.Scalar,
	challenges []frontend.Variable,
	expected *sw_grumpkin.Scalar,
) error {
	return verifySumcheckDegree7(api, fq, coeffs, challenges, expected)
}
