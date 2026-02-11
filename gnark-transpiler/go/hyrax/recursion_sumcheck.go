// Package hyrax implements the recursion sumcheck verifier using emulated Fq arithmetic.
//
// The recursion stages (a17-a21) operate on Fq (BN254 base field) values, not Fr (scalar field).
// This circuit uses gnark's emulated field arithmetic via sw_grumpkin.ScalarField to properly
// compute polynomial evaluations in Fq within the BN254 Groth16 circuit (which operates in Fr).
//
// BN254/Grumpkin 2-cycle relationship:
// - BN254 scalar field Fr = Grumpkin base field
// - BN254 base field Fq = Grumpkin scalar field
// So sw_grumpkin.ScalarField gives us emulated Fq arithmetic.
package hyrax

import (
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/algebra/native/sw_grumpkin"
	"github.com/consensys/gnark/std/math/emulated"
)

// NumRecursionRounds is the number of sumcheck rounds in recursion stage 1
const NumRecursionRounds = 11

// RecursionSumcheckCircuit verifies a single recursion sumcheck (e.g., PackedGtExpVerifier).
//
// The sumcheck has 11 rounds with degree-7 polynomials. For each round:
//   - Coefficients [c0, c2, c3, c4, c5, c6, c7] are provided as witness (in Fq)
//   - c1 is reconstructed from the binding: p(0) + p(1) = prev_round_eval
//   - Challenge r is derived from Poseidon transcript (in Fr, converted to Fq)
//   - Polynomial evaluation p(r) = c0 + c1*r + c2*r^2 + ... + c7*r^7 is computed in Fq
//
// The assertion checks that the final evaluation matches the claimed value.
type RecursionSumcheckCircuit struct {
	// Coefficients for each round: [c0, c2, c3, c4, c5, c6, c7]
	// Note: c1 is reconstructed, so we have 7 coefficients per round
	// Layout: Coeffs[round][i] where i=0 is c0, i=1 is c2, i=2 is c3, etc.
	Coeffs [NumRecursionRounds][7]sw_grumpkin.Scalar `gnark:",public"`

	// Challenges for each round (derived from Poseidon, values in Fr)
	// These need to be converted to Fq for polynomial evaluation
	Challenges [NumRecursionRounds]frontend.Variable `gnark:",public"`

	// Initial claimed sum (the value we're checking against)
	// This is p(0) + p(1) for round 0, which should equal the claimed evaluation
	InitialClaim sw_grumpkin.Scalar `gnark:",public"`

	// Expected final evaluation at the combined challenge point
	FinalEval sw_grumpkin.Scalar `gnark:",public"`
}

// Define implements the circuit constraints for recursion sumcheck verification.
func (c *RecursionSumcheckCircuit) Define(api frontend.API) error {
	// Initialize emulated scalar field for Grumpkin scalars (= BN254 Fq)
	fq, err := emulated.NewField[sw_grumpkin.ScalarField](api)
	if err != nil {
		return err
	}

	// Start with the initial claim
	prevEval := &c.InitialClaim

	// Process each sumcheck round
	for round := 0; round < NumRecursionRounds; round++ {
		// Get coefficients for this round
		// Layout: c[0]=c0, c[1]=c2, c[2]=c3, c[3]=c4, c[4]=c5, c[5]=c6, c[6]=c7
		coeffs := &c.Coeffs[round]

		// Convert challenge from Fr (native) to Fq (emulated)
		// Decompose to bits and reconstruct in emulated field
		rBits := api.ToBinary(c.Challenges[round], 254)
		r := fq.FromBits(rBits...)

		// Reconstruct c1 from the binding constraint:
		// p(0) + p(1) = prevEval
		// where p(0) = c0 and p(1) = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7
		// So: 2*c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 = prevEval
		// c1 = prevEval - 2*c0 - c2 - c3 - c4 - c5 - c6 - c7
		c0 := &coeffs[0]
		c2 := &coeffs[1]
		c3 := &coeffs[2]
		c4 := &coeffs[3]
		c5 := &coeffs[4]
		c6 := &coeffs[5]
		c7 := &coeffs[6]

		// c1 = prevEval - 2*c0 - c2 - c3 - c4 - c5 - c6 - c7
		two := fq.FromBits(api.ToBinary(2, 254)...)
		twoC0 := fq.Mul(two, c0)
		c1 := fq.Sub(prevEval, twoC0)
		c1 = fq.Sub(c1, c2)
		c1 = fq.Sub(c1, c3)
		c1 = fq.Sub(c1, c4)
		c1 = fq.Sub(c1, c5)
		c1 = fq.Sub(c1, c6)
		c1 = fq.Sub(c1, c7)

		// Evaluate p(r) using Horner's method:
		// p(r) = c0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*(c5 + r*(c6 + r*c7))))))
		acc := c7
		acc = fq.Add(c6, fq.Mul(r, acc))
		acc = fq.Add(c5, fq.Mul(r, acc))
		acc = fq.Add(c4, fq.Mul(r, acc))
		acc = fq.Add(c3, fq.Mul(r, acc))
		acc = fq.Add(c2, fq.Mul(r, acc))
		acc = fq.Add(c1, fq.Mul(r, acc))
		acc = fq.Add(c0, fq.Mul(r, acc))

		// This round's evaluation becomes the next round's claim
		prevEval = acc
	}

	// After all rounds, prevEval should equal the final evaluation
	fq.AssertIsEqual(prevEval, &c.FinalEval)

	return nil
}

// RecursionSumcheckWithTranscriptCircuit is the full circuit that includes
// Poseidon transcript challenge derivation along with sumcheck verification.
// This is what we'll use to replace the transpiled a17 assertion.
type RecursionSumcheckWithTranscriptCircuit struct {
	// Coefficients for each round: [c0, c2, c3, c4, c5, c6, c7]
	// These are Fq elements (Grumpkin scalar field)
	Coeffs [NumRecursionRounds][7]sw_grumpkin.Scalar `gnark:",public"`

	// Transcript state before recursion stage (provided as input)
	// This is the Poseidon state after processing all prior protocol messages
	TranscriptState frontend.Variable `gnark:",public"`

	// Initial claimed sum
	InitialClaim sw_grumpkin.Scalar `gnark:",public"`

	// Expected final evaluation
	FinalEval sw_grumpkin.Scalar `gnark:",public"`
}

// ComputeRecursionSumcheck verifies the sumcheck given external challenges.
// This is a helper that can be called from the main circuit.
func ComputeRecursionSumcheck(
	api frontend.API,
	fq *emulated.Field[sw_grumpkin.ScalarField],
	coeffs [][7]*sw_grumpkin.Scalar,
	challenges []frontend.Variable,
	initialClaim *sw_grumpkin.Scalar,
	finalEval *sw_grumpkin.Scalar,
) error {
	if len(coeffs) != len(challenges) {
		panic("coeffs and challenges must have same length")
	}

	prevEval := initialClaim

	for round := 0; round < len(coeffs); round++ {
		c := coeffs[round]
		c0, c2, c3, c4, c5, c6, c7 := c[0], c[1], c[2], c[3], c[4], c[5], c[6]

		// Convert challenge from Fr to Fq
		rBits := api.ToBinary(challenges[round], 254)
		r := fq.FromBits(rBits...)

		// Reconstruct c1
		two := fq.FromBits(api.ToBinary(2, 254)...)
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

	fq.AssertIsEqual(prevEval, finalEval)
	return nil
}

// VerifyRecursionSumcheckStage1 verifies the Recursion Stage 1 sumcheck using emulated Fq.
//
// This function is designed to be called from the main circuit to replace the
// transpiled a17 assertion which incorrectly uses Fr arithmetic on Fq values.
//
// Parameters:
// - api: gnark frontend API
// - coeffs: polynomial coefficients for each round [c0, c2, c3, c4, c5, c6, c7] (Fq values)
// - challenges: sumcheck challenges from Poseidon transcript (Fr values, but fit in 128 bits)
// - expectedOutputClaim: the expected final evaluation (Fq value)
//
// The initial claim for PackedGtExpVerifier is 0.
func VerifyRecursionSumcheckStage1(
	api frontend.API,
	coeffs [11][7]sw_grumpkin.Scalar,
	challenges [11]frontend.Variable,
	expectedOutputClaim sw_grumpkin.Scalar,
) error {
	// Initialize emulated scalar field for Grumpkin scalars (= BN254 Fq)
	fq, err := emulated.NewField[sw_grumpkin.ScalarField](api)
	if err != nil {
		return err
	}

	// Initial claim is 0 for PackedGtExpVerifier
	zero := fq.Zero()
	prevEval := zero

	// Process each sumcheck round
	for round := 0; round < 11; round++ {
		c0 := &coeffs[round][0]
		c2 := &coeffs[round][1]
		c3 := &coeffs[round][2]
		c4 := &coeffs[round][3]
		c5 := &coeffs[round][4]
		c6 := &coeffs[round][5]
		c7 := &coeffs[round][6]

		// Convert challenge from Fr (native) to Fq (emulated)
		rBits := api.ToBinary(challenges[round], 254)
		r := fq.FromBits(rBits...)

		// Reconstruct c1 = prevEval - 2*c0 - c2 - c3 - c4 - c5 - c6 - c7
		two := fq.FromBits(api.ToBinary(2, 254)...)
		twoC0 := fq.Mul(two, c0)
		c1 := fq.Sub(prevEval, twoC0)
		c1 = fq.Sub(c1, c2)
		c1 = fq.Sub(c1, c3)
		c1 = fq.Sub(c1, c4)
		c1 = fq.Sub(c1, c5)
		c1 = fq.Sub(c1, c6)
		c1 = fq.Sub(c1, c7)

		// Evaluate p(r) using Horner's method:
		// p(r) = c0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*(c5 + r*(c6 + r*c7))))))
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

	// Final check: computed evaluation should equal expected output claim
	fq.AssertIsEqual(prevEval, &expectedOutputClaim)

	return nil
}
