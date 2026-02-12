// Package hyrax implements recursion sumcheck verification with transcript-derived challenges.
//
// This file contains circuits that derive Fiat-Shamir challenges from the Poseidon transcript
// rather than accepting them as public inputs. This enables "real transcript deduction"
// where the circuit computes challenges internally.
//
// Transcript protocol for each sumcheck round (matches Rust CompressedUniPoly::append_to_transcript):
//   1. AppendMessage("UniPoly_begin")
//   2. For each coefficient c0, c2, c3, ...: AppendScalar(c_i)
//   3. AppendMessage("UniPoly_end")
//   4. ChallengeScalar() -> r_i
//
// Note: This file uses emulated.BN254Fp directly (same as sw_grumpkin.ScalarField / Grumpkin Fr)
// to be compatible with FqTranscript which uses BN254Fp for emulated arithmetic.
package hyrax

import (
	"jolt_verifier/poseidon"

	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/emulated"
)

// Pre-computed message constants (right-padded to 32 bytes)
var (
	// "UniPoly_begin" padded to 32 bytes
	UniPolyBeginMsg = [32]byte{
		'U', 'n', 'i', 'P', 'o', 'l', 'y', '_', 'b', 'e', 'g', 'i', 'n',
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	}
	// "UniPoly_end" padded to 32 bytes
	UniPolyEndMsg = [32]byte{
		'U', 'n', 'i', 'P', 'o', 'l', 'y', '_', 'e', 'n', 'd',
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	}
)

// FqElem is an alias for BN254 base field element (= Grumpkin scalar field)
// Using FqElem instead of Fq to avoid conflict with the variable in full_recursion_test.go
type FqElem = emulated.Element[emulated.BN254Fp]

// TranscriptDerivedStage1Circuit verifies Stage 1 sumcheck with transcript-derived challenges.
// Instead of accepting challenges as public inputs, this circuit:
// 1. Takes the transcript state at the start of Stage 1 as input
// 2. For each round, derives the challenge by:
//   - Appending "UniPoly_begin"
//   - Appending coefficients c0, c2, c3, c4, c5, c6, c7
//   - Appending "UniPoly_end"
//   - Squeezing a challenge
//
// 3. Verifies the sumcheck using derived challenges
type TranscriptDerivedStage1Circuit struct {
	// Coefficients for each round: [c0, c2, c3, c4, c5, c6, c7]
	// These are secret (private) inputs - the prover provides them
	Coeffs [Stage1Rounds][Stage1CoeffsNum]FqElem `gnark:",secret"`

	// Initial transcript state (at the start of this stage)
	// This would be the state after stages 1-7 of the main Jolt verification
	InitialTranscriptState FqElem            `gnark:",public"`
	InitialNRounds         frontend.Variable `gnark:",public"`

	// Expected final evaluation (public commitment to verify against)
	Expected FqElem `gnark:",public"`
}

func (c *TranscriptDerivedStage1Circuit) Define(api frontend.API) error {
	fq, err := emulated.NewField[emulated.BN254Fp](api)
	if err != nil {
		return err
	}

	// Initialize transcript from given state
	transcript := poseidon.NewFqTranscriptFromState(api, fq, &c.InitialTranscriptState, c.InitialNRounds)

	// Verify sumcheck with transcript-derived challenges
	return verifySumcheckDegree7WithTranscript(api, fq, transcript, c.Coeffs[:], &c.Expected)
}

// verifySumcheckDegree7WithTranscript verifies a degree-7 sumcheck while deriving challenges.
func verifySumcheckDegree7WithTranscript(
	api frontend.API,
	fq *emulated.Field[emulated.BN254Fp],
	transcript *poseidon.FqTranscript,
	coeffs [][7]FqElem,
	expected *FqElem,
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

		// Transcript protocol:
		// 1. Append "UniPoly_begin"
		transcript.AppendMessage(UniPolyBeginMsg)

		// 2. Append coefficients (c0, c2, c3, c4, c5, c6, c7)
		transcript.AppendScalar(c0)
		transcript.AppendScalar(c2)
		transcript.AppendScalar(c3)
		transcript.AppendScalar(c4)
		transcript.AppendScalar(c5)
		transcript.AppendScalar(c6)
		transcript.AppendScalar(c7)

		// 3. Append "UniPoly_end"
		transcript.AppendMessage(UniPolyEndMsg)

		// 4. Derive challenge
		r := transcript.ChallengeScalar()

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

// TranscriptDerivedAllStagesCircuit verifies all recursion sumcheck stages with transcript-derived challenges.
type TranscriptDerivedAllStagesCircuit struct {
	// Stage 1: 11 rounds, degree 7
	Stage1Coeffs [Stage1Rounds][Stage1CoeffsNum]FqElem `gnark:",secret"`

	// Stage 2: 11 rounds, degree 6
	Stage2Coeffs [Stage2Rounds][Stage2CoeffsNum]FqElem `gnark:",secret"`

	// Stage 4: 22 rounds, degree 2
	Stage4Coeffs [Stage4Rounds][Stage4CoeffsNum]FqElem `gnark:",secret"`

	// Stage 5: 88 rounds, degree 2
	Stage5Coeffs [Stage5Rounds][Stage5CoeffsNum]FqElem `gnark:",secret"`

	// Initial transcript state
	InitialTranscriptState FqElem            `gnark:",public"`
	InitialNRounds         frontend.Variable `gnark:",public"`

	// Expected final evaluations (public)
	Stage1Expected FqElem `gnark:",public"`
	Stage2Expected FqElem `gnark:",public"`
	Stage4Expected FqElem `gnark:",public"`
	Stage5Expected FqElem `gnark:",public"`
}

func (c *TranscriptDerivedAllStagesCircuit) Define(api frontend.API) error {
	fq, err := emulated.NewField[emulated.BN254Fp](api)
	if err != nil {
		return err
	}

	// Initialize transcript from given state
	transcript := poseidon.NewFqTranscriptFromState(api, fq, &c.InitialTranscriptState, c.InitialNRounds)

	// Stage 1: degree 7
	if err := verifySumcheckDegree7WithTranscript(api, fq, transcript, c.Stage1Coeffs[:], &c.Stage1Expected); err != nil {
		return err
	}

	// Stage 2: degree 6
	if err := verifySumcheckDegree6WithTranscript(api, fq, transcript, c.Stage2Coeffs[:], &c.Stage2Expected); err != nil {
		return err
	}

	// Stage 4: degree 2
	if err := verifySumcheckDegree2WithTranscript(api, fq, transcript, c.Stage4Coeffs[:], &c.Stage4Expected); err != nil {
		return err
	}

	// Stage 5: degree 2
	if err := verifySumcheckDegree2WithTranscript(api, fq, transcript, c.Stage5Coeffs[:], &c.Stage5Expected); err != nil {
		return err
	}

	return nil
}

// verifySumcheckDegree6WithTranscript verifies a degree-6 sumcheck while deriving challenges.
func verifySumcheckDegree6WithTranscript(
	api frontend.API,
	fq *emulated.Field[emulated.BN254Fp],
	transcript *poseidon.FqTranscript,
	coeffs [][6]FqElem,
	expected *FqElem,
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

		// Transcript protocol
		transcript.AppendMessage(UniPolyBeginMsg)
		transcript.AppendScalar(c0)
		transcript.AppendScalar(c2)
		transcript.AppendScalar(c3)
		transcript.AppendScalar(c4)
		transcript.AppendScalar(c5)
		transcript.AppendScalar(c6)
		transcript.AppendMessage(UniPolyEndMsg)

		r := transcript.ChallengeScalar()

		// Reconstruct c1 = prevEval - 2*c0 - c2 - c3 - c4 - c5 - c6
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

// verifySumcheckDegree2WithTranscript verifies a degree-2 sumcheck while deriving challenges.
func verifySumcheckDegree2WithTranscript(
	api frontend.API,
	fq *emulated.Field[emulated.BN254Fp],
	transcript *poseidon.FqTranscript,
	coeffs [][2]FqElem,
	expected *FqElem,
) error {
	numRounds := len(coeffs)
	prevEval := fq.Zero()
	twoBits := api.ToBinary(2, 254)
	two := fq.FromBits(twoBits...)

	for round := 0; round < numRounds; round++ {
		c0 := &coeffs[round][0]
		c2 := &coeffs[round][1]

		// Transcript protocol
		transcript.AppendMessage(UniPolyBeginMsg)
		transcript.AppendScalar(c0)
		transcript.AppendScalar(c2)
		transcript.AppendMessage(UniPolyEndMsg)

		r := transcript.ChallengeScalar()

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
