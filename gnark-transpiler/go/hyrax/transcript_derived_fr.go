// Package hyrax implements recursion sumcheck verification with native Fr transcript.
//
// This file provides an alternative to the emulated Fq transcript approach.
// Instead of computing Poseidon over Fq (~30K constraints/hash), we use native Fr Poseidon
// (~250 constraints/hash) and convert challenges from Fr to Fq when needed.
//
// The sumcheck coefficients and polynomial evaluation remain in emulated Fq (unavoidable),
// but the transcript operations are ~120x cheaper.
//
// Constraint comparison for Stage 1 (11 rounds, ~10 hashes/round):
//   - Fq transcript: ~110 hashes × 30K = ~3.3M constraints
//   - Fr transcript: ~110 hashes × 250 = ~27.5K constraints
//
// The Fr→Fq conversion for challenges costs ~254 constraints per challenge (bit decomposition).
package hyrax

import (
	"jolt_verifier/poseidon"

	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/emulated"
)

// TranscriptDerivedFrStage1Circuit verifies Stage 1 sumcheck with NATIVE Fr transcript.
// This is much more efficient than FqTranscript because native Poseidon is ~120x cheaper.
//
// Key insight: The transcript just needs to produce consistent challenges.
// It doesn't need to operate in the same field as the sumcheck computation.
type TranscriptDerivedFrStage1Circuit struct {
	// Coefficients for each round: [c0, c2, c3, c4, c5, c6, c7]
	// These are Fq elements (Grumpkin scalars) - must use emulated arithmetic
	Coeffs [Stage1Rounds][Stage1CoeffsNum]FqElem `gnark:",secret"`

	// Initial transcript state (native Fr)
	InitialTranscriptState frontend.Variable `gnark:",public"`
	InitialNRounds         frontend.Variable `gnark:",public"`

	// Expected final evaluation (Fq element)
	Expected FqElem `gnark:",public"`
}

func (c *TranscriptDerivedFrStage1Circuit) Define(api frontend.API) error {
	fq, err := emulated.NewField[emulated.BN254Fp](api)
	if err != nil {
		return err
	}

	// Initialize NATIVE Fr transcript
	transcript := poseidon.NewFrTranscriptFromState(api, c.InitialTranscriptState, c.InitialNRounds)

	// Verify sumcheck with Fr transcript (challenges converted to Fq)
	return verifySumcheckDegree7WithFrTranscript(api, fq, transcript, c.Coeffs[:], &c.Expected)
}

// verifySumcheckDegree7WithFrTranscript verifies a degree-7 sumcheck using native Fr transcript.
// Challenges are derived in Fr (cheap) and converted to Fq (cheap bit conversion).
func verifySumcheckDegree7WithFrTranscript(
	api frontend.API,
	fq *emulated.Field[emulated.BN254Fp],
	transcript *poseidon.FrTranscript,
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

		// Transcript protocol using NATIVE Fr Poseidon:
		// 1. Append "UniPoly_begin"
		transcript.AppendMessage(UniPolyBeginMsg)

		// 2. Append coefficients - need to convert Fq to Fr for transcript
		// The coefficients are Fq elements, we convert them to Fr via bits
		transcript.AppendScalarFq(fq, c0)
		transcript.AppendScalarFq(fq, c2)
		transcript.AppendScalarFq(fq, c3)
		transcript.AppendScalarFq(fq, c4)
		transcript.AppendScalarFq(fq, c5)
		transcript.AppendScalarFq(fq, c6)
		transcript.AppendScalarFq(fq, c7)

		// 3. Append "UniPoly_end"
		transcript.AppendMessage(UniPolyEndMsg)

		// 4. Derive challenge in Fr, convert to Fq
		rFr := transcript.ChallengeScalar()
		rBits := api.ToBinary(rFr, 254)
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

// TranscriptDerivedFrAllStagesCircuit verifies all recursion sumcheck stages with native Fr transcript.
type TranscriptDerivedFrAllStagesCircuit struct {
	// Stage 1: 11 rounds, degree 7
	Stage1Coeffs [Stage1Rounds][Stage1CoeffsNum]FqElem `gnark:",secret"`

	// Stage 2: 11 rounds, degree 6
	Stage2Coeffs [Stage2Rounds][Stage2CoeffsNum]FqElem `gnark:",secret"`

	// Stage 4: 22 rounds, degree 2
	Stage4Coeffs [Stage4Rounds][Stage4CoeffsNum]FqElem `gnark:",secret"`

	// Stage 5: 88 rounds, degree 2
	Stage5Coeffs [Stage5Rounds][Stage5CoeffsNum]FqElem `gnark:",secret"`

	// Initial transcript state (native Fr)
	InitialTranscriptState frontend.Variable `gnark:",public"`
	InitialNRounds         frontend.Variable `gnark:",public"`

	// Expected final evaluations (Fq elements)
	Stage1Expected FqElem `gnark:",public"`
	Stage2Expected FqElem `gnark:",public"`
	Stage4Expected FqElem `gnark:",public"`
	Stage5Expected FqElem `gnark:",public"`
}

func (c *TranscriptDerivedFrAllStagesCircuit) Define(api frontend.API) error {
	fq, err := emulated.NewField[emulated.BN254Fp](api)
	if err != nil {
		return err
	}

	// Initialize native Fr transcript
	transcript := poseidon.NewFrTranscriptFromState(api, c.InitialTranscriptState, c.InitialNRounds)

	// Stage 1: degree 7
	if err := verifySumcheckDegree7WithFrTranscript(api, fq, transcript, c.Stage1Coeffs[:], &c.Stage1Expected); err != nil {
		return err
	}

	// Stage 2: degree 6
	if err := verifySumcheckDegree6WithFrTranscript(api, fq, transcript, c.Stage2Coeffs[:], &c.Stage2Expected); err != nil {
		return err
	}

	// Stage 4: degree 2
	if err := verifySumcheckDegree2WithFrTranscript(api, fq, transcript, c.Stage4Coeffs[:], &c.Stage4Expected); err != nil {
		return err
	}

	// Stage 5: degree 2
	if err := verifySumcheckDegree2WithFrTranscript(api, fq, transcript, c.Stage5Coeffs[:], &c.Stage5Expected); err != nil {
		return err
	}

	return nil
}

// verifySumcheckDegree6WithFrTranscript verifies a degree-6 sumcheck using native Fr transcript.
func verifySumcheckDegree6WithFrTranscript(
	api frontend.API,
	fq *emulated.Field[emulated.BN254Fp],
	transcript *poseidon.FrTranscript,
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
		transcript.AppendScalarFq(fq, c0)
		transcript.AppendScalarFq(fq, c2)
		transcript.AppendScalarFq(fq, c3)
		transcript.AppendScalarFq(fq, c4)
		transcript.AppendScalarFq(fq, c5)
		transcript.AppendScalarFq(fq, c6)
		transcript.AppendMessage(UniPolyEndMsg)

		rFr := transcript.ChallengeScalar()
		rBits := api.ToBinary(rFr, 254)
		r := fq.FromBits(rBits...)

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

// verifySumcheckDegree2WithFrTranscript verifies a degree-2 sumcheck using native Fr transcript.
func verifySumcheckDegree2WithFrTranscript(
	api frontend.API,
	fq *emulated.Field[emulated.BN254Fp],
	transcript *poseidon.FrTranscript,
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
		transcript.AppendScalarFq(fq, c0)
		transcript.AppendScalarFq(fq, c2)
		transcript.AppendMessage(UniPolyEndMsg)

		rFr := transcript.ChallengeScalar()
		rBits := api.ToBinary(rFr, 254)
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
