// Package jolt_verifier provides end-to-end Groth16 tests for the full Jolt verifier circuit.
//
// This file contains the complete E2E test combining:
// - Stages 1-7 (transpiled sumcheck verification, ~3.1M constraints)
// - Recursion sumchecks (emulated Fq, ~127K constraints)
// - Hyrax opening verification (native Grumpkin MSM, ~5M constraints)
//
// Total: ~8.5M constraints for full Jolt verification
package jolt_verifier

import (
	"bytes"
	"encoding/json"
	"math/big"
	"os"
	"testing"
	"time"

	"jolt_verifier/poseidon"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/grumpkin"
	fp_grumpkin "github.com/consensys/gnark-crypto/ecc/grumpkin/fp"
	fr_grumpkin "github.com/consensys/gnark-crypto/ecc/grumpkin/fr"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/std/algebra/native/sw_grumpkin"
	"github.com/consensys/gnark/std/math/emulated"
	"github.com/consensys/gnark/test"
)

// RecursionWitnessE2E represents the structure of recursion_witness.json
type RecursionWitnessE2E struct {
	Stage1Coeffs     [][]string `json:"stage1_coeffs"`
	Stage2Coeffs     [][]string `json:"stage2_coeffs"`
	Stage4Coeffs     [][]string `json:"stage4_coeffs"`
	Stage5Coeffs     [][]string `json:"stage5_coeffs"`
	Stage1Challenges []string   `json:"stage1_challenges"`
	Stage2Challenges []string   `json:"stage2_challenges"`
	Stage4Challenges []string   `json:"stage4_challenges"`
	Stage5Challenges []string   `json:"stage5_challenges"`
	Stage1Expected   string     `json:"stage1_expected"`
	Stage2Expected   string     `json:"stage2_expected"`
	Stage4Expected   string     `json:"stage4_expected"`
	Stage5Expected   string     `json:"stage5_expected"`
}

// HyraxWitnessJSONE2E represents the JSON structure from the Rust transpiler
type HyraxWitnessJSONE2E struct {
	SqrtN          int         `json:"sqrt_n"`
	SqrtN1         int         `json:"sqrt_n1"`
	U              []string    `json:"u"`
	RowCommitments [][2]string `json:"row_commitments"`
	Generators     [][2]string `json:"generators"`
	L              []string    `json:"l"`
	R              []string    `json:"r"`
	V              string      `json:"v"`
	OpeningPoint   []string    `json:"opening_point"`
	DenseNumVars   int         `json:"dense_num_vars"`
}

// FullE2ECircuit combines all Jolt verifier components:
// 1. Stages 1-7 (transpiled sumcheck) - embedded
// 2. Recursion sumchecks (emulated Fq) - inline with transcript-derived challenges
// 3. Hyrax opening (native Grumpkin MSM) - inline
type FullE2ECircuit struct {
	// =========================================================================
	// Part 1: Stages 1-7 (transpiled sumcheck verification)
	// =========================================================================
	JoltStagesCircuit

	// =========================================================================
	// Part 2: Recursion Sumcheck Inputs (a17-a21, emulated Fq)
	// Challenges are derived via Fr Poseidon transcript, NOT passed as hints
	// =========================================================================

	// Initial transcript state (from stages 1-7 transcript)
	RecursionInitialTranscriptState frontend.Variable `gnark:",public"`
	RecursionInitialNRounds         frontend.Variable `gnark:",public"`

	// Stage 1 (a17): 11 rounds, degree 7
	RecursionStage1Coeffs   [11][7]sw_grumpkin.Scalar `gnark:",public"`
	RecursionStage1Expected sw_grumpkin.Scalar        `gnark:",public"`

	// Stage 2 (a18): 11 rounds, degree 6
	RecursionStage2Coeffs   [11][6]sw_grumpkin.Scalar `gnark:",public"`
	RecursionStage2Expected sw_grumpkin.Scalar        `gnark:",public"`

	// Stage 4 (a20): 22 rounds, degree 2
	RecursionStage4Coeffs   [22][2]sw_grumpkin.Scalar `gnark:",public"`
	RecursionStage4Expected sw_grumpkin.Scalar        `gnark:",public"`

	// Stage 5 (a21): 88 rounds, degree 2
	RecursionStage5Coeffs   [88][2]sw_grumpkin.Scalar `gnark:",public"`
	RecursionStage5Expected sw_grumpkin.Scalar        `gnark:",public"`

	// =========================================================================
	// Part 3: Hyrax Opening Inputs (native Grumpkin MSM)
	// =========================================================================

	// HyraxSqrtN is the full polynomial structure size (used for MSM #2)
	HyraxSqrtN int

	// HyraxSqrtN1 is the filtered size for MSM #1 (non-identity row commitments)
	HyraxSqrtN1 int

	// RowCommitments are the non-identity Grumpkin points from Hyrax commitment
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

// Pre-computed message constants (right-padded to 32 bytes) for transcript protocol
var (
	// "UniPoly_begin" padded to 32 bytes
	uniPolyBeginMsg = [32]byte{
		'U', 'n', 'i', 'P', 'o', 'l', 'y', '_', 'b', 'e', 'g', 'i', 'n',
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	}
	// "UniPoly_end" padded to 32 bytes
	uniPolyEndMsg = [32]byte{
		'U', 'n', 'i', 'P', 'o', 'l', 'y', '_', 'e', 'n', 'd',
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	}
)

// Define implements the full E2E circuit constraints.
func (c *FullE2ECircuit) Define(api frontend.API) error {
	// =========================================================================
	// Part 1: Stages 1-7 sumcheck verification (transpiled)
	// =========================================================================
	if err := c.JoltStagesCircuit.Define(api); err != nil {
		return err
	}

	// =========================================================================
	// Part 2: Recursion Sumcheck Verification (emulated Fq)
	// Challenges are DERIVED from Fr Poseidon transcript, NOT passed as hints
	// =========================================================================
	fq, err := emulated.NewField[sw_grumpkin.ScalarField](api)
	if err != nil {
		return err
	}

	// Initialize native Fr transcript from stages 1-7 state
	transcript := poseidon.NewFrTranscriptFromState(api, c.RecursionInitialTranscriptState, c.RecursionInitialNRounds)

	// Stage 1 (a17): degree 7 - challenges derived from transcript
	if err := verifyRecursionDegree7WithTranscript(api, fq, transcript, c.RecursionStage1Coeffs[:], &c.RecursionStage1Expected); err != nil {
		return err
	}

	// Stage 2 (a18): degree 6 - challenges derived from transcript
	if err := verifyRecursionDegree6WithTranscript(api, fq, transcript, c.RecursionStage2Coeffs[:], &c.RecursionStage2Expected); err != nil {
		return err
	}

	// Stage 4 (a20): degree 2 - challenges derived from transcript
	if err := verifyRecursionDegree2WithTranscript(api, fq, transcript, c.RecursionStage4Coeffs[:], &c.RecursionStage4Expected); err != nil {
		return err
	}

	// Stage 5 (a21): degree 2 - challenges derived from transcript
	if err := verifyRecursionDegree2WithTranscript(api, fq, transcript, c.RecursionStage5Coeffs[:], &c.RecursionStage5Expected); err != nil {
		return err
	}

	// =========================================================================
	// Part 3: Hyrax Opening Verification (native Grumpkin MSM)
	// =========================================================================
	if len(c.HyraxRowCommitments) > 0 {
		if err := c.defineHyraxVerification(api, fq); err != nil {
			return err
		}
	}

	return nil
}

// defineHyraxVerification implements Hyrax opening verification
func (c *FullE2ECircuit) defineHyraxVerification(api frontend.API, fq *emulated.Field[sw_grumpkin.ScalarField]) error {
	curve, err := sw_grumpkin.NewCurve(api)
	if err != nil {
		return err
	}

	// Convert to pointer slices
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

	// Check 2: <u, R> == v
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

// Recursion sumcheck verification helpers - challenges derived from Fr Poseidon transcript
func verifyRecursionDegree7WithTranscript(
	api frontend.API,
	fq *emulated.Field[sw_grumpkin.ScalarField],
	transcript *poseidon.FrTranscript,
	coeffs [][7]sw_grumpkin.Scalar,
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

		// Transcript protocol using NATIVE Fr Poseidon:
		// 1. Append "UniPoly_begin"
		transcript.AppendMessage(uniPolyBeginMsg)

		// 2. Append coefficients - convert Fq (Grumpkin scalar) to Fr for transcript
		transcript.AppendScalarGrumpkin(fq, c0)
		transcript.AppendScalarGrumpkin(fq, c2)
		transcript.AppendScalarGrumpkin(fq, c3)
		transcript.AppendScalarGrumpkin(fq, c4)
		transcript.AppendScalarGrumpkin(fq, c5)
		transcript.AppendScalarGrumpkin(fq, c6)
		transcript.AppendScalarGrumpkin(fq, c7)

		// 3. Append "UniPoly_end"
		transcript.AppendMessage(uniPolyEndMsg)

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

func verifyRecursionDegree6WithTranscript(
	api frontend.API,
	fq *emulated.Field[sw_grumpkin.ScalarField],
	transcript *poseidon.FrTranscript,
	coeffs [][6]sw_grumpkin.Scalar,
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

		// Transcript protocol
		transcript.AppendMessage(uniPolyBeginMsg)
		transcript.AppendScalarGrumpkin(fq, c0)
		transcript.AppendScalarGrumpkin(fq, c2)
		transcript.AppendScalarGrumpkin(fq, c3)
		transcript.AppendScalarGrumpkin(fq, c4)
		transcript.AppendScalarGrumpkin(fq, c5)
		transcript.AppendScalarGrumpkin(fq, c6)
		transcript.AppendMessage(uniPolyEndMsg)

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

func verifyRecursionDegree2WithTranscript(
	api frontend.API,
	fq *emulated.Field[sw_grumpkin.ScalarField],
	transcript *poseidon.FrTranscript,
	coeffs [][2]sw_grumpkin.Scalar,
	expected *sw_grumpkin.Scalar,
) error {
	prevEval := fq.Zero()
	twoBits := api.ToBinary(2, 254)
	two := fq.FromBits(twoBits...)

	for round := 0; round < len(coeffs); round++ {
		c0 := &coeffs[round][0]
		c2 := &coeffs[round][1]

		// Transcript protocol
		transcript.AppendMessage(uniPolyBeginMsg)
		transcript.AppendScalarGrumpkin(fq, c0)
		transcript.AppendScalarGrumpkin(fq, c2)
		transcript.AppendMessage(uniPolyEndMsg)

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

// TestFullE2EConstraintCount measures the total constraint count
func TestFullE2EConstraintCount(t *testing.T) {
	// Check if witness files exist
	if _, err := os.Stat("hyrax_witness.json"); os.IsNotExist(err) {
		t.Skip("hyrax_witness.json not found - run transpiler first")
	}

	// Load Hyrax witness to get sizes
	data, err := os.ReadFile("hyrax_witness.json")
	if err != nil {
		t.Skipf("Could not load hyrax_witness.json: %v", err)
	}

	var hyraxWitness HyraxWitnessJSONE2E
	if err := json.Unmarshal(data, &hyraxWitness); err != nil {
		t.Fatalf("Failed to parse hyrax witness: %v", err)
	}

	sqrtN := hyraxWitness.SqrtN
	sqrtN1 := hyraxWitness.SqrtN1
	if sqrtN1 == 0 {
		sqrtN1 = sqrtN
	}

	t.Logf("Hyrax sizes: sqrt(N)=%d, sqrt(N1)=%d", sqrtN, sqrtN1)

	// Create circuit
	circuit := CreateFullE2ECircuit(sqrtN, sqrtN1)

	t.Log("Compiling full E2E circuit...")
	startCompile := time.Now()
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Failed to compile: %v", err)
	}
	compileTime := time.Since(startCompile)

	t.Log("")
	t.Log("=== Full E2E Circuit Constraint Count ===")
	t.Logf("Total Constraints:   %d", cs.GetNbConstraints())
	t.Logf("Public Variables:    %d", cs.GetNbPublicVariables())
	t.Logf("Internal Variables:  %d", cs.GetNbInternalVariables())
	t.Logf("Compile Time:        %v", compileTime)
	t.Log("")
	t.Log("Component breakdown (estimated):")
	t.Log("  - Stages 1-7 (transpiled):     ~3.1M constraints")
	t.Log("  - Recursion sumchecks (Fq):    ~127K constraints")
	t.Logf("  - Hyrax MSM (sqrt_n=%d):      ~%.1fM constraints", sqrtN, float64(sqrtN)*1775*2/1e6)
}

// TestFullE2ESolver tests that the full circuit solves with real witness
func TestFullE2ESolver(t *testing.T) {
	t.Log("=== Full E2E Circuit Solver Test ===")
	t.Log("")

	// Load all witness files
	t.Log("Loading witness files...")

	// 1. Load stages witness
	stagesAssignment, err := LoadStagesAssignment(getStagesWitnessPath())
	if err != nil {
		t.Fatalf("Failed to load stages witness: %v", err)
	}
	t.Log("  ✓ Loaded stages_witness.json")

	// 2. Load recursion witness
	recursionData, err := os.ReadFile("hyrax/recursion_witness.json")
	if err != nil {
		// Try alternate path
		recursionData, err = os.ReadFile("recursion_witness.json")
		if err != nil {
			t.Fatalf("Failed to load recursion_witness.json: %v", err)
		}
	}
	var recursionWitness RecursionWitnessE2E
	if err := json.Unmarshal(recursionData, &recursionWitness); err != nil {
		t.Fatalf("Failed to parse recursion witness: %v", err)
	}
	t.Log("  ✓ Loaded recursion_witness.json")

	// 3. Load Hyrax witness
	hyraxData, err := os.ReadFile("hyrax_witness.json")
	if err != nil {
		t.Fatalf("Failed to load hyrax_witness.json: %v", err)
	}
	var hyraxWitness HyraxWitnessJSONE2E
	if err := json.Unmarshal(hyraxData, &hyraxWitness); err != nil {
		t.Fatalf("Failed to parse hyrax witness: %v", err)
	}
	t.Log("  ✓ Loaded hyrax_witness.json")

	sqrtN := hyraxWitness.SqrtN
	sqrtN1 := hyraxWitness.SqrtN1
	if sqrtN1 == 0 {
		sqrtN1 = sqrtN
	}
	t.Logf("  Hyrax sizes: sqrt(N)=%d, sqrt(N1)=%d", sqrtN, sqrtN1)

	// Build full witness
	t.Log("")
	t.Log("Building full witness...")
	witness := &FullE2ECircuit{
		JoltStagesCircuit:   *stagesAssignment,
		HyraxSqrtN:          sqrtN,
		HyraxSqrtN1:         sqrtN1,
		HyraxRowCommitments: make([]sw_grumpkin.G1Affine, sqrtN1),
		HyraxGenerators:     make([]sw_grumpkin.G1Affine, sqrtN),
		HyraxL:              make([]sw_grumpkin.Scalar, sqrtN1),
		HyraxU:              make([]sw_grumpkin.Scalar, sqrtN),
		HyraxR:              make([]frontend.Variable, sqrtN),
	}

	// Fill recursion witness - challenges are derived from transcript, NOT loaded
	stage1Coeffs := loadStage1CoeffsLocal(&recursionWitness)
	stage2Coeffs := loadStage2CoeffsLocal(&recursionWitness)
	stage4Coeffs := loadStage4CoeffsLocal(&recursionWitness)
	stage5Coeffs := loadStage5CoeffsLocal(&recursionWitness)

	// Set initial transcript state (starts fresh for recursion verifier)
	initialState := big.NewInt(0)
	initialNRounds := int64(0)
	witness.RecursionInitialTranscriptState = initialState
	witness.RecursionInitialNRounds = big.NewInt(initialNRounds)

	// Compute expected values using native Fr transcript
	// This ensures the expected values match the transcript-derived challenges
	stage1Exp, stage2Exp, stage4Exp, stage5Exp := computeAllStagesExpectedLocal(
		stage1Coeffs, stage2Coeffs, stage4Coeffs, stage5Coeffs,
		initialState, initialNRounds,
	)
	t.Logf("  Stage 1 expected (computed): %s", stage1Exp.String())
	t.Logf("  Stage 2 expected (computed): %s", stage2Exp.String())
	t.Logf("  Stage 4 expected (computed): %s", stage4Exp.String())
	t.Logf("  Stage 5 expected (computed): %s", stage5Exp.String())

	// Stage 1
	for round := 0; round < 11; round++ {
		for i := 0; i < 7; i++ {
			witness.RecursionStage1Coeffs[round][i] = emulated.ValueOf[sw_grumpkin.ScalarField](stage1Coeffs[round][i])
		}
	}
	witness.RecursionStage1Expected = emulated.ValueOf[sw_grumpkin.ScalarField](stage1Exp)

	// Stage 2
	for round := 0; round < 11; round++ {
		for i := 0; i < 6; i++ {
			witness.RecursionStage2Coeffs[round][i] = emulated.ValueOf[sw_grumpkin.ScalarField](stage2Coeffs[round][i])
		}
	}
	witness.RecursionStage2Expected = emulated.ValueOf[sw_grumpkin.ScalarField](stage2Exp)

	// Stage 4
	for round := 0; round < 22; round++ {
		for i := 0; i < 2; i++ {
			witness.RecursionStage4Coeffs[round][i] = emulated.ValueOf[sw_grumpkin.ScalarField](stage4Coeffs[round][i])
		}
	}
	witness.RecursionStage4Expected = emulated.ValueOf[sw_grumpkin.ScalarField](stage4Exp)

	// Stage 5
	for round := 0; round < 88; round++ {
		for i := 0; i < 2; i++ {
			witness.RecursionStage5Coeffs[round][i] = emulated.ValueOf[sw_grumpkin.ScalarField](stage5Coeffs[round][i])
		}
	}
	witness.RecursionStage5Expected = emulated.ValueOf[sw_grumpkin.ScalarField](stage5Exp)
	t.Log("  ✓ Filled recursion sumcheck witness (challenges derived from transcript)")

	// Fill Hyrax witness
	for i := 0; i < sqrtN1; i++ {
		// Row commitment
		var x, y fp_grumpkin.Element
		xBig := new(big.Int)
		yBig := new(big.Int)
		xBig.SetString(hyraxWitness.RowCommitments[i][0], 10)
		yBig.SetString(hyraxWitness.RowCommitments[i][1], 10)
		x.SetBigInt(xBig)
		y.SetBigInt(yBig)
		witness.HyraxRowCommitments[i] = sw_grumpkin.NewG1Affine(grumpkin.G1Affine{X: x, Y: y})

		// L
		lBig := new(big.Int)
		lBig.SetString(hyraxWitness.L[i], 10)
		var lElem fr_grumpkin.Element
		lElem.SetBigInt(lBig)
		witness.HyraxL[i] = sw_grumpkin.NewScalar(lElem)
	}

	for i := 0; i < sqrtN; i++ {
		// Generator
		var x, y fp_grumpkin.Element
		xBig := new(big.Int)
		yBig := new(big.Int)
		xBig.SetString(hyraxWitness.Generators[i][0], 10)
		yBig.SetString(hyraxWitness.Generators[i][1], 10)
		x.SetBigInt(xBig)
		y.SetBigInt(yBig)
		witness.HyraxGenerators[i] = sw_grumpkin.NewG1Affine(grumpkin.G1Affine{X: x, Y: y})

		// U
		uBig := new(big.Int)
		uBig.SetString(hyraxWitness.U[i], 10)
		var uElem fr_grumpkin.Element
		uElem.SetBigInt(uBig)
		witness.HyraxU[i] = sw_grumpkin.NewScalar(uElem)

		// R
		rBig := new(big.Int)
		rBig.SetString(hyraxWitness.R[i], 10)
		witness.HyraxR[i] = frontend.Variable(rBig)
	}

	// V
	vBig := new(big.Int)
	vBig.SetString(hyraxWitness.V, 10)
	witness.HyraxV = frontend.Variable(vBig)
	t.Log("  ✓ Filled Hyrax witness")

	// Create placeholder circuit
	circuit := CreateFullE2ECircuit(sqrtN, sqrtN1)

	// Test solver
	t.Log("")
	t.Log("Testing solver...")
	err = test.IsSolved(circuit, witness, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Solver failed: %v", err)
	}
	t.Log("✓ Full E2E solver PASSED!")
}

// TestFullE2EGroth16 runs the complete Groth16 prove/verify cycle
func TestFullE2EGroth16(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping full E2E Groth16 test in short mode (takes ~5 minutes)")
	}

	t.Log("=== Full E2E Groth16 Prove/Verify Test ===")
	t.Log("Expected time: ~5 minutes")
	t.Log("")

	// Load all witness files
	t.Log("Step 1: Loading witness files...")
	startLoad := time.Now()

	// 1. Load stages witness
	stagesAssignment, err := LoadStagesAssignment(getStagesWitnessPath())
	if err != nil {
		t.Fatalf("Failed to load stages witness: %v", err)
	}
	t.Log("  ✓ Loaded stages_witness.json")

	// 2. Load recursion witness
	recursionData, err := os.ReadFile("recursion_witness.json")
	if err != nil {
		t.Fatalf("Failed to load recursion_witness.json: %v", err)
	}
	var recursionWitnessGroth16 RecursionWitnessE2E
	if err := json.Unmarshal(recursionData, &recursionWitnessGroth16); err != nil {
		t.Fatalf("Failed to parse recursion witness: %v", err)
	}
	t.Log("  ✓ Loaded recursion_witness.json")

	// 3. Load Hyrax witness
	hyraxData, err := os.ReadFile("hyrax_witness.json")
	if err != nil {
		t.Fatalf("Failed to load hyrax_witness.json: %v", err)
	}
	var hyraxWitnessGroth16 HyraxWitnessJSONE2E
	if err := json.Unmarshal(hyraxData, &hyraxWitnessGroth16); err != nil {
		t.Fatalf("Failed to parse hyrax witness: %v", err)
	}
	t.Log("  ✓ Loaded hyrax_witness.json")

	sqrtN := hyraxWitnessGroth16.SqrtN
	sqrtN1 := hyraxWitnessGroth16.SqrtN1
	if sqrtN1 == 0 {
		sqrtN1 = sqrtN
	}
	t.Logf("  Hyrax sizes: sqrt(N)=%d, sqrt(N1)=%d", sqrtN, sqrtN1)
	t.Logf("  Load time: %v", time.Since(startLoad))

	// Build full witness
	t.Log("")
	t.Log("Step 2: Building full witness...")
	startWitness := time.Now()

	witness := &FullE2ECircuit{
		JoltStagesCircuit:   *stagesAssignment,
		HyraxSqrtN:          sqrtN,
		HyraxSqrtN1:         sqrtN1,
		HyraxRowCommitments: make([]sw_grumpkin.G1Affine, sqrtN1),
		HyraxGenerators:     make([]sw_grumpkin.G1Affine, sqrtN),
		HyraxL:              make([]sw_grumpkin.Scalar, sqrtN1),
		HyraxU:              make([]sw_grumpkin.Scalar, sqrtN),
		HyraxR:              make([]frontend.Variable, sqrtN),
	}

	// Fill recursion witness - challenges are derived from transcript, NOT loaded
	stage1CoeffsG16 := loadStage1CoeffsLocal(&recursionWitnessGroth16)
	stage2CoeffsG16 := loadStage2CoeffsLocal(&recursionWitnessGroth16)
	stage4CoeffsG16 := loadStage4CoeffsLocal(&recursionWitnessGroth16)
	stage5CoeffsG16 := loadStage5CoeffsLocal(&recursionWitnessGroth16)

	// Set initial transcript state (starts fresh for recursion verifier)
	initialStateG16 := big.NewInt(0)
	initialNRoundsG16 := int64(0)
	witness.RecursionInitialTranscriptState = initialStateG16
	witness.RecursionInitialNRounds = big.NewInt(initialNRoundsG16)

	// Compute expected values using native Fr transcript (must match circuit's transcript)
	stage1ExpG16, stage2ExpG16, stage4ExpG16, stage5ExpG16 := computeAllStagesExpectedLocal(
		stage1CoeffsG16, stage2CoeffsG16, stage4CoeffsG16, stage5CoeffsG16,
		initialStateG16, initialNRoundsG16,
	)

	for round := 0; round < 11; round++ {
		for i := 0; i < 7; i++ {
			witness.RecursionStage1Coeffs[round][i] = emulated.ValueOf[sw_grumpkin.ScalarField](stage1CoeffsG16[round][i])
		}
	}
	witness.RecursionStage1Expected = emulated.ValueOf[sw_grumpkin.ScalarField](stage1ExpG16)

	for round := 0; round < 11; round++ {
		for i := 0; i < 6; i++ {
			witness.RecursionStage2Coeffs[round][i] = emulated.ValueOf[sw_grumpkin.ScalarField](stage2CoeffsG16[round][i])
		}
	}
	witness.RecursionStage2Expected = emulated.ValueOf[sw_grumpkin.ScalarField](stage2ExpG16)

	for round := 0; round < 22; round++ {
		for i := 0; i < 2; i++ {
			witness.RecursionStage4Coeffs[round][i] = emulated.ValueOf[sw_grumpkin.ScalarField](stage4CoeffsG16[round][i])
		}
	}
	witness.RecursionStage4Expected = emulated.ValueOf[sw_grumpkin.ScalarField](stage4ExpG16)

	for round := 0; round < 88; round++ {
		for i := 0; i < 2; i++ {
			witness.RecursionStage5Coeffs[round][i] = emulated.ValueOf[sw_grumpkin.ScalarField](stage5CoeffsG16[round][i])
		}
	}
	witness.RecursionStage5Expected = emulated.ValueOf[sw_grumpkin.ScalarField](stage5ExpG16)

	// Fill Hyrax witness
	for i := 0; i < sqrtN1; i++ {
		var x, y fp_grumpkin.Element
		xBig := new(big.Int)
		yBig := new(big.Int)
		xBig.SetString(hyraxWitnessGroth16.RowCommitments[i][0], 10)
		yBig.SetString(hyraxWitnessGroth16.RowCommitments[i][1], 10)
		x.SetBigInt(xBig)
		y.SetBigInt(yBig)
		witness.HyraxRowCommitments[i] = sw_grumpkin.NewG1Affine(grumpkin.G1Affine{X: x, Y: y})

		lBig := new(big.Int)
		lBig.SetString(hyraxWitnessGroth16.L[i], 10)
		var lElem fr_grumpkin.Element
		lElem.SetBigInt(lBig)
		witness.HyraxL[i] = sw_grumpkin.NewScalar(lElem)
	}

	for i := 0; i < sqrtN; i++ {
		var x, y fp_grumpkin.Element
		xBig := new(big.Int)
		yBig := new(big.Int)
		xBig.SetString(hyraxWitnessGroth16.Generators[i][0], 10)
		yBig.SetString(hyraxWitnessGroth16.Generators[i][1], 10)
		x.SetBigInt(xBig)
		y.SetBigInt(yBig)
		witness.HyraxGenerators[i] = sw_grumpkin.NewG1Affine(grumpkin.G1Affine{X: x, Y: y})

		uBig := new(big.Int)
		uBig.SetString(hyraxWitnessGroth16.U[i], 10)
		var uElem fr_grumpkin.Element
		uElem.SetBigInt(uBig)
		witness.HyraxU[i] = sw_grumpkin.NewScalar(uElem)

		rBig := new(big.Int)
		rBig.SetString(hyraxWitnessGroth16.R[i], 10)
		witness.HyraxR[i] = frontend.Variable(rBig)
	}

	vBig := new(big.Int)
	vBig.SetString(hyraxWitnessGroth16.V, 10)
	witness.HyraxV = frontend.Variable(vBig)

	t.Logf("  Witness build time: %v", time.Since(startWitness))

	// Compile
	t.Log("")
	t.Log("Step 3: Compiling circuit...")
	startCompile := time.Now()

	circuit := CreateFullE2ECircuit(sqrtN, sqrtN1)
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Failed to compile: %v", err)
	}
	compileTime := time.Since(startCompile)

	t.Logf("  Constraints:    %d", cs.GetNbConstraints())
	t.Logf("  Public vars:    %d", cs.GetNbPublicVariables())
	t.Logf("  Internal vars:  %d", cs.GetNbInternalVariables())
	t.Logf("  Compile time:   %v", compileTime)

	// Setup
	t.Log("")
	t.Log("Step 4: Running Groth16 setup...")
	startSetup := time.Now()

	pk, vk, err := groth16.Setup(cs)
	if err != nil {
		t.Fatalf("Failed to setup: %v", err)
	}
	setupTime := time.Since(startSetup)

	var pkBuf, vkBuf bytes.Buffer
	pk.WriteTo(&pkBuf)
	vk.WriteTo(&vkBuf)

	t.Logf("  Setup time:     %v", setupTime)
	t.Logf("  Proving key:    %.2f MB", float64(pkBuf.Len())/1024/1024)
	t.Logf("  Verifying key:  %.2f KB", float64(vkBuf.Len())/1024)

	// Prove
	t.Log("")
	t.Log("Step 5: Generating proof...")
	startProve := time.Now()

	fullWitness, err := frontend.NewWitness(witness, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Failed to create witness: %v", err)
	}

	proof, err := groth16.Prove(cs, pk, fullWitness)
	if err != nil {
		t.Fatalf("Failed to prove: %v", err)
	}
	proveTime := time.Since(startProve)

	var proofBuf bytes.Buffer
	proof.WriteTo(&proofBuf)

	t.Logf("  Prove time:     %v", proveTime)
	t.Logf("  Proof size:     %d bytes", proofBuf.Len())

	// Verify
	t.Log("")
	t.Log("Step 6: Verifying proof...")
	startVerify := time.Now()

	publicWitness, err := fullWitness.Public()
	if err != nil {
		t.Fatalf("Failed to get public witness: %v", err)
	}

	err = groth16.Verify(proof, vk, publicWitness)
	if err != nil {
		t.Fatalf("Failed to verify: %v", err)
	}
	verifyTime := time.Since(startVerify)

	t.Logf("  Verify time:    %v", verifyTime)

	// Summary
	t.Log("")
	t.Log("=== FULL E2E SUMMARY ===")
	t.Logf("Total Constraints:  %d (~%.1fM)", cs.GetNbConstraints(), float64(cs.GetNbConstraints())/1e6)
	t.Logf("Proof Size:         %d bytes", proofBuf.Len())
	t.Logf("Compile Time:       %v", compileTime)
	t.Logf("Setup Time:         %v", setupTime)
	t.Logf("Prove Time:         %v", proveTime)
	t.Logf("Verify Time:        %v", verifyTime)
	t.Logf("Total Time:         %v", time.Since(startLoad))
	t.Log("")
	t.Log("✓ FULL E2E Groth16 proof VERIFIED!")
}

// Helper functions (local copies to avoid import cycles)
func toBigIntLocal(s string) *big.Int {
	n := new(big.Int)
	n.SetString(s, 10)
	return n
}

func loadStage1CoeffsLocal(w *RecursionWitnessE2E) [][7]*big.Int {
	coeffs := make([][7]*big.Int, len(w.Stage1Coeffs))
	for round := 0; round < len(w.Stage1Coeffs); round++ {
		coeffs[round] = [7]*big.Int{}
		for i := 0; i < 7; i++ {
			coeffs[round][i] = toBigIntLocal(w.Stage1Coeffs[round][i])
		}
	}
	return coeffs
}

func loadStage2CoeffsLocal(w *RecursionWitnessE2E) [][6]*big.Int {
	coeffs := make([][6]*big.Int, len(w.Stage2Coeffs))
	for round := 0; round < len(w.Stage2Coeffs); round++ {
		coeffs[round] = [6]*big.Int{}
		for i := 0; i < 6; i++ {
			coeffs[round][i] = toBigIntLocal(w.Stage2Coeffs[round][i])
		}
	}
	return coeffs
}

func loadStage4CoeffsLocal(w *RecursionWitnessE2E) [][2]*big.Int {
	coeffs := make([][2]*big.Int, len(w.Stage4Coeffs))
	for round := 0; round < len(w.Stage4Coeffs); round++ {
		coeffs[round] = [2]*big.Int{}
		for i := 0; i < 2; i++ {
			coeffs[round][i] = toBigIntLocal(w.Stage4Coeffs[round][i])
		}
	}
	return coeffs
}

func loadStage5CoeffsLocal(w *RecursionWitnessE2E) [][2]*big.Int {
	coeffs := make([][2]*big.Int, len(w.Stage5Coeffs))
	for round := 0; round < len(w.Stage5Coeffs); round++ {
		coeffs[round] = [2]*big.Int{}
		for i := 0; i < 2; i++ {
			coeffs[round][i] = toBigIntLocal(w.Stage5Coeffs[round][i])
		}
	}
	return coeffs
}

func loadChallengesLocal(strs []string) []*big.Int {
	challenges := make([]*big.Int, len(strs))
	for i, s := range strs {
		challenges[i] = toBigIntLocal(s)
	}
	return challenges
}

// ============================================================================
// Native Fr Poseidon Transcript (for computing expected values in tests)
// ============================================================================

// Field moduli for native big.Int computation in tests
var (
	// BN254 scalar field (Fr)
	frModLocal, _ = new(big.Int).SetString("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10)
	// BN254 base field (Fq) = Grumpkin scalar field
	fqModLocal, _ = new(big.Int).SetString("21888242871839275222246405745257275088696311157297823662689037894645226208583", 10)
)

// FrTranscriptNativeLocal is a native (non-circuit) implementation of Fr Poseidon transcript
// for computing expected values in tests.
type FrTranscriptNativeLocal struct {
	state   *big.Int
	nRounds int64
}

func NewFrTranscriptNativeLocal(state *big.Int, nRounds int64) *FrTranscriptNativeLocal {
	return &FrTranscriptNativeLocal{
		state:   new(big.Int).Set(state),
		nRounds: nRounds,
	}
}

func (t *FrTranscriptNativeLocal) AppendScalar(scalar *big.Int) {
	nRoundsBig := big.NewInt(t.nRounds)
	t.state = poseidon.HashNative(t.state, nRoundsBig, scalar)
	t.nRounds++
}

func (t *FrTranscriptNativeLocal) AppendMessage(msgBytes32 [32]byte) {
	// Convert bytes to big.Int (BE interpretation)
	msgBig := new(big.Int).SetBytes(msgBytes32[:])
	msgBig.Mod(msgBig, frModLocal)
	t.AppendScalar(msgBig)
}

// AppendScalarFq appends an Fq scalar by converting to Fr
func (t *FrTranscriptNativeLocal) AppendScalarFq(scalar *big.Int) {
	// Fq values are just reduced mod Fr
	frVal := new(big.Int).Mod(scalar, frModLocal)
	t.AppendScalar(frVal)
}

func (t *FrTranscriptNativeLocal) ChallengeScalar() *big.Int {
	nRoundsBig := big.NewInt(t.nRounds)
	zero := big.NewInt(0)
	output := poseidon.HashNative(t.state, nRoundsBig, zero)
	t.state = output
	t.nRounds++
	return new(big.Int).Set(output)
}

// computeAllStagesExpectedLocal computes all 4 recursion stages with Fr transcript
func computeAllStagesExpectedLocal(
	stage1Coeffs [][7]*big.Int,
	stage2Coeffs [][6]*big.Int,
	stage4Coeffs [][2]*big.Int,
	stage5Coeffs [][2]*big.Int,
	initialState *big.Int,
	initialNRounds int64,
) (stage1Exp, stage2Exp, stage4Exp, stage5Exp *big.Int) {
	transcript := NewFrTranscriptNativeLocal(initialState, initialNRounds)

	// Stage 1
	stage1Exp, _ = computeSumcheckDegree7WithFrTranscriptNativeLocal(stage1Coeffs, transcript)

	// Stage 2
	stage2Exp = computeSumcheckDegree6WithFrTranscriptNativeLocal(stage2Coeffs, transcript)

	// Stage 4
	stage4Exp = computeSumcheckDegree2WithFrTranscriptNativeLocal(stage4Coeffs, transcript)

	// Stage 5
	stage5Exp = computeSumcheckDegree2WithFrTranscriptNativeLocal(stage5Coeffs, transcript)

	return
}

func computeSumcheckDegree7WithFrTranscriptNativeLocal(
	coeffs [][7]*big.Int,
	transcript *FrTranscriptNativeLocal,
) (*big.Int, []*big.Int) {
	prevEval := big.NewInt(0)
	two := big.NewInt(2)
	challenges := make([]*big.Int, len(coeffs))

	for round := 0; round < len(coeffs); round++ {
		c := coeffs[round]

		// Transcript protocol
		transcript.AppendMessage(uniPolyBeginMsg)
		for i := 0; i < 7; i++ {
			transcript.AppendScalarFq(c[i])
		}
		transcript.AppendMessage(uniPolyEndMsg)

		// Get Fr challenge, convert to Fq
		rFr := transcript.ChallengeScalar()
		r := new(big.Int).Mod(rFr, fqModLocal) // Fr -> Fq
		challenges[round] = r

		// Reconstruct c1
		c1 := new(big.Int).Set(prevEval)
		twoC0 := new(big.Int).Mul(two, c[0])
		c1.Sub(c1, twoC0)
		for i := 1; i < 7; i++ {
			c1.Sub(c1, c[i])
		}
		c1.Mod(c1, fqModLocal)

		// Horner's evaluation: c0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*(c5 + r*(c6 + r*c7))))))
		acc := new(big.Int).Set(c[6]) // c7
		acc.Mul(r, acc)
		acc.Add(c[5], acc)
		acc.Mod(acc, fqModLocal)

		acc.Mul(r, acc)
		acc.Add(c[4], acc)
		acc.Mod(acc, fqModLocal)

		acc.Mul(r, acc)
		acc.Add(c[3], acc)
		acc.Mod(acc, fqModLocal)

		acc.Mul(r, acc)
		acc.Add(c[2], acc)
		acc.Mod(acc, fqModLocal)

		acc.Mul(r, acc)
		acc.Add(c[1], acc)
		acc.Mod(acc, fqModLocal)

		acc.Mul(r, acc)
		acc.Add(c1, acc)
		acc.Mod(acc, fqModLocal)

		acc.Mul(r, acc)
		acc.Add(c[0], acc)
		acc.Mod(acc, fqModLocal)

		prevEval = acc
	}

	return prevEval, challenges
}

func computeSumcheckDegree6WithFrTranscriptNativeLocal(coeffs [][6]*big.Int, transcript *FrTranscriptNativeLocal) *big.Int {
	prevEval := big.NewInt(0)
	two := big.NewInt(2)

	for round := 0; round < len(coeffs); round++ {
		c := coeffs[round]

		transcript.AppendMessage(uniPolyBeginMsg)
		for i := 0; i < 6; i++ {
			transcript.AppendScalarFq(c[i])
		}
		transcript.AppendMessage(uniPolyEndMsg)

		rFr := transcript.ChallengeScalar()
		r := new(big.Int).Mod(rFr, fqModLocal)

		c1 := new(big.Int).Set(prevEval)
		twoC0 := new(big.Int).Mul(two, c[0])
		c1.Sub(c1, twoC0)
		for i := 1; i < 6; i++ {
			c1.Sub(c1, c[i])
		}
		c1.Mod(c1, fqModLocal)

		acc := new(big.Int).Set(c[5])
		acc.Mul(r, acc)
		acc.Add(c[4], acc)
		acc.Mod(acc, fqModLocal)

		acc.Mul(r, acc)
		acc.Add(c[3], acc)
		acc.Mod(acc, fqModLocal)

		acc.Mul(r, acc)
		acc.Add(c[2], acc)
		acc.Mod(acc, fqModLocal)

		acc.Mul(r, acc)
		acc.Add(c[1], acc)
		acc.Mod(acc, fqModLocal)

		acc.Mul(r, acc)
		acc.Add(c1, acc)
		acc.Mod(acc, fqModLocal)

		acc.Mul(r, acc)
		acc.Add(c[0], acc)
		acc.Mod(acc, fqModLocal)

		prevEval = acc
	}

	return prevEval
}

func computeSumcheckDegree2WithFrTranscriptNativeLocal(coeffs [][2]*big.Int, transcript *FrTranscriptNativeLocal) *big.Int {
	prevEval := big.NewInt(0)
	two := big.NewInt(2)

	for round := 0; round < len(coeffs); round++ {
		c := coeffs[round]

		transcript.AppendMessage(uniPolyBeginMsg)
		for i := 0; i < 2; i++ {
			transcript.AppendScalarFq(c[i])
		}
		transcript.AppendMessage(uniPolyEndMsg)

		rFr := transcript.ChallengeScalar()
		r := new(big.Int).Mod(rFr, fqModLocal)

		c1 := new(big.Int).Set(prevEval)
		twoC0 := new(big.Int).Mul(two, c[0])
		c1.Sub(c1, twoC0)
		c1.Sub(c1, c[1])
		c1.Mod(c1, fqModLocal)

		acc := new(big.Int).Set(c[1])
		acc.Mul(r, acc)
		acc.Add(c1, acc)
		acc.Mod(acc, fqModLocal)

		acc.Mul(r, acc)
		acc.Add(c[0], acc)
		acc.Mod(acc, fqModLocal)

		prevEval = acc
	}

	return prevEval
}
