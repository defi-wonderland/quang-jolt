package hyrax

import (
	"math/big"
	"testing"
	"time"

	"jolt_verifier/poseidon"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/std/math/emulated"
	"github.com/consensys/gnark/test"
)

// FrTranscriptNative is a native (non-circuit) implementation of Fr Poseidon transcript
// for computing expected values in tests.
type FrTranscriptNative struct {
	state   *big.Int
	nRounds int64
}

func NewFrTranscriptNative(state *big.Int, nRounds int64) *FrTranscriptNative {
	return &FrTranscriptNative{
		state:   new(big.Int).Set(state),
		nRounds: nRounds,
	}
}

func (t *FrTranscriptNative) AppendScalar(scalar *big.Int) {
	nRoundsBig := big.NewInt(t.nRounds)
	t.state = poseidon.HashNative(t.state, nRoundsBig, scalar)
	t.nRounds++
}

func (t *FrTranscriptNative) AppendMessage(msgBytes32 [32]byte) {
	// Convert bytes to big.Int (BE interpretation)
	msgBig := new(big.Int).SetBytes(msgBytes32[:])
	msgBig.Mod(msgBig, FrMod)
	t.AppendScalar(msgBig)
}

// AppendScalarFq appends an Fq scalar by converting to Fr
func (t *FrTranscriptNative) AppendScalarFq(scalar *big.Int) {
	// Fq values are just reduced mod Fr
	frVal := new(big.Int).Mod(scalar, FrMod)
	t.AppendScalar(frVal)
}

func (t *FrTranscriptNative) ChallengeScalar() *big.Int {
	nRoundsBig := big.NewInt(t.nRounds)
	zero := big.NewInt(0)
	output := poseidon.HashNative(t.state, nRoundsBig, zero)
	t.state = output
	t.nRounds++
	return new(big.Int).Set(output)
}

// computeSumcheckWithFrTranscript computes sumcheck with Fr transcript-derived challenges
func computeSumcheckDegree7WithFrTranscriptNative(
	coeffs [][7]*big.Int,
	transcript *FrTranscriptNative,
) (*big.Int, []*big.Int) {
	prevEval := big.NewInt(0)
	two := big.NewInt(2)
	challenges := make([]*big.Int, len(coeffs))

	for round := 0; round < len(coeffs); round++ {
		c := coeffs[round]

		// Transcript protocol
		transcript.AppendMessage(UniPolyBeginMsg)
		for i := 0; i < 7; i++ {
			transcript.AppendScalarFq(c[i])
		}
		transcript.AppendMessage(UniPolyEndMsg)

		// Get Fr challenge, convert to Fq
		rFr := transcript.ChallengeScalar()
		r := new(big.Int).Mod(rFr, FqMod) // Fr -> Fq
		challenges[round] = r

		// Reconstruct c1
		c1 := new(big.Int).Set(prevEval)
		twoC0 := new(big.Int).Mul(two, c[0])
		c1.Sub(c1, twoC0)
		for i := 1; i < 7; i++ {
			c1.Sub(c1, c[i])
		}
		c1.Mod(c1, FqMod)

		// Horner's evaluation
		acc := new(big.Int).Set(c[6]) // c7
		acc.Mul(r, acc)
		acc.Add(c[5], acc)
		acc.Mod(acc, FqMod)

		acc.Mul(r, acc)
		acc.Add(c[4], acc)
		acc.Mod(acc, FqMod)

		acc.Mul(r, acc)
		acc.Add(c[3], acc)
		acc.Mod(acc, FqMod)

		acc.Mul(r, acc)
		acc.Add(c[2], acc)
		acc.Mod(acc, FqMod)

		acc.Mul(r, acc)
		acc.Add(c[1], acc)
		acc.Mod(acc, FqMod)

		acc.Mul(r, acc)
		acc.Add(c1, acc)
		acc.Mod(acc, FqMod)

		acc.Mul(r, acc)
		acc.Add(c[0], acc)
		acc.Mod(acc, FqMod)

		prevEval = acc
	}

	return prevEval, challenges
}

// TestFrTranscriptStage1Solver tests that the circuit solves correctly
func TestFrTranscriptStage1Solver(t *testing.T) {
	w := loadRecursionWitness(t)
	stage1Coeffs := loadStage1Coeffs(w)

	// Use initial state = 0, nRounds = 0
	initialState := big.NewInt(0)
	initialNRounds := int64(0)

	// Compute expected using Fr transcript
	transcript := NewFrTranscriptNative(initialState, initialNRounds)
	expected, challenges := computeSumcheckDegree7WithFrTranscriptNative(stage1Coeffs, transcript)

	t.Logf("Initial state: %s", initialState.String())
	t.Logf("Expected final: %s", expected.String())
	t.Logf("First challenge (Fq): %s", challenges[0].String())

	// Build circuit witness
	circuit := &TranscriptDerivedFrStage1Circuit{}
	witness := &TranscriptDerivedFrStage1Circuit{}

	for round := 0; round < Stage1Rounds; round++ {
		for i := 0; i < Stage1CoeffsNum; i++ {
			witness.Coeffs[round][i] = emulated.ValueOf[emulated.BN254Fp](stage1Coeffs[round][i])
		}
	}

	witness.InitialTranscriptState = initialState
	witness.InitialNRounds = initialNRounds
	witness.Expected = emulated.ValueOf[emulated.BN254Fp](expected)

	err := test.IsSolved(circuit, witness, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("FrTranscriptStage1 solver failed: %v", err)
	}
	t.Log("FrTranscriptStage1 solver PASSED!")
}

// TestFrTranscriptStage1Groth16 tests full Groth16 prove/verify
func TestFrTranscriptStage1Groth16(t *testing.T) {
	w := loadRecursionWitness(t)
	stage1Coeffs := loadStage1Coeffs(w)

	// Compute expected
	initialState := big.NewInt(0)
	initialNRounds := int64(0)
	transcript := NewFrTranscriptNative(initialState, initialNRounds)
	expected, _ := computeSumcheckDegree7WithFrTranscriptNative(stage1Coeffs, transcript)

	t.Log("=== Compiling circuit ===")
	startCompile := time.Now()
	circuit := &TranscriptDerivedFrStage1Circuit{}
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Failed to compile: %v", err)
	}
	t.Logf("Compilation took %v", time.Since(startCompile))
	t.Logf("Constraints: %d", cs.GetNbConstraints())

	t.Log("=== Running Groth16 setup ===")
	startSetup := time.Now()
	pk, vk, err := groth16.Setup(cs)
	if err != nil {
		t.Fatalf("Setup failed: %v", err)
	}
	t.Logf("Setup took %v", time.Since(startSetup))

	// Build witness
	witness := &TranscriptDerivedFrStage1Circuit{}
	for round := 0; round < Stage1Rounds; round++ {
		for i := 0; i < Stage1CoeffsNum; i++ {
			witness.Coeffs[round][i] = emulated.ValueOf[emulated.BN254Fp](stage1Coeffs[round][i])
		}
	}
	witness.InitialTranscriptState = initialState
	witness.InitialNRounds = initialNRounds
	witness.Expected = emulated.ValueOf[emulated.BN254Fp](expected)

	fullWitness, err := frontend.NewWitness(witness, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Failed to create witness: %v", err)
	}

	t.Log("=== Proving ===")
	startProve := time.Now()
	proof, err := groth16.Prove(cs, pk, fullWitness)
	if err != nil {
		t.Fatalf("Prove failed: %v", err)
	}
	t.Logf("Prove took %v", time.Since(startProve))

	t.Log("=== Verifying ===")
	publicWitness, err := fullWitness.Public()
	if err != nil {
		t.Fatalf("Failed to get public witness: %v", err)
	}

	startVerify := time.Now()
	err = groth16.Verify(proof, vk, publicWitness)
	if err != nil {
		t.Fatalf("Verify failed: %v", err)
	}
	t.Logf("Verify took %v", time.Since(startVerify))

	t.Log("✅ FrTranscriptStage1 Groth16 PASSED!")
}

// computeAllStagesWithFrTranscript computes all 4 recursion stages with Fr transcript
func computeAllStagesWithFrTranscriptNative(
	stage1Coeffs [][7]*big.Int,
	stage2Coeffs [][6]*big.Int,
	stage4Coeffs [][2]*big.Int,
	stage5Coeffs [][2]*big.Int,
	initialState *big.Int,
	initialNRounds int64,
) (stage1Exp, stage2Exp, stage4Exp, stage5Exp *big.Int) {
	transcript := NewFrTranscriptNative(initialState, initialNRounds)

	// Stage 1
	stage1Exp, _ = computeSumcheckDegree7WithFrTranscriptNative(stage1Coeffs, transcript)

	// Stage 2
	stage2Exp = computeSumcheckDegree6WithFrTranscriptNative(stage2Coeffs, transcript)

	// Stage 4
	stage4Exp = computeSumcheckDegree2WithFrTranscriptNative(stage4Coeffs, transcript)

	// Stage 5
	stage5Exp = computeSumcheckDegree2WithFrTranscriptNative(stage5Coeffs, transcript)

	return
}

func computeSumcheckDegree6WithFrTranscriptNative(coeffs [][6]*big.Int, transcript *FrTranscriptNative) *big.Int {
	prevEval := big.NewInt(0)
	two := big.NewInt(2)

	for round := 0; round < len(coeffs); round++ {
		c := coeffs[round]

		transcript.AppendMessage(UniPolyBeginMsg)
		for i := 0; i < 6; i++ {
			transcript.AppendScalarFq(c[i])
		}
		transcript.AppendMessage(UniPolyEndMsg)

		rFr := transcript.ChallengeScalar()
		r := new(big.Int).Mod(rFr, FqMod)

		c1 := new(big.Int).Set(prevEval)
		twoC0 := new(big.Int).Mul(two, c[0])
		c1.Sub(c1, twoC0)
		for i := 1; i < 6; i++ {
			c1.Sub(c1, c[i])
		}
		c1.Mod(c1, FqMod)

		acc := new(big.Int).Set(c[5])
		acc.Mul(r, acc)
		acc.Add(c[4], acc)
		acc.Mod(acc, FqMod)

		acc.Mul(r, acc)
		acc.Add(c[3], acc)
		acc.Mod(acc, FqMod)

		acc.Mul(r, acc)
		acc.Add(c[2], acc)
		acc.Mod(acc, FqMod)

		acc.Mul(r, acc)
		acc.Add(c[1], acc)
		acc.Mod(acc, FqMod)

		acc.Mul(r, acc)
		acc.Add(c1, acc)
		acc.Mod(acc, FqMod)

		acc.Mul(r, acc)
		acc.Add(c[0], acc)
		acc.Mod(acc, FqMod)

		prevEval = acc
	}

	return prevEval
}

func computeSumcheckDegree2WithFrTranscriptNative(coeffs [][2]*big.Int, transcript *FrTranscriptNative) *big.Int {
	prevEval := big.NewInt(0)
	two := big.NewInt(2)

	for round := 0; round < len(coeffs); round++ {
		c := coeffs[round]

		transcript.AppendMessage(UniPolyBeginMsg)
		for i := 0; i < 2; i++ {
			transcript.AppendScalarFq(c[i])
		}
		transcript.AppendMessage(UniPolyEndMsg)

		rFr := transcript.ChallengeScalar()
		r := new(big.Int).Mod(rFr, FqMod)

		c1 := new(big.Int).Set(prevEval)
		twoC0 := new(big.Int).Mul(two, c[0])
		c1.Sub(c1, twoC0)
		c1.Sub(c1, c[1])
		c1.Mod(c1, FqMod)

		acc := new(big.Int).Set(c[1])
		acc.Mul(r, acc)
		acc.Add(c1, acc)
		acc.Mod(acc, FqMod)

		acc.Mul(r, acc)
		acc.Add(c[0], acc)
		acc.Mod(acc, FqMod)

		prevEval = acc
	}

	return prevEval
}

// TestFrTranscriptAllStagesSolver tests all stages solver
func TestFrTranscriptAllStagesSolver(t *testing.T) {
	w := loadRecursionWitness(t)
	stage1Coeffs := loadStage1Coeffs(w)
	stage2Coeffs := loadStage2Coeffs(w)
	stage4Coeffs := loadStage4Coeffs(w)
	stage5Coeffs := loadStage5Coeffs(w)

	initialState := big.NewInt(0)
	initialNRounds := int64(0)

	stage1Exp, stage2Exp, stage4Exp, stage5Exp := computeAllStagesWithFrTranscriptNative(
		stage1Coeffs, stage2Coeffs, stage4Coeffs, stage5Coeffs,
		initialState, initialNRounds,
	)

	t.Logf("Stage 1 expected: %s", stage1Exp.String())
	t.Logf("Stage 2 expected: %s", stage2Exp.String())
	t.Logf("Stage 4 expected: %s", stage4Exp.String())
	t.Logf("Stage 5 expected: %s", stage5Exp.String())

	circuit := &TranscriptDerivedFrAllStagesCircuit{}
	witness := &TranscriptDerivedFrAllStagesCircuit{}

	for round := 0; round < Stage1Rounds; round++ {
		for i := 0; i < Stage1CoeffsNum; i++ {
			witness.Stage1Coeffs[round][i] = emulated.ValueOf[emulated.BN254Fp](stage1Coeffs[round][i])
		}
	}
	for round := 0; round < Stage2Rounds; round++ {
		for i := 0; i < Stage2CoeffsNum; i++ {
			witness.Stage2Coeffs[round][i] = emulated.ValueOf[emulated.BN254Fp](stage2Coeffs[round][i])
		}
	}
	for round := 0; round < Stage4Rounds; round++ {
		for i := 0; i < Stage4CoeffsNum; i++ {
			witness.Stage4Coeffs[round][i] = emulated.ValueOf[emulated.BN254Fp](stage4Coeffs[round][i])
		}
	}
	for round := 0; round < Stage5Rounds; round++ {
		for i := 0; i < Stage5CoeffsNum; i++ {
			witness.Stage5Coeffs[round][i] = emulated.ValueOf[emulated.BN254Fp](stage5Coeffs[round][i])
		}
	}

	witness.InitialTranscriptState = initialState
	witness.InitialNRounds = initialNRounds
	witness.Stage1Expected = emulated.ValueOf[emulated.BN254Fp](stage1Exp)
	witness.Stage2Expected = emulated.ValueOf[emulated.BN254Fp](stage2Exp)
	witness.Stage4Expected = emulated.ValueOf[emulated.BN254Fp](stage4Exp)
	witness.Stage5Expected = emulated.ValueOf[emulated.BN254Fp](stage5Exp)

	err := test.IsSolved(circuit, witness, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("FrTranscriptAllStages solver failed: %v", err)
	}
	t.Log("FrTranscriptAllStages solver PASSED!")
}

// TestFrTranscriptAllStagesGroth16 tests full Groth16 for all stages
func TestFrTranscriptAllStagesGroth16(t *testing.T) {
	w := loadRecursionWitness(t)
	stage1Coeffs := loadStage1Coeffs(w)
	stage2Coeffs := loadStage2Coeffs(w)
	stage4Coeffs := loadStage4Coeffs(w)
	stage5Coeffs := loadStage5Coeffs(w)

	initialState := big.NewInt(0)
	initialNRounds := int64(0)

	stage1Exp, stage2Exp, stage4Exp, stage5Exp := computeAllStagesWithFrTranscriptNative(
		stage1Coeffs, stage2Coeffs, stage4Coeffs, stage5Coeffs,
		initialState, initialNRounds,
	)

	t.Log("=== Compiling all stages circuit ===")
	startCompile := time.Now()
	circuit := &TranscriptDerivedFrAllStagesCircuit{}
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Failed to compile: %v", err)
	}
	t.Logf("Compilation took %v", time.Since(startCompile))
	t.Logf("Constraints: %d", cs.GetNbConstraints())

	t.Log("=== Running Groth16 setup ===")
	startSetup := time.Now()
	pk, vk, err := groth16.Setup(cs)
	if err != nil {
		t.Fatalf("Setup failed: %v", err)
	}
	t.Logf("Setup took %v", time.Since(startSetup))

	// Build witness
	witness := &TranscriptDerivedFrAllStagesCircuit{}
	for round := 0; round < Stage1Rounds; round++ {
		for i := 0; i < Stage1CoeffsNum; i++ {
			witness.Stage1Coeffs[round][i] = emulated.ValueOf[emulated.BN254Fp](stage1Coeffs[round][i])
		}
	}
	for round := 0; round < Stage2Rounds; round++ {
		for i := 0; i < Stage2CoeffsNum; i++ {
			witness.Stage2Coeffs[round][i] = emulated.ValueOf[emulated.BN254Fp](stage2Coeffs[round][i])
		}
	}
	for round := 0; round < Stage4Rounds; round++ {
		for i := 0; i < Stage4CoeffsNum; i++ {
			witness.Stage4Coeffs[round][i] = emulated.ValueOf[emulated.BN254Fp](stage4Coeffs[round][i])
		}
	}
	for round := 0; round < Stage5Rounds; round++ {
		for i := 0; i < Stage5CoeffsNum; i++ {
			witness.Stage5Coeffs[round][i] = emulated.ValueOf[emulated.BN254Fp](stage5Coeffs[round][i])
		}
	}

	witness.InitialTranscriptState = initialState
	witness.InitialNRounds = initialNRounds
	witness.Stage1Expected = emulated.ValueOf[emulated.BN254Fp](stage1Exp)
	witness.Stage2Expected = emulated.ValueOf[emulated.BN254Fp](stage2Exp)
	witness.Stage4Expected = emulated.ValueOf[emulated.BN254Fp](stage4Exp)
	witness.Stage5Expected = emulated.ValueOf[emulated.BN254Fp](stage5Exp)

	fullWitness, err := frontend.NewWitness(witness, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Failed to create witness: %v", err)
	}

	t.Log("=== Proving ===")
	startProve := time.Now()
	proof, err := groth16.Prove(cs, pk, fullWitness)
	if err != nil {
		t.Fatalf("Prove failed: %v", err)
	}
	t.Logf("Prove took %v", time.Since(startProve))

	t.Log("=== Verifying ===")
	publicWitness, err := fullWitness.Public()
	if err != nil {
		t.Fatalf("Failed to get public witness: %v", err)
	}

	startVerify := time.Now()
	err = groth16.Verify(proof, vk, publicWitness)
	if err != nil {
		t.Fatalf("Verify failed: %v", err)
	}
	t.Logf("Verify took %v", time.Since(startVerify))

	t.Log("✅ FrTranscriptAllStages Groth16 PASSED!")
	t.Logf("Total constraints: %d", cs.GetNbConstraints())
}
