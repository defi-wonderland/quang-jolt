package hyrax

import (
	"encoding/json"
	"math/big"
	"os"
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

// Field moduli for native big.Int computation in tests
var (
	// BN254 scalar field (Fr)
	FrMod, _ = new(big.Int).SetString("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10)
	// BN254 base field (Fq) = Grumpkin scalar field
	FqMod, _ = new(big.Int).SetString("21888242871839275222246405745257275088696311157297823662689037894645226208583", 10)
)

// hashNativeLocal computes native Poseidon hash for tests (using circom constants)
func hashNativeLocal(in1, in2, in3 *big.Int) *big.Int {
	state := [4]*big.Int{big.NewInt(0), new(big.Int).Set(in1), new(big.Int).Set(in2), new(big.Int).Set(in3)}

	cConsts := poseidon.GetCConstants()
	mMatrix := poseidon.GetMMatrix()

	fullRounds := 8
	partialRounds := 56
	width := 4
	halfFull := fullRounds / 2

	exp5 := func(x *big.Int) *big.Int {
		x2 := new(big.Int).Mul(x, x)
		x2.Mod(x2, FrMod)
		x4 := new(big.Int).Mul(x2, x2)
		x4.Mod(x4, FrMod)
		x5 := new(big.Int).Mul(x4, x)
		x5.Mod(x5, FrMod)
		return x5
	}

	mix := func(st [4]*big.Int) [4]*big.Int {
		var result [4]*big.Int
		for i := 0; i < width; i++ {
			acc := new(big.Int)
			for j := 0; j < width; j++ {
				term := new(big.Int).Mul(mMatrix[j][i], st[j])
				acc.Add(acc, term)
			}
			acc.Mod(acc, FrMod)
			result[i] = acc
		}
		return result
	}

	constIdx := 0

	// First half full rounds
	for r := 0; r < halfFull; r++ {
		for i := 0; i < width; i++ {
			state[i] = new(big.Int).Add(state[i], cConsts[constIdx])
			state[i].Mod(state[i], FrMod)
			constIdx++
		}
		for i := 0; i < width; i++ {
			state[i] = exp5(state[i])
		}
		state = mix(state)
	}

	// Partial rounds
	for r := 0; r < partialRounds; r++ {
		for i := 0; i < width; i++ {
			state[i] = new(big.Int).Add(state[i], cConsts[constIdx])
			state[i].Mod(state[i], FrMod)
			constIdx++
		}
		state[0] = exp5(state[0])
		state = mix(state)
	}

	// Last half full rounds
	for r := 0; r < halfFull; r++ {
		for i := 0; i < width; i++ {
			state[i] = new(big.Int).Add(state[i], cConsts[constIdx])
			state[i].Mod(state[i], FrMod)
			constIdx++
		}
		for i := 0; i < width; i++ {
			state[i] = exp5(state[i])
		}
		state = mix(state)
	}

	return state[0]
}

// RecursionWitness represents the structure of recursion_witness.json
type RecursionWitness struct {
	Stage1Coeffs [][]string `json:"stage1_coeffs"`
	Stage2Coeffs [][]string `json:"stage2_coeffs"`
	Stage4Coeffs [][]string `json:"stage4_coeffs"`
	Stage5Coeffs [][]string `json:"stage5_coeffs"`
}

func loadRecursionWitness(t *testing.T) *RecursionWitness {
	data, err := os.ReadFile("recursion_witness.json")
	if err != nil {
		t.Skipf("Could not load recursion_witness.json: %v", err)
	}
	var w RecursionWitness
	if err := json.Unmarshal(data, &w); err != nil {
		t.Fatalf("Failed to parse recursion witness: %v", err)
	}
	return &w
}

func toBigInt(s string) *big.Int {
	n := new(big.Int)
	n.SetString(s, 10)
	return n
}

func loadStage1Coeffs(w *RecursionWitness) [][7]*big.Int {
	coeffs := make([][7]*big.Int, len(w.Stage1Coeffs))
	for round := 0; round < len(w.Stage1Coeffs); round++ {
		coeffs[round] = [7]*big.Int{}
		for i := 0; i < 7; i++ {
			coeffs[round][i] = toBigInt(w.Stage1Coeffs[round][i])
		}
	}
	return coeffs
}

func loadStage2Coeffs(w *RecursionWitness) [][6]*big.Int {
	coeffs := make([][6]*big.Int, len(w.Stage2Coeffs))
	for round := 0; round < len(w.Stage2Coeffs); round++ {
		coeffs[round] = [6]*big.Int{}
		for i := 0; i < 6; i++ {
			coeffs[round][i] = toBigInt(w.Stage2Coeffs[round][i])
		}
	}
	return coeffs
}

func loadStage4Coeffs(w *RecursionWitness) [][2]*big.Int {
	coeffs := make([][2]*big.Int, len(w.Stage4Coeffs))
	for round := 0; round < len(w.Stage4Coeffs); round++ {
		coeffs[round] = [2]*big.Int{}
		for i := 0; i < 2; i++ {
			coeffs[round][i] = toBigInt(w.Stage4Coeffs[round][i])
		}
	}
	return coeffs
}

func loadStage5Coeffs(w *RecursionWitness) [][2]*big.Int {
	coeffs := make([][2]*big.Int, len(w.Stage5Coeffs))
	for round := 0; round < len(w.Stage5Coeffs); round++ {
		coeffs[round] = [2]*big.Int{}
		for i := 0; i < 2; i++ {
			coeffs[round][i] = toBigInt(w.Stage5Coeffs[round][i])
		}
	}
	return coeffs
}

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
	t.state = hashNativeLocal(t.state, nRoundsBig, scalar)
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
	output := hashNativeLocal(t.state, nRoundsBig, zero)
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
