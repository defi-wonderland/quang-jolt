package hyrax

import (
	"encoding/json"
	"math/big"
	"os"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/std/algebra/native/sw_grumpkin"
	"github.com/consensys/gnark/std/math/emulated"
	"github.com/consensys/gnark/test"
)

// Fq modulus for computations
var Fq, _ = new(big.Int).SetString("21888242871839275222246405745257275088696311157297823662689037894645226208583", 10)

// RecursionWitness represents the structure of recursion_witness.json
type RecursionWitness struct {
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

// loadRecursionWitness loads the recursion witness JSON file
func loadRecursionWitness(t *testing.T) *RecursionWitness {
	data, err := os.ReadFile("../recursion_witness.json")
	if err != nil {
		t.Skipf("Could not load recursion witness file: %v", err)
	}

	var witness RecursionWitness
	if err := json.Unmarshal(data, &witness); err != nil {
		t.Fatal(err)
	}
	return &witness
}

func toBigInt(s string) *big.Int {
	n := new(big.Int)
	n.SetString(s, 10)
	return n
}

// ================================================
// Coefficient loading helpers (from RecursionWitness)
// ================================================

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

func loadChallenges(strs []string) []*big.Int {
	challenges := make([]*big.Int, len(strs))
	for i, s := range strs {
		challenges[i] = toBigInt(s)
	}
	return challenges
}

// ================================================
// Sumcheck evaluation helpers for each polynomial degree
// ================================================

// computeSumcheckEvalDegree7 computes sumcheck for degree-7 polynomials
// coeffs[round] = [c0, c2, c3, c4, c5, c6, c7]
func computeSumcheckEvalDegree7(coeffs [][7]*big.Int, challenges []*big.Int) *big.Int {
	prevEval := big.NewInt(0)
	two := big.NewInt(2)

	for round := 0; round < len(coeffs); round++ {
		c := coeffs[round]
		r := challenges[round]

		// Reconstruct c1 = prevEval - 2*c0 - c2 - c3 - c4 - c5 - c6 - c7
		c1 := new(big.Int).Set(prevEval)
		twoC0 := new(big.Int).Mul(two, c[0])
		c1.Sub(c1, twoC0)
		for i := 1; i < 7; i++ {
			c1.Sub(c1, c[i])
		}
		c1.Mod(c1, Fq)

		// Horner's evaluation: p(r) = c0 + r*(c1 + r*(c2 + ... + r*c7)))
		acc := new(big.Int).Set(c[6]) // c7
		acc.Mul(r, acc)
		acc.Add(c[5], acc) // + c6
		acc.Mod(acc, Fq)

		acc.Mul(r, acc)
		acc.Add(c[4], acc) // + c5
		acc.Mod(acc, Fq)

		acc.Mul(r, acc)
		acc.Add(c[3], acc) // + c4
		acc.Mod(acc, Fq)

		acc.Mul(r, acc)
		acc.Add(c[2], acc) // + c3
		acc.Mod(acc, Fq)

		acc.Mul(r, acc)
		acc.Add(c[1], acc) // + c2
		acc.Mod(acc, Fq)

		acc.Mul(r, acc)
		acc.Add(c1, acc) // + c1
		acc.Mod(acc, Fq)

		acc.Mul(r, acc)
		acc.Add(c[0], acc) // + c0
		acc.Mod(acc, Fq)

		prevEval = acc
	}

	return prevEval
}

// computeSumcheckEvalDegree6 computes sumcheck for degree-6 polynomials
// coeffs[round] = [c0, c2, c3, c4, c5, c6]
func computeSumcheckEvalDegree6(coeffs [][6]*big.Int, challenges []*big.Int) *big.Int {
	prevEval := big.NewInt(0)
	two := big.NewInt(2)

	for round := 0; round < len(coeffs); round++ {
		c := coeffs[round]
		r := challenges[round]

		// Reconstruct c1 = prevEval - 2*c0 - c2 - c3 - c4 - c5 - c6
		c1 := new(big.Int).Set(prevEval)
		twoC0 := new(big.Int).Mul(two, c[0])
		c1.Sub(c1, twoC0)
		for i := 1; i < 6; i++ {
			c1.Sub(c1, c[i])
		}
		c1.Mod(c1, Fq)

		// Horner's evaluation
		acc := new(big.Int).Set(c[5]) // c6
		acc.Mul(r, acc)
		acc.Add(c[4], acc) // + c5
		acc.Mod(acc, Fq)

		acc.Mul(r, acc)
		acc.Add(c[3], acc) // + c4
		acc.Mod(acc, Fq)

		acc.Mul(r, acc)
		acc.Add(c[2], acc) // + c3
		acc.Mod(acc, Fq)

		acc.Mul(r, acc)
		acc.Add(c[1], acc) // + c2
		acc.Mod(acc, Fq)

		acc.Mul(r, acc)
		acc.Add(c1, acc) // + c1
		acc.Mod(acc, Fq)

		acc.Mul(r, acc)
		acc.Add(c[0], acc) // + c0
		acc.Mod(acc, Fq)

		prevEval = acc
	}

	return prevEval
}

// computeSumcheckEvalDegree2 computes sumcheck for degree-2 polynomials
// coeffs[round] = [c0, c2]
func computeSumcheckEvalDegree2(coeffs [][2]*big.Int, challenges []*big.Int) *big.Int {
	prevEval := big.NewInt(0)
	two := big.NewInt(2)

	for round := 0; round < len(coeffs); round++ {
		c := coeffs[round]
		r := challenges[round]

		// Reconstruct c1 = prevEval - 2*c0 - c2
		c1 := new(big.Int).Set(prevEval)
		twoC0 := new(big.Int).Mul(two, c[0])
		c1.Sub(c1, twoC0)
		c1.Sub(c1, c[1])
		c1.Mod(c1, Fq)

		// Horner's: p(r) = c0 + r*(c1 + r*c2)
		acc := new(big.Int).Set(c[1]) // c2
		acc.Mul(r, acc)
		acc.Add(c1, acc) // + c1
		acc.Mod(acc, Fq)

		acc.Mul(r, acc)
		acc.Add(c[0], acc) // + c0
		acc.Mod(acc, Fq)

		prevEval = acc
	}

	return prevEval
}

// ================================================
// Constraint count tests (no witness needed)
// ================================================

func TestStage1Constraints(t *testing.T) {
	circuit := &Stage1OnlyCircuit{}
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Failed to compile: %v", err)
	}
	t.Logf("Stage 1 (11 rounds, degree 7): %d constraints", cs.GetNbConstraints())
}

func TestStage2Constraints(t *testing.T) {
	circuit := &Stage2OnlyCircuit{}
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Failed to compile: %v", err)
	}
	t.Logf("Stage 2 (11 rounds, degree 6): %d constraints", cs.GetNbConstraints())
}

func TestStage4Constraints(t *testing.T) {
	circuit := &Stage4OnlyCircuit{}
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Failed to compile: %v", err)
	}
	t.Logf("Stage 4 (22 rounds, degree 2): %d constraints", cs.GetNbConstraints())
}

func TestStage5Constraints(t *testing.T) {
	circuit := &Stage5OnlyCircuit{}
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Failed to compile: %v", err)
	}
	t.Logf("Stage 5 (88 rounds, degree 2): %d constraints", cs.GetNbConstraints())
}

func TestAllStagesConstraints(t *testing.T) {
	circuit := &AllRecursionStagesCircuit{}
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Failed to compile: %v", err)
	}

	t.Logf("All Recursion Sumcheck Stages combined:")
	t.Logf("  Stage 1: 11 rounds, degree 7")
	t.Logf("  Stage 2: 11 rounds, degree 6")
	t.Logf("  Stage 4: 22 rounds, degree 2")
	t.Logf("  Stage 5: 88 rounds, degree 2")
	t.Logf("  Total rounds: %d", 11+11+22+88)
	t.Logf("  Total constraints: %d", cs.GetNbConstraints())
}

// ================================================
// Solver tests (fast, no Groth16)
// ================================================

func TestStage1Solver(t *testing.T) {
	w := loadRecursionWitness(t)
	stage1Coeffs := loadStage1Coeffs(w)
	challenges := loadChallenges(w.Stage1Challenges)
	expected := toBigInt(w.Stage1Expected)

	circuit := &Stage1OnlyCircuit{}
	witness := &Stage1OnlyCircuit{}
	for round := 0; round < 11; round++ {
		for i := 0; i < 7; i++ {
			witness.Coeffs[round][i] = emulated.ValueOf[sw_grumpkin.ScalarField](stage1Coeffs[round][i])
		}
		witness.Challenges[round] = challenges[round]
	}
	witness.Expected = emulated.ValueOf[sw_grumpkin.ScalarField](expected)

	err := test.IsSolved(circuit, witness, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Stage 1 solver failed: %v", err)
	}
	t.Log("Stage 1 solver PASSED!")
}

func TestStage2Solver(t *testing.T) {
	w := loadRecursionWitness(t)
	stage2Coeffs := loadStage2Coeffs(w)
	challenges := loadChallenges(w.Stage2Challenges)
	expected := toBigInt(w.Stage2Expected)

	circuit := &Stage2OnlyCircuit{}
	witness := &Stage2OnlyCircuit{}
	for round := 0; round < 11; round++ {
		for i := 0; i < 6; i++ {
			witness.Coeffs[round][i] = emulated.ValueOf[sw_grumpkin.ScalarField](stage2Coeffs[round][i])
		}
		witness.Challenges[round] = challenges[round]
	}
	witness.Expected = emulated.ValueOf[sw_grumpkin.ScalarField](expected)

	err := test.IsSolved(circuit, witness, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Stage 2 solver failed: %v", err)
	}
	t.Log("Stage 2 solver PASSED!")
}

func TestStage4Solver(t *testing.T) {
	w := loadRecursionWitness(t)
	stage4Coeffs := loadStage4Coeffs(w)
	challenges := loadChallenges(w.Stage4Challenges)
	expected := toBigInt(w.Stage4Expected)

	circuit := &Stage4OnlyCircuit{}
	witness := &Stage4OnlyCircuit{}
	for round := 0; round < 22; round++ {
		for i := 0; i < 2; i++ {
			witness.Coeffs[round][i] = emulated.ValueOf[sw_grumpkin.ScalarField](stage4Coeffs[round][i])
		}
		witness.Challenges[round] = challenges[round]
	}
	witness.Expected = emulated.ValueOf[sw_grumpkin.ScalarField](expected)

	err := test.IsSolved(circuit, witness, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Stage 4 solver failed: %v", err)
	}
	t.Log("Stage 4 solver PASSED!")
}

func TestStage5Solver(t *testing.T) {
	w := loadRecursionWitness(t)
	stage5Coeffs := loadStage5Coeffs(w)
	challenges := loadChallenges(w.Stage5Challenges)
	expected := toBigInt(w.Stage5Expected)

	circuit := &Stage5OnlyCircuit{}
	witness := &Stage5OnlyCircuit{}
	for round := 0; round < 88; round++ {
		for i := 0; i < 2; i++ {
			witness.Coeffs[round][i] = emulated.ValueOf[sw_grumpkin.ScalarField](stage5Coeffs[round][i])
		}
		witness.Challenges[round] = challenges[round]
	}
	witness.Expected = emulated.ValueOf[sw_grumpkin.ScalarField](expected)

	err := test.IsSolved(circuit, witness, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Stage 5 solver failed: %v", err)
	}
	t.Log("Stage 5 solver PASSED!")
}

func TestAllStagesSolver(t *testing.T) {
	w := loadRecursionWitness(t)

	// Load all coefficients
	stage1Coeffs := loadStage1Coeffs(w)
	stage2Coeffs := loadStage2Coeffs(w)
	stage4Coeffs := loadStage4Coeffs(w)
	stage5Coeffs := loadStage5Coeffs(w)

	// Load challenges and expected values
	stage1Challenges := loadChallenges(w.Stage1Challenges)
	stage2Challenges := loadChallenges(w.Stage2Challenges)
	stage4Challenges := loadChallenges(w.Stage4Challenges)
	stage5Challenges := loadChallenges(w.Stage5Challenges)

	stage1Expected := toBigInt(w.Stage1Expected)
	stage2Expected := toBigInt(w.Stage2Expected)
	stage4Expected := toBigInt(w.Stage4Expected)
	stage5Expected := toBigInt(w.Stage5Expected)

	// Build witness
	circuit := &AllRecursionStagesCircuit{}
	witness := &AllRecursionStagesCircuit{}

	// Stage 1
	for round := 0; round < 11; round++ {
		for i := 0; i < 7; i++ {
			witness.Stage1Coeffs[round][i] = emulated.ValueOf[sw_grumpkin.ScalarField](stage1Coeffs[round][i])
		}
		witness.Stage1Challenges[round] = stage1Challenges[round]
	}
	witness.Stage1Expected = emulated.ValueOf[sw_grumpkin.ScalarField](stage1Expected)

	// Stage 2
	for round := 0; round < 11; round++ {
		for i := 0; i < 6; i++ {
			witness.Stage2Coeffs[round][i] = emulated.ValueOf[sw_grumpkin.ScalarField](stage2Coeffs[round][i])
		}
		witness.Stage2Challenges[round] = stage2Challenges[round]
	}
	witness.Stage2Expected = emulated.ValueOf[sw_grumpkin.ScalarField](stage2Expected)

	// Stage 4
	for round := 0; round < 22; round++ {
		for i := 0; i < 2; i++ {
			witness.Stage4Coeffs[round][i] = emulated.ValueOf[sw_grumpkin.ScalarField](stage4Coeffs[round][i])
		}
		witness.Stage4Challenges[round] = stage4Challenges[round]
	}
	witness.Stage4Expected = emulated.ValueOf[sw_grumpkin.ScalarField](stage4Expected)

	// Stage 5
	for round := 0; round < 88; round++ {
		for i := 0; i < 2; i++ {
			witness.Stage5Coeffs[round][i] = emulated.ValueOf[sw_grumpkin.ScalarField](stage5Coeffs[round][i])
		}
		witness.Stage5Challenges[round] = stage5Challenges[round]
	}
	witness.Stage5Expected = emulated.ValueOf[sw_grumpkin.ScalarField](stage5Expected)

	err := test.IsSolved(circuit, witness, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("All stages solver failed: %v", err)
	}
	t.Log("All stages solver PASSED!")
}

// ================================================
// Full Groth16 prove/verify tests
// ================================================

func TestAllStagesGroth16(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Groth16 test in short mode")
	}

	w := loadRecursionWitness(t)

	// Load all coefficients
	stage1Coeffs := loadStage1Coeffs(w)
	stage2Coeffs := loadStage2Coeffs(w)
	stage4Coeffs := loadStage4Coeffs(w)
	stage5Coeffs := loadStage5Coeffs(w)

	// Load challenges and expected values
	stage1Challenges := loadChallenges(w.Stage1Challenges)
	stage2Challenges := loadChallenges(w.Stage2Challenges)
	stage4Challenges := loadChallenges(w.Stage4Challenges)
	stage5Challenges := loadChallenges(w.Stage5Challenges)

	stage1Expected := toBigInt(w.Stage1Expected)
	stage2Expected := toBigInt(w.Stage2Expected)
	stage4Expected := toBigInt(w.Stage4Expected)
	stage5Expected := toBigInt(w.Stage5Expected)

	t.Logf("Stage 1 expected: %s...", stage1Expected.String()[:40])
	t.Logf("Stage 2 expected: %s...", stage2Expected.String()[:40])
	t.Logf("Stage 4 expected: %s...", stage4Expected.String()[:40])
	t.Logf("Stage 5 expected: %s...", stage5Expected.String()[:40])

	// Compile
	circuit := &AllRecursionStagesCircuit{}
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	t.Logf("All stages combined constraints: %d", cs.GetNbConstraints())

	// Setup
	t.Log("Running Groth16 setup...")
	pk, vk, err := groth16.Setup(cs)
	if err != nil {
		t.Fatalf("Setup failed: %v", err)
	}

	// Build witness
	witness := &AllRecursionStagesCircuit{}

	// Stage 1
	for round := 0; round < 11; round++ {
		for i := 0; i < 7; i++ {
			witness.Stage1Coeffs[round][i] = emulated.ValueOf[sw_grumpkin.ScalarField](stage1Coeffs[round][i])
		}
		witness.Stage1Challenges[round] = stage1Challenges[round]
	}
	witness.Stage1Expected = emulated.ValueOf[sw_grumpkin.ScalarField](stage1Expected)

	// Stage 2
	for round := 0; round < 11; round++ {
		for i := 0; i < 6; i++ {
			witness.Stage2Coeffs[round][i] = emulated.ValueOf[sw_grumpkin.ScalarField](stage2Coeffs[round][i])
		}
		witness.Stage2Challenges[round] = stage2Challenges[round]
	}
	witness.Stage2Expected = emulated.ValueOf[sw_grumpkin.ScalarField](stage2Expected)

	// Stage 4
	for round := 0; round < 22; round++ {
		for i := 0; i < 2; i++ {
			witness.Stage4Coeffs[round][i] = emulated.ValueOf[sw_grumpkin.ScalarField](stage4Coeffs[round][i])
		}
		witness.Stage4Challenges[round] = stage4Challenges[round]
	}
	witness.Stage4Expected = emulated.ValueOf[sw_grumpkin.ScalarField](stage4Expected)

	// Stage 5
	for round := 0; round < 88; round++ {
		for i := 0; i < 2; i++ {
			witness.Stage5Coeffs[round][i] = emulated.ValueOf[sw_grumpkin.ScalarField](stage5Coeffs[round][i])
		}
		witness.Stage5Challenges[round] = stage5Challenges[round]
	}
	witness.Stage5Expected = emulated.ValueOf[sw_grumpkin.ScalarField](stage5Expected)

	fullWitness, err := frontend.NewWitness(witness, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("NewWitness failed: %v", err)
	}

	// Prove
	t.Log("Generating proof...")
	proof, err := groth16.Prove(cs, pk, fullWitness)
	if err != nil {
		t.Fatalf("Prove failed: %v", err)
	}
	t.Log("Proof generated!")

	// Verify
	t.Log("Verifying proof...")
	publicWitness, err := fullWitness.Public()
	if err != nil {
		t.Fatalf("Public witness failed: %v", err)
	}

	err = groth16.Verify(proof, vk, publicWitness)
	if err != nil {
		t.Fatalf("Verify failed: %v", err)
	}
	t.Log("ALL STAGES Groth16 proof VERIFIED!")
}
