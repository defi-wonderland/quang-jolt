package jolt_verifier

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math/big"
	"os"
	"testing"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/grumpkin"
	fp_grumpkin "github.com/consensys/gnark-crypto/ecc/grumpkin/fp"
	fr_grumpkin "github.com/consensys/gnark-crypto/ecc/grumpkin/fr"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/std/algebra/native/sw_grumpkin"
	"github.com/consensys/gnark/test"
)

// TestCombinedCircuitStagesOnly tests the combined circuit with only stages 1-7 (no Hyrax).
// This verifies that embedding JoltStagesCircuit works correctly.
func TestCombinedCircuitStagesOnly(t *testing.T) {
	t.Log("=== Combined Circuit Test (Stages Only) ===")
	t.Log("Testing that combined circuit works without Hyrax fields populated...")
	t.Log("")

	// Load stages witness
	witnessPath := getStagesWitnessPath()
	stagesAssignment, err := LoadStagesAssignment(witnessPath)
	if err != nil {
		t.Fatalf("Failed to load stages witness: %v", err)
	}
	t.Logf("Loaded stages witness from: %s", witnessPath)

	// Create combined assignment with only stages fields populated
	// Hyrax fields must be initialized (gnark requires all fields assigned)
	// but empty slices will cause defineHyraxVerification to skip
	assignment := &JoltCombinedCircuit{
		JoltStagesCircuit:   *stagesAssignment,
		HyraxSqrtN:          0,
		HyraxRowCommitments: []sw_grumpkin.G1Affine{},
		HyraxGenerators:     []sw_grumpkin.G1Affine{},
		HyraxL:              []sw_grumpkin.Scalar{},
		HyraxU:              []sw_grumpkin.Scalar{},
		HyraxR:              []frontend.Variable{},
		HyraxV:              0, // Unused when Hyrax fields are empty
	}

	// Test solver - circuit definition must match assignment structure
	t.Log("")
	t.Log("Testing solver...")
	circuit := &JoltCombinedCircuit{
		HyraxRowCommitments: []sw_grumpkin.G1Affine{},
		HyraxGenerators:     []sw_grumpkin.G1Affine{},
		HyraxL:              []sw_grumpkin.Scalar{},
		HyraxU:              []sw_grumpkin.Scalar{},
		HyraxR:              []frontend.Variable{},
	}
	err = test.IsSolved(circuit, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Combined circuit solver failed: %v", err)
	}
	t.Log("Solver passed!")
}

// TestCombinedCircuitCompilation tests that the combined circuit compiles
// and reports constraint counts for both components.
func TestCombinedCircuitCompilation(t *testing.T) {
	t.Log("=== Combined Circuit Compilation Test ===")
	t.Log("")

	// Test with small Hyrax size to get constraint count for both parts
	sqrtN := 4 // Small size for quick test

	// Create circuit definition
	circuit := &JoltCombinedCircuit{
		HyraxSqrtN:          sqrtN,
		HyraxRowCommitments: make([]sw_grumpkin.G1Affine, sqrtN),
		HyraxGenerators:     make([]sw_grumpkin.G1Affine, sqrtN),
		HyraxL:              make([]sw_grumpkin.Scalar, sqrtN),
		HyraxU:              make([]sw_grumpkin.Scalar, sqrtN),
		HyraxR:              make([]frontend.Variable, sqrtN),
	}

	t.Log("Compiling combined circuit (stages + small Hyrax)...")
	start := time.Now()
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Compilation failed: %v", err)
	}
	t.Logf("Compiled in %v", time.Since(start))

	constraints := cs.GetNbConstraints()
	publicVars := cs.GetNbPublicVariables()
	secretVars := cs.GetNbSecretVariables()

	t.Log("")
	t.Logf("=== Constraint Breakdown ===")
	t.Logf("Total constraints:     %d", constraints)
	t.Logf("Public variables:      %d", publicVars)
	t.Logf("Secret variables:      %d", secretVars)

	// Approximate breakdown (stages is ~3.1M, small Hyrax is ~15k)
	stagesConstraints := 3112261 // Known from TestStagesCircuitProveVerify
	hyraxConstraints := constraints - stagesConstraints
	if hyraxConstraints > 0 {
		t.Logf("")
		t.Logf("Estimated breakdown:")
		t.Logf("  Stages 1-7:  ~%d constraints", stagesConstraints)
		t.Logf("  Hyrax (sqrt(N)=%d): ~%d constraints", sqrtN, hyraxConstraints)
	}
}

// TestCombinedCircuitConstraintEstimate estimates total constraints for production config.
func TestCombinedCircuitConstraintEstimate(t *testing.T) {
	t.Log("=== Combined Circuit Constraint Estimate ===")
	t.Log("")

	// Known constraint counts
	stagesConstraints := 3112261 // From actual transpilation

	// Hyrax constraint counts from TestConstraintCount in hyrax package
	hyraxSizes := map[int]int{
		4:    15163,
		16:   58295,
		64:   228910,
		256:  901533,
		1024: 3545963,
		2048: 7025654, // Actual fib(50) proof size
	}

	t.Log("Production configuration estimates:")
	t.Log("")
	t.Logf("Stages 1-7 (sumcheck): %d constraints", stagesConstraints)
	t.Log("")
	t.Log("Stage 8 (Hyrax PCS) by polynomial size:")

	// Print in sorted order
	sortedSizes := []int{4, 16, 64, 256, 1024, 2048}
	for _, sqrtN := range sortedSizes {
		hyraxConstr := hyraxSizes[sqrtN]
		N := sqrtN * sqrtN
		total := stagesConstraints + hyraxConstr
		t.Logf("  N=%d (sqrt(N)=%d): Hyrax=%d, Total=%d", N, sqrtN, hyraxConstr, total)
	}

	t.Log("")
	t.Log("For typical Jolt trace (N=2^20, sqrt(N)=1024):")
	typicalTotal := stagesConstraints + hyraxSizes[1024]
	t.Logf("  Total: %d constraints (~%.1fM)", typicalTotal, float64(typicalTotal)/1e6)
	t.Logf("  Proof size: 164 bytes (Groth16)")

	t.Log("")
	t.Log("For fib(50) proof (N=2^22, sqrt(N)=2048):")
	fib50Total := stagesConstraints + hyraxSizes[2048]
	t.Logf("  Total: %d constraints (~%.1fM)", fib50Total, float64(fib50Total)/1e6)
	t.Logf("  Proof size: 164 bytes (Groth16)")
}

// BenchmarkCombinedCircuitCompilation benchmarks circuit compilation.
func BenchmarkCombinedCircuitCompilation(b *testing.B) {
	// Create minimal circuit for benchmarking compilation
	circuit := &JoltCombinedCircuit{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
		if err != nil {
			b.Fatalf("Compilation failed: %v", err)
		}
	}
}

// TestCombinedCircuitFullProveVerify tests the full prove/verify cycle.
// This is a long-running test (~2+ minutes) so it's skipped by default.
func TestCombinedCircuitFullProveVerify(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping full prove/verify in short mode")
	}

	t.Log("=== Combined Circuit Full Prove/Verify Test ===")
	t.Log("This test takes ~2 minutes...")
	t.Log("")

	// Load stages witness
	witnessPath := getStagesWitnessPath()
	stagesAssignment, err := LoadStagesAssignment(witnessPath)
	if err != nil {
		t.Fatalf("Failed to load stages witness: %v", err)
	}

	// Create combined assignment (stages only for now)
	assignment := &JoltCombinedCircuit{
		JoltStagesCircuit: *stagesAssignment,
	}

	// Circuit definition
	var circuit JoltCombinedCircuit

	// Compile
	t.Log("Compiling circuit...")
	start := time.Now()
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		t.Fatalf("Compilation failed: %v", err)
	}
	t.Logf("Compiled: %d constraints [%v]", cs.GetNbConstraints(), time.Since(start))

	// Setup
	t.Log("Running Groth16 setup...")
	start = time.Now()
	pk, vk, err := groth16.Setup(cs)
	if err != nil {
		t.Fatalf("Setup failed: %v", err)
	}
	t.Logf("Setup complete [%v]", time.Since(start))

	// Create witness
	witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Witness creation failed: %v", err)
	}

	// Prove
	t.Log("Generating proof...")
	start = time.Now()
	proof, err := groth16.Prove(cs, pk, witness)
	if err != nil {
		t.Fatalf("Prove failed: %v", err)
	}
	t.Logf("Proof generated [%v]", time.Since(start))

	// Get proof size
	var proofBuf bytes.Buffer
	_, err = proof.WriteTo(&proofBuf)
	if err == nil {
		t.Logf("Proof size: %d bytes", proofBuf.Len())
	}

	// Verify
	publicWitness, err := witness.Public()
	if err != nil {
		t.Fatalf("Public witness extraction failed: %v", err)
	}

	t.Log("Verifying proof...")
	start = time.Now()
	err = groth16.Verify(proof, vk, publicWitness)
	if err != nil {
		t.Fatalf("Verification failed: %v", err)
	}
	t.Logf("Verified [%v]", time.Since(start))

	t.Log("")
	t.Log("Combined circuit prove/verify passed!")
}

// printConstraintBreakdown is a helper to print detailed constraint info
func printConstraintBreakdown(t *testing.T, constraints, stagesEstimate int) {
	t.Log("")
	t.Log("Constraint breakdown:")
	hyraxEstimate := constraints - stagesEstimate
	t.Logf("  Stages 1-7 (estimated): %d (%.1f%%)", stagesEstimate, 100*float64(stagesEstimate)/float64(constraints))
	t.Logf("  Hyrax (estimated):      %d (%.1f%%)", hyraxEstimate, 100*float64(hyraxEstimate)/float64(constraints))
	t.Logf("  Total:                  %d", constraints)
}

func init() {
	// Suppress verbose output during tests
	fmt.Println("Combined circuit tests loaded")
}

// HyraxWitnessJSON represents the JSON structure from the Rust transpiler
type HyraxWitnessJSON struct {
	SqrtN          int         `json:"sqrt_n"`  // Full size for MSM #2
	SqrtN1         int         `json:"sqrt_n1"` // Filtered size for MSM #1
	U              []string    `json:"u"`
	RowCommitments [][2]string `json:"row_commitments"` // Filtered (no identity points)
	Generators     [][2]string `json:"generators"`      // Full size
	L              []string    `json:"l"`               // Filtered
	R              []string    `json:"r"`               // Full size
	V              string      `json:"v"`
	OpeningPoint   []string    `json:"opening_point"`
	DenseNumVars   int         `json:"dense_num_vars"`
}

// LoadHyraxWitness loads Hyrax witness data from JSON file
func LoadHyraxWitness(path string) (*HyraxWitnessJSON, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var witness HyraxWitnessJSON
	if err := json.Unmarshal(data, &witness); err != nil {
		return nil, err
	}
	return &witness, nil
}

// TestCombinedCircuitFullWithHyrax tests the full combined circuit with real Hyrax data.
// This is the ultimate test: ~8.4M constraints for fib(50) proof (with identity point filtering).
func TestCombinedCircuitFullWithHyrax(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping full combined circuit test in short mode (takes ~30+ minutes)")
	}

	t.Log("=== Full Combined Circuit Test (Stages 1-7 + Hyrax) ===")
	t.Log("This test takes ~10-15 minutes for compilation, setup, and prove...")
	t.Log("")

	// Load stages witness
	stagesWitnessPath := getStagesWitnessPath()
	stagesAssignment, err := LoadStagesAssignment(stagesWitnessPath)
	if err != nil {
		t.Fatalf("Failed to load stages witness: %v", err)
	}
	t.Logf("Loaded stages witness from: %s", stagesWitnessPath)

	// Load Hyrax witness
	hyraxWitnessPath := "hyrax_witness.json"
	hyraxWitness, err := LoadHyraxWitness(hyraxWitnessPath)
	if err != nil {
		t.Fatalf("Failed to load Hyrax witness: %v", err)
	}

	sqrtN := hyraxWitness.SqrtN   // Full size for MSM #2
	sqrtN1 := hyraxWitness.SqrtN1 // Filtered size for MSM #1 (non-identity row commitments)
	if sqrtN1 == 0 {
		sqrtN1 = sqrtN // Backwards compatibility
	}
	t.Logf("Loaded Hyrax witness: sqrt(N) = %d (full), sqrt(N1) = %d (filtered)", sqrtN, sqrtN1)

	// Parse Hyrax witness data into circuit types
	t.Log("Parsing Hyrax witness data...")

	// Filtered arrays (sqrtN1): row commitments and L
	rowCommitments := make([]sw_grumpkin.G1Affine, sqrtN1)
	L := make([]sw_grumpkin.Scalar, sqrtN1)

	for i := 0; i < sqrtN1; i++ {
		// Parse row commitment
		var x, y fp_grumpkin.Element
		xBig := new(big.Int)
		yBig := new(big.Int)
		xBig.SetString(hyraxWitness.RowCommitments[i][0], 10)
		yBig.SetString(hyraxWitness.RowCommitments[i][1], 10)
		x.SetBigInt(xBig)
		y.SetBigInt(yBig)
		rowCommitments[i] = sw_grumpkin.NewG1Affine(grumpkin.G1Affine{X: x, Y: y})

		// Parse L
		lBig := new(big.Int)
		lBig.SetString(hyraxWitness.L[i], 10)
		var lElem fr_grumpkin.Element
		lElem.SetBigInt(lBig)
		L[i] = sw_grumpkin.NewScalar(lElem)
	}

	// Full arrays (sqrtN): generators, U, R
	generators := make([]sw_grumpkin.G1Affine, sqrtN)
	U := make([]sw_grumpkin.Scalar, sqrtN)
	R := make([]frontend.Variable, sqrtN)

	for i := 0; i < sqrtN; i++ {
		// Parse generator
		var x, y fp_grumpkin.Element
		xBig := new(big.Int)
		yBig := new(big.Int)
		xBig.SetString(hyraxWitness.Generators[i][0], 10)
		yBig.SetString(hyraxWitness.Generators[i][1], 10)
		x.SetBigInt(xBig)
		y.SetBigInt(yBig)
		generators[i] = sw_grumpkin.NewG1Affine(grumpkin.G1Affine{X: x, Y: y})

		// Parse U
		uBig := new(big.Int)
		uBig.SetString(hyraxWitness.U[i], 10)
		var uElem fr_grumpkin.Element
		uElem.SetBigInt(uBig)
		U[i] = sw_grumpkin.NewScalar(uElem)

		// Parse R
		rBig := new(big.Int)
		rBig.SetString(hyraxWitness.R[i], 10)
		R[i] = frontend.Variable(rBig)
	}

	// Parse V
	vBig := new(big.Int)
	vBig.SetString(hyraxWitness.V, 10)
	V := frontend.Variable(vBig)

	t.Logf("Parsed %d row commitments + L (filtered), %d generators + U + R (full)", sqrtN1, sqrtN)

	// Create combined assignment
	assignment := &JoltCombinedCircuit{
		JoltStagesCircuit:   *stagesAssignment,
		HyraxSqrtN:          sqrtN,
		HyraxSqrtN1:         sqrtN1,
		HyraxRowCommitments: rowCommitments, // sqrtN1 elements (filtered)
		HyraxGenerators:     generators,     // sqrtN elements (full)
		HyraxL:              L,              // sqrtN1 elements (filtered)
		HyraxU:              U,              // sqrtN elements (full)
		HyraxR:              R,              // sqrtN elements (full)
		HyraxV:              V,
	}

	// Create circuit definition with correct sizes
	circuit := &JoltCombinedCircuit{
		HyraxSqrtN:          sqrtN,
		HyraxSqrtN1:         sqrtN1,
		HyraxRowCommitments: make([]sw_grumpkin.G1Affine, sqrtN1), // Filtered size
		HyraxGenerators:     make([]sw_grumpkin.G1Affine, sqrtN),  // Full size
		HyraxL:              make([]sw_grumpkin.Scalar, sqrtN1),   // Filtered size
		HyraxU:              make([]sw_grumpkin.Scalar, sqrtN),    // Full size
		HyraxR:              make([]frontend.Variable, sqrtN),     // Full size
	}

	// Compile
	t.Log("")
	t.Log("Compiling circuit (~8.4M constraints)...")
	start := time.Now()
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Compilation failed: %v", err)
	}
	compileTime := time.Since(start)
	t.Logf("Compiled: %d constraints [%v]", cs.GetNbConstraints(), compileTime)

	// Setup
	t.Log("")
	t.Log("Running Groth16 setup...")
	start = time.Now()
	pk, vk, err := groth16.Setup(cs)
	if err != nil {
		t.Fatalf("Setup failed: %v", err)
	}
	setupTime := time.Since(start)
	t.Logf("Setup complete [%v]", setupTime)

	// Create witness
	t.Log("")
	t.Log("Creating witness...")
	start = time.Now()
	witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Witness creation failed: %v", err)
	}
	t.Logf("Witness created [%v]", time.Since(start))

	// Prove
	t.Log("")
	t.Log("Generating proof...")
	start = time.Now()
	proof, err := groth16.Prove(cs, pk, witness)
	if err != nil {
		t.Fatalf("Prove failed: %v", err)
	}
	proveTime := time.Since(start)
	t.Logf("Proof generated [%v]", proveTime)

	// Get proof size
	var proofBuf bytes.Buffer
	_, err = proof.WriteTo(&proofBuf)
	if err == nil {
		t.Logf("Proof size: %d bytes", proofBuf.Len())
	}

	// Verify
	publicWitness, err := witness.Public()
	if err != nil {
		t.Fatalf("Public witness extraction failed: %v", err)
	}

	t.Log("")
	t.Log("Verifying proof...")
	start = time.Now()
	err = groth16.Verify(proof, vk, publicWitness)
	if err != nil {
		t.Fatalf("Verification failed: %v", err)
	}
	verifyTime := time.Since(start)
	t.Logf("Verified [%v]", verifyTime)

	t.Log("")
	t.Log("=== SUMMARY ===")
	t.Logf("Constraints:    %d (~%.1fM)", cs.GetNbConstraints(), float64(cs.GetNbConstraints())/1e6)
	t.Logf("Compile time:   %v", compileTime)
	t.Logf("Setup time:     %v", setupTime)
	t.Logf("Prove time:     %v", proveTime)
	t.Logf("Verify time:    %v", verifyTime)
	t.Logf("Proof size:     %d bytes", proofBuf.Len())
	t.Log("")
	t.Log("Full combined circuit prove/verify PASSED!")
}
