package hyrax

import (
	"encoding/json"
	"fmt"
	"math/big"
	"os"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/grumpkin"
	fp_grumpkin "github.com/consensys/gnark-crypto/ecc/grumpkin/fp"
	fr_grumpkin "github.com/consensys/gnark-crypto/ecc/grumpkin/fr"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/std/algebra/native/sw_grumpkin"
)

// TestConstraintCount measures constraint counts for various MSM sizes.
// This is the key benchmark for validating Hyrax feasibility.
func TestConstraintCount(t *testing.T) {
	// Test multiple sqrt(N) sizes
	// 1024 is the target for N = 2^20 (1M coefficients)
	// 2048 is the actual size from fib(50) proof
	sizes := []int{4, 16, 64, 256, 1024, 2048}

	for _, sqrtN := range sizes {
		t.Run(fmt.Sprintf("sqrtN=%d", sqrtN), func(t *testing.T) {
			// Create circuit with placeholder arrays of the right size
			circuit := createPlaceholderCircuit(sqrtN)

			// Compile to R1CS to count constraints
			cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
			if err != nil {
				t.Fatalf("Failed to compile circuit: %v", err)
			}

			nbConstraints := cs.GetNbConstraints()
			nbPublicWires := cs.GetNbPublicVariables()
			nbSecretWires := cs.GetNbSecretVariables()

			t.Logf("sqrt(N) = %d (N = %d)", sqrtN, sqrtN*sqrtN)
			t.Logf("  Constraints: %d", nbConstraints)
			t.Logf("  Public wires: %d", nbPublicWires)
			t.Logf("  Secret wires: %d", nbSecretWires)
			t.Logf("  Constraints per scalar mul: ~%d", nbConstraints/(2*sqrtN))
		})
	}
}

// TestSingleScalarMul measures constraints for a single scalar multiplication.
// This isolates the cost of one Grumpkin scalar mul.
func TestSingleScalarMul(t *testing.T) {
	circuit := &SingleScalarMulCircuit{}

	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Failed to compile circuit: %v", err)
	}

	t.Logf("Single Grumpkin scalar mul:")
	t.Logf("  Constraints: %d", cs.GetNbConstraints())
	t.Logf("  Public wires: %d", cs.GetNbPublicVariables())
}

// SingleScalarMulCircuit tests a single scalar multiplication.
type SingleScalarMulCircuit struct {
	Point  sw_grumpkin.G1Affine `gnark:",public"`
	Scalar sw_grumpkin.Scalar   `gnark:",public"`
	Result sw_grumpkin.G1Affine `gnark:",public"`
}

func (c *SingleScalarMulCircuit) Define(api frontend.API) error {
	curve, err := sw_grumpkin.NewCurve(api)
	if err != nil {
		return err
	}

	result := curve.ScalarMul(&c.Point, &c.Scalar)
	curve.AssertIsEqual(result, &c.Result)

	return nil
}

// TestPointAddition measures constraints for a single point addition.
func TestPointAddition(t *testing.T) {
	circuit := &PointAddCircuit{}

	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Failed to compile circuit: %v", err)
	}

	t.Logf("Single Grumpkin point addition:")
	t.Logf("  Constraints: %d", cs.GetNbConstraints())
}

// PointAddCircuit tests a single point addition.
type PointAddCircuit struct {
	P      sw_grumpkin.G1Affine `gnark:",public"`
	Q      sw_grumpkin.G1Affine `gnark:",public"`
	Result sw_grumpkin.G1Affine `gnark:",public"`
}

func (c *PointAddCircuit) Define(api frontend.API) error {
	curve, err := sw_grumpkin.NewCurve(api)
	if err != nil {
		return err
	}

	result := curve.Add(&c.P, &c.Q)
	curve.AssertIsEqual(result, &c.Result)

	return nil
}

// TestFullProveVerify runs a complete prove/verify cycle with small parameters.
// This validates that the circuit is correctly constructed.
func TestFullProveVerify(t *testing.T) {
	sqrtN := 4 // Small size for testing

	// Generate random test data
	rowCommitments, generators, L, U, R, V := generateTestData(sqrtN)

	// Create circuit
	circuit := createPlaceholderCircuit(sqrtN)

	// Create witness
	witness := &HyraxVerifierCircuit{
		SqrtN:          sqrtN,
		RowCommitments: rowCommitments,
		Generators:     generators,
		L:              L,
		U:              U,
		R:              R,
		V:              V,
	}

	// Compile
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Failed to compile: %v", err)
	}

	// Setup
	pk, vk, err := groth16.Setup(cs)
	if err != nil {
		t.Fatalf("Failed to setup: %v", err)
	}

	// Create witness
	fullWitness, err := frontend.NewWitness(witness, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Failed to create witness: %v", err)
	}

	// Prove
	proof, err := groth16.Prove(cs, pk, fullWitness)
	if err != nil {
		t.Fatalf("Failed to prove: %v", err)
	}

	// Verify
	publicWitness, err := fullWitness.Public()
	if err != nil {
		t.Fatalf("Failed to get public witness: %v", err)
	}

	err = groth16.Verify(proof, vk, publicWitness)
	if err != nil {
		t.Fatalf("Failed to verify: %v", err)
	}

	t.Logf("Full prove/verify cycle succeeded for sqrt(N) = %d", sqrtN)
}

// TestMSMRelationship verifies that our test data satisfies the critical Hyrax constraint:
// Com(u) == C' where C' = Σ L[a] · C[a] and Com(u) = Σ u[j] · G[j]
//
// This is the binding property that makes Hyrax secure.
func TestMSMRelationship(t *testing.T) {
	sqrtN := 4

	// Extract concrete L values
	lValues := make([]*big.Int, sqrtN)
	for a := 0; a < sqrtN; a++ {
		lValues[a] = big.NewInt(int64(a + 1)) // L[a] = a + 1 (from generateTestData)
	}

	// Extract concrete U values (computed as u[j] = Σ_a L[a] · F[a][j])
	uValues := make([]*big.Int, sqrtN)
	for j := 0; j < sqrtN; j++ {
		var sum int64 = 0
		for a := 0; a < sqrtN; a++ {
			F_aj := int64((a + 1) * (j + 1)) // F[a][j] = (a+1)*(j+1)
			sum += int64(a+1) * F_aj         // L[a] * F[a][j]
		}
		uValues[j] = big.NewInt(sum)
	}

	// Recompute generators
	generatorsConcrete := make([]grumpkin.G1Affine, sqrtN)
	_, gen := grumpkin.Generators()
	for j := 0; j < sqrtN; j++ {
		var scalar big.Int
		scalar.SetInt64(int64(j + 1))
		generatorsConcrete[j].ScalarMultiplication(&gen, &scalar)
	}

	// Recompute row commitments
	F := make([][]int64, sqrtN)
	for a := 0; a < sqrtN; a++ {
		F[a] = make([]int64, sqrtN)
		for b := 0; b < sqrtN; b++ {
			F[a][b] = int64((a + 1) * (b + 1))
		}
	}

	rowCommitmentsConcrete := make([]grumpkin.G1Affine, sqrtN)
	for a := 0; a < sqrtN; a++ {
		var sum grumpkin.G1Affine
		for j := 0; j < sqrtN; j++ {
			var scalar big.Int
			scalar.SetInt64(F[a][j])
			var term grumpkin.G1Affine
			term.ScalarMultiplication(&generatorsConcrete[j], &scalar)
			sum.Add(&sum, &term)
		}
		rowCommitmentsConcrete[a] = sum
	}

	// Compute C' = Σ L[a] · C[a]
	var Cprime grumpkin.G1Affine
	for a := 0; a < sqrtN; a++ {
		var term grumpkin.G1Affine
		term.ScalarMultiplication(&rowCommitmentsConcrete[a], lValues[a])
		Cprime.Add(&Cprime, &term)
	}

	// Compute Com(u) = Σ u[j] · G[j]
	var ComU grumpkin.G1Affine
	for j := 0; j < sqrtN; j++ {
		var term grumpkin.G1Affine
		term.ScalarMultiplication(&generatorsConcrete[j], uValues[j])
		ComU.Add(&ComU, &term)
	}

	// Check equality
	if !Cprime.Equal(&ComU) {
		t.Fatalf("MSM relationship violated: Com(u) != C'\nCprime: %v\nComU: %v", Cprime, ComU)
	}

	t.Logf("MSM relationship verified: Com(u) == C'")
	t.Logf("  C' = %s", Cprime.String())
	t.Logf("  Com(u) = %s", ComU.String())

	// Also verify the dot product
	var dotProduct int64 = 0
	for j := 0; j < sqrtN; j++ {
		dotProduct += uValues[j].Int64() * int64(j+1) // u[j] * R[j] where R[j] = j+1
	}
	t.Logf("  <u, R> = %d", dotProduct)
}

// TestRejectsInvalidProof verifies that the circuit rejects an invalid witness.
// This is a soundness test - we create data where Com(u) != C'.
func TestRejectsInvalidProof(t *testing.T) {
	sqrtN := 4

	// Get valid test data
	rowCommitments, generators, L, U, R, _ := generateTestData(sqrtN)

	// Corrupt one of the U values (this will make Com(u) != C')
	var corruptedU []sw_grumpkin.Scalar
	corruptedU = append(corruptedU, U...)
	var badScalar fr_grumpkin.Element
	badScalar.SetInt64(99999) // Different from the correct value
	corruptedU[0] = sw_grumpkin.NewScalar(badScalar)

	// Also need to update V to match the corrupted U for the dot product
	// (otherwise the circuit might fail on the dot product check instead)
	var newV int64 = 99999 * 1 // corruptedU[0] * R[0]
	for j := 1; j < sqrtN; j++ {
		// Original u[j] = Σ_a L[a] * F[a][j]
		var sum int64 = 0
		for a := 0; a < sqrtN; a++ {
			F_aj := int64((a + 1) * (j + 1))
			sum += int64(a+1) * F_aj
		}
		newV += sum * int64(j+1)
	}

	// Create circuit
	circuit := createPlaceholderCircuit(sqrtN)

	// Create invalid witness
	witness := &HyraxVerifierCircuit{
		SqrtN:          sqrtN,
		RowCommitments: rowCommitments,
		Generators:     generators,
		L:              L,
		U:              corruptedU,
		R:              R,
		V:              frontend.Variable(newV),
	}

	// Compile
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Failed to compile: %v", err)
	}

	// Setup
	pk, _, err := groth16.Setup(cs)
	if err != nil {
		t.Fatalf("Failed to setup: %v", err)
	}

	// Create witness
	fullWitness, err := frontend.NewWitness(witness, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Failed to create witness: %v", err)
	}

	// Prove - this should fail because Com(u) != C'
	_, err = groth16.Prove(cs, pk, fullWitness)
	if err == nil {
		t.Fatalf("Expected proof to fail with invalid witness, but it succeeded!")
	}

	t.Logf("Circuit correctly rejected invalid proof: %v", err)
}

// createPlaceholderCircuit creates a circuit with the right array sizes for compilation.
// sqrtN is the full size, sqrtN1 is the filtered size for MSM #1 (non-identity row commitments).
func createPlaceholderCircuit(sqrtN int) *HyraxVerifierCircuit {
	return createPlaceholderCircuitWithSizes(sqrtN, sqrtN)
}

// createPlaceholderCircuitWithSizes creates a circuit with different sizes for filtered MSM #1.
func createPlaceholderCircuitWithSizes(sqrtN, sqrtN1 int) *HyraxVerifierCircuit {
	return &HyraxVerifierCircuit{
		SqrtN:          sqrtN,
		SqrtN1:         sqrtN1,
		RowCommitments: make([]sw_grumpkin.G1Affine, sqrtN1), // Filtered size
		Generators:     make([]sw_grumpkin.G1Affine, sqrtN),  // Full size
		L:              make([]sw_grumpkin.Scalar, sqrtN1),   // Filtered size
		U:              make([]sw_grumpkin.Scalar, sqrtN),    // Full size
		R:              make([]frontend.Variable, sqrtN),     // Full size
		V:              0,
	}
}

// HyraxWitnessJSON represents the JSON structure from the Rust transpiler
type HyraxWitnessJSON struct {
	SqrtN          int          `json:"sqrt_n"`   // Full size for MSM #2
	SqrtN1         int          `json:"sqrt_n1"`  // Filtered size for MSM #1
	U              []string     `json:"u"`        // Full size
	RowCommitments [][2]string  `json:"row_commitments"` // Filtered (no identity points)
	Generators     [][2]string  `json:"generators"`      // Full size
	L              []string     `json:"l"`               // Filtered
	R              []string     `json:"r"`               // Full size
	V              string       `json:"v"`
	OpeningPoint   []string     `json:"opening_point"`
	DenseNumVars   int          `json:"dense_num_vars"`
}

// TestRealHyraxWitness tests the Hyrax circuit with real witness data from the Jolt proof.
// This is a sanity check that the witness data is correctly extracted.
func TestRealHyraxWitness(t *testing.T) {
	// Load witness from JSON file
	witnessPath := "../hyrax_witness.json"
	data, err := os.ReadFile(witnessPath)
	if err != nil {
		t.Skipf("Hyrax witness file not found at %s - run transpiler first", witnessPath)
	}

	var witnessJSON HyraxWitnessJSON
	if err := json.Unmarshal(data, &witnessJSON); err != nil {
		t.Fatalf("Failed to parse witness JSON: %v", err)
	}

	sqrtN := witnessJSON.SqrtN   // Full size for MSM #2
	sqrtN1 := witnessJSON.SqrtN1 // Filtered size for MSM #1 (may be same as sqrtN if no filtering)
	if sqrtN1 == 0 {
		// Backwards compatibility: if sqrtN1 not present, assume same as sqrtN
		sqrtN1 = sqrtN
	}

	t.Logf("Loaded Hyrax witness:")
	t.Logf("  sqrt(N) = %d (full size for MSM #2)", sqrtN)
	t.Logf("  sqrt(N1) = %d (filtered size for MSM #1)", sqrtN1)
	t.Logf("  U length: %d (expected: %d)", len(witnessJSON.U), sqrtN)
	t.Logf("  RowCommitments length: %d (expected: %d)", len(witnessJSON.RowCommitments), sqrtN1)
	t.Logf("  Generators length: %d (expected: %d)", len(witnessJSON.Generators), sqrtN)
	t.Logf("  L length: %d (expected: %d)", len(witnessJSON.L), sqrtN1)
	t.Logf("  R length: %d (expected: %d)", len(witnessJSON.R), sqrtN)

	// Validate data structure sizes
	// U, Generators, R use full size (sqrtN)
	if len(witnessJSON.U) != sqrtN {
		t.Errorf("U length mismatch: got %d, expected %d", len(witnessJSON.U), sqrtN)
	}
	if len(witnessJSON.Generators) != sqrtN {
		t.Errorf("Generators length mismatch: got %d, expected %d", len(witnessJSON.Generators), sqrtN)
	}
	if len(witnessJSON.R) != sqrtN {
		t.Errorf("R length mismatch: got %d, expected %d", len(witnessJSON.R), sqrtN)
	}

	// RowCommitments, L use filtered size (sqrtN1)
	if len(witnessJSON.RowCommitments) != sqrtN1 {
		t.Errorf("RowCommitments length mismatch: got %d, expected %d", len(witnessJSON.RowCommitments), sqrtN1)
	}
	if len(witnessJSON.L) != sqrtN1 {
		t.Errorf("L length mismatch: got %d, expected %d", len(witnessJSON.L), sqrtN1)
	}

	// Parse a few values to make sure the format is correct
	var uVal big.Int
	if _, ok := uVal.SetString(witnessJSON.U[0], 10); !ok {
		t.Errorf("Failed to parse U[0]: %s", witnessJSON.U[0])
	}
	t.Logf("  U[0] = %s", uVal.String())

	var vVal big.Int
	if _, ok := vVal.SetString(witnessJSON.V, 10); !ok {
		t.Errorf("Failed to parse V: %s", witnessJSON.V)
	}
	t.Logf("  V = %s", vVal.String())

	// Parse first row commitment point
	var x, y big.Int
	if _, ok := x.SetString(witnessJSON.RowCommitments[0][0], 10); !ok {
		t.Errorf("Failed to parse RowCommitments[0].x")
	}
	if _, ok := y.SetString(witnessJSON.RowCommitments[0][1], 10); !ok {
		t.Errorf("Failed to parse RowCommitments[0].y")
	}
	t.Logf("  RowCommitments[0] = (%s..., %s...)", x.String()[:20], y.String()[:20])

	t.Logf("Witness data structure validated successfully")
}

// TestRealHyraxCircuitSmall tests the Hyrax circuit with a small subset of real data.
// This is useful to verify the circuit works before running with full 2048 elements.
func TestRealHyraxCircuitSmall(t *testing.T) {
	// Load witness from JSON file
	witnessPath := "../hyrax_witness.json"
	data, err := os.ReadFile(witnessPath)
	if err != nil {
		t.Skipf("Hyrax witness file not found at %s - run transpiler first", witnessPath)
	}

	var witnessJSON HyraxWitnessJSON
	if err := json.Unmarshal(data, &witnessJSON); err != nil {
		t.Fatalf("Failed to parse witness JSON: %v", err)
	}

	// Use a small subset (4 elements) to quickly test circuit compilation
	testSqrtN := 4
	if witnessJSON.SqrtN < testSqrtN {
		t.Skipf("Witness sqrt(N) = %d is smaller than test size %d", witnessJSON.SqrtN, testSqrtN)
	}

	t.Logf("Testing with small subset: sqrt(N) = %d (from full %d)", testSqrtN, witnessJSON.SqrtN)

	// Parse subset of data
	rowCommitments := make([]sw_grumpkin.G1Affine, testSqrtN)
	generators := make([]sw_grumpkin.G1Affine, testSqrtN)
	L := make([]sw_grumpkin.Scalar, testSqrtN)
	U := make([]sw_grumpkin.Scalar, testSqrtN)
	R := make([]frontend.Variable, testSqrtN)

	for i := 0; i < testSqrtN; i++ {
		// Parse row commitment
		var x, y fp_grumpkin.Element
		xBig := new(big.Int)
		yBig := new(big.Int)
		xBig.SetString(witnessJSON.RowCommitments[i][0], 10)
		yBig.SetString(witnessJSON.RowCommitments[i][1], 10)
		x.SetBigInt(xBig)
		y.SetBigInt(yBig)
		rowCommitments[i] = sw_grumpkin.NewG1Affine(grumpkin.G1Affine{X: x, Y: y})

		// Parse generator
		xBig.SetString(witnessJSON.Generators[i][0], 10)
		yBig.SetString(witnessJSON.Generators[i][1], 10)
		x.SetBigInt(xBig)
		y.SetBigInt(yBig)
		generators[i] = sw_grumpkin.NewG1Affine(grumpkin.G1Affine{X: x, Y: y})

		// Parse L
		lBig := new(big.Int)
		lBig.SetString(witnessJSON.L[i], 10)
		var lElem fr_grumpkin.Element
		lElem.SetBigInt(lBig)
		L[i] = sw_grumpkin.NewScalar(lElem)

		// Parse U
		uBig := new(big.Int)
		uBig.SetString(witnessJSON.U[i], 10)
		var uElem fr_grumpkin.Element
		uElem.SetBigInt(uBig)
		U[i] = sw_grumpkin.NewScalar(uElem)

		// Parse R
		rBig := new(big.Int)
		rBig.SetString(witnessJSON.R[i], 10)
		R[i] = frontend.Variable(rBig)
	}

	// Parse V - note: this won't be valid for the subset, use placeholder
	// The real V is for the full vector, not a subset
	V := frontend.Variable(0)

	t.Logf("Parsed %d elements from witness", testSqrtN)

	// Create circuit with test size
	circuit := createPlaceholderCircuit(testSqrtN)

	// Compile to check constraint count
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Failed to compile circuit: %v", err)
	}
	t.Logf("Circuit compiled: %d constraints", cs.GetNbConstraints())

	// Create witness struct - this is just to test parsing, the actual verification
	// would require consistent data (which our subset doesn't have)
	_ = &HyraxVerifierCircuit{
		SqrtN:          testSqrtN,
		RowCommitments: rowCommitments,
		Generators:     generators,
		L:              L,
		U:              U,
		R:              R,
		V:              V,
	}

	t.Logf("Successfully parsed witness data into circuit types")
}

// TestRealHyraxCircuitFull tests the full Hyrax circuit with complete real witness data.
// This takes a long time (~10+ minutes) but validates the witness is correct.
func TestRealHyraxCircuitFull(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping full Hyrax circuit test in short mode (takes ~10+ minutes)")
	}

	// Load witness from JSON file
	witnessPath := "../hyrax_witness.json"
	data, err := os.ReadFile(witnessPath)
	if err != nil {
		t.Skipf("Hyrax witness file not found at %s - run transpiler first", witnessPath)
	}

	var witnessJSON HyraxWitnessJSON
	if err := json.Unmarshal(data, &witnessJSON); err != nil {
		t.Fatalf("Failed to parse witness JSON: %v", err)
	}

	sqrtN := witnessJSON.SqrtN   // Full size for MSM #2 (generators × U) and dot product
	sqrtN1 := witnessJSON.SqrtN1 // Filtered size for MSM #1 (row_commitments × L)
	if sqrtN1 == 0 {
		sqrtN1 = sqrtN // Backwards compatibility
	}

	t.Logf("Testing with full witness:")
	t.Logf("  sqrt(N) = %d (full size for MSM #2)", sqrtN)
	t.Logf("  sqrt(N1) = %d (filtered size for MSM #1)", sqrtN1)

	// Parse row commitments and L (filtered size: sqrtN1)
	rowCommitments := make([]sw_grumpkin.G1Affine, sqrtN1)
	L := make([]sw_grumpkin.Scalar, sqrtN1)

	for i := 0; i < sqrtN1; i++ {
		// Parse row commitment
		var x, y fp_grumpkin.Element
		xBig := new(big.Int)
		yBig := new(big.Int)
		xBig.SetString(witnessJSON.RowCommitments[i][0], 10)
		yBig.SetString(witnessJSON.RowCommitments[i][1], 10)
		x.SetBigInt(xBig)
		y.SetBigInt(yBig)
		rowCommitments[i] = sw_grumpkin.NewG1Affine(grumpkin.G1Affine{X: x, Y: y})

		// Parse L
		lBig := new(big.Int)
		lBig.SetString(witnessJSON.L[i], 10)
		var lElem fr_grumpkin.Element
		lElem.SetBigInt(lBig)
		L[i] = sw_grumpkin.NewScalar(lElem)
	}

	// Parse generators, U, R (full size: sqrtN)
	generators := make([]sw_grumpkin.G1Affine, sqrtN)
	U := make([]sw_grumpkin.Scalar, sqrtN)
	R := make([]frontend.Variable, sqrtN)

	for i := 0; i < sqrtN; i++ {
		// Parse generator
		var x, y fp_grumpkin.Element
		xBig := new(big.Int)
		yBig := new(big.Int)
		xBig.SetString(witnessJSON.Generators[i][0], 10)
		yBig.SetString(witnessJSON.Generators[i][1], 10)
		x.SetBigInt(xBig)
		y.SetBigInt(yBig)
		generators[i] = sw_grumpkin.NewG1Affine(grumpkin.G1Affine{X: x, Y: y})

		// Parse U
		uBig := new(big.Int)
		uBig.SetString(witnessJSON.U[i], 10)
		var uElem fr_grumpkin.Element
		uElem.SetBigInt(uBig)
		U[i] = sw_grumpkin.NewScalar(uElem)

		// Parse R
		rBig := new(big.Int)
		rBig.SetString(witnessJSON.R[i], 10)
		R[i] = frontend.Variable(rBig)
	}

	// Parse V
	vBig := new(big.Int)
	vBig.SetString(witnessJSON.V, 10)
	V := frontend.Variable(vBig)

	t.Logf("Parsed %d row commitments + L (filtered), %d generators + U + R (full)", sqrtN1, sqrtN)

	// Create circuit with both sizes
	circuit := createPlaceholderCircuitWithSizes(sqrtN, sqrtN1)

	// Create witness struct
	witness := &HyraxVerifierCircuit{
		SqrtN:          sqrtN,
		SqrtN1:         sqrtN1,
		RowCommitments: rowCommitments, // sqrtN1 elements
		Generators:     generators,     // sqrtN elements
		L:              L,              // sqrtN1 elements
		U:              U,              // sqrtN elements
		R:              R,              // sqrtN elements
		V:              V,
	}

	// Compile
	t.Log("Compiling circuit...")
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Failed to compile circuit: %v", err)
	}
	t.Logf("Circuit compiled: %d constraints", cs.GetNbConstraints())

	// Setup
	t.Log("Running Groth16 setup...")
	pk, vk, err := groth16.Setup(cs)
	if err != nil {
		t.Fatalf("Failed to setup: %v", err)
	}
	t.Log("Setup complete")

	// Create full witness
	fullWitness, err := frontend.NewWitness(witness, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Failed to create witness: %v", err)
	}

	// Prove
	t.Log("Generating proof...")
	proof, err := groth16.Prove(cs, pk, fullWitness)
	if err != nil {
		t.Fatalf("Failed to prove: %v", err)
	}
	t.Log("Proof generated")

	// Verify
	publicWitness, err := fullWitness.Public()
	if err != nil {
		t.Fatalf("Failed to get public witness: %v", err)
	}

	err = groth16.Verify(proof, vk, publicWitness)
	if err != nil {
		t.Fatalf("Failed to verify: %v", err)
	}

	t.Logf("Full Hyrax prove/verify cycle succeeded for sqrt(N) = %d", sqrtN)
}

// generateTestData creates valid Hyrax test vectors that satisfy the protocol constraints.
//
// For valid Hyrax verification, we need:
//   1. C' = Σ L[a] · RowCommitments[a]     (MSM #1 - verifier computes)
//   2. Com(u) = Σ u[j] · Generators[j]    (MSM #2 - verifier computes)
//   3. Com(u) == C'                         (Check 1 - binding)
//   4. <u, R> == v                          (Check 2 - evaluation)
//
// To satisfy Check 1 (Com(u) == C'), we construct row commitments such that
// when weighted by L, they produce the same point as u weighted by generators.
//
// Strategy: Use a simple coefficient matrix F where we can compute everything.
func generateTestData(sqrtN int) (
	rowCommitments []sw_grumpkin.G1Affine,
	generators []sw_grumpkin.G1Affine,
	L []sw_grumpkin.Scalar,
	U []sw_grumpkin.Scalar,
	R []frontend.Variable,
	V frontend.Variable,
) {
	// Get the Grumpkin generator point
	_, g1Gen := grumpkin.Generators()

	// Generate deterministic SRS generators: G[j] = (j+1) · G
	generators = make([]sw_grumpkin.G1Affine, sqrtN)
	generatorsConcrete := make([]grumpkin.G1Affine, sqrtN)
	for j := 0; j < sqrtN; j++ {
		var scalar big.Int
		scalar.SetInt64(int64(j + 1))
		generatorsConcrete[j].ScalarMultiplication(&g1Gen, &scalar)
		generators[j] = sw_grumpkin.NewG1Affine(generatorsConcrete[j])
	}

	// Create a simple coefficient matrix F (sqrtN x sqrtN)
	// F[a][b] = (a+1) * (b+1) for simplicity
	F := make([][]int64, sqrtN)
	for a := 0; a < sqrtN; a++ {
		F[a] = make([]int64, sqrtN)
		for b := 0; b < sqrtN; b++ {
			F[a][b] = int64((a + 1) * (b + 1))
		}
	}

	// Compute row commitments: C[a] = Σ_j F[a][j] · G[j]
	rowCommitments = make([]sw_grumpkin.G1Affine, sqrtN)
	rowCommitmentsConcrete := make([]grumpkin.G1Affine, sqrtN)
	for a := 0; a < sqrtN; a++ {
		// C[a] = Σ_j F[a][j] · G[j]
		var sum grumpkin.G1Affine
		sum.Set(&grumpkin.G1Affine{}) // identity (point at infinity)
		for j := 0; j < sqrtN; j++ {
			var scalar big.Int
			scalar.SetInt64(F[a][j])
			var term grumpkin.G1Affine
			term.ScalarMultiplication(&generatorsConcrete[j], &scalar)
			sum.Add(&sum, &term)
		}
		rowCommitmentsConcrete[a] = sum
		rowCommitments[a] = sw_grumpkin.NewG1Affine(rowCommitmentsConcrete[a])
	}

	// Generate L (eq vector) - simple values for testing
	// L[a] = a + 1
	L = make([]sw_grumpkin.Scalar, sqrtN)
	lValues := make([]int64, sqrtN)
	for a := 0; a < sqrtN; a++ {
		lValues[a] = int64(a + 1)
		var lElem fr_grumpkin.Element
		lElem.SetInt64(lValues[a])
		L[a] = sw_grumpkin.NewScalar(lElem)
	}

	// Compute u = L^T · F (the projection vector)
	// u[j] = Σ_a L[a] · F[a][j]
	U = make([]sw_grumpkin.Scalar, sqrtN)
	uValues := make([]int64, sqrtN)
	for j := 0; j < sqrtN; j++ {
		var sum int64 = 0
		for a := 0; a < sqrtN; a++ {
			sum += lValues[a] * F[a][j]
		}
		uValues[j] = sum
		var uElem fr_grumpkin.Element
		uElem.SetInt64(uValues[j])
		U[j] = sw_grumpkin.NewScalar(uElem)
	}

	// Generate R (eq vector) - simple values for testing
	// R[b] = b + 1
	R = make([]frontend.Variable, sqrtN)
	rValues := make([]int64, sqrtN)
	for b := 0; b < sqrtN; b++ {
		rValues[b] = int64(b + 1)
		R[b] = frontend.Variable(rValues[b])
	}

	// Compute v = <u, R> = Σ_j u[j] · R[j]
	var vValue int64 = 0
	for j := 0; j < sqrtN; j++ {
		vValue += uValues[j] * rValues[j]
	}
	V = frontend.Variable(vValue)

	// Verification: Com(u) should equal C' = Σ L[a] · C[a]
	// This is guaranteed by construction because:
	//   C' = Σ_a L[a] · (Σ_j F[a][j] · G[j])
	//      = Σ_j (Σ_a L[a] · F[a][j]) · G[j]
	//      = Σ_j u[j] · G[j]
	//      = Com(u)

	return
}
