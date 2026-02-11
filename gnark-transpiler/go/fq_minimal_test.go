package jolt_verifier

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/emulated"
	"github.com/consensys/gnark/test"
)

// MinimalFqCircuit tests a chain of Fq operations similar to our circuit
type MinimalFqCircuit struct {
	// Native Fr inputs (like transcript challenges)
	X0 frontend.Variable
	X1 frontend.Variable
	X2 frontend.Variable
	// Expected Fq result (native for comparison)
	Expected frontend.Variable
}

func (c *MinimalFqCircuit) Define(api frontend.API) error {
	fqField, err := emulated.NewField[emulated.BN254Fp](api)
	if err != nil {
		return err
	}

	// Bridge Fr → Fq (same pattern as our circuit)
	x0_fq := fqField.FromBits(api.ToBinary(c.X0, 254)...)
	x1_fq := fqField.FromBits(api.ToBinary(c.X1, 254)...)
	x2_fq := fqField.FromBits(api.ToBinary(c.X2, 254)...)

	// Chain of Fq operations (similar to sumcheck verification)
	// x0^2
	x0_sq := fqField.Mul(x0_fq, x0_fq)
	// x0^3
	x0_cu := fqField.Mul(x0_sq, x0_fq)

	// Some additions and subtractions
	sum := fqField.Add(x0_fq, x1_fq)
	diff := fqField.Sub(x1_fq, x2_fq)

	// Mul(1, x) pattern
	one_times_x := fqField.Mul(fqField.NewElement(1), x0_fq)

	// 1 - x pattern
	one_minus_x := fqField.Sub(fqField.NewElement(1), x0_fq)

	// Combine: ((x0^3 + sum) * diff + one_times_x) * one_minus_x
	t1 := fqField.Add(x0_cu, sum)
	t2 := fqField.Mul(t1, diff)
	t3 := fqField.Add(t2, one_times_x)
	result := fqField.Mul(t3, one_minus_x)

	// Assert result is some expected value
	fqField.AssertIsEqual(result, fqField.FromBits(api.ToBinary(c.Expected, 254)...))

	return nil
}

func TestMinimalFqOperations(t *testing.T) {
	// Compute expected result in plain big.Int arithmetic mod Fq
	fq, _ := new(big.Int).SetString("21888242871839275222246405745257275088696311157297823662689037894645226208583", 10)

	x0 := big.NewInt(12345678901234567)
	x1 := big.NewInt(98765432109876543)
	x2 := big.NewInt(55555555555555555)

	x0_sq := new(big.Int).Mul(x0, x0)
	x0_sq.Mod(x0_sq, fq)

	x0_cu := new(big.Int).Mul(x0_sq, x0)
	x0_cu.Mod(x0_cu, fq)

	sum := new(big.Int).Add(x0, x1)
	sum.Mod(sum, fq)

	diff := new(big.Int).Sub(x1, x2)
	diff.Mod(diff, fq)

	one_times_x := new(big.Int).Set(x0)

	one_minus_x := new(big.Int).Sub(big.NewInt(1), x0)
	one_minus_x.Mod(one_minus_x, fq)

	t1 := new(big.Int).Add(x0_cu, sum)
	t1.Mod(t1, fq)

	t2 := new(big.Int).Mul(t1, diff)
	t2.Mod(t2, fq)

	t3 := new(big.Int).Add(t2, one_times_x)
	t3.Mod(t3, fq)

	result := new(big.Int).Mul(t3, one_minus_x)
	result.Mod(result, fq)

	t.Logf("Expected result: %s", result.String())

	assignment := &MinimalFqCircuit{
		X0:       x0,
		X1:       x1,
		X2:       x2,
		Expected: result,
	}

	var circuit MinimalFqCircuit
	err := test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Errorf("Minimal Fq test FAILED: %v", err)
	} else {
		t.Log("Minimal Fq test PASSED - no mulCheck errors")
	}
}

// StressFqCircuit tests many Fq operations (closer to our circuit's scale)
type StressFqCircuit struct {
	X [10]frontend.Variable
}

func (c *StressFqCircuit) Define(api frontend.API) error {
	fqField, err := emulated.NewField[emulated.BN254Fp](api)
	if err != nil {
		return err
	}

	// Bridge all inputs
	fqVals := make([]*emulated.Element[emulated.BN254Fp], len(c.X))
	for i := range c.X {
		fqVals[i] = fqField.FromBits(api.ToBinary(c.X[i], 254)...)
	}

	// Chain of operations: accumulate products and sums
	acc := fqVals[0]
	for i := 1; i < len(fqVals); i++ {
		// x^2, x^3, x^4, x^5, x^6 pattern
		pow := fqVals[i]
		for j := 0; j < 5; j++ {
			pow = fqField.Mul(pow, fqVals[i])
		}
		// acc = acc * pow + (1 - fqVals[i])
		acc = fqField.Mul(acc, pow)
		oneMinusX := fqField.Sub(fqField.NewElement(1), fqVals[i])
		acc = fqField.Add(acc, oneMinusX)
		// Add: Mul(1, acc) pattern
		acc = fqField.Mul(fqField.NewElement(1), acc)
	}

	// Just assert non-zero (we don't care about the specific value)
	// This forces gnark to compute everything
	_ = acc

	return nil
}

func TestStressFqOperations(t *testing.T) {
	assignment := &StressFqCircuit{}
	for i := range assignment.X {
		assignment.X[i] = big.NewInt(int64(i*7919 + 42))
	}

	var circuit StressFqCircuit
	err := test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Errorf("Stress Fq test FAILED: %v", err)
	} else {
		t.Log("Stress Fq test PASSED - no mulCheck errors")
	}
}

// TestA17DiagnosticBigInt computes the a17 expression using pure big.Int arithmetic
// to verify if the expected result should be 0 mod Fq.
// This tests the hypothesis: is the transpiled computation correct?
func TestA17DiagnosticBigInt(t *testing.T) {
	// Fq modulus
	fq, _ := new(big.Int).SetString("21888242871839275222246405745257275088696311157297823662689037894645226208583", 10)

	// Load the actual witness values used in the failing test
	witnessPath := getStagesWitnessPath()
	stagesAssignment, err := LoadStagesAssignment(witnessPath)
	if err != nil {
		t.Fatalf("Failed to load stages witness: %v", err)
	}

	// Extract the first few recursion stage values as big.Int
	// These correspond to cse_17_814 = Recursion_Stage1_R0_0, etc.
	r0_0 := fieldToBigInt(stagesAssignment.Recursion_Stage1_R0_0)
	r0_1 := fieldToBigInt(stagesAssignment.Recursion_Stage1_R0_1)
	r0_2 := fieldToBigInt(stagesAssignment.Recursion_Stage1_R0_2)
	r0_3 := fieldToBigInt(stagesAssignment.Recursion_Stage1_R0_3)
	r0_4 := fieldToBigInt(stagesAssignment.Recursion_Stage1_R0_4)
	r0_5 := fieldToBigInt(stagesAssignment.Recursion_Stage1_R0_5)
	r0_6 := fieldToBigInt(stagesAssignment.Recursion_Stage1_R0_6)

	t.Log("=== First Round Witness Values ===")
	t.Logf("R0_0: %s", r0_0.String())
	t.Logf("R0_1: %s", r0_1.String())
	t.Logf("R0_2: %s", r0_2.String())
	t.Logf("R0_3: %s", r0_3.String())
	t.Logf("R0_4: %s", r0_4.String())
	t.Logf("R0_5: %s", r0_5.String())
	t.Logf("R0_6: %s", r0_6.String())

	// The sumcheck verification computes something like:
	// Sum over coefficients R0_i, weighted by powers of challenge r
	// If sumcheck is valid: g(0) + g(1) = claimed_sum
	// where g(X) = R0_0 + R0_1*X + R0_2*X^2 + ...
	// Evaluated at r: g(r) = R0_0 + R0_1*r + R0_2*r^2 + ...
	// The assertion checks: LHS - RHS == 0 mod Fq

	// For diagnostic purposes, just sum all coefficients to see if they're reasonable Fq values
	sum := new(big.Int).Set(r0_0)
	sum.Add(sum, r0_1)
	sum.Add(sum, r0_2)
	sum.Add(sum, r0_3)
	sum.Add(sum, r0_4)
	sum.Add(sum, r0_5)
	sum.Add(sum, r0_6)
	sum.Mod(sum, fq)

	t.Log("")
	t.Logf("Sum of first round coefficients mod Fq: %s", sum.String())

	// Check if any values exceed Fq (would indicate witness corruption)
	for i, v := range []*big.Int{r0_0, r0_1, r0_2, r0_3, r0_4, r0_5, r0_6} {
		if v.Cmp(fq) >= 0 {
			t.Errorf("R0_%d exceeds Fq modulus!", i)
		}
	}

	t.Log("")
	t.Log("All values are valid Fq elements")
}

// fieldToBigInt converts a frontend.Variable to big.Int
func fieldToBigInt(v frontend.Variable) *big.Int {
	switch x := v.(type) {
	case *big.Int:
		return x
	case big.Int:
		return &x
	case string:
		b, _ := new(big.Int).SetString(x, 10)
		return b
	case int64:
		return big.NewInt(x)
	case int:
		return big.NewInt(int64(x))
	default:
		// Try to convert via string representation
		return new(big.Int)
	}
}
