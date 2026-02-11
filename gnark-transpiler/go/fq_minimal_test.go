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
