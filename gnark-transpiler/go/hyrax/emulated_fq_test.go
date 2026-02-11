package hyrax

import (
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/std/algebra/native/sw_grumpkin"
	"github.com/consensys/gnark/std/math/emulated"
)

// EmulatedMulCircuit tests a single emulated Fq multiplication
type EmulatedMulCircuit struct {
	A sw_grumpkin.Scalar `gnark:",public"`
	B sw_grumpkin.Scalar `gnark:",public"`
	C sw_grumpkin.Scalar `gnark:",public"` // C = A * B
}

func (c *EmulatedMulCircuit) Define(api frontend.API) error {
	scalarField, err := emulated.NewField[sw_grumpkin.ScalarField](api)
	if err != nil {
		return err
	}

	result := scalarField.Mul(&c.A, &c.B)
	scalarField.AssertIsEqual(result, &c.C)

	return nil
}

// EmulatedAddCircuit tests a single emulated Fq addition
type EmulatedAddCircuit struct {
	A sw_grumpkin.Scalar `gnark:",public"`
	B sw_grumpkin.Scalar `gnark:",public"`
	C sw_grumpkin.Scalar `gnark:",public"` // C = A + B
}

func (c *EmulatedAddCircuit) Define(api frontend.API) error {
	scalarField, err := emulated.NewField[sw_grumpkin.ScalarField](api)
	if err != nil {
		return err
	}

	result := scalarField.Add(&c.A, &c.B)
	scalarField.AssertIsEqual(result, &c.C)

	return nil
}

// EmulatedPolyEvalCircuit tests evaluating a degree-7 polynomial in emulated Fq
// This is what we'd need for the recursion sumcheck
type EmulatedPolyEvalCircuit struct {
	// Coefficients c0, c1, c2, ..., c7
	Coeffs [8]sw_grumpkin.Scalar `gnark:",public"`
	// Challenge point r
	R sw_grumpkin.Scalar `gnark:",public"`
	// Expected result p(r)
	Result sw_grumpkin.Scalar `gnark:",public"`
}

func (c *EmulatedPolyEvalCircuit) Define(api frontend.API) error {
	scalarField, err := emulated.NewField[sw_grumpkin.ScalarField](api)
	if err != nil {
		return err
	}

	// Compute p(r) = c0 + c1*r + c2*r^2 + ... + c7*r^7
	// Using Horner's method: p(r) = c0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*(c5 + r*(c6 + r*c7))))))

	// Start with c7
	acc := &c.Coeffs[7]

	// Work backwards: acc = c_i + r * acc
	for i := 6; i >= 0; i-- {
		// acc = r * acc
		acc = scalarField.Mul(&c.R, acc)
		// acc = c_i + acc
		acc = scalarField.Add(&c.Coeffs[i], acc)
	}

	scalarField.AssertIsEqual(acc, &c.Result)

	return nil
}

func TestEmulatedMulConstraints(t *testing.T) {
	circuit := &EmulatedMulCircuit{}

	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Failed to compile circuit: %v", err)
	}

	t.Logf("Single emulated Fq multiplication:")
	t.Logf("  Constraints: %d", cs.GetNbConstraints())
}

func TestEmulatedAddConstraints(t *testing.T) {
	circuit := &EmulatedAddCircuit{}

	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Failed to compile circuit: %v", err)
	}

	t.Logf("Single emulated Fq addition:")
	t.Logf("  Constraints: %d", cs.GetNbConstraints())
}

func TestEmulatedPolyEvalConstraints(t *testing.T) {
	circuit := &EmulatedPolyEvalCircuit{}

	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Failed to compile circuit: %v", err)
	}

	t.Logf("Degree-7 polynomial evaluation in emulated Fq:")
	t.Logf("  Constraints: %d", cs.GetNbConstraints())
	t.Logf("  (This is what ONE sumcheck round would cost)")
}
