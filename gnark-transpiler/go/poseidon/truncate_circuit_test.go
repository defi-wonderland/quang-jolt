package poseidon

import (
	"fmt"
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/test"
)

// Truncate128Circuit tests the Truncate128 function in a circuit context
type Truncate128Circuit struct {
	Input    frontend.Variable
	Expected frontend.Variable `gnark:",public"`
}

func (c *Truncate128Circuit) Define(api frontend.API) error {
	result := Truncate128(api, c.Input)
	api.AssertIsEqual(result, c.Expected)
	return nil
}

func TestTruncate128InCircuit(t *testing.T) {
	fmt.Println("=== Testing Truncate128 in Circuit Context ===")

	assert := test.NewAssert(t)

	// Test case: Truncate128(5317...) should give 226766...
	// This matches Rust challenge_scalar_128_bits: from_le_bytes_mod_order(reverse(le16)) = from_be_bytes(le16)
	input, _ := new(big.Int).SetString("5317387130258456662214331362918410991734007599705406860481038345552731150762", 10)
	expected, _ := new(big.Int).SetString("226766529495067233327141020382834624226", 10)

	fmt.Printf("Input:    %s\n", input.String())
	fmt.Printf("Expected: %s\n", expected.String())

	var circuit Truncate128Circuit
	assignment := &Truncate128Circuit{
		Input:    input,
		Expected: expected,
	}

	assert.ProverSucceeded(&circuit, assignment, test.WithCurves(ecc.BN254))
	fmt.Println("Truncate128 circuit test passed!")
}

// Truncate128ReverseCircuit tests the Truncate128Reverse function
type Truncate128ReverseCircuit struct {
	Input    frontend.Variable
	Expected frontend.Variable `gnark:",public"`
}

func (c *Truncate128ReverseCircuit) Define(api frontend.API) error {
	result := Truncate128Reverse(api, c.Input)
	api.AssertIsEqual(result, c.Expected)
	return nil
}

func TestTruncate128ReverseInCircuit(t *testing.T) {
	fmt.Println("\n=== Testing Truncate128Reverse in Circuit Context ===")

	assert := test.NewAssert(t)

	// Test case: Truncate128Reverse(5317...) should give 18953...
	input, _ := new(big.Int).SetString("5317387130258456662214331362918410991734007599705406860481038345552731150762", 10)
	expected, _ := new(big.Int).SetString("18953534306744322081161596255095256148977956028501736402142324914023056147461", 10)

	fmt.Printf("Input:    %s\n", input.String())
	fmt.Printf("Expected: %s\n", expected.String())

	var circuit Truncate128ReverseCircuit
	assignment := &Truncate128ReverseCircuit{
		Input:    input,
		Expected: expected,
	}

	assert.ProverSucceeded(&circuit, assignment, test.WithCurves(ecc.BN254))
	fmt.Println("Truncate128Reverse circuit test passed!")
}

// ByteReverseCircuit tests the ByteReverse function
type ByteReverseCircuit struct {
	Input    frontend.Variable
	Expected frontend.Variable `gnark:",public"`
}

func (c *ByteReverseCircuit) Define(api frontend.API) error {
	result := ByteReverse(api, c.Input)
	api.AssertIsEqual(result, c.Expected)
	return nil
}

func TestByteReverseInCircuit(t *testing.T) {
	fmt.Println("\n=== Testing ByteReverse in Circuit Context ===")

	assert := test.NewAssert(t)

	// Test case from Rust debug_truncate
	input, _ := new(big.Int).SetString("339982277569909227036194045546146182803753403674367144357178929037303311570", 10)
	expected, _ := new(big.Int).SetString("7625174665854580928319602534203856263707274111151135344721492977228796706812", 10)

	fmt.Printf("Input:    %s\n", input.String())
	fmt.Printf("Expected: %s\n", expected.String())

	var circuit ByteReverseCircuit
	assignment := &ByteReverseCircuit{
		Input:    input,
		Expected: expected,
	}

	assert.ProverSucceeded(&circuit, assignment, test.WithCurves(ecc.BN254))
	fmt.Println("ByteReverse circuit test passed!")
}
