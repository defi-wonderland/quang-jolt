package poseidon

import (
	"fmt"
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/test"
)

// TestCircuit for testing Poseidon hash
type TestPoseidonCircuit struct {
	In1      frontend.Variable
	In2      frontend.Variable
	In3      frontend.Variable
	Expected frontend.Variable `gnark:",public"`
}

func (c *TestPoseidonCircuit) Define(api frontend.API) error {
	result := Hash(api, c.In1, c.In2, c.In3)
	api.AssertIsEqual(result, c.Expected)
	return nil
}

func TestPoseidonHashZeros(t *testing.T) {
	// Rust: hash([0, 0, 0]) = 5317387130258456662214331362918410991734007599705406860481038345552731150762
	assert := test.NewAssert(t)

	var circuit TestPoseidonCircuit

	expected := new(big.Int)
	expected.SetString("5317387130258456662214331362918410991734007599705406860481038345552731150762", 10)

	assignment := &TestPoseidonCircuit{
		In1:      0,
		In2:      0,
		In3:      0,
		Expected: expected,
	}

	assert.ProverSucceeded(&circuit, assignment, test.WithCurves(ecc.BN254))
}

func TestPoseidonHash123(t *testing.T) {
	// Rust: hash([1, 2, 3]) = 6542985608222806190361240322586112750744169038454362455181422643027100751666
	assert := test.NewAssert(t)

	var circuit TestPoseidonCircuit

	expected := new(big.Int)
	expected.SetString("6542985608222806190361240322586112750744169038454362455181422643027100751666", 10)

	assignment := &TestPoseidonCircuit{
		In1:      1,
		In2:      2,
		In3:      3,
		Expected: expected,
	}

	assert.ProverSucceeded(&circuit, assignment, test.WithCurves(ecc.BN254))
}

func TestPoseidonManual(t *testing.T) {
	// Print the first few constants to verify they match
	fmt.Println("=== Go Poseidon Constants ===")
	fmt.Println("cConstants[0] =", cConstants[0].String())
	fmt.Println("cConstants[1] =", cConstants[1].String())
	fmt.Println("cConstants[2] =", cConstants[2].String())
	fmt.Println("cConstants[3] =", cConstants[3].String())

	// Expected from light-poseidon circom t=4:
	// ark[0] = 11633431549750490989983886834189948010834808234699737327785600195936805266405
	expected := new(big.Int)
	expected.SetString("11633431549750490989983886834189948010834808234699737327785600195936805266405", 10)

	if cConstants[0].Cmp(expected) != 0 {
		t.Errorf("cConstants[0] mismatch!\nGot:      %s\nExpected: %s", cConstants[0].String(), expected.String())
	} else {
		fmt.Println("cConstants[0] matches light-poseidon!")
	}
}

// TestTruncate128ReverseHint tests the Montgomery form conversion for challenges
func TestTruncate128ReverseHint(t *testing.T) {
	fmt.Println("=== Testing Truncate128ReverseHint ===")

	// Test with a known hash output from Rust
	// We need to trace what truncate128ReverseHint does with a specific input

	// The initial state after hashing "Jolt" label is created by:
	// hash([1953263434, 0, 0]) where 1953263434 = "Jolt" as LE u32
	// Then we hash multiple times for the transcript

	// Let's test the hint function directly
	testInputs := []string{
		"5317387130258456662214331362918410991734007599705406860481038345552731150762", // hash([0,0,0])
		"6542985608222806190361240322586112750744169038454362455181422643027100751666", // hash([1,2,3])
		"1953263434", // "Jolt" label
	}

	for _, inputStr := range testInputs {
		input, _ := new(big.Int).SetString(inputStr, 10)
		outputs := make([]*big.Int, 1)
		outputs[0] = new(big.Int)

		err := truncate128ReverseHint(nil, []*big.Int{input}, outputs)
		if err != nil {
			t.Errorf("truncate128ReverseHint failed: %v", err)
			continue
		}

		fmt.Printf("Input:  %s\n", input.String())
		fmt.Printf("Output: %s\n\n", outputs[0].String())
	}
}

// TestByteReverseHint tests the byte reversal operation
func TestByteReverseHint(t *testing.T) {
	fmt.Println("=== Testing ByteReverseHint ===")

	// Test with witness values from the proof
	testInputs := []string{
		"0",
		"339982277569909227036194045546146182803753403674367144357178929037303311570", // Stage1_Uni_Skip_Coeff_1
		"10223046733663547134027576661019044998752347965522502640886883481748386943546", // Stage1_Sumcheck_R0_0
	}

	// Expected results from Rust debug_truncate
	expectedResults := []string{
		"0",
		"7625174665854580928319602534203856263707274111151135344721492977228796706812",
		"4378126927267714964301483578480114419378979696694852036373749382991194003989",
	}

	for i, inputStr := range testInputs {
		input, _ := new(big.Int).SetString(inputStr, 10)
		expected, _ := new(big.Int).SetString(expectedResults[i], 10)
		outputs := make([]*big.Int, 1)
		outputs[0] = new(big.Int)

		err := byteReverseHint(nil, []*big.Int{input}, outputs)
		if err != nil {
			t.Errorf("byteReverseHint failed: %v", err)
			continue
		}

		fmt.Printf("Input:    %s\n", input.String())
		fmt.Printf("Got:      %s\n", outputs[0].String())
		fmt.Printf("Expected: %s\n", expected.String())
		if outputs[0].Cmp(expected) != 0 {
			t.Errorf("MISMATCH! ByteReverseHint gives wrong result")

			// Debug: print intermediate steps
			inputBytes := input.Bytes()
			fmt.Printf("  Input BE bytes (%d): %02x\n", len(inputBytes), inputBytes)

			le := make([]byte, 32)
			for j := 0; j < len(inputBytes) && j < 32; j++ {
				le[j] = inputBytes[len(inputBytes)-1-j]
			}
			fmt.Printf("  LE bytes (index 0 = LSB): %02x\n", le)

			reversed := make([]byte, 32)
			for j := 0; j < 32; j++ {
				reversed[j] = le[31-j]
			}
			fmt.Printf("  Reversed (index 0 = was MSB): %02x\n", reversed)

			// The correct interpretation: 'reversed' is LE bytes for the result
			// To convert to big.Int (which needs BE), we reverse 'reversed'
			correctBE := make([]byte, 32)
			for j := 0; j < 32; j++ {
				correctBE[j] = reversed[31-j]
			}
			fmt.Printf("  Correct BE for big.Int: %02x\n", correctBE)

			// Check: what does big.Int.SetBytes(reversed) directly give?
			directResult := new(big.Int).SetBytes(reversed)
			fmt.Printf("  SetBytes(reversed) directly: %s\n", directResult.String())

			// What does SetBytes(correctBE) give?
			correctResult := new(big.Int).SetBytes(correctBE)
			fmt.Printf("  SetBytes(correctBE): %s\n", correctResult.String())

			// Expected bytes (what Rust would give)
			expectedBytes := expected.Bytes()
			fmt.Printf("  Expected as BE bytes: %02x\n", expectedBytes)
		} else {
			fmt.Println("  MATCH!")
		}
		fmt.Println()
	}
}

// TestTruncate128Hint tests the simple truncation (no Montgomery conversion)
func TestTruncate128Hint(t *testing.T) {
	fmt.Println("=== Testing Truncate128Hint (no Montgomery) ===")

	testInputs := []string{
		"5317387130258456662214331362918410991734007599705406860481038345552731150762",
		"6542985608222806190361240322586112750744169038454362455181422643027100751666",
	}

	for _, inputStr := range testInputs {
		input, _ := new(big.Int).SetString(inputStr, 10)
		outputs := make([]*big.Int, 1)
		outputs[0] = new(big.Int)

		err := truncate128Hint(nil, []*big.Int{input}, outputs)
		if err != nil {
			t.Errorf("truncate128Hint failed: %v", err)
			continue
		}

		fmt.Printf("Input:  %s\n", input.String())
		fmt.Printf("Truncate128: %s\n\n", outputs[0].String())
	}
}
