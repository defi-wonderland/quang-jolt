package poseidon

import (
	"fmt"
	"math/big"
	"testing"
)

func TestNativePoseidonZeros(t *testing.T) {
	// Rust: hash([0, 0, 0]) = 5317387130258456662214331362918410991734007599705406860481038345552731150762
	expected, _ := new(big.Int).SetString("5317387130258456662214331362918410991734007599705406860481038345552731150762", 10)

	result := HashNative(big.NewInt(0), big.NewInt(0), big.NewInt(0))

	fmt.Printf("Hash([0, 0, 0]) = %s\n", result.String())
	fmt.Printf("Expected:       = %s\n", expected.String())

	if result.Cmp(expected) != 0 {
		t.Errorf("MISMATCH!")
	} else {
		fmt.Println("MATCH!")
	}
}

func TestNativePoseidonJoltLabel(t *testing.T) {
	// Test the initial "Jolt" label hash
	// 1953263434 = "Jolt" as LE u32 = 0x746C6F4A
	joltLabel := big.NewInt(1953263434)
	result := HashNative(joltLabel, big.NewInt(0), big.NewInt(0))

	fmt.Printf("Hash([1953263434, 0, 0]) = %s\n", result.String())
	// TODO: Verify against Rust
}

func TestNativePoseidon123(t *testing.T) {
	// Rust: hash([1, 2, 3]) = 6542985608222806190361240322586112750744169038454362455181422643027100751666
	expected, _ := new(big.Int).SetString("6542985608222806190361240322586112750744169038454362455181422643027100751666", 10)

	result := HashNative(big.NewInt(1), big.NewInt(2), big.NewInt(3))

	fmt.Printf("Hash([1, 2, 3]) = %s\n", result.String())
	fmt.Printf("Expected:       = %s\n", expected.String())

	if result.Cmp(expected) != 0 {
		t.Errorf("MISMATCH!")
	} else {
		fmt.Println("MATCH!")
	}
}

func TestNativeTruncate128(t *testing.T) {
	// Test truncate128 native implementation
	testInputs := []string{
		"5317387130258456662214331362918410991734007599705406860481038345552731150762",
		"6542985608222806190361240322586112750744169038454362455181422643027100751666",
	}

	// Expected: matches Rust from_le_bytes_mod_order(reverse(le16))
	// Verified with Python: int.from_bytes(reversed(le16), 'little')
	expectedResults := []string{
		"226766529495067233327141020382834624226",
		"67578852428328158974664969564539724787",
	}

	fmt.Println("=== Testing Native Truncate128 ===")
	for i, inputStr := range testInputs {
		input, _ := new(big.Int).SetString(inputStr, 10)
		expected, _ := new(big.Int).SetString(expectedResults[i], 10)

		result := Truncate128Native(input)

		fmt.Printf("Input:    %s\n", input.String())
		fmt.Printf("Got:      %s\n", result.String())
		fmt.Printf("Expected: %s\n", expected.String())

		if result.Cmp(expected) != 0 {
			t.Errorf("MISMATCH at index %d!", i)
		} else {
			fmt.Println("MATCH!")
		}
		fmt.Println()
	}
}

func TestNativeAppendU64(t *testing.T) {
	// Test that AppendU64TransformNative matches the hint function
	testValues := []uint64{0, 4096, 8192, 1024, 32768}

	fmt.Println("=== Testing Native AppendU64Transform ===")
	for _, val := range testValues {
		native := AppendU64TransformNative(val)

		// Also test hint function
		outputs := make([]*big.Int, 1)
		outputs[0] = new(big.Int)
		appendU64TransformHint(nil, []*big.Int{big.NewInt(int64(val))}, outputs)

		fmt.Printf("val=%d:\n", val)
		fmt.Printf("  Native: %s\n", native.String())
		fmt.Printf("  Hint:   %s\n", outputs[0].String())

		if native.Cmp(outputs[0]) != 0 {
			t.Errorf("MISMATCH for val=%d!", val)
		} else {
			fmt.Println("  MATCH!")
		}
	}
}

func TestNativeByteReverse(t *testing.T) {
	testInputs := []string{
		"0",
		"339982277569909227036194045546146182803753403674367144357178929037303311570",
	}

	expectedResults := []string{
		"0",
		"7625174665854580928319602534203856263707274111151135344721492977228796706812",
	}

	fmt.Println("=== Testing Native ByteReverse ===")
	for i, inputStr := range testInputs {
		input, _ := new(big.Int).SetString(inputStr, 10)
		expected, _ := new(big.Int).SetString(expectedResults[i], 10)

		result := ByteReverseNative(input)

		fmt.Printf("Input:    %s\n", input.String())
		fmt.Printf("Got:      %s\n", result.String())
		fmt.Printf("Expected: %s\n", expected.String())

		if result.Cmp(expected) != 0 {
			t.Errorf("MISMATCH at index %d!", i)
		} else {
			fmt.Println("MATCH!")
		}
		fmt.Println()
	}
}

func TestDebugHashChain(t *testing.T) {
	// Test hash chain
	fmt.Println("=== Testing Hash Chain ===")
	
	// Start with "Jolt" label = 1953263434
	h := HashNative(big.NewInt(1953263434), big.NewInt(0), big.NewInt(0))
	fmt.Printf("Hash([Jolt,0,0]): %s\n", h.String())
	
	// Round 0: hash(state, 0, 0)
	h = HashNative(h, big.NewInt(0), big.NewInt(0))
	fmt.Printf("Round 0 hash: %s\n", h.String())
	
	// Test Truncate128
	t128 := Truncate128Native(h)
	fmt.Printf("Truncate128(h): %s\n", t128.String())
	
	// Test Truncate128Reverse - needs the native function
}
