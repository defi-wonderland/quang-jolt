package jolt_verifier

import (
	"fmt"
	"math/big"
	"testing"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/emulated"
	"github.com/consensys/gnark/test"
	"jolt_verifier/poseidon"
)

// SingleHashFqCircuit tests exactly 1 call to poseidon.HashFq
type SingleHashFqCircuit struct {
	In1      frontend.Variable
	In2      frontend.Variable
	In3      frontend.Variable
	Expected frontend.Variable
}

func (c *SingleHashFqCircuit) Define(api frontend.API) error {
	fqField, err := emulated.NewField[emulated.BN254Fp](api)
	if err != nil {
		return err
	}

	result := poseidon.HashFq(api, fqField, c.In1, c.In2, c.In3)
	api.AssertIsEqual(result, c.Expected)
	return nil
}

// computeHashFqBigInt computes Poseidon Fq hash using pure big.Int arithmetic
func computeHashFqBigInt(in1, in2, in3 *big.Int) *big.Int {
	fq, _ := new(big.Int).SetString("21888242871839275222246405745257275088696311157297823662689037894645226208583", 10)

	// Load constants from the Go package (use the same ones)
	rc := poseidon.GetFqRoundConstants()
	mds := poseidon.GetFqMdsMatrix()

	state := [4]*big.Int{new(big.Int), new(big.Int).Set(in1), new(big.Int).Set(in2), new(big.Int).Set(in3)}

	halfFull := 4 // FqFullRounds / 2

	exp5 := func(x *big.Int) *big.Int {
		x2 := new(big.Int).Mul(x, x)
		x2.Mod(x2, fq)
		x4 := new(big.Int).Mul(x2, x2)
		x4.Mod(x4, fq)
		r := new(big.Int).Mul(x4, x)
		r.Mod(r, fq)
		return r
	}

	mix := func(st [4]*big.Int) [4]*big.Int {
		var result [4]*big.Int
		for i := 0; i < 4; i++ {
			acc := new(big.Int)
			for j := 0; j < 4; j++ {
				term := new(big.Int).Mul(mds[i][j], st[j])
				acc.Add(acc, term)
			}
			result[i] = acc.Mod(acc, fq)
		}
		return result
	}

	// First half of full rounds
	for r := 0; r < halfFull; r++ {
		for i := 0; i < 4; i++ {
			state[i].Add(state[i], rc[r*4+i])
			state[i].Mod(state[i], fq)
		}
		for i := 0; i < 4; i++ {
			state[i] = exp5(state[i])
		}
		state = mix(state)
	}

	// Partial rounds
	for r := 0; r < 56; r++ {
		rcOffset := halfFull*4 + r*4
		for i := 0; i < 4; i++ {
			state[i].Add(state[i], rc[rcOffset+i])
			state[i].Mod(state[i], fq)
		}
		state[0] = exp5(state[0])
		state = mix(state)
	}

	// Second half of full rounds
	for r := 0; r < halfFull; r++ {
		rcOffset := (halfFull+56)*4 + r*4
		for i := 0; i < 4; i++ {
			state[i].Add(state[i], rc[rcOffset+i])
			state[i].Mod(state[i], fq)
		}
		for i := 0; i < 4; i++ {
			state[i] = exp5(state[i])
		}
		state = mix(state)
	}

	return state[0]
}

func TestSingleHashFq(t *testing.T) {
	in1 := big.NewInt(42)
	in2 := big.NewInt(123)
	in3 := big.NewInt(456)

	t.Log("Computing expected result with big.Int...")
	expected := computeHashFqBigInt(in1, in2, in3)
	t.Logf("Expected: %s", expected.String())

	assignment := &SingleHashFqCircuit{
		In1:      in1,
		In2:      in2,
		In3:      in3,
		Expected: expected,
	}

	t.Log("Running gnark solver...")
	start := time.Now()
	var circuit SingleHashFqCircuit
	err := test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
	elapsed := time.Since(start)
	t.Logf("Solver took: %v", elapsed)

	if err != nil {
		t.Errorf("SingleHashFq FAILED: %v", err)
	} else {
		t.Log("SingleHashFq PASSED")
	}
}

func TestTimingHashFq(t *testing.T) {
	// Test with increasingly large values to check timing
	fq, _ := new(big.Int).SetString("21888242871839275222246405745257275088696311157297823662689037894645226208583", 10)

	for _, label := range []string{"small", "large"} {
		var in1, in2, in3 *big.Int
		if label == "small" {
			in1, in2, in3 = big.NewInt(1), big.NewInt(2), big.NewInt(3)
		} else {
			in1, _ = new(big.Int).SetString("12345678901234567890123456789012345678901234567890123456789012345", 10)
			in1.Mod(in1, fq)
			in2, _ = new(big.Int).SetString("98765432109876543210987654321098765432109876543210987654321098765", 10)
			in2.Mod(in2, fq)
			in3, _ = new(big.Int).SetString("55555555555555555555555555555555555555555555555555555555555555555", 10)
			in3.Mod(in3, fq)
		}

		expected := computeHashFqBigInt(in1, in2, in3)

		assignment := &SingleHashFqCircuit{
			In1:      in1,
			In2:      in2,
			In3:      in3,
			Expected: expected,
		}

		start := time.Now()
		var circuit SingleHashFqCircuit
		err := test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
		elapsed := time.Since(start)

		if err != nil {
			t.Errorf("[%s] FAILED: %v", label, err)
		} else {
			t.Logf("[%s] PASSED in %v", label, elapsed)
		}
	}

	fmt.Printf("Estimated total for 24 hashes: extrapolate from above\n")
}
