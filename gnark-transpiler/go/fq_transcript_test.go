package jolt_verifier

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/emulated"
	"github.com/consensys/gnark/test"
	"jolt_verifier/poseidon"
)

// FqTranscriptTestCircuit tests the FqTranscript against known values.
type FqTranscriptTestCircuit struct {
	// Initial label (as native Fr, will be converted to Fq)
	Label frontend.Variable

	// Values to append
	Value1 frontend.Variable
	Value2 frontend.Variable

	// Expected challenge after appending values
	ExpectedChallenge frontend.Variable
}

func (c *FqTranscriptTestCircuit) Define(api frontend.API) error {
	fqField, err := emulated.NewField[emulated.BN254Fp](api)
	if err != nil {
		return err
	}

	// Create transcript with label
	transcript := poseidon.NewFqTranscript(api, fqField, c.Label)

	// Append values
	transcript.AppendScalarNative(c.Value1)
	transcript.AppendScalarNative(c.Value2)

	// Get challenge
	challenge := transcript.ChallengeScalarNative()

	// Assert matches expected
	api.AssertIsEqual(challenge, c.ExpectedChallenge)

	return nil
}

// computeFqTranscriptBigInt computes the expected transcript output using big.Int
func computeFqTranscriptBigInt(label, value1, value2 *big.Int) *big.Int {
	fq, _ := new(big.Int).SetString("21888242871839275222246405745257275088696311157297823662689037894645226208583", 10)

	// Get constants
	rc := poseidon.GetFqRoundConstants()
	mds := poseidon.GetFqMdsMatrix()

	// Helper: Poseidon hash of 3 inputs
	hashFq := func(in1, in2, in3 *big.Int) *big.Int {
		state := [4]*big.Int{new(big.Int), new(big.Int).Set(in1), new(big.Int).Set(in2), new(big.Int).Set(in3)}

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

		halfFull := 4

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

	// Initialize transcript: state = hash(label, 0, 0)
	zero := big.NewInt(0)
	state := hashFq(label, zero, zero)
	nRounds := big.NewInt(0)

	// Append value1: state = hash(state, nRounds, value1)
	state = hashFq(state, nRounds, value1)
	nRounds.Add(nRounds, big.NewInt(1))

	// Append value2: state = hash(state, nRounds, value2)
	state = hashFq(state, nRounds, value2)
	nRounds.Add(nRounds, big.NewInt(1))

	// Challenge: output = hash(state, nRounds, 0)
	challenge := hashFq(state, nRounds, zero)

	return challenge
}

func TestFqTranscriptBasic(t *testing.T) {
	// Test with simple values
	label := big.NewInt(12345) // Simple label
	value1 := big.NewInt(100)
	value2 := big.NewInt(200)

	// Compute expected challenge using big.Int reference
	expected := computeFqTranscriptBigInt(label, value1, value2)
	t.Logf("Expected challenge: %s", expected.String())

	assignment := &FqTranscriptTestCircuit{
		Label:             label,
		Value1:            value1,
		Value2:            value2,
		ExpectedChallenge: expected,
	}

	var circuit FqTranscriptTestCircuit
	err := test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Errorf("FqTranscript basic test FAILED: %v", err)
	} else {
		t.Log("FqTranscript basic test PASSED")
	}
}

func TestFqTranscriptLargeValues(t *testing.T) {
	// Test with large field-sized values
	fq, _ := new(big.Int).SetString("21888242871839275222246405745257275088696311157297823662689037894645226208583", 10)

	label, _ := new(big.Int).SetString("12345678901234567890", 10)
	value1, _ := new(big.Int).SetString("9876543210987654321098765432109876543210", 10)
	value1.Mod(value1, fq)
	value2, _ := new(big.Int).SetString("1111111111111111111122222222222222222222", 10)
	value2.Mod(value2, fq)

	expected := computeFqTranscriptBigInt(label, value1, value2)
	t.Logf("Expected challenge (large values): %s", expected.String())

	assignment := &FqTranscriptTestCircuit{
		Label:             label,
		Value1:            value1,
		Value2:            value2,
		ExpectedChallenge: expected,
	}

	var circuit FqTranscriptTestCircuit
	err := test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Errorf("FqTranscript large values test FAILED: %v", err)
	} else {
		t.Log("FqTranscript large values test PASSED")
	}
}

// FqTranscriptMultiChallengeCircuit tests multiple challenge derivations
type FqTranscriptMultiChallengeCircuit struct {
	Label             frontend.Variable
	Values            [3]frontend.Variable
	ExpectedChallenge frontend.Variable // Challenge after all appends
}

func (c *FqTranscriptMultiChallengeCircuit) Define(api frontend.API) error {
	fqField, err := emulated.NewField[emulated.BN254Fp](api)
	if err != nil {
		return err
	}

	transcript := poseidon.NewFqTranscript(api, fqField, c.Label)

	// Append all values
	for i := 0; i < len(c.Values); i++ {
		transcript.AppendScalarNative(c.Values[i])
	}

	// Get challenge
	challenge := transcript.ChallengeScalarNative()
	api.AssertIsEqual(challenge, c.ExpectedChallenge)

	return nil
}

func computeMultiChallengeBigInt(label *big.Int, values []*big.Int) *big.Int {
	fq, _ := new(big.Int).SetString("21888242871839275222246405745257275088696311157297823662689037894645226208583", 10)
	rc := poseidon.GetFqRoundConstants()
	mds := poseidon.GetFqMdsMatrix()

	hashFq := func(in1, in2, in3 *big.Int) *big.Int {
		state := [4]*big.Int{new(big.Int), new(big.Int).Set(in1), new(big.Int).Set(in2), new(big.Int).Set(in3)}

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

		halfFull := 4
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
		for r := 0; r < 56; r++ {
			rcOffset := halfFull*4 + r*4
			for i := 0; i < 4; i++ {
				state[i].Add(state[i], rc[rcOffset+i])
				state[i].Mod(state[i], fq)
			}
			state[0] = exp5(state[0])
			state = mix(state)
		}
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

	zero := big.NewInt(0)
	state := hashFq(label, zero, zero)
	nRounds := big.NewInt(0)

	for _, val := range values {
		state = hashFq(state, nRounds, val)
		nRounds.Add(nRounds, big.NewInt(1))
	}

	challenge := hashFq(state, nRounds, zero)
	return challenge
}

func TestFqTranscriptMultiAppend(t *testing.T) {
	label := big.NewInt(999)
	values := []*big.Int{big.NewInt(1), big.NewInt(2), big.NewInt(3)}

	expected := computeMultiChallengeBigInt(label, values)
	t.Logf("Expected challenge (multi-append): %s", expected.String())

	assignment := &FqTranscriptMultiChallengeCircuit{
		Label:             label,
		Values:            [3]frontend.Variable{values[0], values[1], values[2]},
		ExpectedChallenge: expected,
	}

	var circuit FqTranscriptMultiChallengeCircuit
	err := test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Errorf("FqTranscript multi-append test FAILED: %v", err)
	} else {
		t.Log("FqTranscript multi-append test PASSED")
	}
}
