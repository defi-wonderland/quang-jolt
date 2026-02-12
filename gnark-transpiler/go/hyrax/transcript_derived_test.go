package hyrax

import (
	"encoding/json"
	"math/big"
	"os"
	"testing"

	"jolt_verifier/poseidon"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/std/math/emulated"
	"github.com/consensys/gnark/test"
)

// Fq modulus (BN254 base field = Grumpkin scalar field)
var FqMod, _ = new(big.Int).SetString("21888242871839275222246405745257275088696311157297823662689037894645226208583", 10)

// computeHashFqBigInt computes Poseidon hash of 3 Fq elements using big.Int
// This matches the circuit HashFq function.
func computeHashFqBigInt(in1, in2, in3 *big.Int) *big.Int {
	state := [4]*big.Int{big.NewInt(0), new(big.Int).Set(in1), new(big.Int).Set(in2), new(big.Int).Set(in3)}

	halfFull := poseidon.FqFullRounds / 2 // 4
	roundConstants := poseidon.GetFqRoundConstants()

	// First half of full rounds
	for r := 0; r < halfFull; r++ {
		for i := 0; i < poseidon.FqWidth; i++ {
			rc := roundConstants[r*poseidon.FqWidth+i]
			state[i].Add(state[i], rc)
			state[i].Mod(state[i], FqMod)
		}
		for i := 0; i < poseidon.FqWidth; i++ {
			state[i] = exp5BigInt(state[i])
		}
		state = mdsMixBigInt(state)
	}

	// Partial rounds
	for r := 0; r < poseidon.FqPartialRounds; r++ {
		rcOffset := halfFull*poseidon.FqWidth + r*poseidon.FqWidth
		for i := 0; i < poseidon.FqWidth; i++ {
			rc := roundConstants[rcOffset+i]
			state[i].Add(state[i], rc)
			state[i].Mod(state[i], FqMod)
		}
		state[0] = exp5BigInt(state[0])
		state = mdsMixBigInt(state)
	}

	// Second half of full rounds
	for r := 0; r < halfFull; r++ {
		rcOffset := (halfFull+poseidon.FqPartialRounds)*poseidon.FqWidth + r*poseidon.FqWidth
		for i := 0; i < poseidon.FqWidth; i++ {
			rc := roundConstants[rcOffset+i]
			state[i].Add(state[i], rc)
			state[i].Mod(state[i], FqMod)
		}
		for i := 0; i < poseidon.FqWidth; i++ {
			state[i] = exp5BigInt(state[i])
		}
		state = mdsMixBigInt(state)
	}

	return state[0]
}

func exp5BigInt(x *big.Int) *big.Int {
	x2 := new(big.Int).Mul(x, x)
	x2.Mod(x2, FqMod)
	x4 := new(big.Int).Mul(x2, x2)
	x4.Mod(x4, FqMod)
	x5 := new(big.Int).Mul(x4, x)
	x5.Mod(x5, FqMod)
	return x5
}

func mdsMixBigInt(state [4]*big.Int) [4]*big.Int {
	mdsMatrix := poseidon.GetFqMdsMatrix()
	result := [4]*big.Int{big.NewInt(0), big.NewInt(0), big.NewInt(0), big.NewInt(0)}
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			term := new(big.Int).Mul(mdsMatrix[i][j], state[j])
			result[i].Add(result[i], term)
		}
		result[i].Mod(result[i], FqMod)
	}
	return result
}

// FqTranscriptBigInt is a reference implementation of the transcript using big.Int
type FqTranscriptBigInt struct {
	state   *big.Int
	nRounds int64
}

func NewFqTranscriptBigInt(state *big.Int, nRounds int64) *FqTranscriptBigInt {
	return &FqTranscriptBigInt{
		state:   new(big.Int).Set(state),
		nRounds: nRounds,
	}
}

func (t *FqTranscriptBigInt) AppendScalar(scalar *big.Int) {
	nRoundsBig := big.NewInt(t.nRounds)
	t.state = computeHashFqBigInt(t.state, nRoundsBig, scalar)
	t.nRounds++
}

func (t *FqTranscriptBigInt) AppendMessage(msgBytes32 [32]byte) {
	// Convert bytes to big.Int (BE interpretation - matches gnark's SetBytes)
	msgBig := new(big.Int).SetBytes(msgBytes32[:])
	msgBig.Mod(msgBig, FqMod)
	t.AppendScalar(msgBig)
}

func (t *FqTranscriptBigInt) ChallengeScalar() *big.Int {
	nRoundsBig := big.NewInt(t.nRounds)
	zero := big.NewInt(0)
	output := computeHashFqBigInt(t.state, nRoundsBig, zero)
	t.state = output
	t.nRounds++
	return new(big.Int).Set(output)
}

func reverseBytes(b []byte) []byte {
	result := make([]byte, len(b))
	for i := range b {
		result[i] = b[len(b)-1-i]
	}
	return result
}

// computeSumcheckWithTranscript computes the expected final evaluation using the reference transcript
func computeSumcheckDegree7WithTranscriptBigInt(
	coeffs [][7]*big.Int,
	transcript *FqTranscriptBigInt,
) (*big.Int, []*big.Int) {
	prevEval := big.NewInt(0)
	two := big.NewInt(2)
	challenges := make([]*big.Int, len(coeffs))

	for round := 0; round < len(coeffs); round++ {
		c := coeffs[round]

		// Transcript protocol
		transcript.AppendMessage(UniPolyBeginMsg)
		for i := 0; i < 7; i++ {
			transcript.AppendScalar(c[i])
		}
		transcript.AppendMessage(UniPolyEndMsg)

		r := transcript.ChallengeScalar()
		challenges[round] = r

		// Reconstruct c1
		c1 := new(big.Int).Set(prevEval)
		twoC0 := new(big.Int).Mul(two, c[0])
		c1.Sub(c1, twoC0)
		for i := 1; i < 7; i++ {
			c1.Sub(c1, c[i])
		}
		c1.Mod(c1, FqMod)

		// Horner's evaluation
		acc := new(big.Int).Set(c[6]) // c7
		acc.Mul(r, acc)
		acc.Add(c[5], acc) // + c6
		acc.Mod(acc, FqMod)

		acc.Mul(r, acc)
		acc.Add(c[4], acc) // + c5
		acc.Mod(acc, FqMod)

		acc.Mul(r, acc)
		acc.Add(c[3], acc) // + c4
		acc.Mod(acc, FqMod)

		acc.Mul(r, acc)
		acc.Add(c[2], acc) // + c3
		acc.Mod(acc, FqMod)

		acc.Mul(r, acc)
		acc.Add(c[1], acc) // + c2
		acc.Mod(acc, FqMod)

		acc.Mul(r, acc)
		acc.Add(c1, acc) // + c1
		acc.Mod(acc, FqMod)

		acc.Mul(r, acc)
		acc.Add(c[0], acc) // + c0
		acc.Mod(acc, FqMod)

		prevEval = acc
	}

	return prevEval, challenges
}

// TestTranscriptDerivedStage1Constraints tests the constraint count
func TestTranscriptDerivedStage1Constraints(t *testing.T) {
	circuit := &TranscriptDerivedStage1Circuit{}
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Failed to compile: %v", err)
	}
	t.Logf("TranscriptDerivedStage1Circuit (11 rounds, degree 7): %d constraints", cs.GetNbConstraints())
}

// TestTranscriptDerivedAllStagesConstraints tests the combined constraint count
func TestTranscriptDerivedAllStagesConstraints(t *testing.T) {
	circuit := &TranscriptDerivedAllStagesCircuit{}
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Failed to compile: %v", err)
	}
	t.Logf("TranscriptDerivedAllStagesCircuit: %d constraints", cs.GetNbConstraints())
}

// TestTranscriptDerivedStage1Solver tests Stage 1 with transcript-derived challenges
func TestTranscriptDerivedStage1Solver(t *testing.T) {
	w := loadRecursionWitness(t)
	stage1Coeffs := loadStage1Coeffs(w)

	// Compute expected using reference transcript
	// Start with initial state = 0, nRounds = 0 (for this test)
	initialState := big.NewInt(12345) // arbitrary initial state for testing
	initialNRounds := int64(100)      // arbitrary initial round count
	transcript := NewFqTranscriptBigInt(initialState, initialNRounds)
	expected, challenges := computeSumcheckDegree7WithTranscriptBigInt(stage1Coeffs, transcript)

	t.Logf("Initial state: %s", initialState.String())
	t.Logf("Expected final: %s", expected.String())
	t.Logf("First challenge: %s", challenges[0].String())

	// Build circuit witness
	circuit := &TranscriptDerivedStage1Circuit{}
	witness := &TranscriptDerivedStage1Circuit{}

	for round := 0; round < 11; round++ {
		for i := 0; i < 7; i++ {
			witness.Coeffs[round][i] = emulated.ValueOf[emulated.BN254Fp](stage1Coeffs[round][i])
		}
	}

	witness.InitialTranscriptState = emulated.ValueOf[emulated.BN254Fp](initialState)
	witness.InitialNRounds = initialNRounds
	witness.Expected = emulated.ValueOf[emulated.BN254Fp](expected)

	err := test.IsSolved(circuit, witness, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("TranscriptDerivedStage1 solver failed: %v", err)
	}
	t.Log("TranscriptDerivedStage1 solver PASSED!")
}

// loadRecursionWitnessLocal is a duplicate helper for this test file
func loadRecursionWitnessLocal(t *testing.T) *RecursionWitness {
	data, err := os.ReadFile("../recursion_witness.json")
	if err != nil {
		t.Skipf("Could not load recursion witness file: %v", err)
	}

	var witness RecursionWitness
	if err := json.Unmarshal(data, &witness); err != nil {
		t.Fatal(err)
	}
	return &witness
}
