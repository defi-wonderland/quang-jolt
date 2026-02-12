// Package poseidon provides Poseidon hash implementations for gnark circuits.
//
// This file implements the Fq Poseidon transcript for recursion sumcheck verification.
// The transcript uses Poseidon hash over BN254 Fq (base field) with:
// - Width = 4 (3 inputs + 1 capacity)
// - 8 full rounds, 56 partial rounds
// - Alpha = 5 (S-box exponent)
//
// Hash structure: hash(state, n_rounds, data) for domain separation.
// This matches the Rust PoseidonTranscript implementation.
package poseidon

import (
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/emulated"
)

// FqTranscript implements the Fiat-Shamir transcript using Poseidon over Fq.
// Used for recursion sumcheck verification where operations are in Grumpkin's scalar field (= BN254 Fq).
type FqTranscript struct {
	api     frontend.API
	fqField *emulated.Field[emulated.BN254Fp]

	// state is the 254-bit running state (stored as emulated Fq element)
	state *emulated.Element[emulated.BN254Fp]

	// nRounds counter for domain separation
	nRounds frontend.Variable
}

// NewFqTranscript creates a new Fq Poseidon transcript with the given label.
// Matches Rust: hash(label_padded, 0, 0) to initialize state.
func NewFqTranscript(api frontend.API, fqField *emulated.Field[emulated.BN254Fp], label frontend.Variable) *FqTranscript {
	// Initial hash: state = hash(label, 0, 0)
	// Label is assumed to be already padded/converted to a field element
	zero := fqField.Zero()
	labelFq := fqField.FromBits(api.ToBinary(label, 254)...)

	initialState := hashFqInternal(api, fqField, labelFq, zero, zero)

	return &FqTranscript{
		api:     api,
		fqField: fqField,
		state:   initialState,
		nRounds: 0,
	}
}

// NewFqTranscriptFromState creates a transcript with pre-initialized state.
// Useful when continuing from a known transcript state (e.g., after stages 1-7).
func NewFqTranscriptFromState(
	api frontend.API,
	fqField *emulated.Field[emulated.BN254Fp],
	state *emulated.Element[emulated.BN254Fp],
	nRounds frontend.Variable,
) *FqTranscript {
	return &FqTranscript{
		api:     api,
		fqField: fqField,
		state:   state,
		nRounds: nRounds,
	}
}

// AppendScalar absorbs an Fq scalar into the transcript.
// Matches Rust: new_state = hash(state, n_rounds, scalar)
func (t *FqTranscript) AppendScalar(scalar *emulated.Element[emulated.BN254Fp]) {
	nRoundsFq := t.fqField.FromBits(t.api.ToBinary(t.nRounds, 254)...)
	t.state = hashFqInternal(t.api, t.fqField, t.state, nRoundsFq, scalar)
	t.nRounds = t.api.Add(t.nRounds, 1)
}

// AppendMessage absorbs a constant message (right-padded to 32 bytes) into the transcript.
// Used for domain separation labels like "UniPoly_begin", "UniPoly_end".
// The message is interpreted as LE bytes and converted to Fq field element.
func (t *FqTranscript) AppendMessage(msgBytes32 [32]byte) {
	// Convert bytes to big.Int then to Fq element
	// Rust uses from_le_bytes_mod_order, so we interpret as LE
	msgFq := t.fqField.NewElement(msgBytes32[:])
	t.AppendScalar(msgFq)
}

// AppendMessageString is a convenience method that pads a string to 32 bytes.
func (t *FqTranscript) AppendMessageString(msg string) {
	var padded [32]byte
	copy(padded[:], msg)
	t.AppendMessage(padded)
}

// AppendScalarNative absorbs a native Fr scalar by converting to Fq.
func (t *FqTranscript) AppendScalarNative(scalar frontend.Variable) {
	scalarFq := t.fqField.FromBits(t.api.ToBinary(scalar, 254)...)
	t.AppendScalar(scalarFq)
}

// ChallengeScalar squeezes a challenge from the transcript.
// Matches Rust: challenge = hash(state, n_rounds, 0), then update state.
// Returns the challenge as an emulated Fq element.
func (t *FqTranscript) ChallengeScalar() *emulated.Element[emulated.BN254Fp] {
	nRoundsFq := t.fqField.FromBits(t.api.ToBinary(t.nRounds, 254)...)
	zero := t.fqField.Zero()

	// Hash to get random output
	output := hashFqInternal(t.api, t.fqField, t.state, nRoundsFq, zero)

	// Update state with the output
	t.state = output
	t.nRounds = t.api.Add(t.nRounds, 1)

	return output
}

// ChallengeScalarNative squeezes a challenge and converts to native Fr.
// Used when the challenge needs to be used in native Fr operations.
func (t *FqTranscript) ChallengeScalarNative() frontend.Variable {
	challenge := t.ChallengeScalar()
	reduced := t.fqField.Reduce(challenge)
	bits := t.fqField.ToBits(reduced)
	return t.api.FromBinary(bits[:254]...)
}

// ChallengeScalar128Bits squeezes a 128-bit challenge (lower bits only).
// Matches Rust challenge_scalar_128_bits: takes lower 128 bits of hash output.
// This is used for sumcheck challenges where 128-bit security suffices.
func (t *FqTranscript) ChallengeScalar128Bits() *emulated.Element[emulated.BN254Fp] {
	nRoundsFq := t.fqField.FromBits(t.api.ToBinary(t.nRounds, 254)...)
	zero := t.fqField.Zero()

	// Hash to get random output
	output := hashFqInternal(t.api, t.fqField, t.state, nRoundsFq, zero)

	// Update state
	t.state = output
	t.nRounds = t.api.Add(t.nRounds, 1)

	// Truncate to 128 bits: extract bits, zero out upper bits, reconstruct
	reduced := t.fqField.Reduce(output)
	bits := t.fqField.ToBits(reduced)

	// Take only lower 128 bits
	truncated := t.fqField.FromBits(bits[:128]...)

	return truncated
}

// GetState returns the current transcript state (for debugging/testing).
func (t *FqTranscript) GetState() *emulated.Element[emulated.BN254Fp] {
	return t.state
}

// GetNRounds returns the current round counter (for debugging/testing).
func (t *FqTranscript) GetNRounds() frontend.Variable {
	return t.nRounds
}

// hashFqInternal computes Poseidon hash of 3 Fq elements.
// This is the core hash function used by the transcript.
func hashFqInternal(
	api frontend.API,
	fqField *emulated.Field[emulated.BN254Fp],
	in1, in2, in3 *emulated.Element[emulated.BN254Fp],
) *emulated.Element[emulated.BN254Fp] {
	state := [FqWidth]*emulated.Element[emulated.BN254Fp]{fqField.Zero(), in1, in2, in3}

	halfFull := FqFullRounds / 2 // 4

	// First half of full rounds
	for r := 0; r < halfFull; r++ {
		for i := 0; i < FqWidth; i++ {
			rc := fqField.NewElement(fqRoundConstants[r*FqWidth+i])
			state[i] = fqField.Add(state[i], rc)
		}
		for i := 0; i < FqWidth; i++ {
			state[i] = fqEmulatedExp5(fqField, state[i])
		}
		state = fqEmulatedMix(fqField, state)
	}

	// Partial rounds
	for r := 0; r < FqPartialRounds; r++ {
		rcOffset := halfFull*FqWidth + r*FqWidth
		for i := 0; i < FqWidth; i++ {
			rc := fqField.NewElement(fqRoundConstants[rcOffset+i])
			state[i] = fqField.Add(state[i], rc)
		}
		state[0] = fqEmulatedExp5(fqField, state[0])
		state = fqEmulatedMix(fqField, state)
	}

	// Second half of full rounds
	for r := 0; r < halfFull; r++ {
		rcOffset := (halfFull+FqPartialRounds)*FqWidth + r*FqWidth
		for i := 0; i < FqWidth; i++ {
			rc := fqField.NewElement(fqRoundConstants[rcOffset+i])
			state[i] = fqField.Add(state[i], rc)
		}
		for i := 0; i < FqWidth; i++ {
			state[i] = fqEmulatedExp5(fqField, state[i])
		}
		state = fqEmulatedMix(fqField, state)
	}

	// Return first element of state (capacity element, standard Poseidon output)
	return fqField.Reduce(state[0])
}
