// Package poseidon provides Poseidon hash implementations for gnark circuits.
//
// This file implements the native Fr Poseidon transcript.
// Uses the circom-optimized Poseidon over BN254 scalar field (Fr).
// This is ~120x cheaper than the emulated Fq transcript.
//
// For recursion sumcheck verification where coefficients are in Fq but we want
// cheap Fiat-Shamir, we use this Fr transcript and convert challenges Fr→Fq.
package poseidon

import (
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/algebra/native/sw_grumpkin"
	"github.com/consensys/gnark/std/math/emulated"
)

// FrTranscript implements the Fiat-Shamir transcript using native Poseidon over Fr.
// This is much cheaper than FqTranscript (~250 vs ~30K constraints per hash).
type FrTranscript struct {
	api frontend.API

	// state is the running state (native Fr element)
	state frontend.Variable

	// nRounds counter for domain separation
	nRounds frontend.Variable
}

// NewFrTranscript creates a new native Fr Poseidon transcript with the given label.
func NewFrTranscript(api frontend.API, label frontend.Variable) *FrTranscript {
	// Initial hash: state = hash(label, 0, 0)
	zero := frontend.Variable(0)
	initialState := Hash(api, label, zero, zero)

	return &FrTranscript{
		api:     api,
		state:   initialState,
		nRounds: zero,
	}
}

// NewFrTranscriptFromState creates a transcript with pre-initialized state.
func NewFrTranscriptFromState(
	api frontend.API,
	state frontend.Variable,
	nRounds frontend.Variable,
) *FrTranscript {
	return &FrTranscript{
		api:     api,
		state:   state,
		nRounds: nRounds,
	}
}

// AppendScalar absorbs a native Fr scalar into the transcript.
func (t *FrTranscript) AppendScalar(scalar frontend.Variable) {
	t.state = Hash(t.api, t.state, t.nRounds, scalar)
	t.nRounds = t.api.Add(t.nRounds, 1)
}

// AppendMessage absorbs a constant message (32 bytes) into the transcript.
// The message is interpreted as big-endian bytes converted to a field element.
func (t *FrTranscript) AppendMessage(msgBytes32 [32]byte) {
	// Convert bytes to field element constant
	// gnark interprets bytes as big-endian when creating constants
	msgVar := frontend.Variable(msgBytes32[:])
	t.AppendScalar(msgVar)
}

// AppendScalarFq absorbs an emulated Fq scalar by converting to Fr.
// Uses bit decomposition: Fq → bits → Fr (cheap, ~254 constraints).
func (t *FrTranscript) AppendScalarFq(fq *emulated.Field[emulated.BN254Fp], scalar *emulated.Element[emulated.BN254Fp]) {
	// Reduce to get canonical representation
	reduced := fq.Reduce(scalar)
	// Convert to bits
	bits := fq.ToBits(reduced)
	// Reconstruct as native Fr (take lower 254 bits)
	frVal := t.api.FromBinary(bits[:254]...)
	t.AppendScalar(frVal)
}

// AppendScalarGrumpkin absorbs a Grumpkin scalar (= BN254 Fq) by converting to Fr.
// This is the same as AppendScalarFq but uses sw_grumpkin.ScalarField type.
// Uses bit decomposition: Fq → bits → Fr (cheap, ~254 constraints).
func (t *FrTranscript) AppendScalarGrumpkin(fq *emulated.Field[sw_grumpkin.ScalarField], scalar *sw_grumpkin.Scalar) {
	// Reduce to get canonical representation
	reduced := fq.Reduce(scalar)
	// Convert to bits
	bits := fq.ToBits(reduced)
	// Reconstruct as native Fr (take lower 254 bits)
	frVal := t.api.FromBinary(bits[:254]...)
	t.AppendScalar(frVal)
}

// ChallengeScalar squeezes a challenge from the transcript.
// Returns a native Fr element.
func (t *FrTranscript) ChallengeScalar() frontend.Variable {
	zero := frontend.Variable(0)

	// Hash to get random output
	output := Hash(t.api, t.state, t.nRounds, zero)

	// Update state
	t.state = output
	t.nRounds = t.api.Add(t.nRounds, 1)

	return output
}

// GetState returns the current transcript state.
func (t *FrTranscript) GetState() frontend.Variable {
	return t.state
}

// GetNRounds returns the current round counter.
func (t *FrTranscript) GetNRounds() frontend.Variable {
	return t.nRounds
}
