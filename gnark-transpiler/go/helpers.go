package jolt_verifier

import (
	"math/big"

	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/emulated"
)

// bigInt creates a *big.Int from a string, for constants too large for int64
func bigInt(s string) *big.Int {
	n, _ := new(big.Int).SetString(s, 10)
	return n
}

// fqToNative converts an emulated Fq element to a native Fr element.
// Uses bit decomposition: Fq -> bits -> Fr.
// Must Reduce first to ensure canonical representation.
func fqToNative(api frontend.API, fqField *emulated.Field[emulated.BN254Fp], x *emulated.Element[emulated.BN254Fp]) frontend.Variable {
	reduced := fqField.Reduce(x)
	bits := fqField.ToBits(reduced)
	return api.FromBinary(bits[:254]...)
}
