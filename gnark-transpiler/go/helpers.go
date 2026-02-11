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

// fqToNative converts an emulated BN254 Fq element back to a native Fr variable.
// Uses ToBits for proper range-checked decomposition, then FromBinary to reconstruct natively.
func fqToNative(api frontend.API, fqField *emulated.Field[emulated.BN254Fp], elem *emulated.Element[emulated.BN254Fp]) frontend.Variable {
	bits := fqField.ToBits(elem)
	return api.FromBinary(bits...)
}
