// Package poseidon - native (non-circuit) Poseidon implementation for debugging
package poseidon

import (
	"math/big"
)

// BN254 prime
var bn254Prime, _ = new(big.Int).SetString("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10)

// NativeState is a 4-element state array for native Poseidon
type NativeState = [BN254_SPONGE_WIDTH]*big.Int

// HashNative computes Poseidon hash of 3 field elements natively (for debugging)
func HashNative(in1, in2, in3 *big.Int) *big.Int {
	// Initialize state: [0, in1, in2, in3]
	state := NativeState{
		big.NewInt(0),
		new(big.Int).Set(in1),
		new(big.Int).Set(in2),
		new(big.Int).Set(in3),
	}

	result := poseidonNative(state)
	return result[0]
}

func poseidonNative(state NativeState) NativeState {
	state = arkNative(state, 0)
	state = fullRoundsNative(state, true)
	state = partialRoundsNative(state)
	state = fullRoundsNative(state, false)
	return state
}

func fullRoundsNative(state NativeState, isFirst bool) NativeState {
	for i := 0; i < BN254_FULL_ROUNDS/2-1; i++ {
		state = exp5stateNative(state)
		if isFirst {
			state = arkNative(state, (i+1)*BN254_SPONGE_WIDTH)
		} else {
			state = arkNative(state, (BN254_FULL_ROUNDS/2+1)*BN254_SPONGE_WIDTH+BN254_PARTIAL_ROUNDS+i*BN254_SPONGE_WIDTH)
		}
		state = mixNative(state, mMatrix)
	}

	state = exp5stateNative(state)
	if isFirst {
		state = arkNative(state, (BN254_FULL_ROUNDS/2)*BN254_SPONGE_WIDTH)
		state = mixNative(state, pMatrix)
	} else {
		state = mixNative(state, mMatrix)
	}

	return state
}

func partialRoundsNative(state NativeState) NativeState {
	for i := 0; i < BN254_PARTIAL_ROUNDS; i++ {
		state[0] = exp5Native(state[0])
		state[0] = modAdd(state[0], cConstants[(BN254_FULL_ROUNDS/2+1)*BN254_SPONGE_WIDTH+i])

		newState0 := big.NewInt(0)
		for j := 0; j < BN254_SPONGE_WIDTH; j++ {
			term := modMul(sConstants[(BN254_SPONGE_WIDTH*2-1)*i+j], state[j])
			newState0 = modAdd(newState0, term)
		}

		for k := 1; k < BN254_SPONGE_WIDTH; k++ {
			term := modMul(state[0], sConstants[(BN254_SPONGE_WIDTH*2-1)*i+BN254_SPONGE_WIDTH+k-1])
			state[k] = modAdd(state[k], term)
		}
		state[0] = newState0
	}

	return state
}

func arkNative(state NativeState, it int) NativeState {
	var result NativeState
	for i := 0; i < len(state); i++ {
		result[i] = modAdd(state[i], cConstants[it+i])
	}
	return result
}

func exp5Native(x *big.Int) *big.Int {
	x2 := modMul(x, x)
	x4 := modMul(x2, x2)
	return modMul(x4, x)
}

func exp5stateNative(state NativeState) NativeState {
	var result NativeState
	for i := 0; i < BN254_SPONGE_WIDTH; i++ {
		result[i] = exp5Native(state[i])
	}
	return result
}

func mixNative(state_ NativeState, constantMatrix [][]*big.Int) NativeState {
	var result NativeState
	for i := 0; i < BN254_SPONGE_WIDTH; i++ {
		result[i] = big.NewInt(0)
	}

	for i := 0; i < BN254_SPONGE_WIDTH; i++ {
		for j := 0; j < BN254_SPONGE_WIDTH; j++ {
			term := modMul(constantMatrix[j][i], state_[j])
			result[i] = modAdd(result[i], term)
		}
	}

	return result
}

// modAdd computes (a + b) mod p
func modAdd(a, b *big.Int) *big.Int {
	result := new(big.Int).Add(a, b)
	return result.Mod(result, bn254Prime)
}

// modMul computes (a * b) mod p
func modMul(a, b *big.Int) *big.Int {
	result := new(big.Int).Mul(a, b)
	return result.Mod(result, bn254Prime)
}

// ByteReverseNative performs byte-reversal natively (same as byteReverseHint)
func ByteReverseNative(x *big.Int) *big.Int {
	inputBytes := x.Bytes()
	le := make([]byte, 32)
	for i := 0; i < len(inputBytes) && i < 32; i++ {
		le[i] = inputBytes[len(inputBytes)-1-i]
	}

	reversed := make([]byte, 32)
	for i := 0; i < 32; i++ {
		reversed[i] = le[31-i]
	}

	be := make([]byte, 32)
	for i := 0; i < 32; i++ {
		be[i] = reversed[31-i]
	}
	result := new(big.Int).SetBytes(be)
	result.Mod(result, bn254Prime)
	return result
}

// Truncate128Native performs truncation to 128 bits natively (same as truncate128Hint)
func Truncate128Native(x *big.Int) *big.Int {
	inputBytes := x.Bytes()
	le := make([]byte, 32)
	for i := 0; i < len(inputBytes) && i < 32; i++ {
		le[i] = inputBytes[len(inputBytes)-1-i]
	}
	le16 := le[:16]

	// Rust behavior: reverse 16 bytes, then from_le_bytes_mod_order
	// from_le_bytes_mod_order(reverse([b0,...,b15])) = from_le([b15,...,b0])
	// = b15 + b14*256 + ... + b0*256^15
	//
	// Go SetBytes interprets as big-endian:
	// SetBytes([b0,...,b15]) = b0*256^15 + b1*256^14 + ... + b15
	//
	// Since from_le(reverse(x)) = from_be(x), and SetBytes is from_be,
	// we pass le16 directly without reversing.
	result := new(big.Int).SetBytes(le16)
	return result
}

// AppendU64TransformNative transforms a u64 value the way the transcript does
func AppendU64TransformNative(val uint64) *big.Int {
	// Transform: pad u64 to 32 bytes (LE), reverse bytes, interpret as LE
	le := make([]byte, 32)
	for i := 0; i < 8; i++ {
		le[i] = byte(val >> (8 * i))
	}
	// remaining 24 bytes are 0

	reversed := make([]byte, 32)
	for i := 0; i < 32; i++ {
		reversed[i] = le[31-i]
	}

	be := make([]byte, 32)
	for i := 0; i < 32; i++ {
		be[i] = reversed[31-i]
	}

	result := new(big.Int).SetBytes(be)
	result.Mod(result, bn254Prime)
	return result
}

// Truncate128ReverseNative computes Truncate128Reverse natively (same as truncate128ReverseHint)
// This matches Rust's MontU128Challenge::into() for ark_bn254::Fr.
func Truncate128ReverseNative(x *big.Int) *big.Int {
	// Convert to 32-byte LE representation
	inputBytes := x.Bytes() // big-endian from big.Int
	le := make([]byte, 32)
	for i := 0; i < len(inputBytes) && i < 32; i++ {
		le[i] = inputBytes[len(inputBytes)-1-i]
	}

	// Take first 16 bytes (low 128 bits in LE format)
	le16 := make([]byte, 16)
	copy(le16, le[:16])

	// Reverse 16 bytes (like Rust buf.reverse())
	for i := 0; i < 8; i++ {
		le16[i], le16[15-i] = le16[15-i], le16[i]
	}

	// Interpret as BE to get u128 value (like Rust u128::from_be_bytes)
	value128 := new(big.Int).SetBytes(le16)

	// Apply 125-bit mask for MontU128Challenge
	mask125 := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 125), big.NewInt(1))
	valueMasked := new(big.Int).And(value128, mask125)

	// Compute BigInt representation [0, 0, low, high]
	// This equals: low * 2^128 + high * 2^192
	mask64 := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 64), big.NewInt(1))
	low := new(big.Int).And(valueMasked, mask64)
	high := new(big.Int).Rsh(valueMasked, 64)

	twoPow128 := new(big.Int).Lsh(big.NewInt(1), 128)
	twoPow192 := new(big.Int).Lsh(big.NewInt(1), 192)

	bigintValue := new(big.Int).Add(
		new(big.Int).Mul(low, twoPow128),
		new(big.Int).Mul(high, twoPow192),
	)

	// R^-1 mod p for BN254 Fr Montgomery arithmetic
	rInv, _ := new(big.Int).SetString("9915499612839321149637521777990102151350674507940716049588462388200839649614", 10)

	// Multiply by R^-1 mod p (what from_bigint_unchecked does)
	result := new(big.Int).Mul(bigintValue, rInv)
	result.Mod(result, bn254Prime)

	return result
}
