# Fq Poseidon HashFq Bug Fix

**Date:** February 11, 2026
**Branch:** `fq-emulated-wip`

## Summary

Fixed the `HashFq` function in `poseidon/poseidon.go` that was producing incorrect hash outputs when using gnark's emulated Fq arithmetic.

## The Bug

**Symptom:** Circuit output differed from big.Int reference by exactly `fq - fr`.

```
Circuit:  8417226212671951959673014861015479017793435941711031633485612824435516734249
Expected: 15477005650161725593319355367930180892414620950823348980660970931768572935650
```

**Root Cause:** gnark's emulated field `Add` does NOT automatically reduce. The MDS matrix multiplication accumulates 4 terms:

```
result[i] = sum_{j=0}^{3} mds[i][j] * state[j]
```

Each term is ~2^254, so the sum can exceed 6×Fq. Without calling `Reduce()`:
- The emulated element's limbs contain unreduced values
- `ToBits()` produces bits for the unreduced value
- The final result is wrong by multiples of `fq - fr`

## The Fix

### 1. Added `Reduce()` after MDS accumulation

In `fqEmulatedMix`:
```go
func fqEmulatedMix(fqField *emulated.Field[emulated.BN254Fp], state [FqWidth]*emulated.Element[emulated.BN254Fp]) [FqWidth]*emulated.Element[emulated.BN254Fp] {
    var result [FqWidth]*emulated.Element[emulated.BN254Fp]
    for i := 0; i < FqWidth; i++ {
        acc := fqField.Zero()
        for j := 0; j < FqWidth; j++ {
            mds := fqField.NewElement(fqMdsMatrix[i][j])
            term := fqField.Mul(mds, state[j])
            acc = fqField.Add(acc, term)
        }
        // CRITICAL: Reduce after accumulation
        result[i] = fqField.Reduce(acc)
    }
    return result
}
```

### 2. Added `Reduce()` before `ToBits()` in HashFq

```go
// Must Reduce first to get canonical representation for ToBits
reduced := fqField.Reduce(state[0])
bits := fqField.ToBits(reduced)
return api.FromBinary(bits[:254]...)
```

### 3. Changed constant loading pattern

Changed from `emulated.ValueOf[emulated.BN254Fp](...)` with `&rc` to `fqField.NewElement(...)` which returns a stable pointer. This wasn't the main bug but is cleaner.

## Testing

All tests pass after the fix:

| Test | Status |
|------|--------|
| `TestSingleHashFq` | ✅ PASSED |
| `TestTimingHashFq` (small) | ✅ PASSED |
| `TestTimingHashFq` (large) | ✅ PASSED |
| `TestMulAddFourTerms` | ✅ PASSED |
| `TestDebugHashFqRound0Full` | ✅ PASSED |

### Test Details

- **Input:** (42, 123, 456)
- **Expected:** `15477005650161725593319355367930180892414620950823348980660970931768572935650`
- **Circuit output:** Matches expected ✅

## Files Modified

| File | Changes |
|------|---------|
| `poseidon/poseidon.go` | Fixed `HashFq` and `fqEmulatedMix` with `Reduce()` calls |
| `hashfq_debug_test.go` | Added debugging tests to isolate the bug |
| `hashfq_single_test.go` | Core test for HashFq correctness |

## Key Insight

**gnark's emulated field `Add` doesn't automatically reduce.** For operations that accumulate many terms (like MDS matrix multiplication), you must call `Reduce()` before:
- `ToBits()`
- Values are used in subsequent operations that depend on canonical representation

## Next Steps

1. Copy fix to `feat/e2e-manual-recursion` branch
2. Integrate with full recursion sumcheck circuit
3. Run E2E Groth16 tests

## Running Tests

```bash
cd /Users/home/dev/parti/cryptography/zkVMs/quangvdao/quang-jolt/gnark-transpiler/go
go test -v -run TestSingleHashFq
go test -v -run TestTimingHashFq
go test -v -run TestMulAddFourTerms
```
