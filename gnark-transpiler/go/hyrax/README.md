# Hyrax Verifier Circuit Benchmark

This package implements an isolated **Hyrax polynomial commitment verifier circuit** in gnark for benchmarking Grumpkin MSM constraint costs.

## Purpose

This is a standalone experiment to validate the **feasibility** of Hyrax verification inside a BN254 Groth16 circuit, leveraging the BN254/Grumpkin 2-cycle.

**Key question answered**: How many constraints does it cost to verify a Hyrax opening with sqrt(N) = 1024 (N = 1M polynomial coefficients)?

**Answer**: ~3.55M constraints with full-precision dot product arithmetic.

## Background

Jolt uses the Hyrax polynomial commitment scheme with Grumpkin curve commitments. When building a recursive verifier for Jolt proofs inside a BN254 Groth16 circuit, the Hyrax verification is one of the most expensive components due to the two MSMs required.

The BN254/Grumpkin 2-cycle enables efficient verification:
- **Grumpkin base field** = BN254 scalar field (native in circuit)
- **Grumpkin scalar field** = BN254 base field (emulated)

This means Grumpkin point coordinates are native, but scalars require emulated field arithmetic.

## Protocol Implemented

The circuit implements the Hyrax opening verification protocol:

1. **MSM #1**: `C' = sum(L[a] * RowCommitments[a])` - Verifier derives target commitment
2. **MSM #2**: `Com(u) = sum(u[j] * Generators[j])` - Verifier commits to u
3. **Check 1**: `Com(u) == C'` - Binding property
4. **Check 2**: `<u, R> == v` - Evaluation correctness (dot product)

Where:
- `L`, `R` are eq vectors from the evaluation point
- `u` is the prover's claimed projection vector
- `v` is the claimed polynomial evaluation

## Running the Tests

```bash
# Run all tests
go test -v ./...

# Run constraint benchmarks
go test -v -run TestConstraintCount

# Run full prove/verify cycle
go test -v -run TestFullProveVerify

# Run MSM relationship validation
go test -v -run TestMSMRelationship
```

## Benchmark Results

| sqrt(N) | N (coefficients) | N (power of 2) | Constraints | Per Scalar Mul |
|---------|------------------|----------------|-------------|----------------|
| 4       | 16               | 2⁴             | 15,163      | ~1,895         |
| 16      | 256              | 2⁸             | 58,295      | ~1,821         |
| 64      | 4,096            | 2¹²            | 228,910     | ~1,788         |
| 256     | 65,536           | 2¹⁶            | 901,533     | ~1,760         |
| **1024**| **1,048,576**    | **2²⁰**        | **3,545,963** | **~1,731**   |

### Primitive Costs

| Operation           | Constraints |
|---------------------|-------------|
| Point addition      | 5           |
| Single scalar mul   | 1,776       |
| Bit decomposition   | ~254        |
| Emulated field mul  | ~500-1000   |

## Requirements

This benchmark requires **gnark master branch** (or v0.12.0+ when released) for `sw_grumpkin` support:

```go
require github.com/consensys/gnark v0.14.1-0.20260126121332-407111efab55
```

The `sw_grumpkin` package provides native Grumpkin curve operations in BN254 circuits.

## Files

- `hyrax_verifier.go` - The circuit implementation
- `hyrax_verifier_test.go` - Tests and benchmarks
- `NOTES.md` - Technical implementation notes and trade-off analysis

## Validation

This implementation has been validated against Jolt's actual Hyrax implementation in `jolt-core/src/poly/commitment/hyrax.rs`:

| Component | Jolt (Rust) | gnark (Go) | Match |
|-----------|-------------|------------|-------|
| Curve     | `ark_grumpkin` | `sw_grumpkin` | Yes |
| MSM #1    | `VariableBaseMSM::msm` on row commitments | `curve.MultiScalarMul` | Yes |
| MSM #2    | `VariableBaseMSM::msm_field_elements` on generators | `curve.MultiScalarMul` | Yes |
| Check 1   | `homomorphically_derived == product_commitment` | `curve.AssertIsEqual` | Yes |
| Check 2   | `dot_product == opening` | `scalarField.AssertIsEqual` | Yes |

## Related Documentation

For deeper technical context, see the documentation in the base Jolt repository:
- `docs/08_Stage8_Curve_Recursion_Challenges.md` - BN254/Grumpkin 2-cycle explanation
- `docs/09_RecursionVerifier_Stages_and_Hyrax.md` - Hyrax protocol details
- `docs/10_Hyrax_gnark_Implementation_Plan.md` - Full implementation plan and API analysis
