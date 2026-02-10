# RecursionVerifier Genericization for Transpilation

## Overview

This document describes the work to make the `RecursionVerifier` and its constituent stages generic over the field type `F`, enabling symbolic transpilation of recursion stages to Gnark.

## Architecture

The Jolt zkVM verifier has a two-layer proof structure:

1. **Stages 1-7**: Main verifier stages operating over Fr (BN254 scalar field)
2. **Recursion SNARK**: Additional stages operating over Fq (Grumpkin scalar field = BN254 base field)
3. **Hyrax PCS**: Final polynomial commitment opening (native Gnark, not transpiled)

The recursion SNARK has 5 sumcheck stages:
- **Stage 1**: Packed GT exponentiation sumcheck
- **Stage 2**: Batched constraint sumchecks (shift rho, claim reductions, curve ops)
- **Stage 3**: Direct evaluation protocol (virtualization)
- **Stage 4**: Jagged transform sumcheck
- **Stage 5**: Jagged assist sumcheck

## Changes Made

### 1. RecursionVerifier Genericization (jolt-core/src/zkvm/recursion/recursion_verifier.rs)

Added a fully generic `verify_sumchecks_symbolic` method that operates over any `F: JoltField`:

```rust
pub fn verify_sumchecks_symbolic<F: JoltField, T: Transcript, A: OpeningAccumulator<F>>(
    &self,
    stage1_proof: &SumcheckInstanceProof<F, T>,
    stage2_proof: &SumcheckInstanceProof<F, T>,
    stage3_m_eval: F,
    stage4_proof: &SumcheckInstanceProof<F, T>,
    stage5_proof: &JaggedAssistProof<F, T>,
    transcript: &mut T,
    accumulator: &mut A,
) -> Result<(), Box<dyn std::error::Error>>
```

The existing `_generic` methods now delegate to `_symbolic` variants with `F = Fq`.

### 2. Stage Verifiers Made Generic

- **PackedGtExpVerifier<F>**: Already generic, supports symbolic execution
- **ShiftRhoVerifier<F>**: Generic over F, degree 3
- **PackedGtExpClaimReductionVerifier<F>**: Generic over F, degree 2
- **DirectEvaluationVerifier**: Generic for Stage 3

### 3. symbolize_recursion_proof Function (gnark-transpiler/src/symbolic_proof.rs)

Created a function to convert the real recursion proof (Fq elements) to symbolic MleAst variables:

```rust
pub fn symbolize_recursion_proof(
    real_proof: &RV64IMACProof,
    alloc: &mut VarAllocator,
) -> RecursionProof<MleAst, PoseidonAstTranscript, AstCommitmentScheme>
```

This handles:
- Stage 1-5 sumcheck proofs (coefficient symbolization)
- Stage 3 M evaluation
- Stage 5 claimed evaluations

### 4. MleOpeningAccumulator Update (gnark-transpiler/src/mle_opening_accumulator.rs)

Modified `append_virtual` in the `OpeningAccumulator<MleAst>` trait implementation to handle missing claims gracefully for recursion stages:

```rust
fn append_virtual<T: Transcript>(&mut self, ...) {
    // For recursion stages, claims may not be pre-populated.
    // Insert a symbolic zero claim if one doesn't exist.
    if let Some((stored_point, claim)) = self.openings.get_mut(&key) {
        transcript.append_scalar(claim);
        *stored_point = point;
    } else {
        let symbolic_claim = MleAst::zero();
        transcript.append_scalar(&symbolic_claim);
        self.openings.insert(key, (point, symbolic_claim));
    }
}
```

### 5. main.rs Integration (gnark-transpiler/src/main.rs)

Added recursion transpilation after stages 1-7:
- Builds `RecursionVerifierInput` from proof metadata
- Creates symbolic recursion proof with `symbolize_recursion_proof`
- Runs `verify_sumchecks_symbolic` with MleAst
- Merges recursion assertions with stages 1-7 assertions

## Current Status

**Transpilation output:**
- Stages 1-7: 17 assertions (working)
- Recursion stages: 1 assertion (partial - Stage 2 incomplete)
- **Total: 18 assertions**

## Known Limitations

### Stage 2 Curve-Specific Verifiers

Stage 2 includes curve-specific verifiers that have not yet been made generic:
- **G1ScalarMulVerifier**: Degree 6, uses curve constants
- **G2ScalarMulVerifier**: Uses Fq2 extension field
- **GtMulVerifier**: Uses Fq12 tower extension
- **G1AddVerifier**: Uses curve addition formulas
- **G2AddVerifier**: Uses Fq2 curve addition

**Impact:** The real proof was generated with all Stage 2 verifiers (max degree 6), but symbolic verification only runs the generic verifiers (ShiftRho degree 3, PackedGtExpReduction degree 2). This causes a degree mismatch error: `InvalidInputLength(3, 6)`.

### Resolution Path

To fully transpile recursion stages, the curve-specific verifiers need to be made generic. Two approaches:

1. **Full Genericization**: Make each verifier generic over F, embedding curve constants as field elements
2. **Symbolic Stubs**: Create simplified symbolic verifiers that only track constraint structure

The recommended approach is #1, as it maintains verification correctness while enabling transpilation.

## Testing

Run the transpiler:
```bash
cargo run -p gnark-transpiler --release
```

Expected output:
```
=== Stages 1-7 Assertions ===
  Assertions: 17

=== Running Symbolic Recursion Verification (Stages 9-13) ===
  Constraint count: 332
  num_s_vars: 15, num_constraint_vars: 11
  Recursion symbolic variables: 2366
  Recursion verification error: InvalidInputLength(3, 6)  # Expected - Stage 2 limitation

=== Recursion Assertions ===
  Assertions: 1

=== Total Accumulated Assertions ===
  Total assertions: 18
```

## Files Modified

| File | Changes |
|------|---------|
| `jolt-core/src/zkvm/recursion/recursion_verifier.rs` | Added `verify_sumchecks_symbolic` and stage-specific `_symbolic` methods |
| `gnark-transpiler/src/symbolic_proof.rs` | Added `symbolize_recursion_proof` function |
| `gnark-transpiler/src/mle_opening_accumulator.rs` | Updated `append_virtual` for recursion support |
| `gnark-transpiler/src/main.rs` | Integrated recursion transpilation |

---

*Last updated: February 10, 2026*
