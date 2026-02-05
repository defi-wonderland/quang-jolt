# Hyrax Verifier Implementation Notes

This document captures technical considerations and trade-offs for the Hyrax verifier circuit.

## Overview

This benchmark implements an isolated Hyrax polynomial commitment verifier in gnark to measure constraint costs for Grumpkin MSMs inside a BN254 Groth16 circuit.

**Why this matters**: Jolt uses Hyrax with Grumpkin commitments. Building a recursive Jolt verifier requires implementing Hyrax verification in-circuit. The BN254/Grumpkin 2-cycle enables this - Grumpkin's base field equals BN254's scalar field, making Grumpkin point coordinates native in BN254 circuits.

**Key result**: ~3.55M constraints for sqrt(N) = 1024, well within practical limits (~50M).

For protocol background, see `docs/09_RecursionVerifier_Stages_and_Hyrax.md` in the base Jolt repo.

## Code Walkthrough

The implementation uses two gnark standard library packages:

```go
import (
    "github.com/consensys/gnark/frontend"
    "github.com/consensys/gnark/std/algebra/native/sw_grumpkin"
    "github.com/consensys/gnark/std/math/emulated"
)
```

### What Each Package Provides

| Package | Purpose | What It Gives Us |
|---------|---------|------------------|
| `frontend` | Core circuit API | `frontend.Variable` (native field elements), `api.Add`, `api.Mul`, `api.ToBinary` |
| `sw_grumpkin` | Grumpkin curve gadget | `G1Affine` (points), `Scalar` (emulated scalars), `Curve.MultiScalarMul` |
| `emulated` | Non-native field arithmetic | `Field[T].Mul`, `Field[T].Add`, `Field[T].FromBits` |

### What Is a "Gadget"?

In gnark, a **gadget** is a pre-built circuit component that abstracts complex operations. The `sw_grumpkin` package is a gadget for Grumpkin curve operations:

```go
// Initialize the Grumpkin curve gadget
curve, err := sw_grumpkin.NewCurve(api)

// Now we can do elliptic curve operations as single function calls
result, err := curve.MultiScalarMul(points, scalars)
curve.AssertIsEqual(point1, point2)
```

Behind the scenes, `MultiScalarMul` generates thousands of constraints (bit decomposition, point doublings, additions), but we interact with it as a simple function call.

### What We're Emulating

The circuit runs in BN254's scalar field (Fr), but Hyrax uses Grumpkin's scalar field for its scalars. These are different fields, so we must **emulate** Grumpkin scalar arithmetic:

```go
// Initialize emulated field for Grumpkin scalars
scalarField, err := emulated.NewField[sw_grumpkin.ScalarField](api)

// Now we can do Grumpkin scalar arithmetic (each op = many constraints)
product := scalarField.Mul(&a, &b)      // ~200 constraints
sum := scalarField.Add(&a, &b)          // ~50 constraints
scalarField.AssertIsEqual(&x, &y)       // ~50 constraints
```

### Circuit Structure

```go
type HyraxVerifierCircuit struct {
    // Grumpkin points — coordinates are native (BN254 Fr)
    RowCommitments []sw_grumpkin.G1Affine  // C_0, ..., C_{√N-1}
    Generators     []sw_grumpkin.G1Affine  // SRS: G_0, ..., G_{√N-1}

    // Grumpkin scalars — must be emulated (Grumpkin Fr ≠ BN254 Fr)
    L []sw_grumpkin.Scalar  // eq vector for left half of z
    U []sw_grumpkin.Scalar  // prover's claimed projection

    // Native BN254 Fr values
    R []frontend.Variable   // eq vector for right half of z
    V frontend.Variable     // claimed evaluation f(z)
}
```

**Key insight**: Points use `G1Affine` (native coordinates), scalars use `Scalar` (emulated). This split is why MSMs are efficient but the dot product is expensive.

### The Define Function

The `Define` method implements the verification logic:

```go
func (c *HyraxVerifierCircuit) Define(api frontend.API) error {
    // 1. Initialize gadgets
    curve, _ := sw_grumpkin.NewCurve(api)
    scalarField, _ := emulated.NewField[sw_grumpkin.ScalarField](api)

    // 2. MSM #1: C' = Σ L[a] · RowCommitments[a]
    Cprime, _ := curve.MultiScalarMul(rowPtrs, lScalars)

    // 3. MSM #2: Com(u) = Σ u[j] · Generators[j]
    ComU, _ := curve.MultiScalarMul(genPtrs, uScalars)

    // 4. Check 1: Com(u) == C' (point equality, native)
    curve.AssertIsEqual(ComU, Cprime)

    // 5. Check 2: ⟨u, R⟩ == v (dot product, emulated)
    dotProduct := scalarField.Zero()
    for i := range c.U {
        rBits := api.ToBinary(c.R[i], 254)      // native → bits
        rEmulated := scalarField.FromBits(rBits...) // bits → emulated
        term := scalarField.Mul(&c.U[i], rEmulated)
        dotProduct = scalarField.Add(dotProduct, term)
    }
    vEmulated := scalarField.FromBits(api.ToBinary(c.V, 254)...)
    scalarField.AssertIsEqual(dotProduct, vEmulated)

    return nil
}
```

### Go Boilerplate: Pointer Conversion

The actual code has extra lines converting slices to pointer slices:

```go
// Our circuit has slices of values:
c.RowCommitments = []G1Affine{pointA, pointB, pointC}

// But MultiScalarMul wants slices of pointers:
func MultiScalarMul(points []*G1Affine, ...)

// So we convert — just creating a list of "where each item lives":
rowPtrs := make([]*G1Affine, len(c.RowCommitments))
for i := range c.RowCommitments {
    rowPtrs[i] = &c.RowCommitments[i]  // & = "address of"
}
```

This is purely a Go API requirement — gnark's function signature demands pointers. It has no effect on the circuit logic or constraint count. Think of it as: instead of handing over copies of all the points, we hand over a list of "go look at room 101, room 102, ..." addresses.

## The Two Fields and Why It Matters

When implementing Hyrax verification in a BN254 Groth16 circuit with Grumpkin MSMs, we work with two distinct but similar fields:

| Field | Modulus | In Circuit | Usage |
|-------|---------|------------|-------|
| BN254 Fr | `21888...495617` | **Native** | Circuit constraints, Grumpkin point coordinates |
| Grumpkin Fr | `21888...208583` | **Emulated** | MSM scalars, dot product operands |

Both are ~254-bit primes but **not identical** (difference ≈ 2^191). This field mismatch is why Check 1 and Check 2 have fundamentally different costs.

### Why Check 1 (MSMs) Benefits from the 2-Cycle

The MSMs compute `C' = Σ L[a] · C[a]` and `Com(u) = Σ u[j] · G[j]`:

```
MSM = Σ scalar[i] × Point[i]  →  Result is a POINT
```

Even though the **scalars** are in the non-native Grumpkin scalar field, the **point operations** happen on Grumpkin point coordinates, which are in Grumpkin's base field = BN254's scalar field = **native**.

In each scalar multiplication `k · P` (double-and-add algorithm):

```
k = 254-bit scalar (in Grumpkin Fr, non-native)
P = point (coordinates in Grumpkin base field = BN254 Fr, native)

1. Decompose k into 254 bits: k = Σ bᵢ · 2ⁱ
2. For each bit (MSB to LSB):
   - DOUBLE: result = 2 · result     (~5 constraints, native point op)
   - If bit=1: ADD: result = result + P  (~5 constraints, native point op)
```

**Why ~5 constraints per point operation?** For Grumpkin (`y² = x³ - 17`), point addition uses:

```
λ = (y₂ - y₁) / (x₂ - x₁)      // slope
x₃ = λ² - x₁ - x₂              // result x
y₃ = λ(x₁ - x₃) - y₁           // result y

Circuit constraints:
1. λ · (x₂ - x₁) = y₂ - y₁     // slope (rewritten to avoid division)
2. x₃ = λ² - x₁ - x₂           // x-coordinate
3. y₃ = λ · (x₁ - x₃) - y₁     // y-coordinate
4-5. Auxiliary for λ² and conditional selection
```

Each is a **single native constraint** because coordinates are in BN254 Fr.

**Constraint breakdown for one scalar mul (~1,776 total):**
- Scalar bit decomposition: ~254 constraints (proving each bᵢ ∈ {0,1})
- ~254 point doublings: ~254 × 5 = ~1,270 constraints
- ~127 point additions (half the bits are 1 on average): ~127 × 5 = ~635 constraints
- Conditional selection logic: ~100 constraints

The native point operations dominate (~1,900 constraints), but they're only ~5 each instead of ~1,000 each (non-native). This is why MSMs benefit massively from the 2-cycle.

### Why Check 2 (Dot Product) Cannot Benefit

The dot product computes `⟨u, R⟩ = Σ u[j] · R[j]`:

```
Dot product = Σ scalar[i] × scalar[i]  →  Result is a SCALAR
```

There are **no elliptic curve points** — it's purely field arithmetic. The computation happens entirely in the Grumpkin scalar field (non-native), so every operation requires emulated arithmetic.

**What the circuit does for each term `u[j] · R[j]`:**

```
u[j] = emulated Grumpkin scalar (already 4 limbs)
R[j] = native BN254 Fr variable

1. Convert R[j] to emulated:
   - Decompose R[j] to 254 bits: ~254 constraints
   - Reconstruct as emulated element: ~50 constraints

2. Multiply u[j] × R[j] in emulated field:
   - Cross-multiply 4×4 limbs: 16 native muls
   - Combine partial products with carries: ~100 constraints
   - Modular reduction: ~100 constraints
   - Total: ~200 constraints per multiplication

3. Accumulate into running sum:
   - Emulated addition: ~50 constraints
```

**Constraint breakdown for dot product (√N = 1024 terms):**
- R[j] bit decomposition: 1024 × 254 = ~260k constraints
- R[j] → emulated conversion: 1024 × 50 = ~50k constraints
- Emulated multiplications: 1024 × 200 = ~205k constraints
- Emulated additions: 1024 × 50 = ~50k constraints
- V conversion + final equality: ~300 constraints
- **Total: ~565k constraints** (matches observed ~600k)

| Operation | What's native | What's emulated | 2-Cycle benefit? |
|-----------|---------------|-----------------|------------------|
| MSM | Point coordinates | Scalars | **Yes** — point ops dominate |
| Dot product | — | Both operands | **No** — pure scalar arithmetic |

### Emulated Field Structure

Grumpkin scalars in gnark are represented as emulated elements:
- **4 limbs** of **64 bits** each (256 bits total, with modular reduction)
- Little-endian order (least significant limb first)
- Accessed via `Element.Limbs[]`

### How FromBits Reconstructs Limbs

When converting native → bits → emulated, `FromBits` packs bits into limbs:

```
Given 254 bits: b₀, b₁, b₂, ..., b₂₅₃

Limb 0 = b₀·2⁰ + b₁·2¹ + b₂·2² + ... + b₆₃·2⁶³
Limb 1 = b₆₄·2⁰ + b₆₅·2¹ + ... + b₁₂₇·2⁶³
Limb 2 = b₁₂₈·2⁰ + b₁₂₉·2¹ + ... + b₁₉₁·2⁶³
Limb 3 = b₁₉₂·2⁰ + b₁₉₃·2¹ + ... + b₂₅₃·2⁶¹
```

The `2ⁱ` values are **constants** (precomputed), not circuit variables. So each limb is just a **linear combination**:

```
limb0 = Σᵢ bᵢ · 2ⁱ  (for i = 0..63)
```

Linear combinations are cheap — gnark handles them as a single R1CS constraint (or even collapses them for free in some cases).

### How Emulated Add/Mul Handle Carries

**Emulated Add — No carries at all!**

```
a = [a₀, a₁, a₂, a₃]   (each limb ≤ 2⁶⁴ - 1)
b = [b₀, b₁, b₂, b₃]

Add(a, b) = [a₀+b₀, a₁+b₁, a₂+b₂, a₃+b₃]
```

The limbs just get bigger. This works because:
- BN254's native field is ~254 bits
- Each sum aᵢ + bᵢ fits in ~65 bits (still << 254)
- We can do **many adds** before risking overflow

gnark tracks a "limb overflow count" internally and only forces reduction when necessary.

**Emulated Mul — Schoolbook + delayed reduction**

For `Mul(a, b)`, gnark uses schoolbook multiplication:

```
        a₃  a₂  a₁  a₀
     ×  b₃  b₂  b₁  b₀
     ─────────────────
        a₀b₀                    → partial for limb 0
        a₀b₁ + a₁b₀             → partial for limb 1
        a₀b₂ + a₁b₁ + a₂b₀      → partial for limb 2
        ... and so on (7 partial products total)
```

This produces a **wide result** (7 limbs for 4×4). Then **modular reduction** happens:

1. Prover computes reduced value outside circuit (using a "hint")
2. Circuit verifies: `wide_result = quotient × modulus + reduced_result`
3. Range checks ensure limbs are properly bounded (< 2⁶⁴)

The range checks are expensive (~50+ constraints each), which is why emulated mul costs ~200 constraints total.

**Key insight**: Carries don't propagate limb-by-limb like hardware addition. Instead, gnark accumulates into wide limbs and does one batched reduction with range checks at the end. This "lazy carry" approach minimizes constraint count.

## Dot Product Implementation Options

The Hyrax protocol requires computing `⟨u, R⟩ == v` where:
- `U` elements are `sw_grumpkin.Scalar` (emulated Grumpkin Fr)
- `R` elements are `frontend.Variable` (native BN254 Fr)
- `V` is the claimed result (native BN254 Fr)

#### Option 1: First Limb Only (Low Precision)

```go
dotProduct := frontend.Variable(0)
for i := range c.U {
    uReduced := scalarField.Reduce(&c.U[i])
    term := api.Mul(uReduced.Limbs[0], c.R[i])
    dotProduct = api.Add(dotProduct, term)
}
api.AssertIsEqual(dotProduct, c.V)
```

**Pros:**
- Minimal constraint overhead (~2.96M for √N=1024)
- Simple implementation

**Cons:**
- Only correct for values < 2^64
- Not suitable for production with arbitrary field elements

#### Option 2: Full Precision via Bit Decomposition (Current Implementation)

```go
dotProduct := scalarField.Zero()
for i := range c.U {
    rBits := api.ToBinary(c.R[i], 254)
    rEmulated := scalarField.FromBits(rBits...)
    term := scalarField.Mul(&c.U[i], rEmulated)
    dotProduct = scalarField.Add(dotProduct, term)
}
vBits := api.ToBinary(c.V, 254)
vEmulated := scalarField.FromBits(vBits...)
scalarField.AssertIsEqual(dotProduct, vEmulated)
```

**Pros:**
- Handles full 254-bit field elements correctly
- No assumptions about value sizes

**Cons:**
- ~20% more constraints (~3.55M for √N=1024)
- Adds bit decomposition overhead per R element (254 constraints each)

#### Option 3: Native Field with Range Assumptions (Not Implemented)

If we can guarantee that U values fit in native field (< BN254 Fr), we could:
1. Reconstruct U from all 4 limbs into native
2. Compute dot product natively

This would require:
- Proving U limbs are properly bounded
- Careful handling of overflow in accumulation

### Current Choice

We use **Option 2 (Full Precision)** because:
1. Correctness is paramount for cryptographic verification
2. The ~600k constraint overhead is acceptable (~3.55M total, still << 50M practical limit)
3. No assumptions needed about input value ranges

**Constraint breakdown for √N = 1024:**
- MSMs (Check 1): ~2.95M — benefits from native point arithmetic
- Dot product (Check 2): ~600k — unavoidable emulated overhead
- **Total: ~3.55M**

## Benchmark Results

### Full Precision Implementation (Current)

| √N | N | N (power of 2) | Constraints | Per Scalar Mul |
|----|---|----------------|-------------|----------------|
| 4 | 16 | 2⁴ | 15,163 | ~1,895 |
| 16 | 256 | 2⁸ | 58,295 | ~1,821 |
| 64 | 4,096 | 2¹² | 228,910 | ~1,788 |
| 256 | 65,536 | 2¹⁶ | 901,533 | ~1,760 |
| **1024** | **1,048,576** | **2²⁰** | **3,545,963** | **~1,731** |

### Comparison with Low Precision (First Limb Only)

| √N | Low Precision | Full Precision | Increase |
|----|---------------|----------------|----------|
| 1024 | 2,955,162 | 3,545,963 | +20% |

### Primitive Costs

| Operation | Constraints |
|-----------|-------------|
| Point addition | 5 |
| Single scalar mul | 1,776 |
| Bit decomposition (254 bits) | ~254 |
| Emulated field mul | ~500-1000 |

## Field Usage in Jolt (Detailed Analysis)

Investigation of the actual Jolt codebase (jolt-core) reveals the following field structure:

### Jolt's Field Hierarchy

| Type | Field | Bits | Purpose |
|------|-------|------|---------|
| Polynomial coefficients | `ark_bn254::Fr` | ~254 | Main computation field |
| Challenge values (L, R) | `MontU128Challenge<Fr>` | 125 | Compressed challenges for performance |
| Dot product result | `ark_bn254::Fr` | ~254 | Result of `⟨u, R⟩` |

**Key insight**: Jolt uses a **compressed challenge field** (`MontU128Challenge`) for performance - challenges are stored as 125-bit values but automatically convert to full `Fr` when needed for arithmetic.

### How This Affects Our Implementation

1. **MSM scalars**: In Jolt, L and U vectors contain challenge values that get converted to full field elements before MSM. Our use of emulated Grumpkin scalars is correct.

2. **Dot product field**: Jolt computes `⟨u, R⟩` in BN254 Fr after converting challenges. Our implementation converts native BN254 Fr to emulated Grumpkin Fr via bit decomposition - this is conservative but correct.

3. **Why the bit decomposition works**: Both BN254 Fr and Grumpkin Fr are ~254-bit primes. Values produced by Jolt's challenge arithmetic are bounded by min(Fr, Fq), so the conversion is lossless.

### Protocol Alignment

Our implementation matches doc 09's Hyrax verification protocol:
- **MSM #1**: C' = Σ L[a] · RowCommitments[a] ✓
- **MSM #2**: Com(u) = Σ u[j] · Generators[j] ✓
- **Check 1**: Com(u) == C' ✓
- **Check 2**: ⟨u, R⟩ == v ✓

The test data construction (matrix F → row commitments → u = L^T · F) correctly models the Hyrax commitment structure.

## gnark Algebra Options

The `algopts` package provides these options for MSM/scalar mul:

| Option | Description | Applicability |
|--------|-------------|---------------|
| `WithNbScalarBits(n)` | Reduce scalar bits if known small | Jolt challenges are 125-bit, could save ~50% |
| `WithFoldingScalarMul()` | For scalars that are powers (1, s, s², ...) | Not applicable to our case |
| `WithCompleteArithmetic()` | Safe addition formulas | Needed if points might be equal |

### Fixed-Base MSM Consideration

MSM #2 uses SRS generators which are **fixed at setup time**. gnark doesn't have explicit precomputation support for this, but:

1. `ScalarMulBase` exists for generator multiplication (single point)
2. For MSM, no special fixed-base path exists in `sw_grumpkin`
3. Potential future optimization: implement precomputed table lookup for fixed generators

### Potential Constraint Savings

If Jolt challenge values are guaranteed ≤ 125 bits:
- Use `WithNbScalarBits(125)` to reduce scalar mul costs
- Estimated saving: ~45% on scalar decomposition (~800 constraints/mul)
- Total potential: ~1.5M fewer constraints for √N=1024

**Not implemented yet** - requires verification that Jolt's `MontU128Challenge` bounds are guaranteed.

## Future Considerations

### Potential Optimizations

1. **Scalar bit width**: Use `WithNbScalarBits(125)` if challenge bounds confirmed
2. **Lazy Reduction**: Accumulate in unreduced form, reduce once at end
3. **Fixed-base precomputation**: Custom MSM for SRS generators
4. **Native dot product**: If U values are guaranteed in both fields' intersection
5. **Reuse U bit decomposition**: gnark's `MultiScalarMul` internally decomposes U scalars to bits for the MSM, but doesn't expose them. A custom implementation could decompose U once and reuse the bits for both the MSM and dot product, potentially saving ~254 × √N constraints

### Validation with Real Data

Current tests use synthetic data. Future work should:
1. Export real Hyrax proofs from Jolt (Rust)
2. Serialize row commitments, U, L, R, V to JSON
3. Load in Go tests to verify against known-good values
4. Confirm challenge value bounds for optimization

### Integration Path

When integrating with transpiled sumcheck stages:
1. Transcript handling: L and R come from Fiat-Shamir challenges
2. Data flow: Sumcheck produces evaluation point z → split into (z_L, z_R) → compute L, R eq vectors
3. Public input layout: Row commitments, generators as public inputs to circuit
