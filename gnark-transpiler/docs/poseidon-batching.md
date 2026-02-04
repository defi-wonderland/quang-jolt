# Poseidon Batching Optimization

This document describes the Poseidon batching optimization implemented in the gnark-transpiler to reduce the number of Poseidon hash operations in the generated gnark circuits.

## Current Status

**Stages Implemented**: 1-5 (Stage 6+ commented out in `transpilable_verifier.rs`)

**Current Configuration**: `POSEIDON_BATCH_SIZE = 1` (width-3, baseline - no batching active)

**Baseline Benchmarks (n=10 fibonacci, Stages 1-5):**
| Metric | Value |
|--------|-------|
| Assertions | 13 |
| R1CS Constraints | 1,531,516 |
| Public Variables | 1,211 |
| Prove Time | ~3.8s |
| Verify Time | ~1.7ms |
| Proof Size | 164 bytes |

**Stage 1 Only Benchmarks (n=10 fibonacci):**
| Metric | Width-3 |
|--------|---------|
| R1CS Constraints | 158,006 |
| Prove Time | 466ms |
| Verify Time | 1.3ms |

The batching infrastructure is fully implemented in Rust. **Width-5 batching requires fixing the Go Poseidon5 implementation** - the current Go code produces different hash outputs than the Rust `light-poseidon` library.

## Known Issue: Width-5 Go Implementation

The Go `Width5Chip` Poseidon implementation produces different outputs than `light-poseidon`. This is likely due to:

1. **Different algorithm structure**: The circom-compatible `light-poseidon` uses an unoptimized algorithm (full MDS in partial rounds), but the round constant layout and application order may differ from the Go implementation.

2. **Round constant indexing**: The Go implementation needs to match exactly how `light-poseidon` indexes into the round constant array during full and partial rounds.

**To fix**: Compare step-by-step intermediate values between Rust and Go implementations, or port the exact algorithm from `light-poseidon`'s source code.

## Overview

The Jolt verifier uses a Poseidon-based transcript (Fiat-Shamir) that absorbs field elements and derives challenges. Each absorption originally required one Poseidon permutation. By batching multiple field elements into a single permutation using wider Poseidon variants, we can significantly reduce the total number of hash operations.

## Background: Poseidon Width

Poseidon is a sponge-based hash function parameterized by:
- **Width (t)**: The state size (number of field elements)
- **Rate (r)**: The number of elements absorbed per permutation (typically t-1 or t-2)
- **Capacity (c)**: Security parameter (c = t - r)

The `light-poseidon` library supports widths 1-12 with pre-audited circom-compatible parameters for BN254.

## Batching Strategy

| Configuration | Width | Inputs | Data Elements | Hash Reduction |
|---------------|-------|--------|---------------|----------------|
| Current (baseline) | 3 | state, n_rounds, d1 | 1 | - |
| Batch-2 | 4 | state, n_rounds, d1, d2 | 2 | ~50% |
| Batch-3 | 5 | state, n_rounds, d1, d2, d3 | 3 | ~67% |

**Domain separation**: The `n_rounds` counter is always included to ensure unique hashes across rounds.

## Implementation

### Rust Side (gnark-transpiler/src/poseidon.rs)

The `PoseidonAstTranscript` buffers incoming field elements and flushes when:
1. The buffer reaches capacity (`POSEIDON_BATCH_SIZE`)
2. A challenge is requested (must flush before deriving)

```rust
pub const POSEIDON_BATCH_SIZE: usize = 1; // Set to 2 or 3 for batching

fn flush_buffer(&mut self) {
    if self.buffer.is_empty() { return; }

    // Pad with zeros if partial batch
    while self.buffer.len() < POSEIDON_BATCH_SIZE {
        self.buffer.push(zero.clone());
    }

    // Create appropriate Poseidon node based on batch size
    self.state = match POSEIDON_BATCH_SIZE {
        1 => MleAst::poseidon(&self.state, &round, &self.buffer[0]),
        2 => MleAst::poseidon4(&self.state, &round, &self.buffer[0], &self.buffer[1]),
        3 => MleAst::poseidon5(&self.state, &round, &self.buffer[0], &self.buffer[1], &self.buffer[2]),
        _ => panic!("Unsupported batch size"),
    };
    self.buffer.clear();
    self.n_rounds += 1;
}
```

### AST Nodes (zklean-extractor/src/mle_ast.rs)

Three Poseidon node variants support different widths:

```rust
pub enum Node {
    Poseidon(Edge, Edge, Edge),           // Width-3: state, n_rounds, d1
    Poseidon4(Edge, Edge, Edge, Edge),    // Width-4: state, n_rounds, d1, d2
    Poseidon5(Edge, Edge, Edge, Edge, Edge), // Width-5: state, n_rounds, d1, d2, d3
}
```

### Go Side (gnark-transpiler/go/poseidon/)

Three hash functions correspond to the AST nodes:

```go
// Width-3: Original implementation
func Hash(api frontend.API, in1, in2, in3 frontend.Variable) frontend.Variable

// Width-4: Batch-2
func Hash4(api frontend.API, in1, in2, in3, in4 frontend.Variable) frontend.Variable

// Width-5: Batch-3
func Hash5(api frontend.API, in1, in2, in3, in4, in5 frontend.Variable) frontend.Variable
```

Each uses the appropriate Poseidon chip with matching constants:
- `Chip` (width-4 with zero padding) for `Hash`
- `Width4Chip` for `Hash4` (reuses existing constants)
- `Width5Chip` for `Hash5` (uses constants5.go)

### Poseidon Constants

Width-5 constants are extracted from `light-poseidon` using `extract_poseidon_constants.rs`:

| Parameter | Width-5 Value |
|-----------|---------------|
| Full rounds | 8 |
| Partial rounds | 60 |
| Round constants | 340 elements |
| MDS matrix | 5x5 |

The constants are stored in `go/poseidon/constants5.go`.

## Code Generation

The code generator (gnark-transpiler/src/codegen.rs) maps AST nodes to Go function calls:

```rust
Node::Poseidon(s, n, d1) =>
    format!("poseidon.Hash(api, {}, {}, {})", ...)

Node::Poseidon4(s, n, d1, d2) =>
    format!("poseidon.Hash4(api, {}, {}, {}, {})", ...)

Node::Poseidon5(s, n, d1, d2, d3) =>
    format!("poseidon.Hash5(api, {}, {}, {}, {}, {})", ...)
```

## Configuration

To enable batching:

1. **Rust side**: In `gnark-transpiler/src/poseidon.rs`, set:
   ```rust
   pub const POSEIDON_BATCH_SIZE: usize = 3; // or 2
   ```

2. **jolt-core side**: In `jolt-core/src/transcripts/poseidon.rs`, set the matching batch size to ensure the AST matches the actual proof computation.

**Important**: Both sides MUST use the same batch size for correctness.

## Files Modified

| File | Purpose |
|------|---------|
| `gnark-transpiler/src/poseidon.rs` | AST transcript with buffering |
| `gnark-transpiler/src/codegen.rs` | Code generation for Poseidon4/5 |
| `gnark-transpiler/src/ast_json.rs` | JSON serialization for Poseidon4/5 |
| `gnark-transpiler/src/bin/transpile_stages.rs` | Analysis functions |
| `gnark-transpiler/go/poseidon/poseidon.go` | Hash4, Hash5, Width5Chip |
| `gnark-transpiler/go/poseidon/constants5.go` | Width-5 Poseidon constants |
| `gnark-transpiler/src/bin/extract_poseidon_constants.rs` | Constant extraction tool |

## Expected Performance

For a typical Jolt proof with ~5,924 Poseidon hashes (Stages 1-5):

| Batch Size | Estimated Hashes | Reduction |
|------------|------------------|-----------|
| 1 (current) | 5,924 | baseline |
| 2 | ~2,962 | ~50% |
| 3 | ~1,975 | ~67% |

Constraint reduction depends on the relative cost of Poseidon permutations vs other operations.

## Security Considerations

- **Domain separation**: The `n_rounds` counter ensures each hash invocation is unique
- **Padding**: Partial batches are zero-padded, which is safe as long as padding is consistent
- **Audited parameters**: All Poseidon parameters come from `light-poseidon`'s circom-compatible presets, which are widely audited

## Future Work

- Benchmark actual constraint reduction with batch sizes 2 and 3
- Consider adaptive batching based on proof structure
- Evaluate width-6+ for even more aggressive batching
