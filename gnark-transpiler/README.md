# Gnark Transpiler

Transpiles the Jolt zkVM verifier (Stages 1-7) into Gnark circuits for Groth16 proving.

## Status

**All sumcheck stages (1-7) working** - 3.1M constraints, 164 byte proofs

## Quick Start

### 1. Generate Proof with Poseidon Transcript

```bash
cd /Users/home/dev/parti/cryptography/zkVMs/quangvdao/quang-jolt
cargo run -p fibonacci --release --features transcript-poseidon -- --save
```

Output files:
- `/tmp/fib_proof.bin` - Serialized Jolt proof
- `/tmp/fib_io_device.bin` - I/O device state
- `/tmp/jolt_verifier_preprocessing.dat` - Verifier preprocessing

### 2. Run Transpiler

```bash
cargo run -p gnark-transpiler --release --bin gnark-transpiler
```

Output files:
- `go/stages16_circuit.go` - Generated Gnark circuit
- `go/stages16_witness.json` - Witness values

### 3. Run Groth16 Prove/Verify

```bash
cd go
go test -v -run TestStages16CircuitProveVerify -timeout 15m
```

Expected output:
```
=== RUN   TestStages16CircuitProveVerify
    Compiled: 3112261 constraints, 1412 public inputs
    Proof generated: 164 bytes [7.77s]
    Verified [1.85ms]
    ✓ Stages 1-7 circuit verification passed!
--- PASS: TestStages16CircuitProveVerify (102.92s)
```

## Other Tests

```bash
# Quick solver test (no proving key generation)
go test -v -run TestStages16CircuitSolver

# Sanity checks (witness corruption rejection)
go test -v -run "TestCorruptedWitnessRejected|TestRandomFuzzing"
```

## Metrics

| Metric | Value |
|--------|-------|
| Constraints | 3,112,261 |
| Assertions | 17 |
| Proof size | 164 bytes |
| Prove time | ~8s |
| Verify time | ~2ms |

## Architecture

```
src/
├── main.rs                    # Transpiler entry point
├── codegen.rs                 # AST to Go code generation
├── mle_opening_accumulator.rs # Symbolic opening accumulator
├── poseidon.rs                # Poseidon transcript for MleAst
├── symbolic_proof.rs          # Proof symbolization
└── witness.rs                 # Witness generation

go/
├── stages16_circuit.go        # Generated circuit (gitignored)
├── stages16_witness.json      # Generated witness (gitignored)
├── stages16_circuit_test.go   # Groth16 tests
├── helpers.go                 # Circuit helper functions
└── poseidon/                  # Poseidon hash implementation
```

## Stage 8 (Hyrax PCS)

Stage 8 cannot be transpiled - it requires native Gnark pairing operations for Hyrax polynomial commitment verification. This will be implemented directly in Go.
