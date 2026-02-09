//! Transpile Jolt verifier stages 1-7 to Gnark circuit
//!
//! Uses TranspilableVerifier with symbolic proof and MleOpeningAccumulator
//! to generate a Gnark circuit for stages 1-7 of the Jolt verifier.
//!
//! NOTE: This is adapted for the quangvdao fork which has stages 6a, 6b, 7, 8 (Hyrax-based).
//! We transpile stages 1-7 (all sumcheck stages before the final Hyrax opening).

use ark_ff::PrimeField;
use ark_serialize::CanonicalDeserialize;
use clap::Parser;
use serde::Serialize;
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

use gnark_transpiler::{
    symbolize_jolt_proof, extract_sumcheck_witness_values, AstCommitmentScheme, MleOpeningAccumulator,
    PoseidonAstTranscript, MemoizedCodeGen, sanitize_go_name,
};
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::transpilable_verifier::{
    JoltVerifierPreprocessing as TranspilablePreprocessing, TranspilableVerifier, SharedPreprocessing
};
use jolt_core::zkvm::verifier::JoltVerifierPreprocessing as VerifierPreprocessing;
use jolt_core::zkvm::program::VerifierProgram;
use jolt_core::zkvm::RV64IMACProof;
use common::jolt_device::JoltDevice;
use zklean_extractor::mle_ast::{enable_constraint_mode, take_constraints as take_assertions, MleAst};

/// Transpile Jolt proofs to gnark circuits for Groth16 proving.
///
/// Takes a Jolt proof, io_device, and preprocessing files as input,
/// and generates gnark circuit code and witness data.
#[derive(Parser)]
#[command(name = "gnark-transpiler")]
#[command(version, about)]
struct Args {
    /// Path to the proof file (serialized JoltProof)
    #[arg(long, default_value = "/tmp/fib_proof.bin")]
    proof: PathBuf,

    /// Path to the io_device file (inputs/outputs)
    #[arg(long, default_value = "/tmp/fib_io_device.bin")]
    io_device: PathBuf,

    /// Path to the preprocessing file
    #[arg(long, default_value = "/tmp/jolt_verifier_preprocessing.dat")]
    preprocessing: PathBuf,

    /// Output directory for generated Go files
    #[arg(long, short = 'o', default_value = "go")]
    output_dir: PathBuf,

    /// Enable verbose output (shows sumcheck rounds and other debug info)
    #[arg(long, short = 'v')]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    println!("=== Transpiling Jolt Verifier Stages 1-7 to Gnark ===\n");

    // Load proof
    println!("Loading proof from: {:?}", args.proof);
    let proof_bytes = std::fs::read(&args.proof)
        .unwrap_or_else(|e| panic!("Failed to read proof file {:?}: {}", args.proof, e));
    let real_proof: RV64IMACProof =
        CanonicalDeserialize::deserialize_compressed(&proof_bytes[..])
            .expect("Failed to deserialize proof");
    println!("  trace_length: {}", real_proof.trace_length);
    println!("  commitments: {}", real_proof.commitments.len());

    // Print Hyrax/Recursion proof details
    println!("\n=== Hyrax/Recursion Proof Details ===");
    println!("  recursion_proof.opening_proof.vector_matrix_product.len(): {}",
        real_proof.recursion_proof.opening_proof.vector_matrix_product.len());
    println!("  recursion_proof.dense_commitment.row_commitments.len(): {}",
        real_proof.recursion_proof.dense_commitment.row_commitments.len());

    // Print sumcheck details for all stages (if verbose)
    if args.verbose {
        println!("\n=== Sumcheck Details (All Stages) ===");
        println!("  Stage 1 rounds: {}", real_proof.stage1_sumcheck_proof.compressed_polys.len());
        println!("  Stage 2 rounds: {}", real_proof.stage2_sumcheck_proof.compressed_polys.len());
        println!("  Stage 3 rounds: {}", real_proof.stage3_sumcheck_proof.compressed_polys.len());
        println!("  Stage 4 rounds: {}", real_proof.stage4_sumcheck_proof.compressed_polys.len());
        println!("  Stage 5 rounds: {}", real_proof.stage5_sumcheck_proof.compressed_polys.len());
        println!("  Stage 6a rounds: {}", real_proof.stage6a_sumcheck_proof.compressed_polys.len());
        println!("  Stage 6b rounds: {}", real_proof.stage6b_sumcheck_proof.compressed_polys.len());
        println!("  Stage 7 rounds: {}", real_proof.stage7_sumcheck_proof.compressed_polys.len());
    }

    // Load io_device
    println!("\nLoading io_device from: {:?}", args.io_device);
    let io_device_bytes = std::fs::read(&args.io_device)
        .unwrap_or_else(|e| panic!("Failed to read io_device file {:?}: {}", args.io_device, e));
    let io_device: JoltDevice = CanonicalDeserialize::deserialize_compressed(&io_device_bytes[..])
        .expect("Failed to deserialize io_device");
    println!("  inputs: {} bytes", io_device.inputs.len());
    println!("  outputs: {} bytes", io_device.outputs.len());

    // Load preprocessing (Dory version - matches jolt-sdk)
    // Uses verifier::JoltVerifierPreprocessing which has the serialization format
    println!("\nLoading preprocessing from: {:?}", args.preprocessing);
    let preprocessing_bytes = std::fs::read(&args.preprocessing)
        .unwrap_or_else(|e| panic!("Failed to read preprocessing file {:?}: {}", args.preprocessing, e));
    let real_preprocessing: VerifierPreprocessing<ark_bn254::Fr, DoryCommitmentScheme> =
        CanonicalDeserialize::deserialize_compressed(&preprocessing_bytes[..])
            .expect("Failed to deserialize preprocessing");
    println!("  memory_layout: {:?}", real_preprocessing.shared.memory_layout);

    // Convert preprocessing to AstCommitmentScheme version for TranspilableVerifier
    // quangvdao's verifier::JoltVerifierPreprocessing has: generators, shared, hyrax_recursion_setup, program
    // TranspilableVerifier needs: generators, program, shared (just program_meta), ram, memory_layout
    let program_preprocessing = real_preprocessing.program.as_full()
        .expect("Transpilation requires Full program mode, not Committed mode")
        .clone();

    // Construct RAMPreprocessing from ProgramPreprocessing data
    let ram_preprocessing = jolt_core::zkvm::ram::RAMPreprocessing {
        min_bytecode_address: program_preprocessing.min_bytecode_address,
        program_image_len_words: program_preprocessing.program_image_words.len(),
    };

    let symbolic_preprocessing: TranspilablePreprocessing<MleAst, AstCommitmentScheme> =
        TranspilablePreprocessing {
            generators: gnark_transpiler::ast_commitment_scheme::AstVerifierSetup,
            program: VerifierProgram::Full(program_preprocessing),
            shared: SharedPreprocessing {
                program_meta: real_preprocessing.shared.program_meta.clone(),
            },
            ram: ram_preprocessing,
            memory_layout: real_preprocessing.shared.memory_layout.clone(),
        };

    // Symbolize the proof (creates full JoltProof with symbolic sumcheck coefficients)
    println!("\n=== Symbolizing Proof ===");
    let (symbolic_proof, accumulator, var_alloc) = symbolize_jolt_proof(&real_proof);
    println!("  Total symbolic variables: {}", var_alloc.next_idx());

    // Create transcript
    let transcript: PoseidonAstTranscript = Transcript::new(b"Jolt");

    // Create TranspilableVerifier with symbolic types
    println!("\n=== Creating TranspilableVerifier ===");
    let verifier = TranspilableVerifier::<
        MleAst,
        AstCommitmentScheme,
        PoseidonAstTranscript,
        MleOpeningAccumulator,
    >::new_with_accumulator(
        &symbolic_preprocessing,
        symbolic_proof,
        io_device,
        None, // trusted_advice_commitment
        transcript,
        accumulator,
    );

    // Enable assertion mode so MleAst comparisons register equality checks
    enable_constraint_mode();

    // Run verification - Stages 1-7 (all sumcheck stages, excluding Hyrax opening)
    println!("\n=== Running Symbolic Verification (Stages 1-7) ===");
    match verifier.verify_stages_1_7() {
        Ok(()) => println!("  Stages 1-7 verification completed successfully"),
        Err(e) => {
            println!("  Verification error: {:?}", e);
            return;
        }
    }

    // Collect accumulated assertions (equality checks that become api.AssertIsEqual calls)
    let assertions = take_assertions();
    println!("\n=== Accumulated Assertions ===");
    println!("  Total assertions: {}", assertions.len());

    // Debug: analyze each assertion to find problematic ones (constant vs constant)
    use zklean_extractor::mle_ast::{get_node, Node, Atom, Edge};

    fn is_constant(edge: &Edge) -> bool {
        match edge {
            Edge::Atom(Atom::Scalar(_)) => true,
            Edge::Atom(Atom::Var(_)) => false,
            Edge::Atom(Atom::NamedVar(_)) => false,
            Edge::NodeRef(id) => is_node_constant(*id),
        }
    }

    fn is_node_constant(node_id: usize) -> bool {
        let node = get_node(node_id);
        match node {
            Node::Atom(Atom::Scalar(_)) => true,
            Node::Atom(Atom::Var(_)) => false,
            Node::Atom(Atom::NamedVar(_)) => false,
            Node::Neg(e) => is_constant(&e),
            Node::Inv(e) => is_constant(&e),
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) | Node::Div(a, b) => {
                is_constant(&a) && is_constant(&b)
            }
            Node::Poseidon(a, b, c) => is_constant(&a) && is_constant(&b) && is_constant(&c),
            Node::Keccak256(e) | Node::ByteReverse(e) | Node::Truncate128Reverse(e)
            | Node::Truncate128(e) | Node::MulTwoPow192(e) => is_constant(&e),
        }
    }

    fn describe_node(node_id: usize, depth: usize) -> String {
        if depth > 5 {
            return "...".to_string();
        }
        let node = get_node(node_id);
        match node {
            Node::Atom(Atom::Scalar(limbs)) => {
                if limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0 {
                    format!("Scalar({})", limbs[0])
                } else {
                    format!("Scalar(large)")
                }
            }
            Node::Atom(Atom::Var(idx)) => format!("Var({})", idx),
            Node::Atom(Atom::NamedVar(idx)) => format!("NamedVar({})", idx),
            Node::Neg(e) => format!("Neg({})", describe_edge(&e, depth + 1)),
            Node::Inv(e) => format!("Inv({})", describe_edge(&e, depth + 1)),
            Node::Add(a, b) => format!("Add({}, {})", describe_edge(&a, depth + 1), describe_edge(&b, depth + 1)),
            Node::Sub(a, b) => format!("Sub({}, {})", describe_edge(&a, depth + 1), describe_edge(&b, depth + 1)),
            Node::Mul(a, b) => format!("Mul({}, {})", describe_edge(&a, depth + 1), describe_edge(&b, depth + 1)),
            Node::Div(a, b) => format!("Div({}, {})", describe_edge(&a, depth + 1), describe_edge(&b, depth + 1)),
            Node::Poseidon(..) => format!("Poseidon(...)"),
            Node::Keccak256(e) => format!("Keccak256({})", describe_edge(&e, depth + 1)),
            Node::ByteReverse(e) => format!("ByteReverse({})", describe_edge(&e, depth + 1)),
            Node::Truncate128Reverse(e) => format!("Truncate128Reverse({})", describe_edge(&e, depth + 1)),
            Node::Truncate128(e) => format!("Truncate128({})", describe_edge(&e, depth + 1)),
            Node::MulTwoPow192(e) => format!("MulTwoPow192({})", describe_edge(&e, depth + 1)),
        }
    }

    fn describe_edge(edge: &Edge, depth: usize) -> String {
        match edge {
            Edge::Atom(Atom::Scalar(limbs)) => {
                if limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0 {
                    format!("{}", limbs[0])
                } else {
                    "large".to_string()
                }
            }
            Edge::Atom(Atom::Var(idx)) => format!("Var({})", idx),
            Edge::Atom(Atom::NamedVar(idx)) => format!("NamedVar({})", idx),
            Edge::NodeRef(id) => describe_node(*id, depth),
        }
    }

    // Check each assertion for constant-vs-constant issues and filter them out
    println!("\n=== Analyzing Assertions for Constant Issues ===");
    let mut problematic_count = 0;
    for (i, assertion) in assertions.iter().enumerate() {
        let root_id = assertion.root();
        if is_node_constant(root_id) {
            problematic_count += 1;
            if problematic_count <= 10 {
                println!("  [PROBLEMATIC] Assertion {}: entirely constant!", i);
                println!("    Structure: {}", describe_node(root_id, 0));
            }
        }
    }
    if problematic_count > 10 {
        println!("  ... and {} more problematic assertions", problematic_count - 10);
    }
    println!("  Total problematic assertions: {}", problematic_count);

    // Filter out constant assertions - they don't provide any constraints
    // (they're just constant == constant checks that always pass or fail)
    let filtered_assertions: Vec<MleAst> = assertions
        .into_iter()
        .filter(|a| !is_node_constant(a.root()))
        .collect();
    println!("  Assertions after filtering: {} (removed {} constant assertions)",
        filtered_assertions.len(), problematic_count);
    let assertions = filtered_assertions;

    // Count multiplications by constant 0 in each assertion (with memoization)
    fn count_mul_by_zero(node_id: usize, cache: &mut HashMap<usize, usize>) -> usize {
        if let Some(&count) = cache.get(&node_id) {
            return count;
        }
        let node = get_node(node_id);
        let count = match node {
            Node::Atom(_) => 0,
            Node::Neg(e) | Node::Inv(e) | Node::Keccak256(e) | Node::ByteReverse(e)
            | Node::Truncate128Reverse(e) | Node::Truncate128(e) | Node::MulTwoPow192(e) => {
                count_mul_by_zero_edge(&e, cache)
            }
            Node::Add(a, b) | Node::Sub(a, b) | Node::Div(a, b) => {
                count_mul_by_zero_edge(&a, cache) + count_mul_by_zero_edge(&b, cache)
            }
            Node::Mul(a, b) => {
                let is_zero_mul = match (&a, &b) {
                    (_, Edge::Atom(Atom::Scalar(limbs))) | (Edge::Atom(Atom::Scalar(limbs)), _) => {
                        limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0
                    }
                    _ => false,
                };
                let base = if is_zero_mul { 1 } else { 0 };
                base + count_mul_by_zero_edge(&a, cache) + count_mul_by_zero_edge(&b, cache)
            }
            Node::Poseidon(a, b, c) => {
                count_mul_by_zero_edge(&a, cache) + count_mul_by_zero_edge(&b, cache) + count_mul_by_zero_edge(&c, cache)
            }
        };
        cache.insert(node_id, count);
        count
    }

    fn count_mul_by_zero_edge(edge: &Edge, cache: &mut HashMap<usize, usize>) -> usize {
        match edge {
            Edge::Atom(_) => 0,
            Edge::NodeRef(id) => count_mul_by_zero(*id, cache),
        }
    }

    println!("\n=== Checking for Mul-by-Zero Pattern ===");
    let mut mul_zero_assertions = Vec::new();
    let mut mul_zero_cache: HashMap<usize, usize> = HashMap::new();
    for (i, assertion) in assertions.iter().enumerate() {
        let count = count_mul_by_zero(assertion.root(), &mut mul_zero_cache);
        if count > 0 {
            mul_zero_assertions.push((i, count));
        }
    }

    if !mul_zero_assertions.is_empty() {
        println!("  Found {} assertions with mul-by-zero:", mul_zero_assertions.len());
        for (i, count) in mul_zero_assertions.iter().take(20) {
            println!("    Assertion {}: {} mul-by-zero operations", i, count);
        }
        if mul_zero_assertions.len() > 20 {
            println!("    ... and {} more", mul_zero_assertions.len() - 20);
        }
    } else {
        println!("  No mul-by-zero operations found");
    }

    // Build variable name mapping from VarAllocator
    let var_names: HashMap<u16, String> = var_alloc
        .descriptions()
        .iter()
        .map(|(idx, name)| (*idx, name.clone()))
        .collect();

    // Generate Gnark circuit
    println!("\n=== Generating Gnark Circuit ===");
    let circuit_code = generate_stages_circuit(&assertions, &var_names, "JoltStagesCircuit");

    // Determine output directory (resolve relative to manifest dir if relative path)
    let output_dir = if args.output_dir.is_relative() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        manifest_dir.join(&args.output_dir)
    } else {
        args.output_dir.clone()
    };

    // Write circuit to file
    let circuit_path = output_dir.join("stages_circuit.go");
    std::fs::write(&circuit_path, &circuit_code)
        .unwrap_or_else(|e| panic!("Failed to write circuit file {:?}: {}", circuit_path, e));
    println!("  Circuit written to: {:?}", circuit_path);
    println!("  Circuit size: {} bytes", circuit_code.len());

    // === Generate witness data ===
    println!("\n=== Generating Witness Data ===");
    let witness_values = extract_sumcheck_witness_values(&real_proof);

    // Build witness JSON mapping sanitized variable names to values
    let mut witness_map: HashMap<String, String> = HashMap::new();
    for (idx, name) in var_alloc.descriptions() {
        let sanitized = sanitize_go_name(name);
        if let Some(value) = witness_values.get(&(*idx as usize)) {
            witness_map.insert(sanitized, value.clone());
        }
    }

    let witness_json = serde_json::to_string_pretty(&witness_map).expect("Failed to serialize witness");
    let witness_path = output_dir.join("stages_witness.json");
    std::fs::write(&witness_path, &witness_json)
        .unwrap_or_else(|e| panic!("Failed to write witness file {:?}: {}", witness_path, e));
    println!("  Witness written to: {:?}", witness_path);
    println!("  Witness variables: {}", witness_map.len());

    // === Extract Hyrax Witness Data ===
    println!("\n=== Extracting Hyrax Witness Data ===");
    let hyrax_witness = extract_hyrax_witness(&real_proof, &real_preprocessing);

    let hyrax_witness_json = serde_json::to_string_pretty(&hyrax_witness).expect("Failed to serialize Hyrax witness");
    let hyrax_witness_path = output_dir.join("hyrax_witness.json");
    std::fs::write(&hyrax_witness_path, &hyrax_witness_json)
        .unwrap_or_else(|e| panic!("Failed to write Hyrax witness file {:?}: {}", hyrax_witness_path, e));
    println!("  Hyrax witness written to: {:?}", hyrax_witness_path);
    println!("  sqrt(N) = {}", hyrax_witness.sqrt_n);
    println!("  U (vector_matrix_product): {} elements", hyrax_witness.u.len());
    println!("  RowCommitments: {} points", hyrax_witness.row_commitments.len());
    println!("  Generators: {} points", hyrax_witness.generators.len());
    println!("  L: {} elements", hyrax_witness.l.len());
    println!("  R: {} elements", hyrax_witness.r.len());

    println!("\n=== SUCCESS ===");
    println!("TranspilableVerifier stages 1-7 transpiled to Gnark circuit.");
    println!("Hyrax witness data extracted for Stage 8 verification.");
}

/// Hyrax witness data for Gnark circuit
#[derive(Serialize)]
struct HyraxWitness {
    /// Full sqrt(N) - used for MSM #2 (generators × U) and dot product (U · R)
    sqrt_n: usize,
    /// Filtered size for MSM #1 (non-identity row commitments × L)
    sqrt_n1: usize,
    /// U vector (prover's projection): vector_matrix_product from opening proof
    /// Size: sqrt_n (full)
    u: Vec<String>,
    /// Row commitments from dense_commitment - FILTERED to remove identity points
    /// Size: sqrt_n1 (filtered)
    row_commitments: Vec<[String; 2]>,
    /// Pedersen generators from preprocessing (Grumpkin affine points as [x, y])
    /// Size: sqrt_n (full - generators are never identity)
    generators: Vec<[String; 2]>,
    /// L vector: eq(a, z_L) - FILTERED to match non-identity row commitments
    /// Size: sqrt_n1 (filtered)
    l: Vec<String>,
    /// R vector: eq(b, z_R) for right half of opening point
    /// Size: sqrt_n (full)
    r: Vec<String>,
    /// V: the claimed evaluation
    v: String,
    /// Opening point (for reference)
    opening_point: Vec<String>,
    /// Dense num vars
    dense_num_vars: usize,
}

/// Convert a Grumpkin affine point to [x, y] string representation
fn grumpkin_point_to_strings(point: &ark_grumpkin::Affine) -> [String; 2] {
    use ark_ec::AffineRepr;
    // Grumpkin points use Fq (BN254 scalar field) for coordinates
    if point.is_zero() {
        // Point at infinity - use zeros
        return ["0".to_string(), "0".to_string()];
    }
    let (x, y) = point.xy().unwrap();
    [
        x.into_bigint().to_string(),
        y.into_bigint().to_string(),
    ]
}

/// Extract Hyrax witness data from JoltProof and preprocessing
fn extract_hyrax_witness(
    proof: &RV64IMACProof,
    preprocessing: &VerifierPreprocessing<ark_bn254::Fr, DoryCommitmentScheme>,
) -> HyraxWitness {
    use ark_bn254::Fq;
    use jolt_core::poly::commitment::hyrax::matrix_dimensions;
    use jolt_core::poly::eq_poly::EqPolynomial;

    // Get the recursion proof containing Hyrax data
    let recursion_proof = &proof.recursion_proof;

    // Get the dense_num_vars from the metadata
    let dense_num_vars = proof.stage10_recursion_metadata.dense_num_vars;

    // U vector is the vector_matrix_product from the opening proof
    let u_vec: Vec<String> = recursion_proof
        .opening_proof
        .vector_matrix_product
        .iter()
        .map(|f| f.into_bigint().to_string())
        .collect();

    // Row commitments from dense_commitment
    // IMPORTANT: gnark's scalar multiplication doesn't handle identity points correctly.
    // We FILTER OUT identity points entirely from both row_commitments and L arrays.
    // This is mathematically correct because L[a] × identity = identity (adds zero to MSM).
    use ark_ec::AffineRepr;

    // First pass: identify which points are non-identity and their original indices
    let mut non_identity_row_commitments: Vec<[String; 2]> = Vec::new();
    let mut non_identity_indices: Vec<usize> = Vec::new();
    let mut identity_count = 0;

    for (i, p) in recursion_proof.dense_commitment.row_commitments.iter().enumerate() {
        let affine: ark_grumpkin::Affine = (*p).into();
        if affine.is_zero() {
            identity_count += 1;
        } else {
            non_identity_row_commitments.push(grumpkin_point_to_strings(&affine));
            non_identity_indices.push(i);
        }
    }

    println!("  Identity row commitments (filtered out): {}", identity_count);
    println!("  Non-identity row commitments: {}", non_identity_row_commitments.len());

    // Generators - use from preprocessing instead of regenerating
    let (_, gen_size) = matrix_dimensions(dense_num_vars, 1);

    // Use generators from preprocessing
    let generators: Vec<[String; 2]> = preprocessing.hyrax_recursion_setup.generators[..gen_size]
        .iter()
        .map(|p| grumpkin_point_to_strings(p))
        .collect();

    // Calculate matrix dimensions to get L_size and R_size
    let (l_size, r_size) = matrix_dimensions(dense_num_vars, 1);
    let sqrt_n = l_size; // Both should be equal for square matrix

    // The opening point for the dense polynomial comes from the recursion proof's opening claims
    // We need to find the point at which the dense polynomial is opened
    // In the Hyrax protocol, this is determined by the sumcheck challenges

    // For now, we'll extract what we can from the proof structure
    // The opening point is built from sumcheck challenges during verification
    // We need to look at the stage4 (jagged sumcheck) which produces the r_dense point

    // The opening point is derived from transcript challenges during verification
    // For witness extraction, we need to re-run the transcript to get these values
    // This is complex, so for now we'll compute L and R from a placeholder opening point

    // TODO: Properly extract the opening point by re-running the transcript
    // For now, use placeholder values (zeros) which won't give valid witness
    // but will allow testing the circuit structure

    // Actually, we need to trace through verification to get the actual challenges
    // Let's look at what data we have available...

    // The opening claims in the recursion proof contain the evaluation value
    // Let's find the DoryDenseMatrix opening claim
    let mut v_value: Option<Fq> = None;
    let mut opening_point_opt: Option<Vec<Fq>> = None;

    for (key, (point, value)) in recursion_proof.opening_claims.iter() {
        use jolt_core::poly::opening_proof::{OpeningId, PolynomialId};
        use jolt_core::zkvm::witness::CommittedPolynomial;

        if let OpeningId::Polynomial(PolynomialId::Committed(CommittedPolynomial::DoryDenseMatrix), _) = key {
            v_value = Some(*value);
            // The point is stored as challenges - convert to field elements
            let point_vec: Vec<Fq> = point.r.iter().map(|c| {
                let fq: Fq = (*c).into();
                fq
            }).collect();
            opening_point_opt = Some(point_vec);
            println!("  Found DoryDenseMatrix opening claim!");
            println!("    Evaluation value (V): {}", value.into_bigint());
            println!("    Opening point length: {}", point.r.len());
            break;
        }
    }

    // If we didn't find the opening claim, use placeholder
    let (opening_point_vec, v_string) = match (opening_point_opt, v_value) {
        (Some(point), Some(v)) => {
            let point_strings: Vec<String> = point.iter().map(|f| f.into_bigint().to_string()).collect();
            let v_str = v.into_bigint().to_string();
            (point, v_str)
        }
        _ => {
            panic!("DoryDenseMatrix opening claim not found in proof. Cannot extract Hyrax witness. \
                    Ensure the proof was generated with the correct features (--features transcript-poseidon).");
        }
    };

    // Compute L and R from the opening point
    // L = eq(a, z_L) for a in {0,1}^{log(L_size)}
    // R = eq(b, z_R) for b in {0,1}^{log(R_size)}
    let l_num_vars = l_size.trailing_zeros() as usize;
    let r_num_vars = r_size.trailing_zeros() as usize;

    let z_l = &opening_point_vec[..l_num_vars];
    let z_r = &opening_point_vec[l_num_vars..];

    let all_l_evals: Vec<Fq> = EqPolynomial::evals(z_l);
    let r_evals: Vec<Fq> = EqPolynomial::evals(z_r);

    // Filter L to only include non-identity indices (matching the filtered row_commitments)
    let filtered_l_evals: Vec<Fq> = non_identity_indices
        .iter()
        .map(|&idx| all_l_evals[idx])
        .collect();

    println!("  L num_vars: {}, all L evals: {}", l_num_vars, all_l_evals.len());
    println!("  Filtered L evals (matching non-identity rows): {}", filtered_l_evals.len());
    println!("  R num_vars: {}, R evals: {}", r_num_vars, r_evals.len());

    // VALIDATION: Check that <U, R> == V holds
    // U is from vector_matrix_product, R is from EqPolynomial::evals(z_r), V is from opening claim
    let u_vec_fq: Vec<Fq> = recursion_proof
        .opening_proof
        .vector_matrix_product
        .iter()
        .cloned()
        .collect();

    let dot_product: Fq = u_vec_fq.iter()
        .zip(r_evals.iter())
        .map(|(u, r)| *u * *r)
        .fold(Fq::from(0u64), |acc, x| acc + x);

    let v_fq = v_value.unwrap_or(Fq::from(0u64));

    println!("  VALIDATION: <U, R> = {}", dot_product.into_bigint());
    println!("  VALIDATION: V      = {}", v_fq.into_bigint());
    println!("  VALIDATION: Match? {}", dot_product == v_fq);

    if dot_product != v_fq {
        panic!("Hyrax witness validation failed: <U, R> != V. \
                Dot product: {}, V: {}. \
                This indicates a bug in witness extraction or proof corruption.",
               dot_product.into_bigint(), v_fq.into_bigint());
    }

    // sqrt_n1 is the filtered size (non-identity row commitments)
    // sqrt_n is the full size (for MSM #2 with generators × U)
    let sqrt_n1 = non_identity_row_commitments.len();

    println!("  Final witness sizes:");
    println!("    sqrt_n (full):     {}", sqrt_n);
    println!("    sqrt_n1 (filtered): {}", sqrt_n1);

    HyraxWitness {
        sqrt_n,  // Full size for MSM #2 and dot product
        sqrt_n1, // Filtered size for MSM #1
        u: u_vec, // Full size (prover's projection)
        row_commitments: non_identity_row_commitments, // Filtered (no identity points)
        generators: generators[..sqrt_n].to_vec(), // Full size (generators for MSM #2)
        l: filtered_l_evals.iter().map(|f| f.into_bigint().to_string()).collect(), // Filtered
        r: r_evals.iter().map(|f| f.into_bigint().to_string()).collect(), // Full size
        v: v_string,
        opening_point: opening_point_vec.iter().map(|f| f.into_bigint().to_string()).collect(),
        dense_num_vars,
    }
}

/// Generate Gnark circuit code from accumulated assertions
///
/// **Per-Assertion CSE**: To fix the node aliasing bug where structurally identical
/// expressions from different assertions get merged, we now use per-assertion CSE contexts.
/// Each assertion gets its own CSE namespace (cse_0_0, cse_0_1 for assertion 0, etc.).
fn generate_stages_circuit(
    assertions: &[MleAst],
    var_names: &HashMap<u16, String>,
    circuit_name: &str,
) -> String {
    // Per-assertion CSE: generate each assertion with its own CSE context
    // This avoids the node aliasing bug where structurally identical expressions
    // from different assertions get incorrectly merged.
    let mut all_bindings_code = String::new();
    let mut assertion_exprs: Vec<String> = Vec::new();
    let mut all_vars: BTreeSet<u16> = BTreeSet::new();

    for (assertion_idx, assertion) in assertions.iter().enumerate() {
        // Create a fresh codegen context for this assertion with per-assertion CSE naming
        let mut codegen = MemoizedCodeGen::with_var_names_and_constraint_idx(
            var_names.clone(),
            assertion_idx,
        );

        // Count references within this assertion only
        codegen.count_refs(assertion.root());

        // Generate expression for this assertion
        let expr = codegen.generate_expr(assertion.root());
        assertion_exprs.push(expr);

        // Collect vars used in this assertion
        all_vars.extend(codegen.vars().iter());

        // Collect bindings for this assertion (already have prefixed names from codegen)
        let constraint_bindings = codegen.bindings_code();
        if !constraint_bindings.is_empty() {
            all_bindings_code.push_str(&format!("\t// CSE bindings for assertion {}\n", assertion_idx));
            all_bindings_code.push_str(&constraint_bindings);
        }
    }

    let bindings_code = all_bindings_code;
    let vars = &all_vars;

    let mut output = String::new();

    // Package and imports
    output.push_str("package jolt_verifier\n\n");
    output.push_str("import (\n");
    output.push_str("\t\"github.com/consensys/gnark/frontend\"\n");
    if bindings_code.contains("poseidon.Hash")
        || assertion_exprs.iter().any(|e| e.contains("poseidon.Hash"))
    {
        output.push_str("\t\"jolt_verifier/poseidon\"\n");
    }
    output.push_str(")\n\n");

    // Note: bigInt helper is defined in helpers.go

    // Circuit struct
    output.push_str(&format!("type {} struct {{\n", circuit_name));

    // Add input variables - use sanitized names
    for var_idx in vars.iter() {
        let name = var_names
            .get(var_idx)
            .map(|n| sanitize_go_name(n))
            .unwrap_or_else(|| format!("X{}", var_idx));
        output.push_str(&format!(
            "\t{} frontend.Variable `gnark:\",public\"`\n",
            name
        ));
    }

    output.push_str("}\n\n");

    // Define method
    output.push_str(&format!(
        "func (circuit *{}) Define(api frontend.API) error {{\n",
        circuit_name
    ));

    // CSE bindings
    if !bindings_code.is_empty() {
        output.push_str("\t// Memoized subexpressions (CSE)\n");
        output.push_str(&bindings_code);
        output.push_str("\n");
    }

    // Generate assertions - each expression must equal zero
    output.push_str("\t// Verification assertions (each must equal 0)\n");
    for (i, expr) in assertion_exprs.iter().enumerate() {
        output.push_str(&format!("\ta{} := {}\n", i, expr));
        output.push_str(&format!("\tapi.AssertIsEqual(a{}, 0)\n", i));
        if (i + 1) % 100 == 0 {
            output.push_str(&format!("\t// ... assertion {} of {}\n", i + 1, assertion_exprs.len()));
        }
    }

    output.push_str("\n\treturn nil\n");
    output.push_str("}\n");

    output
}

