//! Poseidon transcript for symbolic execution (MleAst)
//!
//! This implementation mirrors jolt-core's PoseidonTranscript but records
//! operations in MleAst instead of computing actual hashes. The structure
//! matches exactly to ensure circuit compatibility:
//!
//! - Width-3 Poseidon: poseidon(state, n_rounds, data)
//! - Domain separation via n_rounds counter
//! - Same byte chunking and padding behavior
//!
//! ## Challenge Derivation Modes
//!
//! Two different challenge derivation methods exist, producing different AST nodes:
//!
//! | Jolt Function                    | AST Node             | Used For                    |
//! |----------------------------------|----------------------|-----------------------------|
//! | `challenge_scalar_128_bits()`    | `Truncate128`        | Batching coefficients       |
//! | `challenge_scalar_optimized()`   | `Truncate128Reverse` | Sumcheck challenges (r)     |
//!
//! ### Truncate128 (simple 128-bit truncation)
//! - Takes low 16 bytes of hash, interprets as field element
//! - Used for batching where range doesn't matter
//!
//! ### Truncate128Reverse (optimized Montgomery representation)
//! - Applies 125-bit mask + shift to create `[0, 0, low, high]` limb layout
//! - Multiplies by R⁻¹ because `from_bigint_unchecked` expects Montgomery form
//! - Used for sumcheck challenges where MontU128Challenge optimization applies
//!
//! See `MontU128Challenge` in `jolt-core/src/transcripts/poseidon.rs` for the
//! optimized representation that avoids expensive `into()` conversions.

use ark_ec::CurveGroup;
use ark_serialize::CanonicalSerialize;
use jolt_core::field::JoltField;
use jolt_core::transcripts::Transcript;
use std::borrow::Borrow;
use jolt_core::zkvm::dory_replay::take_pending_dory_absorb_indices;
use zklean_extractor::mle_ast::{set_pending_challenge, take_pending_append, take_pending_commitment_chunks, MleAst};

/// Convert 32 bytes (little-endian) to a [u64; 4] scalar.
/// Uses `from_le_bytes_mod_order` to match the real PoseidonTranscript exactly:
/// values exceeding the BN254 Fr modulus are reduced mod p.
fn bytes_to_scalar(bytes: &[u8; 32]) -> [u64; 4] {
    use ark_ff::PrimeField;
    ark_bn254::Fr::from_le_bytes_mod_order(bytes).into_bigint().0
}

/// Poseidon transcript for symbolic execution.
/// Mirrors jolt-core's PoseidonTranscript structure exactly:
/// - 32-byte state (represented as MleAst)
/// - n_rounds counter for domain separation
/// - Width-3 Poseidon: hash(state, n_rounds, data)
#[derive(Clone)]
pub struct PoseidonAstTranscript {
    /// Current state (symbolic field element)
    state: MleAst,
    /// Round counter for domain separation
    n_rounds: u32,
}

impl Default for PoseidonAstTranscript {
    fn default() -> Self {
        Self {
            state: MleAst::from_u64(0),
            n_rounds: 0,
        }
    }
}

impl PoseidonAstTranscript {
    /// Get the current round counter.
    pub fn n_rounds(&self) -> u32 {
        self.n_rounds
    }

    /// Convert a label to a field element, matching jolt-core's behavior.
    ///
    /// jolt-core does: label_padded[..label.len()].copy_from_slice(label);
    ///                 Fr::from_le_bytes_mod_order(&label_padded)
    ///
    /// For symbolic execution, we compute the actual integer value that
    /// the label bytes represent in little-endian order.
    fn label_to_field(label: &[u8]) -> MleAst {
        assert!(label.len() <= 32, "Label must be <= 32 bytes");

        // Pad label to 32 bytes and convert to [u64; 4]
        let mut padded = [0u8; 32];
        padded[..label.len()].copy_from_slice(label);
        let limbs = bytes_to_scalar(&padded);

        MleAst::from(limbs)
    }

    /// Create a new transcript with initial state.
    ///
    /// Mirrors jolt-core: initial_state = poseidon(label, 0, 0)
    pub fn new_mle(label: &'static [u8]) -> Self {
        let label_field = Self::label_to_field(label);
        let initial_state =
            MleAst::poseidon(&label_field, &MleAst::from_u64(0), &MleAst::from_u64(0));
        Self {
            state: initial_state,
            n_rounds: 0,
        }
    }

    /// Hash a field element with domain separation.
    ///
    /// Mirrors jolt-core: poseidon(state, n_rounds, element)
    fn hash_and_update(&mut self, element: MleAst) {
        let round = MleAst::from_u64(self.n_rounds as u64);
        self.state = MleAst::poseidon(&self.state, &round, &element);
        self.n_rounds += 1;
    }

    /// Derive a challenge as MleAst.
    ///
    /// Mirrors jolt-core: poseidon(state, n_rounds, 0)
    pub fn challenge_mle(&mut self) -> MleAst {
        let round = MleAst::from_u64(self.n_rounds as u64);
        let zero = MleAst::from_u64(0);
        let challenge = MleAst::poseidon(&self.state, &round, &zero);
        self.state = challenge.clone();
        self.n_rounds += 1;
        challenge
    }

    /// Derive multiple challenges as MleAst.
    pub fn challenge_vector_mle(&mut self, len: usize) -> Vec<MleAst> {
        (0..len).map(|_| self.challenge_mle()).collect()
    }

    /// Append symbolic field elements (for commitments/preamble as circuit inputs).
    ///
    /// This mirrors append_bytes but with symbolic inputs instead of concrete bytes.
    /// Each field element corresponds to one 32-byte chunk.
    pub fn append_field_elements(&mut self, elements: &[MleAst]) {
        let round = MleAst::from_u64(self.n_rounds as u64);
        let zero = MleAst::from_u64(0);

        let mut iter = elements.iter();

        // First element: includes n_rounds for domain separation
        let mut current = if let Some(first) = iter.next() {
            MleAst::poseidon(&self.state, &round, first)
        } else {
            // Empty: just hash state with n_rounds and zero
            MleAst::poseidon(&self.state, &round, &zero)
        };

        // Remaining elements: no n_rounds (already accounted for)
        for elem in iter {
            current = MleAst::poseidon(&current, &zero, elem);
        }

        self.state = current;
        self.n_rounds += 1;
    }

    /// Append a single symbolic u64 (for preamble values as circuit inputs).
    ///
    /// This applies the same BE-padding transformation as PoseidonTranscript::append_u64:
    /// - u64 value is placed in bytes 24-31 of a 32-byte array (big-endian padding)
    /// - The 32 bytes are then interpreted as little-endian
    /// - Mathematically, this equals: value * 2^192
    ///
    /// Example: append_u64(4096) stores [0..0, 0x00, 0x00, 0x10, 0x00] (BE) in bytes 24-31
    /// Interpreted as LE 32-byte integer: 4096 * 2^192
    pub fn append_u64_symbolic(&mut self, value: MleAst) {
        // Apply the BE-padding transformation: value * 2^192
        let transformed = MleAst::mul_two_pow_192(&value);
        self.hash_and_update(transformed);
    }
}

/// Implement Jolt's Transcript trait for PoseidonAstTranscript.
///
/// This allows using PoseidonAstTranscript with verify_stage1_with_transcript.
/// The challenge methods return MleAst when F = MleAst.
impl Transcript for PoseidonAstTranscript {
    fn new(label: &'static [u8]) -> Self {
        // Mirror jolt-core: initial_state = poseidon(label, 0, 0)
        let label_field = Self::label_to_field(label);
        let initial_state = MleAst::poseidon(
            &label_field,
            &MleAst::from_u64(0), // n_rounds = 0
            &MleAst::from_u64(0), // zero
        );
        Self {
            state: initial_state,
            n_rounds: 0,
        }
    }

    fn append_message(&mut self, msg: &'static [u8]) {
        // Matches jolt-core: hash_bytes32_and_update which does immediate hashing.
        // This is NOT batchable - it's a direct poseidon call per jolt-core semantics.
        assert!(msg.len() <= 32);
        let mut padded = [0u8; 32];
        padded[..msg.len()].copy_from_slice(msg);
        let limbs = bytes_to_scalar(&padded);
        let chunk_field = MleAst::from(limbs);
        let round = MleAst::from_u64(self.n_rounds as u64);
        self.state = MleAst::poseidon(&self.state, &round, &chunk_field);
        self.n_rounds += 1;
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        // Check for pending Dory IPA symbolic variable indices (thread-local tunneling).
        // If present, reconstruct MleAst::Var from u16 indices and use as symbolic chunks.
        if let Some(var_indices) = take_pending_dory_absorb_indices() {
            let symbolic_chunks: Vec<MleAst> = var_indices.into_iter().map(MleAst::from_var).collect();
            self.append_field_elements(&symbolic_chunks);
            return;
        }

        // Hash all bytes using Poseidon with domain separation via n_rounds.
        // First chunk: hash(state, n_rounds, chunk), includes domain separator.
        // Subsequent chunks: hash(prev, 0, chunk), chained but without redundant n_rounds.
        let round = MleAst::from_u64(self.n_rounds as u64);
        let zero = MleAst::from_u64(0);

        let mut chunks = bytes.chunks(32);

        // First chunk: includes n_rounds for domain separation
        let mut current = if let Some(first_chunk) = chunks.next() {
            let mut padded = [0u8; 32];
            padded[..first_chunk.len()].copy_from_slice(first_chunk);
            let chunk_field = MleAst::from(bytes_to_scalar(&padded));
            MleAst::poseidon(&self.state, &round, &chunk_field)
        } else {
            // Empty bytes: just hash state with n_rounds and zero
            MleAst::poseidon(&self.state, &round, &zero)
        };

        // Remaining chunks: no n_rounds (already accounted for)
        for chunk in chunks {
            let mut padded = [0u8; 32];
            padded[..chunk.len()].copy_from_slice(chunk);
            let chunk_field = MleAst::from(bytes_to_scalar(&padded));
            current = MleAst::poseidon(&current, &zero, &chunk_field);
        }

        self.state = current;
        self.n_rounds += 1;
    }

    fn append_u64(&mut self, x: u64) {
        // Matches jolt-core: hash_bytes32_and_update which does immediate hashing.
        // This is NOT batchable - it's a direct poseidon call per jolt-core semantics.
        //
        // PoseidonTranscript::append_u64 does:
        //   1. Pack u64 into 32-byte array with BE-padding: packed[24..32] = x.to_be_bytes()
        //   2. Interpret packed as LE field element via from_le_bytes_mod_order
        //   3. Mathematically: value * 2^192
        let x_ast = MleAst::from_u64(x);
        let transformed = MleAst::mul_two_pow_192(&x_ast);
        let round = MleAst::from_u64(self.n_rounds as u64);
        self.state = MleAst::poseidon(&self.state, &round, &transformed);
        self.n_rounds += 1;
    }

    fn append_scalar<F: JoltField>(&mut self, scalar: &F) {
        // Matches jolt-core: serialize -> reverse -> hash_bytes32_and_update (immediate hashing).
        // This is NOT batchable - it's a direct poseidon call per jolt-core semantics.
        //
        // Trigger serialization which stores MleAst in thread-local (if F = MleAst)
        let mut buf = vec![];
        let _ = scalar.serialize_uncompressed(&mut buf);

        // Retrieve the MleAst from thread-local (set by MleAst::serialize_with_mode)
        let round = MleAst::from_u64(self.n_rounds as u64);
        if let Some(mle_ast) = take_pending_append() {
            // Apply byte-reverse to match PoseidonTranscript::append_scalar behavior:
            // PoseidonTranscript does: serialize(LE) -> reverse -> from_le_bytes_mod_order -> hash
            let byte_reversed = MleAst::byte_reverse(&mle_ast);
            self.state = MleAst::poseidon(&self.state, &round, &byte_reversed);
            self.n_rounds += 1;
        } else {
            // Fallback for non-MleAst types (shouldn't happen in transpilation)
            self.state = MleAst::poseidon(&self.state, &round, &MleAst::from_u64(0));
            self.n_rounds += 1;
        }
    }

    fn append_serializable<S: CanonicalSerialize>(&mut self, scalar: &S) {
        // The real PoseidonTranscript::append_serializable does:
        // 1. serialize_uncompressed -> bytes (LE)
        // 2. reverse bytes (for EVM compat)
        // 3. append_bytes (which chunks into 32-byte pieces and hashes with chaining)
        //
        // For symbolic execution, serialization stores values in thread-local.
        let mut buf = vec![];
        let _ = scalar.serialize_uncompressed(&mut buf);

        // Check for commitment chunks first (12 MleAst for commitment hashing)
        // AstCommitment::serialize stores 12 chunks in PENDING_COMMITMENT_CHUNKS
        if let Some(chunks) = take_pending_commitment_chunks() {
            // Hash all 12 chunks with proper chaining (like append_bytes does)
            // This matches what PoseidonTranscript::append_serializable does:
            // - First chunk: poseidon(state, n_rounds, chunk_0)
            // - Remaining: poseidon(prev_hash, 0, chunk_i)
            // - Only increment n_rounds once at the end
            self.append_field_elements(&chunks);
            return;
        }

        // Fallback: single MleAst (existing behavior for non-commitment types)
        if let Some(mle_ast) = take_pending_append() {
            // Apply byte-reverse to match PoseidonTranscript::append_serializable behavior
            let byte_reversed = MleAst::byte_reverse(&mle_ast);
            self.hash_and_update(byte_reversed);
        } else {
            // Fallback for non-MleAst types (shouldn't happen in transpilation)
            self.hash_and_update(MleAst::from_u64(0));
        }
    }

    fn append_scalars<F: JoltField>(&mut self, scalars: &[impl Borrow<F>]) {
        self.append_message(b"begin_append_vector");
        for scalar in scalars.iter() {
            self.append_scalar(scalar.borrow());
        }
        self.append_message(b"end_append_vector");
    }

    fn append_point<G: CurveGroup>(&mut self, _point: &G) {
        self.hash_and_update(MleAst::from_u64(0));
    }

    fn append_points<G: CurveGroup>(&mut self, points: &[G]) {
        self.append_message(b"begin_append_vector");
        for _ in points.iter() {
            self.hash_and_update(MleAst::from_u64(0));
        }
        self.append_message(b"end_append_vector");
    }

    fn challenge_u128(&mut self) -> u128 {
        let _ = self.challenge_mle();
        0u128
    }

    fn challenge_scalar<F: JoltField>(&mut self) -> F {
        // Rust PoseidonTranscript::challenge_scalar calls challenge_scalar_128_bits
        // which truncates to 128 bits (NO mask, NO shift) - just plain truncation
        let hash = self.challenge_mle();
        let challenge = MleAst::truncate_128(&hash);
        set_pending_challenge(challenge);
        F::from_bytes(&[0u8; 32])
    }

    fn challenge_scalar_128_bits<F: JoltField>(&mut self) -> F {
        // Truncate to 128 bits without mask or shift
        let hash = self.challenge_mle();
        let challenge = MleAst::truncate_128(&hash);
        set_pending_challenge(challenge);
        F::from_bytes(&[0u8; 16])
    }

    fn challenge_vector<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        (0..len)
            .map(|_| {
                let hash = self.challenge_mle();
                // challenge_vector uses challenge_scalar internally, so no mask/shift
                let challenge = MleAst::truncate_128(&hash);
                set_pending_challenge(challenge);
                F::from_bytes(&[0u8; 32])
            })
            .collect()
    }

    fn challenge_scalar_powers<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        // Get base challenge - uses challenge_scalar which has no mask/shift
        let hash = self.challenge_mle();
        let challenge = MleAst::truncate_128(&hash);
        set_pending_challenge(challenge);
        let base: F = F::from_bytes(&[0u8; 32]);

        // Compute powers: 1, base, base^2, ...
        let mut powers = Vec::with_capacity(len);
        let mut current = F::one();
        for _ in 0..len {
            powers.push(current);
            current = current * base;
        }
        powers
    }

    /// Returns a challenge scalar using the optimized 125-bit representation.
    ///
    /// # Safety Invariant
    /// This function uses `transmute_copy` which is only safe when `F = MleAst`.
    /// For MleAst, `F::Challenge = MleAst` (same type), so the transmute is a no-op.
    /// We add a runtime assertion to catch misuse early.
    fn challenge_scalar_optimized<F: JoltField + 'static>(&mut self) -> F::Challenge {
        // Runtime check: ensure this is only called with F = MleAst
        assert!(
            std::any::TypeId::of::<F>() == std::any::TypeId::of::<MleAst>(),
            "PoseidonAstTranscript only supports F = MleAst for symbolic execution"
        );

        let hash = self.challenge_mle();
        let challenge = MleAst::truncate_128_reverse(&hash);
        set_pending_challenge(challenge);
        // The pending_challenge mechanism: F::from_bytes returns the pending challenge for MleAst
        let f_val: F = F::from_bytes(&[0u8; 16]);
        // SAFETY: We verified F = MleAst above, and for MleAst, F::Challenge = MleAst.
        // Both types are identical, so this transmute is a no-op bit copy.
        unsafe { std::mem::transmute_copy::<F, F::Challenge>(&f_val) }
    }

    /// Returns a vector of challenge scalars using the optimized representation.
    ///
    /// # Safety Invariant
    /// Same as `challenge_scalar_optimized` - only safe when `F = MleAst`.
    fn challenge_vector_optimized<F: JoltField + 'static>(&mut self, len: usize) -> Vec<F::Challenge> {
        // Runtime check: ensure this is only called with F = MleAst
        assert!(
            std::any::TypeId::of::<F>() == std::any::TypeId::of::<MleAst>(),
            "PoseidonAstTranscript only supports F = MleAst for symbolic execution"
        );

        (0..len)
            .map(|_| {
                let hash = self.challenge_mle();
                let challenge = MleAst::truncate_128_reverse(&hash);
                set_pending_challenge(challenge);
                let f_val: F = F::from_bytes(&[0u8; 16]);
                // SAFETY: Verified F = MleAst above
                unsafe { std::mem::transmute_copy::<F, F::Challenge>(&f_val) }
            })
            .collect()
    }

    fn challenge_scalar_powers_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        let _ = self.challenge_mle();
        vec![F::zero(); len]
    }

    fn debug_state(&self, _label: &str) {
        // Debug output disabled for production use
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_to_field() {
        // "jolt" = [0x6a, 0x6f, 0x6c, 0x74] in ASCII
        // As little-endian u64 (padded to 8 bytes): 0x746c6f6a = 1953263466
        let label_field = PoseidonAstTranscript::label_to_field(b"jolt");

        // Check it's a scalar with the expected value
        let root = label_field.root();
        let node = zklean_extractor::mle_ast::get_node(root);
        match node {
            zklean_extractor::mle_ast::Node::Atom(zklean_extractor::mle_ast::Atom::Scalar(v)) => {
                // "jolt" bytes: [0x6a, 0x6f, 0x6c, 0x74, 0, 0, 0, 0]
                // As little-endian u64: 0x00_00_00_00_74_6c_6f_6a = 1953263466
                // As [u64; 4]: [1953263466, 0, 0, 0]
                let expected: [u64; 4] = [1953263466, 0, 0, 0];
                assert_eq!(
                    v, expected,
                    "Label 'jolt' should be {:?} but got {:?}",
                    expected, v
                );
            }
            _ => panic!("Expected Scalar atom, got {:?}", node),
        }
    }

    #[test]
    fn test_transcript_creation() {
        let transcript: PoseidonAstTranscript = Transcript::new(b"test");
        assert_eq!(transcript.n_rounds, 0);
    }

    #[test]
    fn test_transcript_with_jolt_label() {
        // Verify that creating transcript with "Jolt" label produces a Poseidon node
        let transcript: PoseidonAstTranscript = Transcript::new(b"Jolt");
        assert_eq!(transcript.n_rounds, 0);

        // The initial state should be a Poseidon hash node
        let root = transcript.state.root();
        let node = zklean_extractor::mle_ast::get_node(root);
        match node {
            zklean_extractor::mle_ast::Node::Poseidon(_, _, _) => {
                // Expected: poseidon(label, 0, 0)
            }
            _ => panic!("Expected Poseidon node for initial state, got {:?}", node),
        }
    }

    #[test]
    fn test_append_and_challenge() {
        let mut transcript: PoseidonAstTranscript = Transcript::new(b"test");
        transcript.hash_and_update(MleAst::from_u64(42));
        let _challenge = transcript.challenge_mle();
        assert_eq!(transcript.n_rounds, 2); // 1 append + 1 challenge
    }

    #[test]
    fn test_append_scalar_with_mle_ast() {
        use jolt_core::transcripts::Transcript as _;

        let mut transcript: PoseidonAstTranscript = Transcript::new(b"test");

        // Create a variable (not a constant)
        let var = MleAst::from_var(42);

        // Append it to transcript
        transcript.append_scalar(&var);

        // Check that the state now contains a Poseidon node with ByteReverse(variable)
        // append_scalar does: byte_reverse(var) -> hash
        let root = transcript.state.root();
        let node = zklean_extractor::mle_ast::get_node(root);

        match node {
            zklean_extractor::mle_ast::Node::Poseidon(_, _, e3) => {
                // The third argument should be a ByteReverse node containing the variable
                match e3 {
                    zklean_extractor::mle_ast::Edge::NodeRef(byte_rev_id) => {
                        let byte_rev_node = zklean_extractor::mle_ast::get_node(byte_rev_id);
                        match byte_rev_node {
                            zklean_extractor::mle_ast::Node::ByteReverse(inner) => {
                                match inner {
                                    zklean_extractor::mle_ast::Edge::Atom(
                                        zklean_extractor::mle_ast::Atom::Var(idx),
                                    ) => {
                                        assert_eq!(idx, 42, "Expected Var(42), got Var({})", idx);
                                    }
                                    other => panic!(
                                        "Expected Var(42) inside ByteReverse, got {:?}",
                                        other
                                    ),
                                }
                            }
                            other => panic!("Expected ByteReverse node, got {:?}", other),
                        }
                    }
                    other => panic!(
                        "Expected NodeRef to ByteReverse as third Poseidon arg, got {:?}",
                        other
                    ),
                }
            }
            _ => panic!("Expected Poseidon node, got {:?}", node),
        }
    }
}
