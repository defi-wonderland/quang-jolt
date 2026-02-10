//! Dory IPA transcript replay via thread-local tunneling.
//!
//! Used to replay Dory proof bytes through the Poseidon sponge during
//! symbolic execution, bridging the gap between `verify_stage8_wo_pcs`
//! (which skips `PCS::verify_with_hint`) and the actual transcript state.
//!
//! The pattern follows the existing thread-local tunneling in zklean-extractor
//! (PENDING_CHALLENGE, PENDING_APPEND, PENDING_COMMITMENT_CHUNKS).
//!
//! Uses u16 variable indices (not MleAst) to avoid circular dependencies:
//! jolt-core cannot depend on zklean-extractor. The consumer
//! (PoseidonAstTranscript) reconstructs MleAst::Var(idx) from these indices.

use std::cell::RefCell;
use std::collections::VecDeque;

/// Dory replay operation with u16 variable indices.
/// The consumer (PoseidonAstTranscript) creates MleAst::Var(idx) from these.
#[derive(Clone, Debug)]
pub enum DoryReplayOp {
    /// Absorb bytes: variable indices for each 32-byte chunk.
    Absorb { byte_size: usize, var_indices: Vec<u16> },
    /// Squeeze a challenge scalar.
    Squeeze,
}

thread_local! {
    /// Full sequence of Dory IPA replay operations.
    /// Set by main.rs before verify(), consumed by verify_stage8_wo_pcs at the boundary.
    static DORY_REPLAY_OPS: RefCell<Option<VecDeque<DoryReplayOp>>> = RefCell::new(None);
}

thread_local! {
    /// Single-shot pending variable indices for the next append_bytes call.
    /// Set by the replay loop, consumed by PoseidonAstTranscript::append_bytes.
    static PENDING_DORY_ABSORB_INDICES: RefCell<Option<Vec<u16>>> = RefCell::new(None);
}

/// Set the full Dory IPA replay sequence. Called from main.rs before verify().
pub fn set_dory_replay_ops(ops: VecDeque<DoryReplayOp>) {
    DORY_REPLAY_OPS.with(|cell| {
        *cell.borrow_mut() = Some(ops);
    });
}

/// Take all remaining Dory replay ops.
/// Called from verify_stage8_wo_pcs at the boundary.
pub fn take_dory_replay_ops() -> Option<VecDeque<DoryReplayOp>> {
    DORY_REPLAY_OPS.with(|cell| cell.borrow_mut().take())
}

/// Set pending variable indices for the next append_bytes call.
/// Called from the replay loop in verify_stage8_wo_pcs.
pub fn set_pending_dory_absorb_indices(indices: Vec<u16>) {
    PENDING_DORY_ABSORB_INDICES.with(|cell| {
        *cell.borrow_mut() = Some(indices);
    });
}

/// Take pending variable indices.
/// Called from PoseidonAstTranscript::append_bytes.
pub fn take_pending_dory_absorb_indices() -> Option<Vec<u16>> {
    PENDING_DORY_ABSORB_INDICES.with(|cell| cell.borrow_mut().take())
}

// =============================================================================
// Dense commitment replay
// =============================================================================
//
// The recursion proof's dense_commitment (HyraxCommitment) is appended to the
// Poseidon transcript via `append_serializable`. Since it's not symbolized in the
// AstCommitment (which only has 12 chunks for GT elements), we pre-serialize the
// real bytes and replay them.
//
// Format: serialize_uncompressed → reverse → append_bytes (matching PoseidonTranscript).

thread_local! {
    /// Pre-serialized dense commitment bytes (uncompressed + reversed).
    /// Set by main.rs, consumed by verify_stage8_wo_pcs at the dense_commitment boundary.
    static DENSE_COMMITMENT_BYTES: RefCell<Option<Vec<u8>>> = RefCell::new(None);
}

/// Set the pre-serialized dense commitment bytes.
pub fn set_dense_commitment_bytes(bytes: Vec<u8>) {
    DENSE_COMMITMENT_BYTES.with(|cell| {
        *cell.borrow_mut() = Some(bytes);
    });
}

/// Take the pre-serialized dense commitment bytes.
pub fn take_dense_commitment_bytes() -> Option<Vec<u8>> {
    DENSE_COMMITMENT_BYTES.with(|cell| cell.borrow_mut().take())
}
