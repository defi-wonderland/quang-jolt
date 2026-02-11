//! Thread-local flag for Fq (BN254 base field) arithmetic mode.
//!
//! When enabled, MleAst operator impls emit FqMul/FqAdd/FqSub nodes
//! instead of Mul/Add/Sub, so codegen produces emulated field arithmetic
//! in the Gnark circuit.
//!
//! Set by transpilable_verifier.rs around recursion sumchecks,
//! read by zklean-extractor's mle_ast.rs operator impls.

use std::cell::RefCell;

thread_local! {
    static FQ_MODE: RefCell<bool> = RefCell::new(false);
}

/// Enable or disable Fq arithmetic mode.
pub fn set_fq_mode(enabled: bool) {
    FQ_MODE.with(|cell| {
        *cell.borrow_mut() = enabled;
    });
}

/// Check if Fq arithmetic mode is currently enabled.
pub fn is_fq_mode() -> bool {
    FQ_MODE.with(|cell| *cell.borrow())
}
