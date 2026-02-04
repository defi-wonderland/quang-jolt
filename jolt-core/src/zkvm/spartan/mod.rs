use crate::field::JoltField;
use crate::poly::opening_proof::{OpeningAccumulator, VerifierOpeningAccumulator};
use crate::subprotocols::univariate_skip::UniSkipFirstRoundProof;
use crate::transcripts::Transcript;
use crate::zkvm::r1cs::constraints::{
    OUTER_FIRST_ROUND_POLY_NUM_COEFFS, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
    PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS, PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
};
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::spartan::outer::{OuterUniSkipParams, OuterUniSkipVerifier};
use crate::zkvm::spartan::product::{ProductVirtualUniSkipParams, ProductVirtualUniSkipVerifier};

pub mod instruction_input;
pub mod outer;
pub mod product;
pub mod shift;

/// Stage 1a: Verify first round of Spartan outer sum-check with univariate skip
///
/// Generic over accumulator type to support both concrete verification and transpilation.
pub fn verify_stage1_uni_skip<F: JoltField, T: Transcript, A: OpeningAccumulator<F>>(
    proof: &UniSkipFirstRoundProof<F, T>,
    key: &UniformSpartanKey<F>,
    opening_accumulator: &mut A,
    transcript: &mut T,
) -> Result<OuterUniSkipParams<F>, anyhow::Error> {
    let verifier = OuterUniSkipVerifier::new(key, transcript);
    UniSkipFirstRoundProof::verify::<
        OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        OUTER_FIRST_ROUND_POLY_NUM_COEFFS,
        A,
    >(proof, &verifier, opening_accumulator, transcript)
    .map_err(|_| anyhow::anyhow!("UniSkip first-round verification failed"))?;

    Ok(verifier.params)
}

/// Stage 2a: Verify first round of Product Virtual sum-check with univariate skip
///
/// Generic over accumulator type to support both concrete verification and transpilation.
pub fn verify_stage2_uni_skip<F: JoltField, T: Transcript, A: OpeningAccumulator<F>>(
    proof: &UniSkipFirstRoundProof<F, T>,
    opening_accumulator: &mut A,
    transcript: &mut T,
) -> Result<ProductVirtualUniSkipParams<F>, anyhow::Error> {
    let verifier = ProductVirtualUniSkipVerifier::new(opening_accumulator, transcript);
    UniSkipFirstRoundProof::verify::<
        PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS,
        A,
    >(proof, &verifier, opening_accumulator, transcript)
    .map_err(|_| anyhow::anyhow!("ProductVirtual uni-skip first-round verification failed"))?;

    Ok(verifier.params)
}
