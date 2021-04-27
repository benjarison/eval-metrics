use crate::error::EvalError;
use crate::numeric::Float;

///
/// Validates a pair of scores and labels, returning an error if either scores or labels are
/// empty, or if they have lengths that differ
///
/// # Arguments
///
/// * `scores` the vector of scores
/// * `labels` the vector of labels
///
pub fn validate_input<T, U>(scores: &Vec<T>, labels: &Vec<U>) -> Result<(), EvalError> {
    if scores.is_empty() {
        Err(EvalError::invalid_input("Scores are empty"))
    } else if labels.is_empty() {
        Err(EvalError::invalid_input("Labels are empty"))
    } else if scores.len() != labels.len() {
        Err(EvalError::invalid_input("Scores and labels have different lengths"))
    } else {
        Ok(())
    }
}

///
/// Check if the provided value is finite
///
/// # Arguments
///
/// * `value` the float value to check
///
pub fn check_finite<T: Float>(value: T) -> Result<T, EvalError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EvalError::infinite_value())
    }
}
