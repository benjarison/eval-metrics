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
        Err(EvalError::new("Scores are empty"))
    } else if labels.is_empty() {
        Err(EvalError::new("Labels are empty"))
    } else if scores.len() != labels.len() {
        Err(EvalError::new("Scores and labels have different lengths"))
    } else {
        Ok(())
    }
}

///
/// Check for the presence of a NaN value and return an error if found
///
/// # Arguments
///
/// * `value` the float value to check
///
pub fn check_nan<T: Float>(value: T) -> Result<T, EvalError> {
    if value.is_nan() {
        Err(EvalError::nan_value())
    } else {
        Ok(value)
    }
}
