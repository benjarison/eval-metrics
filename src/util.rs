use crate::error::EvalError;

///
/// Validates a pair of scores and labels, returning an error if either scores or labels are
/// empty, or if they have lengths that differ
///
/// # Arguments
///
/// * `scores` the vector of scores
/// * `labels` the vector of labels
///
pub fn validate_data<T, U>(scores: &Vec<T>, labels: &Vec<U>) -> Result<(), EvalError> {
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
