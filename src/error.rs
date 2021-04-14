
///
/// Enumerated evaluation errors
///
#[derive(Clone, Debug)]
pub enum EvalError {
    InvalidInputError(String),
    UndefinedMetricError(String)
}

impl EvalError {

    ///
    /// Constructs a new InvalidInputError with the provided message
    ///
    /// # Arguments
    ///
    /// * `msg` the detailed error message
    ///
    pub fn invalid_input(msg: &str) -> EvalError {
        EvalError::InvalidInputError(String::from(msg))
    }

    ///
    /// Constructs a new UndefinedMetricError with the provided metric name
    ///
    /// # Arguments
    ///
    /// * `metric` the metric name
    ///
    pub fn undefined_metric(msg: &str) -> EvalError {
        EvalError::UndefinedMetricError(String::from(msg))
    }
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::InvalidInputError(msg) => write!(f, "Invalid input: {}", msg),
            EvalError::UndefinedMetricError(msg) => write!(f, "Undefined metric: {}", msg)
        }
    }
}

impl std::error::Error for EvalError {}
