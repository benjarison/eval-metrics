//!
//! Details evaluation error types
//!

///
/// Evaluation error enumeration
///
#[derive(Clone, Debug)]
pub enum EvalError {
    /// An error that results from invalid user input, such as mismatched data lengths
    InvalidInput(String),
    /// An error that indicates that the metric is undefined for some numeric reason
    UndefinedMetric(String)
}

impl EvalError {

    ///
    /// Constructs a new invalid input error with the provided message
    ///
    /// # Arguments
    ///
    /// * `msg` - detailed error message
    ///
    pub fn invalid_input(msg: &str) -> EvalError {
        EvalError::InvalidInput(String::from(msg))
    }

    ///
    /// Constructs a new undefined metric error with the provided metric name
    ///
    /// # Arguments
    ///
    /// * `name` - metric name
    ///
    pub fn undefined_metric(name: &str) -> EvalError {
        EvalError::UndefinedMetric(String::from(name))
    }

    ///
    /// Constructs a new undefined metric error for encountered infinite/NaN values
    ///
    pub fn infinite_value() -> EvalError {
        EvalError::undefined_metric("Infinite or NaN value")
    }

    ///
    /// Constructs a new undefined metric error due to constant input data
    ///
    pub fn constant_input_data() -> EvalError {
        EvalError::undefined_metric("Input data is constant")
    }
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            EvalError::UndefinedMetric(name) => write!(f, "Undefined metric: {}", name)
        }
    }
}

impl std::error::Error for EvalError {}
