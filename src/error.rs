
///
/// Represents a general evaluation error
///
#[derive(Clone, Debug)]
pub struct EvalError {
    /// The error message
    pub msg: String
}

impl EvalError {
    ///
    /// Constructs a new eval error with the provided message
    ///
    /// # Arguments
    ///
    /// * `msg` the error message
    ///
    pub fn new(msg: &str) -> EvalError {
        EvalError {msg: String::from(msg)}
    }

    ///
    /// Indicates that a NaN value was encountered
    ///
    pub fn nan_value() -> EvalError {
        EvalError::new("Encountered NaN value")
    }
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.msg)
    }
}

impl std::error::Error for EvalError {}
