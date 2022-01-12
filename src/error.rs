//!
//! Details evaluation error types
//!

///
/// Evaluation error enumeration
///
#[derive(Clone, Debug)]
pub struct EvalError {
    /// The error message
    msg: String
}

impl EvalError {

    ///
    /// Alerts than an invalid input was provided
    ///
    /// # Arguments
    ///
    /// * `msg` - detailed error message
    ///
    pub fn invalid_input(msg: &str) -> EvalError {
        EvalError {msg: format!("Invalid input: {}", msg)}
    }

    ///
    /// Alerts that an undefined metric was encountered
    ///
    /// # Arguments
    ///
    /// * `name` - metric name
    ///
    pub fn undefined_metric(name: &str) -> EvalError {
        EvalError {msg: format!("Undefined metric: {}", name)}
    }

    ///
    /// Alerts than an infinite/NaN value was encountered
    ///
    pub fn infinite_value() -> EvalError {
        EvalError {msg: String::from("Infinite or NaN value")}
    }

    ///
    /// Alerts that constant input data was encountered
    ///
    pub fn constant_input_data() -> EvalError {
        EvalError {msg: String::from("Constant input data")}
    }

    ///
    /// Alerts than an invalid metric was provided
    ///
    /// # Arguments
    ///
    /// * `name` - metric name
    ///
    pub fn invalid_metric(name: &str) -> EvalError {
        EvalError {msg: format!("Invalid metric: {}", name)}
    }
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.msg)
    }
}

impl std::error::Error for EvalError {}
