//!
//! Details evaluation error types
//!

///
/// Represents a generic evaluation error
///
#[derive(Clone, Debug)]
pub enum EvalError {
    InvalidArgument { message: String },
    InvalidMetric { name: String },
    NonFiniteValue,
    UndefinedMetric { name: String },
}

impl EvalError {
    ///
    /// Alerts than an invalid input was provided
    ///
    /// # Arguments
    ///
    /// * `msg` - detailed error message
    ///
    pub fn invalid_argument(message: impl Into<String>) -> EvalError {
        EvalError::InvalidArgument {
            message: message.into(),
        }
    }

    ///
    /// Alerts than an invalid metric was provided
    ///
    /// # Arguments
    ///
    /// * `name` - metric name
    ///
    pub fn invalid_metric(name: impl Into<String>) -> EvalError {
        EvalError::InvalidMetric { name: name.into() }
    }

    ///
    /// Alerts than an infinite or NaN value was encountered
    ///
    pub fn non_finite_value() -> EvalError {
        EvalError::NonFiniteValue
    }

    ///
    /// Alerts that an undefined metric was encountered
    ///
    /// # Arguments
    ///
    /// * `name` - metric name
    ///
    pub fn undefined_metric(name: impl Into<String>) -> EvalError {
        EvalError::UndefinedMetric { name: name.into() }
    }
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for EvalError {}
