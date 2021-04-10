
///
/// Enumerated evaluation errors
///
#[derive(Clone, Debug)]
pub enum EvalError {
    InvalidInputError(String),
    NanValueError,
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
    pub fn undefined_metric(metric: &str) -> EvalError {
        EvalError::UndefinedMetricError(String::from(metric))
    }
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::InvalidInputError(msg) => write!(f, "Invalid input: {}", msg),
            EvalError::NanValueError => write!(f, "Encountered NaN value"),
            EvalError::UndefinedMetricError(metric) => write!(f, "Undefined metric: {}", metric)
        }
    }
}

impl std::error::Error for EvalError {}
