use crate::util;
use crate::numeric::Float;
use crate::error::EvalError;

///
/// Computes the mean squared error between scores and labels
///
/// # Arguments
///
/// * `scores` the score vector
/// * `label` the label vector
///
pub fn mse<T: Float>(scores: &Vec<T>, labels: &Vec<T>) -> Result<T, EvalError> {
    util::validate_data(scores, labels).and_then(|_| {
        Ok(scores.iter().zip(labels.iter()).fold(T::zero(), |sum, (&a, &b)| {
            let diff = a - b;
            sum + (diff * diff)
        }) / T::from_usize(scores.len()))
    })
}

///
/// Computes the root mean squared error between scores and labels
///
/// # Arguments
///
/// * `scores` the score vector
/// * `label` the label vector
///
pub fn rmse<T: Float>(scores: &Vec<T>, labels: &Vec<T>) -> Result<T, EvalError> {
    mse(scores, labels).map(|m| m.sqrt())
}

///
/// Computes the mean absolute error between scores and labels
///
/// # Arguments
///
/// * `scores` the score vector
/// * `label` the label vector
///
pub fn mae<T: Float>(scores: &Vec<T>, labels: &Vec<T>) -> Result<T, EvalError> {
    util::validate_data(scores, labels).and_then(|_| {
        Ok(scores.iter().zip(labels.iter()).fold(T::zero(), |sum, (&a, &b)| {
            sum + (a - b).abs()
        }) / T::from_usize(scores.len()))
    })
}

///
/// Computes the coefficient of determination between scores and labels
///
/// # Arguments
///
/// * `scores` the score vector
/// * `label` the label vector
///
pub fn rsq<T: Float>(scores: &Vec<T>, labels: &Vec<T>) -> Result<T, EvalError> {
    util::validate_data(scores, labels).and_then(|_| {
        let length = scores.len();
        let label_sum = labels.iter().fold(T::zero(), |s, &v| {s + v});
        let label_mean =  label_sum / T::from_usize(length);
        let den = labels.iter().fold(T::zero(), |sse, &label| {
            sse + (label - label_mean) * (label - label_mean)
        }) / T::from_usize(length);
        if den == T::zero() {
            Err(EvalError::new("Unable to compute r-square due to constant data"))
        } else {
            mse(scores, labels).map(|m| T::one() - (m / den))
        }
    })
}

///
/// Computes the linear correlation between scores and labels
///
/// # Arguments
///
/// * `scores` the score vector
/// * `label` the label vector
///
pub fn corr<T: Float>(scores: &Vec<T>, labels: &Vec<T>) -> Result<T, EvalError> {

    util::validate_data(scores, labels).and_then(|_| {
        let length = scores.len();

        let x_mean = scores.iter().fold(T::zero(), |sum, &v| {sum + v}) / T::from_usize(length);
        let y_mean = labels.iter().fold(T::zero(), |sum, &v| {sum + v}) / T::from_usize(length);

        let mut sxx = T::zero();
        let mut syy = T::zero();
        let mut sxy = T::zero();

        scores.iter().zip(labels.iter()).for_each(|(&x, &y)| {
            let x_diff = x - x_mean;
            let y_diff = y - y_mean;
            sxx += x_diff * x_diff;
            syy += y_diff * y_diff;
            sxy += x_diff * y_diff;
        });

        let den = (sxx * syy).sqrt();
        if den == T::zero() {
            Err(EvalError::new("Unable to compute correlation due to constant data"))
        } else {
            Ok(sxy / den)
        }
    })
}

#[cfg(test)]
mod tests {

    use assert_approx_eq::assert_approx_eq;
    use super::*;

    fn data() -> (Vec<f64>, Vec<f64>) {
        let scores= vec![0.5, 0.2, 0.7, 0.4, 0.1, 0.3, 0.8, 0.9];
        let labels= vec![0.3, 0.1, 0.5, 0.6, 0.2, 0.5, 0.7, 0.6];
        (scores, labels)
    }

    #[test]
    fn test_mse() {
        let (scores, labels) = data();
        assert_approx_eq!(mse(&scores, &labels).unwrap(), 0.035)
    }

    #[test]
    fn test_mse_empty() {
        assert!(mse(&Vec::<f64>::new(), &Vec::<f64>::new()).is_err())
    }

    #[test]
    fn test_mse_unequal_length() {
        assert!(mse(&vec![0.1, 0.2], &vec![0.3, 0.5, 0.8]).is_err())
    }

    #[test]
    fn test_mse_constant() {
        assert_approx_eq!(mse(&vec![1.0; 10], &vec![1.0; 10]).unwrap(), 0.0)
    }

    #[test]
    fn test_rmse() {
        let (scores, labels) = data();
        assert_approx_eq!(rmse(&scores, &labels).unwrap(), 0.035.sqrt())
    }

    #[test]
    fn test_rmse_empty() {
        assert!(rmse(&Vec::<f64>::new(), &Vec::<f64>::new()).is_err())
    }

    #[test]
    fn test_rmse_unequal_length() {
        assert!(rmse(&vec![0.1, 0.2], &vec![0.3, 0.5, 0.8]).is_err())
    }

    #[test]
    fn test_rmse_constant() {
        assert_approx_eq!(rmse(&vec![1.0; 10], &vec![1.0; 10]).unwrap(), 0.0)
    }

    #[test]
    fn test_mae() {
        let (scores, labels) = data();
        assert_approx_eq!(mae(&scores, &labels).unwrap(), 0.175)
    }

    #[test]
    fn test_mae_empty() {
        assert!(mae(&Vec::<f64>::new(), &Vec::<f64>::new()).is_err())
    }

    #[test]
    fn test_mae_unequal_length() {
        assert!(mae(&vec![0.1, 0.2], &vec![0.3, 0.5, 0.8]).is_err())
    }

    #[test]
    fn test_mae_constant() {
        assert_approx_eq!(mae(&vec![1.0; 10], &vec![1.0; 10]).unwrap(), 0.0)
    }

    #[test]
    fn test_rsq() {
        let (scores, labels) = data();
        assert_approx_eq!(rsq(&scores, &labels).unwrap(), 0.12156862745098007)
    }

    #[test]
    fn test_rsq_empty() {
        assert!(rsq(&Vec::<f64>::new(), &Vec::<f64>::new()).is_err())
    }

    #[test]
    fn test_rsq_unequal_length() {
        assert!(rsq(&vec![0.1, 0.2], &vec![0.3, 0.5, 0.8]).is_err())
    }

    #[test]
    fn test_rsq_constant() {
        assert!(rsq(&vec![1.0; 10], &vec![1.0; 10]).is_err())
    }

    #[test]
    fn test_corr() {
        let (scores, labels) = data();
        assert_approx_eq!(corr(&scores, &labels).unwrap(), 0.7473417080949364)
    }

    #[test]
    fn test_corr_empty() {
        assert!(corr(&Vec::<f64>::new(), &Vec::<f64>::new()).is_err())
    }

    #[test]
    fn test_corr_unequal_length() {
        assert!(corr(&vec![0.1, 0.2], &vec![0.3, 0.5, 0.8]).is_err())
    }

    #[test]
    fn test_corr_constant() {
        assert!(corr(&vec![1.0; 10], &vec![1.0; 10]).is_err())
    }
}
