use std::cmp::Ordering;
use crate::util;
use crate::numeric::Float;
use crate::error::EvalError;

///
/// Confusion matrix for binary classification
///
pub struct BinaryConfusionMatrix {
    /// true positive count
    pub tpc: usize,
    /// false positive count
    pub fpc: usize,
    /// true negative count
    pub tnc: usize,
    /// false negative count
    pub fnc: usize,
    /// count sum
    sum: usize
}

impl BinaryConfusionMatrix {

    ///
    /// Computes a new binary confusion matrix from the provided scores and vectors
    ///
    /// # Arguments
    ///
    /// * `scores` a vector of scores scaled to be in the [0, 1] range
    /// * `labels` a vector of boolean labels
    /// * `threshold` the decision threshold value for classifying scores
    ///
    pub fn compute<T: Float>(scores: &Vec<T>,
                             labels: &Vec<bool>,
                             threshold: T) -> Result<BinaryConfusionMatrix, EvalError> {

        util::validate_data(scores, labels).and_then(|_| {
            let mut counts = [0, 0, 0, 0];
            scores.iter().zip(labels).for_each(|(&score, &label)| {
                if score >= threshold && label {
                    counts[3] += 1;
                } else if score >= threshold {
                    counts[2] += 1;
                } else if score < threshold && !label {
                    counts[0] += 1;
                } else {
                    counts[1] += 1;
                }
            });
            let sum = counts.iter().sum();
            Ok(BinaryConfusionMatrix {
                tpc: counts[3],
                fpc: counts[2],
                tnc: counts[0],
                fnc: counts[1],
                sum
            })
        })
    }

    ///
    /// Constructs a binary confusion matrix with the provided counts
    ///
    /// # Arguments
    ///
    /// * `tpc` true positive count
    /// * `fpc` false positive count
    /// * `tnc` true negative count
    /// * `fnc` false negative count
    ///
    pub fn with_counts(tpc: usize,
                       fpc: usize,
                       tnc: usize,
                       fnc: usize) -> Result<BinaryConfusionMatrix, EvalError> {
        // keep this as a Result in case of future implementation changes
        Ok(BinaryConfusionMatrix {tpc, fpc, tnc, fnc, sum: tpc + fpc + tnc + fnc})
    }

    ///
    /// Computes the accuracy metric
    ///
    pub fn accuracy(&self) -> Result<f64, EvalError> {
        let num = self.tpc + self.tnc;
        match self.sum {
            0 => Err(EvalError::new("Unable to compute accuracy (empty counts)")),
            sum => Ok(num as f64 / sum as f64)
        }
    }

    ///
    /// Computes the precision metric
    ///
    pub fn precision(&self) -> Result<f64, EvalError> {
        match self.tpc + self.fpc {
            0 => Err(EvalError::new("Unable to compute precision (TPC + FPC = 0)")),
            den => Ok((self.tpc as f64) / den as f64)
        }
    }

    ///
    /// Computes the recall metric
    ///
    pub fn recall(&self) -> Result<f64, EvalError> {
        match self.tpc + self.fnc {
            0 => Err(EvalError::new("Unable to compute recall (TPC + FNC = 0)")),
            den => Ok((self.tpc as f64) / den as f64)
        }
    }

    ///
    /// Computes the F1 metric
    ///
    pub fn f1(&self) -> Result<f64, EvalError> {
        match (self.precision(), self.recall()) {
            (Ok(p), Ok(r)) => Ok(2.0 * (p * r) / (p + r)),
            (Err(e), _) | (_, Err(e)) => {
                Err(EvalError {msg: format!("Unable to compute F1 due to: {}", e)})
            }
        }
    }

    ///
    /// Computes Matthews correlation coefficient (phi)
    ///
    pub fn mcc(&self) -> Result<f64, EvalError> {
        let n = self.sum as f64;
        let s = (self.tpc + self.fnc) as f64 / n;
        let p = (self.tpc + self.fpc) as f64 / n;
        match (p * s * (1.0 - s) * (1.0 - p)).sqrt() {
            0.0 => Err(EvalError::new("Undefined MCC Metric")),
            den => Ok(((self.tpc as f64 / n) - s * p) / den)
        }
    }
}

///
/// Confusion matrix for multi-class classification, in which predicted counts constitute the rows,
/// and actual (label) counts constitute the columns
///
pub struct MultiConfusionMatrix {
    /// output dimension
    pub dim: usize,
    /// count data
    pub counts: Vec<Vec<usize>>,
    /// count sum
    sum: usize
}

impl MultiConfusionMatrix {

    ///
    /// Computes a new confusion matrix from the provided scores and labels
    ///
    /// # Arguments
    ///
    /// * `scores` the vector of class scores
    /// * `labels` the vector of class labels
    ///
    pub fn compute<T: Float>(scores: &Vec<Vec<T>>,
                             labels: &Vec<usize>) -> Result<MultiConfusionMatrix, EvalError> {

        util::validate_data(scores, labels).and_then(|_| {
            let dim = scores[0].len();
            let mut counts = vec![vec![0; dim]; dim];
            let mut sum = 0;
            for (i, s) in scores.iter().enumerate() {
                let ind = s.iter().enumerate().max_by(|(_, a), (_, b)| {
                    a.partial_cmp(b).unwrap_or(Ordering::Equal)
                }).map(|(mi, _)| mi).ok_or(EvalError::new("Invalid score"))?;
                counts[ind][labels[i]] += 1;
                sum += 1;
            }
            Ok(MultiConfusionMatrix {dim, counts, sum})
        })
    }

    ///
    /// Constructs a multi confusion matrix with the provided counts
    ///
    /// # Arguments
    ///
    /// * `counts` a vector of vector of counts, where each inner vector represents a row in the
    /// confusion matrix
    ///
    pub fn with_counts(counts: Vec<Vec<usize>>) -> Result<MultiConfusionMatrix, EvalError> {
        let dim = counts.len();
        let mut sum = 0;
        for row in &counts {
            sum += row.iter().sum::<usize>();
            if row.len() != dim {
                return Err(EvalError {
                    msg: format!("Invalid column dim ({}) for row dim ({})", row.len(), dim)
                })
            }
        }
        Ok(MultiConfusionMatrix {dim, counts, sum})
    }

    ///
    /// Computes the accuracy metric
    ///
    pub fn accuracy(&self) -> Result<f64, EvalError> {
        match self.sum {
            0 => Err(EvalError::new("Unable to compute accuracy (empty counts)")),
            sum => {
                let mut correct = 0;
                for i in 0..self.dim {
                    correct += self.counts[i][i];
                }
                Ok(correct as f64 / sum as f64)
            }
        }
    }

    ///
    /// Computes the precision metric, which necessarily requires a specified averaging method
    ///
    /// # Arguments
    ///
    /// * `avg` the averaging method, which can be either 'Macro' or 'Weighted'
    ///
    pub fn precision(&self, avg: &Averaging) -> Result<f64, EvalError> {
        self.agg_metric(&self.per_class_precision(), avg)
    }

    ///
    /// Computes the recall metric, which necessarily requires a specified averaging method
    ///
    /// # Arguments
    ///
    /// * `avg` the averaging method, which can be either 'Macro' or 'Weighted'
    ///
    pub fn recall(&self, avg: &Averaging) -> Result<f64, EvalError> {
        self.agg_metric(&self.per_class_recall(), avg)
    }

    ///
    /// Computes the F-1 metric, which necessarily requires a specified averaging method
    ///
    /// # Arguments
    ///
    /// * `avg` the averaging method, which can be either 'Macro' or 'Weighted'
    ///
    pub fn f1(&self, avg: &Averaging) -> Result<f64, EvalError> {
        self.agg_metric(&self.per_class_f1(), avg)
    }

    ///
    /// Computes the Rk metric, also known as the multi-class Matthews correlation coefficient
    /// following the approach in "Comparing two K-category assignments by a K-category
    /// correlation coefficient" (2004)
    ///
    pub fn rk(&self) -> Result<f64, EvalError> {

        let mut t = vec![0.0; self.dim];
        let mut p = vec![0.0; self.dim];
        let mut c = 0.0;
        let s = self.sum as f64;

        for i in 0..self.dim {
            c += self.counts[i][i] as f64;
            for j in 0..self.dim {
                t[j] += self.counts[i][j] as f64;
                p[i] += self.counts[i][j] as f64;
            }
        }

        let tt = t.iter().fold(0.0, |acc, val| acc + (val * val));
        let pp = p.iter().fold(0.0, |acc, val| acc + (val * val));
        let tp = t.iter().zip(p).fold(0.0, |acc, (t_val, p_val)| acc + t_val * p_val);

        let num = (c * s - tp);
        let den = (s * s - pp).sqrt() * (s * s - tt).sqrt();

        if den == 0.0 {
            Err(EvalError::new("Undefined Rk metric"))
        } else {
            Ok(num / den)
        }
    }

    ///
    /// Computes the per-class precision metrics, resulting in a vector of values for each class
    ///
    pub fn per_class_precision(&self) -> Vec<Result<f64, EvalError>> {
        self.per_class_metric(Metric::Precision)
    }

    ///
    /// Computes the per-class recall metrics, resulting in a vector of values for each class
    ///
    pub fn per_class_recall(&self) -> Vec<Result<f64, EvalError>> {
        self.per_class_metric(Metric::Recall)
    }

    ///
    /// Computes the per-class f1 metrics, resulting in a vector of values for each class
    ///
    pub fn per_class_f1(&self) -> Vec<Result<f64, EvalError>> {
        let pcp = self.per_class_precision();
        let pcr = self.per_class_recall();
        pcp.iter().zip(pcr.iter()).map(|pair| {
            match pair {
                (Ok(p), Ok(r)) => Ok(2.0 * (p * r) / (p + r)),
                (Err(e), _) | (_, Err(e)) => {
                    Err(EvalError {msg: format!("Unable to compute per-class F1 due to: {}", e)})
                }
            }
        }).collect()
    }

    fn agg_metric(&self, pcm: &Vec<Result<f64, EvalError>>,
                  avg: &Averaging) -> Result<f64, EvalError> {

        match avg {
            Averaging::Macro => self.macro_metric(pcm),
            Averaging::Weighted => self.weighted_metric(pcm)
        }
    }

    fn per_class_metric(&self, metric: Metric) -> Vec<Result<f64, EvalError>> {
        (0..self.dim).map(|i| {
            let a = self.counts[i][i];
            let b = (0..self.dim).fold(0, |sum, j| {
                sum + match metric {
                    Metric::Precision => self.counts[i][j],
                    Metric::Recall => self.counts[j][i]
                }
            }) - a;
            match a + b {
                0 => Err(EvalError::new("Undefined per-class metric")),
                den => Ok(a as f64 / den as f64)
            }
        }).collect()
    }

    fn macro_metric(&self, pcm: &Vec<Result<f64, EvalError>>) -> Result<f64, EvalError> {
        pcm.iter().try_fold(0.0, |sum, metric| {
            match metric {
                Ok(m) => Ok(sum + m),
                Err(e) => Err(e.clone())
            }
        }).map(|sum| {sum / pcm.len() as f64})
    }

    fn weighted_metric(&self, pcm: &Vec<Result<f64, EvalError>>) -> Result<f64, EvalError> {
        pcm.iter()
            .zip(self.class_counts().iter())
            .try_fold(0.0, |val, (metric, &class)| {
                match metric {
                    Ok(m) => Ok(val + (m * (class as f64) / (self.sum as f64))),
                    Err(e) => Err(e.clone())
                }
            })
    }

    fn class_counts(&self) -> Vec<usize> {
        let mut counts = vec![0; self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                counts[j] += self.counts[i][j];
            }
        }
        counts
    }
}

///
/// Computes the AUC (area under the ROC curve) for binary classification
///
/// # Arguments
///
/// * `scores` the vector of scores
/// * `labels` the vector of labels
///

pub fn auc<T: Float>(scores: &Vec<T>, labels: &Vec<bool>) -> Result<f64, EvalError> {

    util::validate_data(scores, labels).and_then(|_| {
        let length = labels.len();
        let mut pairs: Vec<(&T, &bool)> = scores.iter().zip(labels.iter()).collect();
        pairs.sort_by(|(&s1, _), (&s2, _)| {
            if s1 > s2 {
                Ordering::Greater
            } else if s1 < s2 {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        });

        let mut s0 = 0.0;
        let mut n0 = 0.0;

        for (i, pair) in pairs.iter().enumerate() {
            if *pair.1 {
                n0 += 1.0;
                s0 += i as f64 + 1.0;
            }
        }

        let n1 = length as f64 - n0;
        Ok((s0 - (n0 * (n0 + 1.0)) / 2.0) / (n0 * n1))
    })
}

///
/// Computes the multi-class AUC metric as described by Hand and Till in "A Simple Generalisation
/// of the Area Under the ROC Curve for Multiple Class Classification Problems" (2001)
///
/// # Arguments
///
/// * `scores` the vector of class scores
/// * `labels` the vector of class labels
///

pub fn m_auc<T: Float>(scores: &Vec<Vec<T>>, labels: &Vec<usize>) -> Result<f64, EvalError> {

    util::validate_data(scores, labels).and_then(|_| {
        let dim = scores[0].len();
        let mut m_sum = 0.0;

        fn subset<T: Float>(scr: &Vec<Vec<T>>,
                            lab: &Vec<usize>,
                            j: usize,
                            k: usize) -> (Vec<T>, Vec<bool>) {

            scr.iter().zip(lab.iter()).filter(|(_, &l)| {
                l == j || l == k
            }).map(|(s, &l)| {
                (s[k], l == k)
            }).unzip()
        }

        for j in 0..dim {
            for k in 0..j {
                let (k_scores, k_labels) = subset(scores, labels, j, k);
                let ajk = auc(&k_scores, &k_labels)?;
                let (j_scores, j_labels) = subset(scores, labels, k, j);
                let akj = auc(&j_scores, &j_labels)?;
                m_sum += (ajk + akj) / 2.0;
            }
        }

        Ok(m_sum * 2.0 / (dim * (dim - 1)) as f64)
    })
}

///
/// Specifies the averaging method to use for computing multi-class metrics
///
pub enum Averaging {
    /// Macro average, in which the individual metrics for each class are weighted uniformly
    Macro,
    /// Weighted average, in which the individual metrics for each class are weighted by the number
    /// of occurrences of that label
    Weighted
}

enum Metric {
    Precision,
    Recall
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use super::*;

    fn binary_data() -> (Vec<f64>, Vec<bool>) {
        let scores = vec![0.5, 0.2, 0.7, 0.4, 0.1, 0.3, 0.8, 0.9];
        let labels = vec![false, false, true, false, true, false, false, true];
        (scores, labels)
    }

    fn multi_class_data() -> (Vec<Vec<f64>>, Vec<usize>) {

        let scores = vec![
            vec![0.3, 0.1, 0.6],
            vec![0.5, 0.2, 0.3],
            vec![0.2, 0.7, 0.1],
            vec![0.3, 0.3, 0.4],
            vec![0.5, 0.1, 0.4],
            vec![0.8, 0.1, 0.1],
            vec![0.3, 0.5, 0.2]
        ];
        let labels = vec![2, 1, 1, 2, 0, 2, 0];
        (scores, labels)
    }

    #[test]
    fn test_binary_confusion_matrix() {
        let (scores, labels) = binary_data();
        let matrix = BinaryConfusionMatrix::compute(&scores, &labels, 0.5).unwrap();
        assert_eq!(matrix.tpc, 2);
        assert_eq!(matrix.fpc, 2);
        assert_eq!(matrix.tnc, 3);
        assert_eq!(matrix.fnc, 1);
    }

    #[test]
    fn test_binary_confusion_matrix_empty() {
        assert!(BinaryConfusionMatrix::compute(&Vec::<f64>::new(), &Vec::<bool>::new(), 0.5).is_err());
    }

    #[test]
    fn test_binary_confusion_matrix_unequal_length() {
        assert!(BinaryConfusionMatrix::compute(&vec![0.1, 0.2], &vec![true, false, true], 0.5).is_err());
    }

    #[test]
    fn test_binary_confusion_matrix_with_counts() {
        let matrix = BinaryConfusionMatrix::with_counts(2, 4, 5, 3).unwrap();
        assert_eq!(matrix.tpc, 2);
        assert_eq!(matrix.fpc, 4);
        assert_eq!(matrix.tnc, 5);
        assert_eq!(matrix.fnc, 3);
        assert_eq!(matrix.sum, 14);
    }

    #[test]
    fn test_binary_accuracy() {
        let (scores, labels) = binary_data();
        let matrix = BinaryConfusionMatrix::compute(&scores, &labels, 0.5).unwrap();
        assert_approx_eq!(matrix.accuracy().unwrap(), 0.625);
    }

    #[test]
    fn test_binary_precision() {
        let (scores, labels) = binary_data();
        let matrix = BinaryConfusionMatrix::compute(&scores, &labels, 0.5).unwrap();
        assert_approx_eq!(matrix.precision().unwrap(), 0.5);

        // test edge case where we never predict a positive
        assert!(BinaryConfusionMatrix::compute(
            &vec![0.4, 0.3, 0.1, 0.2, 0.1],
            &vec![true, false, true, false, true],
            0.5
        ).unwrap().precision().is_err());
    }

    #[test]
    fn test_binary_precision_empty() {
        assert!(BinaryConfusionMatrix::compute(&Vec::<f64>::new(), &Vec::<bool>::new(), 0.5).is_err());
    }

    #[test]
    fn test_binary_precision_unequal_length() {
        assert!(BinaryConfusionMatrix::compute(&vec![0.1, 0.2], &vec![true, false, true], 0.5).is_err());
    }

    #[test]
    fn test_binary_recall() {
        let (scores, labels) = binary_data();
        let matrix = BinaryConfusionMatrix::compute(&scores, &labels, 0.5).unwrap();
        assert_approx_eq!(matrix.recall().unwrap(), 2.0 / 3.0);

        // test edge case where we have no positive class
        assert!(BinaryConfusionMatrix::compute(
            &vec![0.4, 0.3, 0.1, 0.8, 0.7],
            &vec![false, false, false, false, false],
            0.5
        ).unwrap().recall().is_err());
    }

    #[test]
    fn test_binary_recall_empty() {
        assert!(BinaryConfusionMatrix::compute(&Vec::<f64>::new(), &Vec::<bool>::new(), 0.5).is_err());
    }

    #[test]
    fn test_binary_recall_unequal_length() {
        assert!(BinaryConfusionMatrix::compute(&vec![0.1, 0.2], &vec![true, false, true], 0.5).is_err());
    }

    #[test]
    fn test_binary_f1() {
        let (scores, labels) = binary_data();
        let matrix = BinaryConfusionMatrix::compute(&scores, &labels, 0.5).unwrap();
        assert_approx_eq!(matrix.f1().unwrap(), 0.5714285714285715);

        // test edge case where we never predict a positive
        assert!(BinaryConfusionMatrix::compute(
            &vec![0.4, 0.3, 0.1, 0.2, 0.1],
            &vec![true, false, true, false, true],
            0.5
        ).unwrap().f1().is_err());

        // test edge case where we have no positive class
        assert!(BinaryConfusionMatrix::compute(
            &vec![0.4, 0.3, 0.1, 0.8, 0.7],
            &vec![false, false, false, false, false],
            0.5
        ).unwrap().f1().is_err());
    }

    #[test]
    fn test_binary_f1_empty() {
        assert!(BinaryConfusionMatrix::compute(&Vec::<f64>::new(), &Vec::<bool>::new(), 0.5).is_err());
    }

    #[test]
    fn test_binary_f1_unequal_length() {
        assert!(BinaryConfusionMatrix::compute(&vec![0.1, 0.2], &vec![true, false, true], 0.5).is_err());
    }

    #[test]
    fn test_auc() {
        let (scores, labels) = binary_data();
        assert_approx_eq!(auc(&scores, &labels).unwrap(), 0.6)
    }

    #[test]
    fn test_auc_empty() {
        assert!(auc(&Vec::<f64>::new(), &Vec::<bool>::new()).is_err());
    }

    #[test]
    fn test_auc_unequal_length() {
        assert!(auc(&vec![0.4, 0.5, 0.2], &vec![true, false, true, false]).is_err());
    }

    #[test]
    fn test_multi_confusion_matrix() {
        let (scores, labels) = multi_class_data();
        let matrix = MultiConfusionMatrix::compute(&scores, &labels).unwrap();
        assert_eq!(matrix.counts, vec![vec![1, 1, 1], vec![1, 1, 0], vec![0, 0, 2]]);
        assert_eq!(matrix.dim, 3);
        assert_eq!(matrix.sum, 7);
    }

    #[test]
    fn test_multi_confusion_matrix_empty() {
        let scores: Vec<Vec<f64>> = vec![];
        let labels = Vec::<usize>::new();
        assert!(MultiConfusionMatrix::compute(&scores, &labels).is_err());
    }

    #[test]
    fn test_multi_confusion_matrix_unequal_length() {
        assert!(MultiConfusionMatrix::compute(&vec![vec![0.2, 0.4, 0.4], vec![0.5, 0.1, 0.4]],
                                      &vec![2, 1, 0]).is_err());
    }

    #[test]
    fn test_multi_confusion_matrix_counts() {
        let counts = vec![vec![6, 3, 1], vec![4, 2, 7], vec![5, 2, 8]];
        let matrix = MultiConfusionMatrix::with_counts(counts).unwrap();
        assert_eq!(matrix.dim, 3);
        assert_eq!(matrix.sum, 38);
        assert_eq!(matrix.counts, vec![vec![6, 3, 1], vec![4, 2, 7], vec![5, 2, 8]]);
    }

    #[test]
    fn test_multi_confusion_matrix_bad_counts() {
        let counts = vec![vec![6, 3, 1], vec![4, 2], vec![5, 2, 8]];
        assert!(MultiConfusionMatrix::with_counts(counts).is_err())
    }

    #[test]
    fn test_multi_accuracy() {
        let (scores, labels) = multi_class_data();
        let matrix = MultiConfusionMatrix::compute(&scores, &labels).unwrap();
        assert_approx_eq!(matrix.accuracy().unwrap(), 0.5714285714285714)
    }

    #[test]
    fn test_multi_precision() {
        let (scores, labels) = multi_class_data();
        let matrix = MultiConfusionMatrix::compute(&scores, &labels).unwrap();
        assert_approx_eq!(matrix.precision(&Averaging::Macro).unwrap(), 0.611111111111111);
        assert_approx_eq!(matrix.precision(&Averaging::Weighted).unwrap(), 2.0 / 3.0);

        assert!(MultiConfusionMatrix::compute(
            &vec![vec![0.6, 0.4, 0.0],
                  vec![0.2, 0.8, 0.0],
                  vec![0.9, 0.1, 0.0],
                  vec![0.3, 0.7, 0.0]],
            &vec![0, 1, 2, 1]
        ).unwrap().precision(&Averaging::Macro).is_err())
    }

    #[test]
    fn test_multi_recall() {
        let (scores, labels) = multi_class_data();
        let matrix = MultiConfusionMatrix::compute(&scores, &labels).unwrap();
        assert_approx_eq!(matrix.recall(&Averaging::Macro).unwrap(), 0.5555555555555555);
        assert_approx_eq!(matrix.recall(&Averaging::Weighted).unwrap(), 0.5714285714285714);

        assert!(MultiConfusionMatrix::compute(
            &vec![vec![0.6, 0.3, 0.1],
                  vec![0.2, 0.5, 0.3],
                  vec![0.8, 0.1, 0.1],
                  vec![0.3, 0.5, 0.2]],
            &vec![0, 1, 0, 1]
        ).unwrap().recall(&Averaging::Macro).is_err())
    }

    #[test]
    fn test_multi_f1() {
        let (scores, labels) = multi_class_data();
        let matrix = MultiConfusionMatrix::compute(&scores, &labels).unwrap();
        assert_approx_eq!(matrix.f1(&Averaging::Macro).unwrap(), 0.5666666666666668);
        assert_approx_eq!(matrix.f1(&Averaging::Weighted).unwrap(), 0.6);

        assert!(MultiConfusionMatrix::compute(
            &vec![vec![0.6, 0.4, 0.0],
                  vec![0.2, 0.8, 0.0],
                  vec![0.3, 0.7, 0.0]],
            &vec![0, 2, 1]
        ).unwrap().f1(&Averaging::Macro).is_err());

        assert!(MultiConfusionMatrix::compute(
            &vec![vec![0.6, 0.3, 0.1],
                  vec![0.2, 0.5, 0.3],
                  vec![0.3, 0.5, 0.2]],
            &vec![1, 0, 1]
        ).unwrap().f1(&Averaging::Macro).is_err());
    }

    #[test]
    fn test_m_auc() {
        let (scores, labels) = multi_class_data();
        assert_approx_eq!(m_auc(&scores, &labels).unwrap(), 0.7222222222222221)
    }

    #[test]
    fn test_m_auc_empty() {
        assert!(m_auc(&Vec::<Vec<f64>>::new(), &Vec::<usize>::new()).is_err());
    }

    #[test]
    fn test_m_auc_unequal_length() {
        assert!(m_auc(&Vec::<Vec<f64>>::new(), &vec![3, 0, 1, 2]).is_err());
    }
}
