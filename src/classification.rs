//!
//! Provides support for both binary and multi-class classification metrics
//!

use std::cmp::Ordering;
use crate::util;
use crate::numeric::Scalar;
use crate::error::EvalError;
use crate::display;

///
/// Confusion matrix for binary classification
///
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BinaryConfusionMatrix {
    /// true positive count
    pub tp_count: usize,
    /// false positive count
    pub fp_count: usize,
    /// true negative count
    pub tn_count: usize,
    /// false negative count
    pub fn_count: usize,
    /// count sum
    sum: usize
}

impl BinaryConfusionMatrix {

    ///
    /// Computes a new binary confusion matrix from the provided scores and labels
    ///
    /// # Arguments
    ///
    /// * `scores` - vector of scores
    /// * `labels` - vector of boolean labels
    /// * `threshold` - decision threshold value for classifying scores
    ///
    /// # Errors
    ///
    /// An invalid input error will be returned if either scores or labels are empty, or if their
    /// lengths do not match. An undefined metric error will be returned if scores contain any value
    /// that is not finite.
    ///
    /// # Examples
    ///
    /// ```
    /// # use eval_metrics::error::EvalError;
    /// # fn main() -> Result<(), EvalError> {
    /// use eval_metrics::classification::BinaryConfusionMatrix;
    /// let scores = vec![0.4, 0.7, 0.1, 0.3, 0.9];
    /// let labels = vec![false, true, false, true, true];
    /// let matrix = BinaryConfusionMatrix::compute(&scores, &labels, 0.5)?;
    /// # Ok(())}
    /// ```
    ///
    pub fn compute<T: Scalar>(scores: &Vec<T>,
                              labels: &Vec<bool>,
                              threshold: T) -> Result<BinaryConfusionMatrix, EvalError> {
        util::validate_input_dims(scores, labels).and_then(|()| {
            let mut counts = [0, 0, 0, 0];
            for (&score, &label) in scores.iter().zip(labels) {
                if !score.is_finite() {
                    return Err(EvalError::infinite_value())
                } else if score >= threshold && label {
                    counts[3] += 1;
                } else if score >= threshold {
                    counts[2] += 1;
                } else if score < threshold && !label {
                    counts[0] += 1;
                } else {
                    counts[1] += 1;
                }
            };
            let sum = counts.iter().sum();
            Ok(BinaryConfusionMatrix {
                tp_count: counts[3],
                fp_count: counts[2],
                tn_count: counts[0],
                fn_count: counts[1],
                sum
            })
        })
    }

    ///
    /// Constructs a binary confusion matrix with the provided counts
    ///
    /// # Arguments
    ///
    /// * `tp_count` - true positive count
    /// * `fp_count` - false positive count
    /// * `tn_count` - true negative count
    /// * `fn_count` - false negative count
    ///
    /// # Errors
    ///
    /// An invalid input error will be returned if all provided counts are zero
    ///
    pub fn from_counts(tp_count: usize,
                       fp_count: usize,
                       tn_count: usize,
                       fn_count: usize) -> Result<BinaryConfusionMatrix, EvalError> {
        match tp_count + fp_count + tn_count + fn_count {
            0 => Err(EvalError::invalid_input("Confusion matrix has all zero counts")),
            sum => Ok(BinaryConfusionMatrix {tp_count, fp_count, tn_count, fn_count, sum})
        }
    }

    ///
    /// Computes accuracy
    ///
    pub fn accuracy(&self) -> Result<f64, EvalError> {
        let num = self.tp_count + self.tn_count;
        match self.sum {
            // This should never happen as long as we prevent empty confusion matrices
            0 => Err(EvalError::undefined_metric("Accuracy")),
            sum => Ok(num as f64 / sum as f64)
        }
    }

    ///
    /// Computes precision
    ///
    pub fn precision(&self) -> Result<f64, EvalError> {
        match self.tp_count + self.fp_count {
            0 => Err(EvalError::undefined_metric("Precision")),
            den => Ok((self.tp_count as f64) / den as f64)
        }
    }

    ///
    /// Computes recall
    ///
    pub fn recall(&self) -> Result<f64, EvalError> {
        match self.tp_count + self.fn_count {
            0 => Err(EvalError::undefined_metric("Recall")),
            den => Ok((self.tp_count as f64) / den as f64)
        }
    }

    ///
    /// Computes F1
    ///
    pub fn f1(&self) -> Result<f64, EvalError> {
        match (self.precision(), self.recall()) {
            (Ok(p), Ok(r)) if p == 0.0 && r == 0.0 => Ok(0.0),
            (Ok(p), Ok(r)) => Ok(2.0 * (p * r) / (p + r)),
            (Err(e), _) => Err(e),
            (_, Err(e)) => Err(e)
        }
    }

    ///
    /// Computes Matthews correlation coefficient (phi)
    ///
    pub fn mcc(&self) -> Result<f64, EvalError> {
        let n = self.sum as f64;
        let s = (self.tp_count + self.fn_count) as f64 / n;
        let p = (self.tp_count + self.fp_count) as f64 / n;
        match (p * s * (1.0 - s) * (1.0 - p)).sqrt() {
            den if den == 0.0 => Err(EvalError::undefined_metric("MCC")),
            den => Ok(((self.tp_count as f64 / n) - s * p) / den)
        }
    }
}

impl std::fmt::Display for BinaryConfusionMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let counts = vec![
            vec![self.tp_count, self.fp_count],
            vec![self.fn_count, self.tn_count]
        ];
        let outcomes = vec![String::from("Positive"), String::from("Negative")];
        write!(f, "{}", display::stringify_confusion_matrix(&counts, &outcomes))
    }
}

///
/// Represents a single point along a roc curve
///
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RocPoint<T: Scalar> {
    /// True positive rate
    pub tp_rate: T,
    /// False positive rate
    pub fp_rate: T,
    /// Score threshold
    pub threshold: T
}

///
/// Represents a full roc curve
///
#[derive(Clone, Debug)]
pub struct RocCurve<T: Scalar> {
    /// Roc curve points
    pub points: Vec<RocPoint<T>>,
    /// Length
    dim: usize
}

impl <T: Scalar> RocCurve<T> {

    ///
    /// Computes the roc curve from the provided data
    ///
    /// # Arguments
    ///
    /// * `scores` - vector of scores
    /// * `labels` - vector of labels
    ///
    /// # Errors
    ///
    /// An invalid input error will be returned if either scores or labels are empty or contain a
    /// single data point, or if their lengths do not match. An undefined metric error will be
    /// returned if scores contain any value that is not finite or if labels are all constant.
    ///
    /// # Examples
    ///
    /// ```
    /// # use eval_metrics::error::EvalError;
    /// # fn main() -> Result<(), EvalError> {
    /// use eval_metrics::classification::RocCurve;
    /// let scores = vec![0.4, 0.7, 0.1, 0.3, 0.9];
    /// let labels = vec![false, true, false, true, true];
    /// let roc = RocCurve::compute(&scores, &labels)?;
    /// # Ok(())}
    /// ```
    ///
    pub fn compute(scores: &Vec<T>, labels: &Vec<bool>) -> Result<RocCurve<T>, EvalError> {
        util::validate_input_dims(scores, labels).and_then(|()| {
            // roc not defined for a single data point
            let n = match scores.len() {
                1 => return Err(EvalError::invalid_input(
                    "Unable to compute roc curve on single data point"
                )),
                len => len
            };
            let (mut pairs, np) = create_pairs(scores, labels)?;
            let nn = n - np;
            sort_pairs_descending(&mut pairs);
            let mut tpc = if pairs[0].1 {1} else {0};
            let mut fpc = 1 - tpc;
            let mut points = Vec::<RocPoint<T>>::new();
            let mut last_tpr = T::zero();
            let mut last_fpr = T::zero();
            let mut trend: Option<RocTrend> = None;

            for i in 1..n {
                if pairs[i].0 != pairs[i-1].0 {
                    let tp_rate = T::from_usize(tpc) / T::from_usize(np);
                    let fp_rate = T::from_usize(fpc) / T::from_usize(nn);
                    if !tp_rate.is_finite() || !fp_rate.is_finite() {
                        return Err(EvalError::undefined_metric("ROC"))
                    }
                    let threshold = pairs[i-1].0;
                    match trend {
                        Some(RocTrend::Horizontal) => if tp_rate > last_tpr {
                            points.push(RocPoint {tp_rate, fp_rate, threshold});
                        } else if let Some(mut point) = points.last_mut() {
                            point.fp_rate = fp_rate;
                            point.threshold = threshold;
                        },
                        Some(RocTrend::Vertical) => if fp_rate > last_fpr {
                            points.push(RocPoint {tp_rate, fp_rate, threshold})
                        } else if let Some(mut point) = points.last_mut() {
                            point.tp_rate = tp_rate;
                            point.threshold = threshold;
                        },
                        _ => points.push(RocPoint {tp_rate, fp_rate, threshold}),
                    }

                    trend = if fp_rate > last_fpr && tp_rate == last_tpr {
                        Some(RocTrend::Horizontal)
                    } else if tp_rate > last_tpr && fp_rate == last_fpr {
                        Some(RocTrend::Vertical)
                    } else {
                        Some(RocTrend::Diagonal)
                    };
                    last_tpr = tp_rate;
                    last_fpr = fp_rate;
                }
                if pairs[i].1 {
                    tpc += 1;
                } else {
                    fpc += 1;
                }
            }

            if let Some(mut point) = points.last_mut() {
                if point.tp_rate != T::one() || point.fp_rate != T::one() {
                    let threshold = pairs.last().unwrap().0;
                    match trend {
                        Some(RocTrend::Horizontal) if point.tp_rate == T::one() => {
                            point.fp_rate = T::one();
                            point.threshold = threshold;
                        },
                        Some(RocTrend::Vertical) if point.fp_rate == T::one() => {
                            point.tp_rate = T::one();
                            point.threshold = threshold;
                        }
                        _ => points.push(RocPoint {
                            tp_rate: T::one(), fp_rate: T::one(), threshold
                        })
                    }
                }
            }

            match points.len() {
                0 => Err(EvalError::constant_input_data()),
                dim => Ok(RocCurve {points, dim})
            }
        })
    }

    ///
    /// Computes AUC from the roc curve
    ///
    pub fn auc(&self) -> T {
        let mut val = self.points[0].tp_rate * self.points[0].fp_rate / T::from_f64(2.0);
        for i in 1..self.dim {
            let fpr_diff = self.points[i].fp_rate - self.points[i-1].fp_rate;
            let a = self.points[i-1].tp_rate * fpr_diff;
            let tpr_diff = self.points[i].tp_rate - self.points[i-1].tp_rate;
            let b = tpr_diff * fpr_diff / T::from_f64(2.0);
            val += a + b;
        }
        return val
    }
}

///
/// Represents a single point along a precision-recall curve
///
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PrPoint<T: Scalar> {
    /// Precision value
    pub precision: T,
    /// Recall value
    pub recall: T,
    /// Score threshold
    pub threshold: T
}

///
/// Represents a full precision-recall curve
///
#[derive(Clone, Debug)]
pub struct PrCurve<T: Scalar> {
    /// PR curve points
    pub points: Vec<PrPoint<T>>,
    /// Length
    dim: usize
}

impl <T: Scalar> PrCurve<T> {

    ///
    /// Computes the precision-recall curve from the provided data
    ///
    /// # Arguments
    ///
    /// * `scores` - vector of scores
    /// * `labels` - vector of labels
    ///
    /// # Errors
    ///
    /// An invalid input error will be returned if either scores or labels are empty or contain a
    /// single data point, or if their lengths do not match. An undefined metric error will be
    /// returned if scores contain any value that is not finite, or if labels are all false.
    ///
    /// # Examples
    ///
    /// ```
    /// # use eval_metrics::error::EvalError;
    /// # fn main() -> Result<(), EvalError> {
    /// use eval_metrics::classification::PrCurve;
    /// let scores = vec![0.4, 0.7, 0.1, 0.3, 0.9];
    /// let labels = vec![false, true, false, true, true];
    /// let pr = PrCurve::compute(&scores, &labels)?;
    /// # Ok(())}
    /// ```
    ///
    pub fn compute(scores: &Vec<T>, labels: &Vec<bool>) -> Result<PrCurve<T>, EvalError> {
        util::validate_input_dims(scores, labels).and_then(|()| {
            let n = match scores.len() {
                1 => return Err(EvalError::invalid_input(
                    "Unable to compute pr curve on single data point"
                )),
                len => len
            };
            let (mut pairs, mut fnc) = create_pairs(scores, labels)?;
            sort_pairs_descending(&mut pairs);
            let mut tpc = 0;
            let mut fpc = 0;
            let mut points = Vec::<PrPoint<T>>::new();
            let mut last_rec = T::zero();

            for i in 0..n {
                if pairs[i].1 {
                    tpc += 1;
                    fnc -= 1;
                } else {
                    fpc += 1;
                }
                if (i < n-1 && pairs[i].0 != pairs[i+1].0) || i == n-1 {
                    let precision = T::from_usize(tpc) / T::from_usize(tpc + fpc);
                    let recall = T::from_usize(tpc) / T::from_usize(tpc + fnc);
                    if !precision.is_finite() || !recall.is_finite() {
                        return Err(EvalError::undefined_metric("PR"))
                    }
                    let threshold = pairs[i].0;
                    if recall != last_rec {
                        points.push(PrPoint {precision, recall, threshold});
                    }
                    last_rec = recall;
                }
            }

            let dim = points.len();
            Ok(PrCurve {points, dim})
        })
    }

    ///
    /// Computes average precision from the PR curve
    ///
    pub fn ap(&self) -> T {
        let mut val = self.points[0].precision * self.points[0].recall;
        for i in 1..self.dim {
            let rec_diff = self.points[i].recall - self.points[i-1].recall;
            val += rec_diff * self.points[i].precision;
        }
        return val;
    }
}


///
/// Confusion matrix for multi-class classification, in which rows represent predicted counts and
/// columns represent labeled counts
///
#[derive(Clone, Debug, Eq, PartialEq)]
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
    /// * `scores` - vector of class scores
    /// * `labels` - vector of class labels (indexed at zero)
    ///
    /// # Errors
    ///
    /// An invalid input error will be returned if either scores or labels are empty, or if their
    /// lengths do not match. An undefined metric error will be returned if scores contain any value
    /// that is not finite.
    ///
    /// # Examples
    ///
    /// ```
    /// # use eval_metrics::error::EvalError;
    /// # fn main() -> Result<(), EvalError> {
    /// use eval_metrics::classification::MultiConfusionMatrix;
    /// let scores = vec![
    ///    vec![0.3, 0.1, 0.6],
    ///    vec![0.5, 0.2, 0.3],
    ///    vec![0.2, 0.7, 0.1],
    ///    vec![0.3, 0.3, 0.4],
    ///    vec![0.5, 0.1, 0.4],
    ///    vec![0.8, 0.1, 0.1],
    ///    vec![0.3, 0.5, 0.2]
    /// ];
    /// let labels = vec![2, 1, 1, 2, 0, 2, 0];
    /// let matrix = MultiConfusionMatrix::compute(&scores, &labels)?;
    /// # Ok(())}
    /// ```
    ///
    pub fn compute<T: Scalar>(scores: &Vec<Vec<T>>,
                              labels: &Vec<usize>) -> Result<MultiConfusionMatrix, EvalError> {
        util::validate_input_dims(scores, labels).and_then(|()| {
            let dim = scores[0].len();
            let mut counts = vec![vec![0; dim]; dim];
            let mut sum = 0;
            for (i, s) in scores.iter().enumerate() {
                if s.iter().any(|v| !v.is_finite()) {
                    return Err(EvalError::infinite_value())
                } else if s.len() != dim {
                    return Err(EvalError::invalid_input("Inconsistent score dimension"))
                } else if labels[i] >= dim {
                    return Err(EvalError::invalid_input("Labels have more classes than scores"))
                }
                let ind = s.iter().enumerate().max_by(|(_, a), (_, b)| {
                    a.partial_cmp(b).unwrap_or(Ordering::Equal)
                }).map(|(mi, _)| mi).ok_or(EvalError::constant_input_data())?;
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
    /// * `counts` - vector of vector of counts, where each inner vector represents a row in the
    /// confusion matrix
    ///
    /// # Errors
    ///
    /// An invalid input error will be returned if the counts are not a square matrix, or if the
    /// counts are all zero
    ///
    /// # Examples
    ///
    /// ```
    /// # use eval_metrics::error::EvalError;
    /// # fn main() -> Result<(), EvalError> {
    /// use eval_metrics::classification::MultiConfusionMatrix;
    /// let counts = vec![
    ///     vec![8, 3, 2],
    ///     vec![1, 5, 3],
    ///     vec![2, 1, 9]
    /// ];
    /// let matrix = MultiConfusionMatrix::from_counts(counts)?;
    /// # Ok(())}
    /// ```
    ///
    pub fn from_counts(counts: Vec<Vec<usize>>) -> Result<MultiConfusionMatrix, EvalError> {
        let dim = counts.len();
        let mut sum = 0;
        for row in &counts {
            sum += row.iter().sum::<usize>();
            if row.len() != dim {
                let msg = format!("Inconsistent column length ({})", row.len());
                return Err(EvalError::invalid_input(msg.as_str()));
            }
        }
        if sum == 0 {
            Err(EvalError::invalid_input("Confusion matrix has all zero counts"))
        } else {
            Ok(MultiConfusionMatrix {dim, counts, sum})
        }
    }

    ///
    /// Computes accuracy
    ///
    pub fn accuracy(&self) -> Result<f64, EvalError> {
        match self.sum {
            // This should never happen as long as we prevent empty confusion matrices
            0 => Err(EvalError::undefined_metric("Accuracy")),
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
    /// Computes precision, which necessarily requires a specified averaging method
    ///
    /// # Arguments
    ///
    /// * `avg` - averaging method, which can be either 'Macro' or 'Weighted'
    ///
    pub fn precision(&self, avg: &Averaging) -> Result<f64, EvalError> {
        self.agg_metric(&self.per_class_precision(), avg)
    }

    ///
    /// Computes recall, which necessarily requires a specified averaging method
    ///
    /// # Arguments
    ///
    /// * `avg` - averaging method, which can be either 'Macro' or 'Weighted'
    ///
    pub fn recall(&self, avg: &Averaging) -> Result<f64, EvalError> {
        self.agg_metric(&self.per_class_recall(), avg)
    }

    ///
    /// Computes F1, which necessarily requires a specified averaging method
    ///
    /// # Arguments
    ///
    /// * `avg` - averaging method, which can be either 'Macro' or 'Weighted'
    ///
    pub fn f1(&self, avg: &Averaging) -> Result<f64, EvalError> {
        self.agg_metric(&self.per_class_f1(), avg)
    }

    ///
    /// Computes Rk, also known as the multi-class Matthews correlation coefficient following the
    /// approach of Gorodkin in "Comparing two K-category assignments by a K-category correlation
    /// coefficient" (2004)
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
        let num = c * s - tp;
        let den = (s * s - pp).sqrt() * (s * s - tt).sqrt();

        if den == 0.0 {
            Err(EvalError::undefined_metric("Rk"))
        } else {
            Ok(num / den)
        }
    }

    ///
    /// Computes per-class accuracy, resulting in a vector of values for each class
    ///
    pub fn per_class_accuracy(&self) -> Vec<Result<f64, EvalError>> {
        self.per_class_binary_metric("accuracy")
    }

    ///
    /// Computes per-class precision, resulting in a vector of values for each class
    ///
    pub fn per_class_precision(&self) -> Vec<Result<f64, EvalError>> {
        self.per_class_binary_metric("precision")
    }

    ///
    /// Computes per-class recall, resulting in a vector of values for each class
    ///
    pub fn per_class_recall(&self) -> Vec<Result<f64, EvalError>> {
        self.per_class_binary_metric("recall")
    }

    ///
    /// Computes per-class F1, resulting in a vector of values for each class
    ///
    pub fn per_class_f1(&self) -> Vec<Result<f64, EvalError>> {
        self.per_class_binary_metric("f1")
    }

    ///
    /// Computes per-class MCC, resulting in a vector of values for each class
    ///
    pub fn per_class_mcc(&self) -> Vec<Result<f64, EvalError>> {
        self.per_class_binary_metric("mcc")
    }

    fn per_class_binary_metric(&self, metric: &str) -> Vec<Result<f64, EvalError>> {
        (0..self.dim).map(|k| {
            let (mut tpc, mut fpc, mut tnc, mut fnc) = (0, 0, 0, 0);
            for i in 0..self.dim {
                for j in 0..self.dim {
                    let count = self.counts[i][j];
                    if i == k && j == k {
                        tpc = count;
                    } else if i == k {
                        fpc += count;
                    } else if j == k {
                        fnc += count;
                    } else {
                        tnc += count;
                    }
                }
            }
            let matrix = BinaryConfusionMatrix::from_counts(tpc, fpc, tnc, fnc)?;
            match metric {
                "accuracy" => matrix.accuracy(),
                "precision" => matrix.precision(),
                "recall" => matrix.recall(),
                "f1" => matrix.f1(),
                "mcc" => matrix.mcc(),
                other => Err(EvalError::invalid_metric(other))
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

impl std::fmt::Display for MultiConfusionMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.dim <= 25 {
            let outcomes = (0..self.dim).map(|i| format!("Class-{}", i + 1)).collect();
            write!(f, "{}", display::stringify_confusion_matrix(&self.counts, &outcomes))
        } else {
            write!(f, "[Confusion matrix is too large to display]")
        }
    }
}

///
/// Computes multi-class AUC as described by Hand and Till in "A Simple Generalisation of the Area
/// Under the ROC Curve for Multiple Class Classification Problems" (2001)
///
/// # Arguments
///
/// * `scores` - vector of class scores
/// * `labels` - vector of class labels
///
/// # Errors
///
/// An invalid input error will be returned if either scores or labels are empty or contain a
/// single data point, or if their lengths do not match. An undefined metric error will be
/// returned if scores contain any value that is not finite, or if any pairwise roc curve is not
/// defined for all distinct class label pairs.
///
/// # Examples
///
/// ```
/// # use eval_metrics::error::EvalError;
/// # fn main() -> Result<(), EvalError> {
/// use eval_metrics::classification::m_auc;
/// let scores = vec![
///    vec![0.3, 0.1, 0.6],
///    vec![0.5, 0.2, 0.3],
///    vec![0.2, 0.7, 0.1],
///    vec![0.3, 0.3, 0.4],
///    vec![0.5, 0.1, 0.4],
///    vec![0.8, 0.1, 0.1],
///    vec![0.3, 0.5, 0.2]
/// ];
/// let labels = vec![2, 1, 1, 2, 0, 2, 0];
/// let metric = m_auc(&scores, &labels)?;
/// # Ok(())}
/// ```

pub fn m_auc<T: Scalar>(scores: &Vec<Vec<T>>, labels: &Vec<usize>) -> Result<T, EvalError> {
    util::validate_input_dims(scores, labels).and_then(|()| {
        let dim = scores[0].len();
        let mut m_sum = T::zero();

        fn subset<T: Scalar>(scr: &Vec<Vec<T>>,
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
                let ajk = RocCurve::compute(&k_scores, &k_labels)?.auc();
                let (j_scores, j_labels) = subset(scores, labels, k, j);
                let akj = RocCurve::compute(&j_scores, &j_labels)?.auc();
                m_sum += (ajk + akj) / T::from_f64(2.0);
            }
        }
        Ok(m_sum * T::from_f64(2.0) / (T::from_usize(dim) * (T::from_usize(dim) - T::one())))
    })
}

///
/// Specifies the averaging method to use for computing multi-class metrics
///
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Averaging {
    /// Macro average, in which the individual metrics for each class are weighted uniformly
    Macro,
    /// Weighted average, in which the individual metrics for each class are weighted by the number
    /// of occurrences of that class
    Weighted
}

enum RocTrend {
    Horizontal,
    Vertical,
    Diagonal
}

fn create_pairs<T: Scalar>(scores: &Vec<T>,
                           labels: &Vec<bool>) -> Result<(Vec<(T, bool)>, usize), EvalError> {
    let n = scores.len();
    let mut pairs = Vec::with_capacity(n);
    let mut num_pos = 0;

    for i in 0..n {
        if !scores[i].is_finite() {
            return Err(EvalError::infinite_value())
        } else if labels[i] {
            num_pos += 1;
        }
        pairs.push((scores[i], labels[i]))
    }
    Ok((pairs, num_pos))
}

fn sort_pairs_descending<T: Scalar>(pairs: &mut Vec<(T, bool)>) {
    pairs.sort_unstable_by(|(s1, _), (s2, _)| {
        if s1 > s2 {
            Ordering::Less
        } else if s1 < s2 {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });
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
        assert_eq!(matrix.tp_count, 2);
        assert_eq!(matrix.fp_count, 2);
        assert_eq!(matrix.tn_count, 3);
        assert_eq!(matrix.fn_count, 1);
    }

    #[test]
    fn test_binary_confusion_matrix_empty() {
        assert!(BinaryConfusionMatrix::compute(
            &Vec::<f64>::new(),
            &Vec::<bool>::new(),
            0.5
        ).is_err());
    }

    #[test]
    fn test_binary_confusion_matrix_unequal_length() {
        assert!(BinaryConfusionMatrix::compute(
            &vec![0.1, 0.2],
            &vec![true, false, true],
            0.5
        ).is_err());
    }

    #[test]
    fn test_binary_confusion_matrix_nan() {
        assert!(BinaryConfusionMatrix::compute(
            &vec![f64::NAN, 0.2, 0.4],
            &vec![true, false, true],
            0.5
        ).is_err());
    }

    #[test]
    fn test_binary_confusion_matrix_with_counts() {
        let matrix = BinaryConfusionMatrix::from_counts(2, 4, 5, 3).unwrap();
        assert_eq!(matrix.tp_count, 2);
        assert_eq!(matrix.fp_count, 4);
        assert_eq!(matrix.tn_count, 5);
        assert_eq!(matrix.fn_count, 3);
        assert_eq!(matrix.sum, 14);
        assert!(BinaryConfusionMatrix::from_counts(0, 0, 0, 0).is_err())
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
        assert!(BinaryConfusionMatrix::compute(
            &Vec::<f64>::new(),
            &Vec::<bool>::new(),
            0.5
        ).is_err());
    }

    #[test]
    fn test_binary_precision_unequal_length() {
        assert!(BinaryConfusionMatrix::compute(
            &vec![0.1, 0.2],
            &vec![true, false, true],
            0.5
        ).is_err());
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
        assert!(BinaryConfusionMatrix::compute(
            &Vec::<f64>::new(),
            &Vec::<bool>::new(),
            0.5
        ).is_err());
    }

    #[test]
    fn test_binary_recall_unequal_length() {
        assert!(BinaryConfusionMatrix::compute(
            &vec![0.1, 0.2],
            &vec![true, false, true],
            0.5
        ).is_err());
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
        assert!(BinaryConfusionMatrix::compute(
            &Vec::<f64>::new(),
            &Vec::<bool>::new(),
            0.5
        ).is_err());
    }

    #[test]
    fn test_binary_f1_unequal_length() {
        assert!(BinaryConfusionMatrix::compute(
            &vec![0.1, 0.2],
            &vec![true, false, true],
            0.5
        ).is_err());
    }

    #[test]
    fn test_binary_f1_0p_0r() {
        let scores = vec![0.1, 0.2, 0.7, 0.8];
        let labels = vec![false, true, false, false];

        assert_eq!(BinaryConfusionMatrix::compute(&scores, &labels, 0.5)
                       .unwrap()
                       .f1()
                       .unwrap(), 0.0
        )
    }

    #[test]
    fn test_mcc() {
        let (scores, labels) = binary_data();
        let matrix = BinaryConfusionMatrix::compute(&scores, &labels, 0.5).unwrap();
        assert_approx_eq!(matrix.mcc().unwrap(), 0.2581988897471611)
    }

    #[test]
    fn test_roc() {
        let (scores, labels) = binary_data();
        let roc = RocCurve::compute(&scores, &labels).unwrap();

        assert_eq!(roc.dim, 5);
        assert_approx_eq!(roc.points[0].tp_rate, 1.0 / 3.0);
        assert_approx_eq!(roc.points[0].fp_rate, 0.0);
        assert_approx_eq!(roc.points[0].threshold, 0.9);
        assert_approx_eq!(roc.points[1].tp_rate, 1.0 / 3.0);
        assert_approx_eq!(roc.points[1].fp_rate, 0.2);
        assert_approx_eq!(roc.points[1].threshold, 0.8);
        assert_approx_eq!(roc.points[2].tp_rate, 2.0 / 3.0);
        assert_approx_eq!(roc.points[2].fp_rate, 0.2);
        assert_approx_eq!(roc.points[2].threshold, 0.7);
        assert_approx_eq!(roc.points[3].tp_rate, 2.0 / 3.0);
        assert_approx_eq!(roc.points[3].fp_rate, 1.0);
        assert_approx_eq!(roc.points[3].threshold, 0.2);
        assert_approx_eq!(roc.points[4].tp_rate, 1.0);
        assert_approx_eq!(roc.points[4].fp_rate, 1.0);
        assert_approx_eq!(roc.points[4].threshold, 0.1);
    }

    #[test]
    fn test_roc_tied_scores() {
        let scores = vec![1.0, 0.1, 1.0, 0.9, 0.5, 0.1, 0.8, 0.9, 1.0, 0.4];
        let labels = vec![true, false, false, false, false, false, true, true, false, false];
        let roc = RocCurve::compute(&scores, &labels).unwrap();
        assert_approx_eq!(roc.points[0].tp_rate, 1.0 / 3.0);
        assert_approx_eq!(roc.points[0].fp_rate, 0.2857142857142857);
        assert_approx_eq!(roc.points[0].threshold, 1.0);
        assert_approx_eq!(roc.points[1].tp_rate, 2.0 / 3.0);
        assert_approx_eq!(roc.points[1].fp_rate, 0.42857142857142855);
        assert_approx_eq!(roc.points[1].threshold, 0.9);
        assert_approx_eq!(roc.points[2].tp_rate, 1.0);
        assert_approx_eq!(roc.points[2].fp_rate, 0.42857142857142855);
        assert_approx_eq!(roc.points[2].threshold, 0.8);
        assert_approx_eq!(roc.points[3].tp_rate, 1.0);
        assert_approx_eq!(roc.points[3].fp_rate, 1.0);
        assert_approx_eq!(roc.points[3].threshold, 0.1);
    }

    #[test]
    fn test_roc_empty() {
        assert!(RocCurve::compute(&Vec::<f64>::new(), &Vec::<bool>::new()).is_err());
    }

    #[test]
    fn test_roc_unequal_length() {
        assert!(RocCurve::compute(
            &vec![0.4, 0.5, 0.2],
            &vec![true, false, true, false]
        ).is_err());
    }

    #[test]
    fn test_roc_nan() {
        assert!(RocCurve::compute(
            &vec![0.4, 0.5, 0.2, f64::NAN],
            &vec![true, false, true, false]
        ).is_err());
    }

    #[test]
    fn test_roc_constant_label() {
        let scores = vec![0.1, 0.4, 0.5, 0.7];
        let labels_true = vec![true; 4];
        let labels_false = vec![false; 4];
        assert!(match RocCurve::compute(&scores, &labels_true) {
            Err(err) if err.msg.contains("Undefined") => true,
            _ => false
        });
        assert!(match RocCurve::compute(&scores, &labels_false) {
            Err(err) if err.msg.contains("Undefined") => true,
            _ => false
        });
    }

    #[test]
    fn test_roc_constant_score() {
        let scores = vec![0.4, 0.4, 0.4, 0.4];
        let labels = vec![true, false, true, false];
        assert!(match RocCurve::compute(&scores, &labels) {
            Err(err) if err.msg.contains("Constant") => true,
            _ => false
        });
    }

    #[test]
    fn test_auc() {
        let (scores, labels) = binary_data();
        assert_approx_eq!(RocCurve::compute(&scores, &labels).unwrap().auc(), 0.6);

        let scores2 = vec![0.2, 0.5, 0.5, 0.3];
        let labels2 = vec![false, true, false, true];
        assert_approx_eq!(RocCurve::compute(&scores2, &labels2).unwrap().auc(), 0.625);
    }

    #[test]
    fn test_auc_tied_scores() {
        let scores = vec![0.1, 0.2, 0.3, 0.3, 0.3, 0.7, 0.8];
        let labels1 = vec![false, false, true, false, true, false, true];
        let labels2 = vec![false, false, true, true, false, false, true];
        let labels3 = vec![false, false, false, true, true, false, true];
        assert_approx_eq!(RocCurve::compute(&scores, &labels1).unwrap().auc(), 0.75);
        assert_approx_eq!(RocCurve::compute(&scores, &labels2).unwrap().auc(), 0.75);
        assert_approx_eq!(RocCurve::compute(&scores, &labels3).unwrap().auc(), 0.75);

        let scores2 = vec![1.0, 0.1, 1.0, 0.9, 0.5, 0.1, 0.8, 0.9, 1.0, 0.4];
        let labels4 = vec![true, false, false, false, false, false, true, true, false, false];
        assert_approx_eq!(RocCurve::compute(&scores2, &labels4).unwrap().auc(), 0.6904761904761905);
    }

    #[test]
    fn test_pr() {
        let (scores, labels) = binary_data();
        let pr = PrCurve::compute(&scores, &labels).unwrap();
        assert_approx_eq!(pr.points[0].precision, 1.0);
        assert_approx_eq!(pr.points[0].recall, 1.0 / 3.0);
        assert_approx_eq!(pr.points[0].threshold, 0.9);
        assert_approx_eq!(pr.points[1].precision, 2.0 / 3.0);
        assert_approx_eq!(pr.points[1].recall, 2.0 / 3.0);
        assert_approx_eq!(pr.points[1].threshold, 0.7);
        assert_approx_eq!(pr.points[2].precision, 0.375);
        assert_approx_eq!(pr.points[2].recall, 1.0);
        assert_approx_eq!(pr.points[2].threshold, 0.1);
    }

    #[test]
    fn test_pr_empty() {
        assert!(PrCurve::compute(&Vec::<f64>::new(), &Vec::<bool>::new()).is_err());
    }

    #[test]
    fn test_pr_unequal_length() {
        assert!(PrCurve::compute(&vec![0.4, 0.5, 0.2], &vec![true, false, true, false]).is_err());
    }

    #[test]
    fn test_pr_nan() {
        assert!(PrCurve::compute(
            &vec![0.4, 0.5, 0.2, f64::NAN],
            &vec![true, false, true, false]
        ).is_err());
    }

    #[test]
    fn test_pr_constant_label() {
        let scores = vec![0.1, 0.4, 0.5, 0.7];
        let labels_true = vec![true; 4];
        let labels_false = vec![false; 4];
        assert!(PrCurve::compute(&scores, &labels_true).is_ok());
        assert!(match PrCurve::compute(&scores, &labels_false) {
            Err(err) if err.msg.contains("Undefined") => true,
            _ => false
        });
    }

    #[test]
    fn test_pr_constant_score() {
        let scores = vec![0.4, 0.4, 0.4, 0.4];
        let labels = vec![true, false, true, false];
        assert!(PrCurve::compute(&scores, &labels).is_ok());
    }

    #[test]
    fn test_ap() {
        let (scores, labels) = binary_data();
        assert_approx_eq!(PrCurve::compute(&scores, &labels).unwrap().ap(), 0.6805555555555556);

        let scores2 = vec![0.2, 0.5, 0.5, 0.3];
        let labels2 = vec![false, true, false, true];
        assert_approx_eq!(PrCurve::compute(&scores2, &labels2).unwrap().ap(), 0.58333333333333);
    }

    #[test]
    fn test_ap_tied_scores() {
        let scores = vec![0.1, 0.2, 0.3, 0.3, 0.3, 0.7, 0.8];
        let labels1 = vec![false, false, true, false, true, false, true];
        let labels2 = vec![false, false, true, true, false, false, true];
        let labels3 = vec![false, false, false, true, true, false, true];
        assert_approx_eq!(PrCurve::compute(&scores, &labels1).unwrap().ap(), 0.7333333333333);
        assert_approx_eq!(PrCurve::compute(&scores, &labels2).unwrap().ap(), 0.7333333333333);
        assert_approx_eq!(PrCurve::compute(&scores, &labels3).unwrap().ap(), 0.7333333333333);
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
    fn test_multi_confusion_matrix_nan() {
        assert!(MultiConfusionMatrix::compute(
            &vec![vec![0.2, 0.4, 0.4], vec![0.5, 0.1, 0.4], vec![0.3, 0.7, f64::NAN]],
            &vec![2, 1, 0]
        ).is_err());
    }

    #[test]
    fn test_multi_confusion_matrix_inconsistent_score_dims() {
        let scores = vec![vec![0.2, 0.4, 0.4], vec![0.5, 0.1, 0.4], vec![0.3, 0.7]];
        let labels = vec![2, 1, 0];
        assert!(MultiConfusionMatrix::compute(&scores, &labels).is_err());
    }

    #[test]
    fn test_multi_confusion_matrix_score_label_dim_mismatch() {
        let scores = vec![vec![0.2, 0.4, 0.4], vec![0.5, 0.1, 0.4], vec![0.3, 0.2, 0.5]];
        let labels = vec![2, 3, 0];
        assert!(MultiConfusionMatrix::compute(&scores, &labels).is_err());
    }

    #[test]
    fn test_multi_confusion_matrix_counts() {
        let counts = vec![vec![6, 3, 1], vec![4, 2, 7], vec![5, 2, 8]];
        let matrix = MultiConfusionMatrix::from_counts(counts).unwrap();
        assert_eq!(matrix.dim, 3);
        assert_eq!(matrix.sum, 38);
        assert_eq!(matrix.counts, vec![vec![6, 3, 1], vec![4, 2, 7], vec![5, 2, 8]]);
    }

    #[test]
    fn test_multi_confusion_matrix_bad_counts() {
        let counts = vec![vec![6, 3, 1], vec![4, 2], vec![5, 2, 8]];
        assert!(MultiConfusionMatrix::from_counts(counts).is_err())
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
    fn test_multi_f1_0p_0r() {
        let scores = multi_class_data().0;
        // every prediction is wrong
        let labels = vec![1, 2, 0, 0, 1, 1, 0];

        assert_eq!(MultiConfusionMatrix::compute(&scores, &labels)
                       .unwrap()
                       .f1(&Averaging::Macro)
                       .unwrap(), 0.0
        )
    }

    #[test]
    fn test_rk() {
        let (scores, labels) = multi_class_data();
        let matrix = MultiConfusionMatrix::compute(&scores, &labels).unwrap();
        assert_approx_eq!(matrix.rk().unwrap(), 0.375)
    }

    #[test]
    fn test_per_class_accuracy() {
        let (scores, labels) = multi_class_data();
        let matrix = MultiConfusionMatrix::compute(&scores, &labels).unwrap();
        let pca = matrix.per_class_accuracy();
        assert_eq!(pca.len(), 3);
        assert_approx_eq!(pca[0].as_ref().unwrap(), 0.5714285714285714);
        assert_approx_eq!(pca[1].as_ref().unwrap(), 0.7142857142857143);
        assert_approx_eq!(pca[2].as_ref().unwrap(), 0.8571428571428571);
    }

    #[test]
    fn test_per_class_precision() {
        let (scores, labels) = multi_class_data();
        let matrix = MultiConfusionMatrix::compute(&scores, &labels).unwrap();
        let pcp = matrix.per_class_precision();
        assert_eq!(pcp.len(), 3);
        assert_approx_eq!(pcp[0].as_ref().unwrap(), 0.3333333333333333);
        assert_approx_eq!(pcp[1].as_ref().unwrap(), 0.5);
        assert_approx_eq!(pcp[2].as_ref().unwrap(), 1.0);
        println!("{}", matrix);
    }

    #[test]
    fn test_per_class_recall() {
        let (scores, labels) = multi_class_data();
        let matrix = MultiConfusionMatrix::compute(&scores, &labels).unwrap();
        let pcr = matrix.per_class_recall();
        assert_eq!(pcr.len(), 3);
        assert_approx_eq!(pcr[0].as_ref().unwrap(), 0.5);
        assert_approx_eq!(pcr[1].as_ref().unwrap(), 0.5);
        assert_approx_eq!(pcr[2].as_ref().unwrap(), 0.6666666666666666);
    }

    #[test]
    fn test_per_class_f1() {
        let (scores, labels) = multi_class_data();
        let matrix = MultiConfusionMatrix::compute(&scores, &labels).unwrap();
        let pcf = matrix.per_class_f1();
        assert_eq!(pcf.len(), 3);
        assert_approx_eq!(pcf[0].as_ref().unwrap(), 0.4);
        assert_approx_eq!(pcf[1].as_ref().unwrap(), 0.5);
        assert_approx_eq!(pcf[2].as_ref().unwrap(), 0.8);
    }

    #[test]
    fn test_per_class_mcc() {
        let (scores, labels) = multi_class_data();
        let matrix = MultiConfusionMatrix::compute(&scores, &labels).unwrap();
        let pcm = matrix.per_class_mcc();
        assert_eq!(pcm.len(), 3);
        assert_approx_eq!(pcm[0].as_ref().unwrap(), 0.09128709291752773);
        assert_approx_eq!(pcm[1].as_ref().unwrap(), 0.3);
        assert_approx_eq!(pcm[2].as_ref().unwrap(), 0.7302967433402215);
    }

    #[test]
    fn test_m_auc() {
        let (scores, labels) = multi_class_data();
        assert_approx_eq!(m_auc(&scores, &labels).unwrap(), 0.673611111111111)
    }

    #[test]
    fn test_m_auc_empty() {
        assert!(m_auc(&Vec::<Vec<f64>>::new(), &Vec::<usize>::new()).is_err());
    }

    #[test]
    fn test_m_auc_unequal_length() {
        assert!(m_auc(&Vec::<Vec<f64>>::new(), &vec![3, 0, 1, 2]).is_err());
    }

    #[test]
    fn test_m_auc_nan() {
        let scores = vec![
            vec![0.3, 0.1, 0.6],
            vec![0.5, f64::NAN, 0.3],
            vec![0.2, 0.7, 0.1],
        ];
        // every prediction is wrong
        let labels = vec![1, 2, 0];
        assert!(m_auc(&scores, &labels).is_err());
    }

    #[test]
    fn test_m_auc_constant_label() {
        let scores = vec![
            vec![0.3, 0.1, 0.6],
            vec![0.5, 0.2, 0.3],
            vec![0.2, 0.7, 0.1],
            vec![0.8, 0.1, 0.1],
        ];

        let labels = vec![1, 1, 1, 1];
        assert!(m_auc(&scores, &labels).is_err())
    }
}
