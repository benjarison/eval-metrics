//!
//! Provides support for both binary and multi-class classification metrics
//!

use std::cmp::Ordering;
use crate::util;
use crate::numeric::Float;
use crate::error::EvalError;
use crate::display;

///
/// Confusion matrix for binary classification
///
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
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
    /// Computes a new binary confusion matrix from the provided scores and labels
    ///
    /// # Arguments
    ///
    /// * `scores` - vector of scores
    /// * `labels` - vector of boolean labels
    /// * `threshold` - decision threshold value for classifying scores
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
    pub fn compute<T: Float>(scores: &Vec<T>,
                             labels: &Vec<bool>,
                             threshold: T) -> Result<BinaryConfusionMatrix, EvalError> {

        util::validate_input(scores, labels).and_then(|_| {
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
    /// * `tpc` - true positive count
    /// * `fpc` - false positive count
    /// * `tnc` - true negative count
    /// * `fnc` - false negative count
    ///
    pub fn with_counts(tpc: usize,
                       fpc: usize,
                       tnc: usize,
                       fnc: usize) -> Result<BinaryConfusionMatrix, EvalError> {
        match tpc + fpc + tnc + fnc {
            0 => Err(EvalError::invalid_input("Confusion matrix has all zero counts")),
            sum => Ok(BinaryConfusionMatrix {tpc, fpc, tnc, fnc, sum})
        }
    }

    ///
    /// Computes the accuracy metric
    ///
    pub fn accuracy(&self) -> Result<f64, EvalError> {
        let num = self.tpc + self.tnc;
        match self.sum {
            0 => Err(EvalError::undefined_metric("Unable to compute accuracy")),
            sum => Ok(num as f64 / sum as f64)
        }
    }

    ///
    /// Computes the precision metric
    ///
    pub fn precision(&self) -> Result<f64, EvalError> {
        match self.tpc + self.fpc {
            0 => Err(EvalError::undefined_metric("Unable to compute precision")),
            den => Ok((self.tpc as f64) / den as f64)
        }
    }

    ///
    /// Computes the recall metric
    ///
    pub fn recall(&self) -> Result<f64, EvalError> {
        match self.tpc + self.fnc {
            0 => Err(EvalError::undefined_metric("Unable to compute recall")),
            den => Ok((self.tpc as f64) / den as f64)
        }
    }

    ///
    /// Computes the F1 metric
    ///
    pub fn f1(&self) -> Result<f64, EvalError> {
        match (self.precision(), self.recall()) {
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
        let s = (self.tpc + self.fnc) as f64 / n;
        let p = (self.tpc + self.fpc) as f64 / n;
        match (p * s * (1.0 - s) * (1.0 - p)).sqrt() {
            den if den == 0.0 => Err(EvalError::undefined_metric("Unable to compute MCC")),
            den => Ok(((self.tpc as f64 / n) - s * p) / den)
        }
    }
}

impl std::fmt::Display for BinaryConfusionMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let counts = vec![vec![self.tpc, self.fpc], vec![self.fnc, self.tnc]];
        let outcomes = vec![String::from("Positive"), String::from("Negative")];
        write!(f, "{}", display::stringify(&counts, &outcomes))
    }
}

///
/// Represents a single point along a roc curve
///
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RocPoint<T: Float> {
    /// True positive rate
    pub tpr: T,
    /// False positive rate
    pub fpr: T,
    /// Score threshold
    pub threshold: T
}

///
/// Represents a full roc curve
///
#[derive(Clone, Debug)]
pub struct RocCurve<T: Float> {
    /// Roc curve points
    pub points: Vec<RocPoint<T>>,
    /// Length
    dim: usize
}

impl <T: Float> RocCurve<T> {

    ///
    /// Computes the roc curve from the provided data
    ///
    /// # Arguments
    ///
    /// * `scores` - vector of scores
    /// * `labels` - vector of labels
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
        util::validate_input(scores, labels).and_then(|_| {
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

            let mut tp = if pairs[0].1 {1} else {0};
            let mut fp = 1 - tp;
            let mut points = Vec::<RocPoint<T>>::new();
            let mut last_tpr = T::zero();

            for i in 1..n {
                if pairs[i].0 != pairs[i-1].0 {
                    let tpr = T::from_usize(tp) / T::from_usize(np);
                    let fpr = T::from_usize(fp) / T::from_usize(nn);
                    if !tpr.is_finite() || !fpr.is_finite() {
                        return Err(EvalError::infinite_value())
                    }
                    let threshold = pairs[i-1].0;
                    match points.last_mut() {
                        Some(mut point) if (point.fpr - fpr).abs() < T::from_f64(1e-10) => {
                            if point.tpr > last_tpr {
                                point.tpr = tpr;
                                point.threshold = threshold;
                            } else {
                                points.push(RocPoint {tpr, fpr, threshold});
                            }
                        },
                        _ => {
                            points.push(RocPoint {tpr, fpr, threshold});
                            last_tpr = tpr;
                        }
                    }
                }
                if pairs[i].1 {
                    tp += 1;
                } else {
                    fp += 1;
                }
            }

            if let Some(point) = points.last() {
                if point.tpr != T::one() || point.fpr != T::one() {
                    let t = pairs.last().unwrap().0;
                    points.push(RocPoint {tpr: T::one(), fpr: T::one(), threshold: t});
                }
            }

            let dim = points.len();
            Ok(RocCurve {points, dim})
        })
    }

    ///
    /// Computes the AUC from the roc curve
    ///
    pub fn auc(&self) -> Result<T, EvalError> {
        let mut sum = self.points[0].tpr * self.points[0].fpr / T::from_f64(2.0);
        for i in 1..self.dim {
            let fpr_diff = self.points[i].fpr - self.points[i-1].fpr;
            let a = self.points[i-1].tpr * fpr_diff;
            let b = (self.points[i].tpr - self.points[i-1].tpr) * fpr_diff / T::from_f64(2.0);
            sum += a + b;
        }
        return Ok(sum)
    }
}

///
/// Represents a single point along a precision-recall curve
///
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PrPoint<T: Float> {
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
pub struct PrCurve<T: Float> {
    /// PR curve points
    pub points: Vec<PrPoint<T>>,
    /// Length
    dim: usize
}

impl <T: Float> PrCurve<T> {

    ///
    /// Computes the precision-recall curve from the provided data
    ///
    /// # Arguments
    ///
    /// * `scores` - vector of scores
    /// * `labels` - vector of labels
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
        util::validate_input(scores, labels).and_then(|_| {
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
                    let pre = T::from_usize(tpc) / T::from_usize(tpc + fpc);
                    let rec = T::from_usize(tpc) / T::from_usize(tpc + fnc);
                    if !pre.is_finite() || !rec.is_finite() {
                        return Err(EvalError::infinite_value())
                    }
                    let threshold = pairs[i].0;
                    if rec != last_rec {
                        points.push(PrPoint {precision: pre, recall: rec, threshold});
                    }
                    last_rec = rec;
                }
            }

            let dim = points.len();
            Ok(PrCurve {points, dim})
        })
    }

    ///
    /// Computes the average precision metric from the PR curve
    ///
    pub fn ap(&self) -> Result<T, EvalError> {
        let mut avg_pre = self.points[0].precision * self.points[0].recall;
        for i in 1..self.dim {
            let rec_diff = self.points[i].recall - self.points[i-1].recall;
            avg_pre += rec_diff * self.points[i].precision;
        }
        return Ok(avg_pre)
    }
}


///
/// Confusion matrix for multi-class classification, in which predicted counts constitute the rows,
/// and actual (label) counts constitute the columns
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
    pub fn compute<T: Float>(scores: &Vec<Vec<T>>,
                             labels: &Vec<usize>) -> Result<MultiConfusionMatrix, EvalError> {

        util::validate_input(scores, labels).and_then(|_| {
            let dim = scores[0].len();
            let mut counts = vec![vec![0; dim]; dim];
            let mut sum = 0;
            for (i, s) in scores.iter().enumerate() {
                if s.iter().any(|v| !v.is_finite()) {
                    return Err(EvalError::infinite_value())
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
    /// let matrix = MultiConfusionMatrix::with_counts(counts)?;
    /// # Ok(())}
    /// ```
    ///
    pub fn with_counts(counts: Vec<Vec<usize>>) -> Result<MultiConfusionMatrix, EvalError> {
        let dim = counts.len();
        let mut sum = 0;
        for row in &counts {
            sum += row.iter().sum::<usize>();
            if row.len() != dim {
                let msg = format!("Inconsistent column length ({})", row.len());
                return Err(EvalError::InvalidInput(msg));
            }
        }
        Ok(MultiConfusionMatrix {dim, counts, sum})
    }

    ///
    /// Computes the accuracy metric
    ///
    pub fn accuracy(&self) -> Result<f64, EvalError> {
        match self.sum {
            0 => Err(EvalError::undefined_metric("Unable to compute accuracy")),
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
    /// * `avg` - averaging method, which can be either 'Macro' or 'Weighted'
    ///
    pub fn precision(&self, avg: &Averaging) -> Result<f64, EvalError> {
        self.agg_metric(&self.per_class_precision(), avg)
    }

    ///
    /// Computes the recall metric, which necessarily requires a specified averaging method
    ///
    /// # Arguments
    ///
    /// * `avg` - averaging method, which can be either 'Macro' or 'Weighted'
    ///
    pub fn recall(&self, avg: &Averaging) -> Result<f64, EvalError> {
        self.agg_metric(&self.per_class_recall(), avg)
    }

    ///
    /// Computes the F-1 metric, which necessarily requires a specified averaging method
    ///
    /// # Arguments
    ///
    /// * `avg` - averaging method, which can be either 'Macro' or 'Weighted'
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

        let num = c * s - tp;
        let den = (s * s - pp).sqrt() * (s * s - tt).sqrt();

        if den == 0.0 {
            Err(EvalError::undefined_metric("Unable to compute Rk"))
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
                (Ok(p), Ok(r)) if *p == 0.0 && *r == 0.0 => {
                    Err(EvalError::undefined_metric("Unable to compute F1 (precision, recall = 0)"))
                },
                (Ok(p), Ok(r)) => Ok(2.0 * (p * r) / (p + r)),
                (Err(e), _) | (_, Err(e)) => {
                    Err(EvalError::UndefinedMetric(format!("Unable to compute F1 ({})", e)))
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
                0 => Err(EvalError::UndefinedMetric(format!("Unable to compute {}", metric.name()))),
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

impl std::fmt::Display for MultiConfusionMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.dim <= 25 {
            let outcomes = (0..self.dim).map(|i| format!("Class-{}", i + 1)).collect();
            write!(f, "{}", display::stringify(&self.counts, &outcomes))
        } else {
            write!(f, "[Confusion matrix is too large to display]")
        }
    }
}

///
/// Computes the multi-class AUC metric as described by Hand and Till in "A Simple Generalisation
/// of the Area Under the ROC Curve for Multiple Class Classification Problems" (2001)
///
/// # Arguments
///
/// * `scores` - vector of class scores
/// * `labels` - vector of class labels
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

pub fn m_auc<T: Float>(scores: &Vec<Vec<T>>, labels: &Vec<usize>) -> Result<T, EvalError> {

    util::validate_input(scores, labels).and_then(|_| {
        let dim = scores[0].len();
        let mut m_sum = T::zero();

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
                let ajk = RocCurve::compute(&k_scores, &k_labels)?.auc()?;
                let (j_scores, j_labels) = subset(scores, labels, k, j);
                let akj = RocCurve::compute(&j_scores, &j_labels)?.auc()?;
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
    /// of occurrences of that label
    Weighted
}

enum Metric {
    Precision,
    Recall
}

impl Metric {
    fn name(&self) -> &'static str {
        match self {
            Metric::Precision => "precision",
            Metric::Recall => "recall"
        }
    }
}

fn create_pairs<T: Float>(scores: &Vec<T>,
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

fn sort_pairs_descending<T: Float>(pairs: &mut Vec<(T, bool)>) {
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
    fn test_binary_confusion_matrix_nan() {
        assert!(BinaryConfusionMatrix::compute(&vec![f64::NAN, 0.2, 0.4], &vec![true, false, true], 0.5).is_err());
    }

    #[test]
    fn test_binary_confusion_matrix_with_counts() {
        let matrix = BinaryConfusionMatrix::with_counts(2, 4, 5, 3).unwrap();
        assert_eq!(matrix.tpc, 2);
        assert_eq!(matrix.fpc, 4);
        assert_eq!(matrix.tnc, 5);
        assert_eq!(matrix.fnc, 3);
        assert_eq!(matrix.sum, 14);
        assert!(BinaryConfusionMatrix::with_counts(0, 0, 0, 0).is_err())
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
    fn test_mcc() {
        let (scores, labels) = binary_data();
        let matrix = BinaryConfusionMatrix::compute(&scores, &labels, 0.5).unwrap();
        assert_approx_eq!(matrix.mcc().unwrap(), 0.2581988897471611)
    }

    #[test]
    fn test_roc() {
        let (scores, labels) = binary_data();
        let roc = RocCurve::compute(&scores, &labels).unwrap();

        assert_eq!(roc.dim, 8);
        assert_approx_eq!(roc.points[0].tpr, 1.0 / 3.0);
        assert_approx_eq!(roc.points[0].fpr, 0.0);
        assert_approx_eq!(roc.points[0].threshold, 0.9);
        assert_approx_eq!(roc.points[1].tpr, 1.0 / 3.0);
        assert_approx_eq!(roc.points[1].fpr, 0.2);
        assert_approx_eq!(roc.points[1].threshold, 0.8);
        assert_approx_eq!(roc.points[2].tpr, 2.0 / 3.0);
        assert_approx_eq!(roc.points[2].fpr, 0.2);
        assert_approx_eq!(roc.points[2].threshold, 0.7);
        assert_approx_eq!(roc.points[3].tpr, 2.0 / 3.0);
        assert_approx_eq!(roc.points[3].fpr, 0.4);
        assert_approx_eq!(roc.points[3].threshold, 0.5);
        assert_approx_eq!(roc.points[4].tpr, 2.0 / 3.0);
        assert_approx_eq!(roc.points[4].fpr, 0.6);
        assert_approx_eq!(roc.points[4].threshold, 0.4);
        assert_approx_eq!(roc.points[5].tpr, 2.0 / 3.0);
        assert_approx_eq!(roc.points[5].fpr, 0.8);
        assert_approx_eq!(roc.points[5].threshold, 0.3);
        assert_approx_eq!(roc.points[6].tpr, 2.0 / 3.0);
        assert_approx_eq!(roc.points[6].fpr, 1.0);
        assert_approx_eq!(roc.points[6].threshold, 0.2);
        assert_approx_eq!(roc.points[7].tpr, 1.0);
        assert_approx_eq!(roc.points[7].fpr, 1.0);
        assert_approx_eq!(roc.points[7].threshold, 0.1);
    }

    #[test]
    fn test_roc_empty() {
        assert!(RocCurve::compute(&Vec::<f64>::new(), &Vec::<bool>::new()).is_err());
    }

    #[test]
    fn test_roc_unequal_length() {
        assert!(RocCurve::compute(&vec![0.4, 0.5, 0.2], &vec![true, false, true, false]).is_err());
    }

    #[test]
    fn test_roc_nan() {
        assert!(RocCurve::compute(&vec![0.4, 0.5, 0.2, f64::NAN], &vec![true, false, true, false]).is_err());
    }

    #[test]
    fn test_roc_constant_label() {
        let scores = vec![0.1, 0.4, 0.5, 0.7];
        let labels_true = vec![true; 4];
        let labels_false = vec![false; 4];
        assert!(RocCurve::compute(&scores, &labels_true).is_err());
        assert!(RocCurve::compute(&scores, &labels_false).is_err());
    }

    #[test]
    fn test_auc() {
        let (scores, labels) = binary_data();
        assert_approx_eq!(RocCurve::compute(&scores, &labels).unwrap().auc().unwrap(), 0.6);

        let scores2 = vec![0.2, 0.5, 0.5, 0.3];
        let labels2 = vec![false, true, false, true];
        assert_approx_eq!(RocCurve::compute(&scores2, &labels2).unwrap().auc().unwrap(), 0.625);
    }

    #[test]
    fn test_auc_tied_scores() {
        let scores = vec![0.1, 0.2, 0.3, 0.3, 0.3, 0.7, 0.8];
        let labels1 = vec![false, false, true, false, true, false, true];
        let labels2 = vec![false, false, true, true, false, false, true];
        let labels3 = vec![false, false, false, true, true, false, true];
        assert_approx_eq!(RocCurve::compute(&scores, &labels1).unwrap().auc().unwrap(), 0.75);
        assert_approx_eq!(RocCurve::compute(&scores, &labels2).unwrap().auc().unwrap(), 0.75);
        assert_approx_eq!(RocCurve::compute(&scores, &labels3).unwrap().auc().unwrap(), 0.75);
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
        assert!(PrCurve::compute(&vec![0.4, 0.5, 0.2, f64::NAN], &vec![true, false, true, false]).is_err());
    }

    #[test]
    fn test_pr_constant_label() {
        let scores = vec![0.1, 0.4, 0.5, 0.7];
        let labels_true = vec![true; 4];
        let labels_false = vec![false; 4];
        assert!(PrCurve::compute(&scores, &labels_true).is_ok());
        assert!(PrCurve::compute(&scores, &labels_false).is_err());
    }

    #[test]
    fn test_ap() {
        let (scores, labels) = binary_data();
        assert_approx_eq!(PrCurve::compute(&scores, &labels).unwrap().ap().unwrap(), 0.6805555555555556);

        let scores2 = vec![0.2, 0.5, 0.5, 0.3];
        let labels2 = vec![false, true, false, true];
        assert_approx_eq!(PrCurve::compute(&scores2, &labels2).unwrap().ap().unwrap(), 0.58333333333333);
    }

    #[test]
    fn test_ap_tied_scores() {
        let scores = vec![0.1, 0.2, 0.3, 0.3, 0.3, 0.7, 0.8];
        let labels1 = vec![false, false, true, false, true, false, true];
        let labels2 = vec![false, false, true, true, false, false, true];
        let labels3 = vec![false, false, false, true, true, false, true];
        assert_approx_eq!(PrCurve::compute(&scores, &labels1).unwrap().ap().unwrap(), 0.7333333333333);
        assert_approx_eq!(PrCurve::compute(&scores, &labels2).unwrap().ap().unwrap(), 0.7333333333333);
        assert_approx_eq!(PrCurve::compute(&scores, &labels3).unwrap().ap().unwrap(), 0.7333333333333);
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
        assert!(MultiConfusionMatrix::compute(&vec![vec![0.2, 0.4, 0.4], vec![0.5, 0.1, 0.4], vec![0.3, 0.7, f64::NAN]],
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
    fn test_multi_f1_0p_0r() {

        let scores = multi_class_data().0;
        // every prediction is wrong
        let labels = vec![1, 2, 0, 0, 1, 1, 0];

        assert!(MultiConfusionMatrix::compute(&scores, &labels)
            .unwrap()
            .f1(&Averaging::Macro)
            .is_err()
        )
    }

    #[test]
    fn test_rk() {
        let (scores, labels) = multi_class_data();
        let matrix = MultiConfusionMatrix::compute(&scores, &labels).unwrap();
        assert_approx_eq!(matrix.rk().unwrap(), 0.375)
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
