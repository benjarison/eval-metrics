# eval-metrics

Evaluation metrics for machine learning

## Supported Metrics

| Metric      | Task                       | Description                                                        |
|-------------|----------------------------|--------------------------------------------------------------------|
| Accuracy    | Binary Classification      | Binary Class Accuracy                                              |
| Precision   | Binary Classification      | Binary Class Precision                                             |
| Recall      | Binary Classification      | Binary Class Recall                                                |
| F-1         | Binary Classification      | Harmonic Mean of Precision and Recall                              |
| MCC         | Binary Classification      | Matthews Correlation Coefficient                                   |
| ROC Curve   | Binary Classification      | Receiver Operating Characteristic Curve                            |
| AUC         | Binary Classification      | Area Under ROC Curve                                               |
| PR Curve    | Binary Classification      | Precision-Recall Curve                                             |
| AP          | Binary Classification      | Average Precision                                                  |
| Accuracy    | Multi-Class Classification | Multi-Class Accuracy                                               |
| Precision   | Multi-Class Classification | Multi-Class Precision                                              |
| Recall      | Multi-Class Classification | Multi-Class Recall                                                 |
| F-1         | Multi-Class Classification | Multi-Class F1                                                     |
| Rk          | Multi-Class Classification | K-Category Correlation Coefficient as described by Gorodkin (2004) |
| M-AUC       | Multi-Class Classification | Multi-Class AUC as described by Hand and Till (2001)               |
| RMSE        | Regression                 | Root Mean Squared Error                                            |
| MSE         | Regression                 | Mean Squared Error                                                 |
| MAE         | Regression                 | Mean Absolute Error                                                |
| R-Square    | Regression                 | Coefficient of Determination                                       |
| Correlation | Regression                 | Linear Correlation Coefficient                                     |

## Usage

### Binary Classification

The `BinaryConfusionMatrix` struct provides functionality for computing common binary classification metrics.

```rust
use eval_metrics::error::EvalError;
use eval_metrics::classification::BinaryConfusionMatrix;

fn main() -> Result<(), EvalError> {

    // note: these scores could also be f32 values
    let scores = vec![0.5, 0.2, 0.7, 0.4, 0.1, 0.3, 0.8, 0.9];
    let labels = vec![false, false, true, false, true, false, false, true];
    let threshold = 0.5;

    // compute confusion matrix from scores and labels
    let matrix = BinaryConfusionMatrix::compute(&scores, &labels, threshold)?;

    // counts
    let tpc = matrix.tp_count;
    let fpc = matrix.fp_count;
    let tnc = matrix.tn_count;
    let fnc = matrix.fn_count;

    // metrics
    let acc = matrix.accuracy()?;
    let pre = matrix.precision()?;
    let rec = matrix.recall()?;
    let f1 = matrix.f1()?;
    let mcc = matrix.mcc()?;

    // print matrix to console
    println!("{}", matrix);
    Ok(())
}
```
```
                            o=========================o
                            |          Label          |
                            o=========================o
                            |  Positive  |  Negative  |
o==============o============o============|============o
|              |  Positive  |    2532    |    2492    |
|  Prediction  |============|------------|------------|
|              |  Negative  |    2457    |    2519    |
o==============o============o=========================o
```

In addition to the metrics derived from the confusion matrix, ROC curves and PR curves can be computed, providing metrics
such as AUC and AP.

```rust
use eval_metrics::error::EvalError;
use eval_metrics::classification::{RocCurve, RocPoint, PrCurve, PrPoint};

fn main() -> Result<(), EvalError> {

    let scores = vec![0.5, 0.2, 0.7, 0.4, 0.1, 0.3, 0.8, 0.9];
    let labels = vec![false, false, true, false, true, false, false, true];

    // construct roc curve
    let roc = RocCurve::compute(&scores, &labels)?;
    // compute auc
    let auc = roc.auc();
    // inspect roc curve points
    roc.points.iter().for_each(|point| {
        let tpr = point.tp_rate;
        let fpr = point.fp_rate;
        let thresh = point.threshold;
    });

    // construct pr curve
    let pr = PrCurve::compute(&scores, &labels)?;
    // compute average precision
    let ap = pr.ap();
    // inspect pr curve points
    pr.points.iter().for_each(|point| {
        let pre = point.precision;
        let rec = point.recall;
        let thresh = point.threshold;
    });
    Ok(())
}
```

### Multi-Class Classification

The `MultiConfusionMatrix` struct provides functionality for computing common multi-class classification metrics. 
Additionally, averaging methods must be explicitly provided for several of these metrics.

```rust
use eval_metrics::error::EvalError;
use eval_metrics::classification::{MultiConfusionMatrix, Averaging};

fn main() -> Result<(), EvalError> {

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

    // compute confusion matrix from scores and labels
    let matrix = MultiConfusionMatrix::compute(&scores, &labels)?;

    // get counts
    let counts = &matrix.counts;

    // metrics
    let acc = matrix.accuracy()?;
    let mac_pre = matrix.precision(&Averaging::Macro)?;
    let wgt_pre = matrix.precision(&Averaging::Weighted)?;
    let mac_rec = matrix.recall(&Averaging::Macro)?;
    let wgt_rec = matrix.recall(&Averaging::Weighted)?;
    let mac_f1 = matrix.f1(&Averaging::Macro)?;
    let wgt_f1 = matrix.f1(&Averaging::Weighted)?;
    let rk = matrix.rk()?;

    // print matrix to console
    println!("{}", matrix);
    Ok(())
}
```
```
                           o===================================o
                           |               Label               |
                           o===================================o
                           |  Class-1  |  Class-2  |  Class-3  |
o==============o===========o===========|===========|===========o
|              |  Class-1  |     1     |     1     |     1     |
|              |===========|-----------|-----------|-----------|
|  Prediction  |  Class-2  |     1     |     1     |     0     |
|              |===========|-----------|-----------|-----------|
|              |  Class-3  |     0     |     0     |     2     |
o==============o===========o===================================o
```

In addition to the metrics derived from the confusion matrix, the M-AUC (multi-class AUC) metric as described by
Hand and Till (2001) is provided as a standalone function:

```rust
let mauc = m_auc(&scores, &labels)?;
```

### Regression

All regression metrics operate on a pair of scores and labels.

```rust
use eval_metrics::error::EvalError;
use eval_metrics::regression::*;

fn main() -> Result<(), EvalError> {

    // note: these could also be f32 values
    let scores = vec![0.4, 0.7, -1.2, 2.5, 0.3];
    let labels = vec![0.2, 1.1, -0.9, 1.3, -0.2];

    // root mean squared error
    let rmse = rmse(&scores, &labels)?;
    // mean squared error
    let mse = mse(&scores, &labels)?;
    // mean absolute error
    let mae = mae(&scores, &labels)?;
    // coefficient of determination
    let rsq = rsq(&scores, &labels)?;
    // pearson correlation coefficient
    let corr = corr(&scores, &labels)?;
    Ok(())
}
```
