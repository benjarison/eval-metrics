# eval-metrics

Evaluation metrics for machine learning

## Design

The goal of this library is to serve as a very lightweight and intuitive collection of functions for computing 
evaluation metrics, with an emphasis on efficiency and explicitly defined behavior. Classification metrics like 
accuracy, precision, recall, and f1 can all be computed (cheaply) from an instantiated `BinaryConfusionMatrix` or 
`MultiConfusionMatrix`. The distinction between binary and multi-class classification is made explicit to underscore 
the fact that these metrics are naturally formulated for the binary case, and that they require additional assumptions 
(namely averaging methods) in the multi-class case. `Result`s are used as return types for all metrics, reflecting the
fact that evaluation metrics can fail to be defined for any number of reasons. If a metric is not well defined, then an 
`Err` type will be returned. Note that this is inherently more conservative than some other libraries which may impute 
values of zero in cases of undefined metrics. 

## Supported Metrics

| Metric      | Task                       | Description                                                        |
|-------------|----------------------------|--------------------------------------------------------------------|
| Accuracy    | Binary Classification      | Binary Class Accuracy                                              |
| Precision   | Binary Classification      | Binary Class Precision                                             |
| Recall      | Binary Classification      | Binary Class Recall                                                |
| F-1         | Binary Classification      | Harmonic Mean of Precision and Recall                              |
| MCC         | Binary Classification      | Matthews Correlation Coefficient                                   |
| AUC         | Binary Classification      | Standard Area Under ROC Curve                                      |
| Accuracy    | Multi-Class Classification | Multi-Class Accuracy                                               |
| Precision   | Multi-Class Classification | Multi-Class Precision (requires specified averaging method)        |
| Recall      | Multi-Class Classification | Multi-Class Recall (requires specified averaging method)           |
| F-1         | Multi-Class Classification | Multi-Class F1 (requires specified averaging method)               |
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
use eval_metrics::classification::{BinaryConfusionMatrix, auc};

// data
let scores = vec![0.5, 0.2, 0.7, 0.4, 0.1, 0.3, 0.8, 0.9]; // note: these could also be f32 values
let labels = vec![false, false, true, false, true, false, false, true];
let threshold = 0.5;

// compute confusion matrix from scores and labels
let matrix = BinaryConfusionMatrix::compute(&scores, &labels, threshold).unwrap();

// metrics
let acc: Result<f64, EvalError> = matrix.accuracy();
let pre: Result<f64, EvalError> = matrix.precision();
let rec: Result<f64, EvalError> = matrix.recall();
let f1: Result<f64, EvalError> = matrix.f1();
let mcc: Result<f64, EvalError> = matrix.mcc();
```

In additional to the metrics derived from the confusion matrix, the AUC (area under the ROC curve) is also supported
as a standalone function.

```rust
let auc: Result<f64, EvalError> = auc(&scores, &labels);
```

### Multi-Class Classification

The `MultiConfusionMatrix` struct provides functionality for computing common multi-class classification metrics. 
Additionally, averaging methods must be explicitly provided for these metrics.

```rust
use eval_metrics::classification::{Averaging, MultiConfusionMatrix, m_auc};

// data
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
let matrix = MultiConfusionMatrix::compute(&scores, &labels).unwrap();

// metrics
let acc: Result<f64, EvalError> = matrix.accuracy();
let mac_pre: Result<f64, EvalError> = matrix.precision(&Averaging::Macro);
let wgt_pre: Result<f64, EvalError> = matrix.precision(&Averaging::Weighted);
let mac_rec: Result<f64, EvalError> = matrix.recall(&Averaging::Macro);
let wgt_rec: Result<f64, EvalError> = matrix.recall(&Averaging::Weighted);
let mac_f1: Result<f64, EvalError> = matrix.f1(&Averaging::Macro);
let wgt_f1: Result<f64, EvalError> = matrix.f1(&Averaging::Weighted);
let rk: Result<f64, EvalError> = matrix.rk();
```

In additional to the metrics derived from the confusion matrix, the M-AUC (multi-class AUC) metric as described by
Hand and Till (2001) is provided as a standalone function:

```rust
let mauc: Result<f64, EvalError> = m_auc(&scores, &labels);
```

### Regression

All regression metrics operate on a pair of scores and labels.

```rust
use eval_metrics::regression::*;

// note: these could also be f32 values
let scores = vec![0.4, 0.7, -1.2, 2.5, 0.3];
let labels = vec![0.2, 1.1, -0.9, 1.3, -0.2];

// root mean squared error
let rmse: Result<f64, EvalError> = rmse(&scores, &labels);
// mean squared error
let mse: Result<f64, EvalError> = mse(&scores, &labels);
// mean absolute error
let mae: Result<f64, EvalError> = mae(&scores, &labels);
// coefficient of determination
let rsq: Result<f64, EvalError> = rsq(&scores, &labels);
// pearson correlation coefficient
let corr: Result<f64, EvalError> = corr(&scores, &labels);
```
