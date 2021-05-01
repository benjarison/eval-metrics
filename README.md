# eval-metrics

Evaluation metrics for machine learning

## Design

The goal of this library is to serve as a very lightweight and intuitive collection of functions for computing 
evaluation metrics, with an emphasis on efficiency and explicitly defined behavior. Classification metrics like 
accuracy, precision, recall, and f1 can all be computed (cheaply) from an instantiated `BinaryConfusionMatrix` or 
`MultiConfusionMatrix`. The distinction between binary and multi-class classification is made explicit to underscore 
the fact that these metrics are naturally formulated for the binary case, and that some require additional assumptions 
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
| ROC Curve   | Binary Classification      | Receiver Operating Characteristic Curve                            |
| AUC         | Binary Classification      | Area Under ROC Curve                                               |
| PR Curve    | Binary Classification      | Precision-Recall Curve                                             |
| AP          | Binary Classification      | Average Precision                                                  |
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
// Note: these scores could also be f32 values
let scores = vec![0.5, 0.2, 0.7, 0.4, 0.1, 0.3, 0.8, 0.9];
let labels = vec![false, false, true, false, true, false, false, true];
let threshold = 0.5;

// compute confusion matrix from scores and labels
let matrix = BinaryConfusionMatrix::compute(&scores, &labels, threshold)?;

// metrics
let acc = matrix.accuracy()?;
let pre = matrix.precision()?;
let rec = matrix.recall()?;
let f1 = matrix.f1()?;
let mcc = matrix.mcc()?;
```
The matrix can be printed to the console as well:

```rust
let matrix = BinaryConfusionMatrix::compute(&scores, &labels, threshold)?;
println!("{}", matrix);
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
// construct roc curve
let roc = RocCurve::compute(&scores, &labels)?;
// compute auc
let auc = roc.auc()?;
// inspect roc curve points
let points: Vec<RocPoint> = roc.points;
points.iter().for_each(|point| {
    println!("{}, {}, {}", 
             point.tpr, // true positive rate
             point.fpr, // false positive rate
             point.threshold); // corresponding score threshold
});

// construct pr curve
let pr = PrCurve::compute(&scores, &labels)?;
// compute average precision
let ap = pr.ap()?;
// inspect pr curve points
let points: Vec<PrPoint> = pr.points;
points.iter().for_each(|point| {
    println!("{}, {}, {}", 
             point.precision, // precision value
             point.recall, // recall value
             point.threshold); // corresponding score threshold
});
```

### Multi-Class Classification

The `MultiConfusionMatrix` struct provides functionality for computing common multi-class classification metrics. 
Additionally, averaging methods must be explicitly provided for these metrics.

```rust
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

// metrics
let acc = matrix.accuracy()?;
let mac_pre = matrix.precision(&Averaging::Macro)?;
let wgt_pre = matrix.precision(&Averaging::Weighted)?;
let mac_rec = matrix.recall(&Averaging::Macro)?;
let wgt_rec = matrix.recall(&Averaging::Weighted)?;
let mac_f1 = matrix.f1(&Averaging::Macro)?;
let wgt_f1 = matrix.f1(&Averaging::Weighted)?;
let rk = matrix.rk()?;
```
The matrix can be printed to the console as well:

```rust
let matrix = MultiConfusionMatrix::compute(&scores, &labels)?;
println!("{}", matrix);
```
```
                           o===================================o
                           |               Label               |
                           o===================================o
                           |  Class-1  |  Class-2  |  Class-3  |
o==============o===========o===========|===========|===========o
|              |  Class-1  |   1138    |   1126    |   1084    |
|              |===========|-----------|-----------|-----------|
|  Prediction  |  Class-2  |   1078    |   1147    |   1126    |
|              |===========|-----------|-----------|-----------|
|              |  Class-3  |   1107    |   1092    |   1102    |
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
use eval_metrics::regression::*;

// Note: these could also be f32 values
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
```
