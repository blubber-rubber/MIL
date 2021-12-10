| dataset | bag distance | int_dist | ext_owa | int_owa | Accuracy | F1 | TP | TN | FP | FN | Sensitivity | False Negative Rate | False Positive Rate | Specificity | Precission | False omission rate | FDR | Negative predictive value |
|---------|--------------|----------|---------|---------|----------|----|----|----|----|----|-------------|---------------------|---------------------|-------------|------------|---------------------|-----|---------------------------|
| westEast | H | euclidean | additive | strict | 0.8 | 0.83 | 10 | 6 | 4 | 0 | 1.0 | 0.0 | 0.4 | 0.6 | 0.71 | 0.0 | 0.29 | 1.0 |
| westEast | H | euclidean | inverse_additive | strict | 0.8 | 0.83 | 10 | 6 | 4 | 0 | 1.0 | 0.0 | 0.4 | 0.6 | 0.71 | 0.0 | 0.29 | 1.0 |
| westEast | AvgH | euclidean | inverse_additive | strict | 0.8 | 0.8 | 8 | 8 | 2 | 2 | 0.8 | 0.2 | 0.2 | 0.8 | 0.8 | 0.2 | 0.2 | 0.8 |
| westEast | AvgH | euclidean | additive | strict | 0.75 | 0.78 | 9 | 6 | 4 | 1 | 0.9 | 0.1 | 0.4 | 0.6 | 0.69 | 0.14 | 0.31 | 0.86 |
| westEast | H | euclidean | exp | strict | 0.75 | 0.78 | 9 | 6 | 4 | 1 | 0.9 | 0.1 | 0.4 | 0.6 | 0.69 | 0.14 | 0.31 | 0.86 |
| westEast | link | euclidean | strict | None | 0.75 | 0.74 | 7 | 8 | 2 | 3 | 0.7 | 0.3 | 0.2 | 0.8 | 0.78 | 0.27 | 0.22 | 0.73 |
| westEast | surj | euclidean | strict | None | 0.75 | 0.74 | 7 | 8 | 2 | 3 | 0.7 | 0.3 | 0.2 | 0.8 | 0.78 | 0.27 | 0.22 | 0.73 |
| westEast | AvgH | euclidean | exp | strict | 0.75 | 0.74 | 7 | 8 | 2 | 3 | 0.7 | 0.3 | 0.2 | 0.8 | 0.78 | 0.27 | 0.22 | 0.73 |
| westEast | MinH | euclidean | additive | strict | 0.65 | 0.67 | 7 | 6 | 4 | 3 | 0.7 | 0.3 | 0.4 | 0.6 | 0.64 | 0.33 | 0.36 | 0.67 |
| westEast | link | euclidean | additive | None | 0.7 | 0.67 | 6 | 8 | 2 | 4 | 0.6 | 0.4 | 0.2 | 0.8 | 0.75 | 0.33 | 0.25 | 0.67 |
| westEast | MinH | euclidean | inverse_additive | strict | 0.65 | 0.67 | 7 | 6 | 4 | 3 | 0.7 | 0.3 | 0.4 | 0.6 | 0.64 | 0.33 | 0.36 | 0.67 |
| westEast | MinH | euclidean | exp | strict | 0.65 | 0.67 | 7 | 6 | 4 | 3 | 0.7 | 0.3 | 0.4 | 0.6 | 0.64 | 0.33 | 0.36 | 0.67 |
| westEast | AvgH | euclidean | strict | strict | 0.65 | 0.63 | 6 | 7 | 3 | 4 | 0.6 | 0.4 | 0.3 | 0.7 | 0.67 | 0.36 | 0.33 | 0.64 |
| westEast | link | euclidean | exp | None | 0.65 | 0.63 | 6 | 7 | 3 | 4 | 0.6 | 0.4 | 0.3 | 0.7 | 0.67 | 0.36 | 0.33 | 0.64 |
| westEast | surj | euclidean | exp | None | 0.6 | 0.6 | 6 | 6 | 4 | 4 | 0.6 | 0.4 | 0.4 | 0.6 | 0.6 | 0.4 | 0.4 | 0.6 |
| westEast | link | euclidean | inverse_additive | None | 0.65 | 0.59 | 5 | 8 | 2 | 5 | 0.5 | 0.5 | 0.2 | 0.8 | 0.71 | 0.38 | 0.29 | 0.62 |
| westEast | surj | euclidean | additive | None | 0.6 | 0.56 | 5 | 7 | 3 | 5 | 0.5 | 0.5 | 0.3 | 0.7 | 0.62 | 0.42 | 0.38 | 0.58 |
| westEast | H | euclidean | strict | strict | 0.55 | 0.53 | 5 | 6 | 4 | 5 | 0.5 | 0.5 | 0.4 | 0.6 | 0.56 | 0.45 | 0.44 | 0.55 |
| westEast | MinH | euclidean | strict | strict | 0.55 | 0.53 | 5 | 6 | 4 | 5 | 0.5 | 0.5 | 0.4 | 0.6 | 0.56 | 0.45 | 0.44 | 0.55 |
| westEast | surj | euclidean | inverse_additive | None | 0.55 | 0.47 | 4 | 7 | 3 | 6 | 0.4 | 0.6 | 0.3 | 0.7 | 0.57 | 0.46 | 0.43 | 0.54 |
| westEast | SumMin | euclidean | inverse_additive | strict | 0.65 | 0.46 | 3 | 10 | 0 | 7 | 0.3 | 0.7 | 0.0 | 1.0 | 1.0 | 0.41 | 0.0 | 0.59 |
| westEast | SumMin | euclidean | exp | strict | 0.65 | 0.46 | 3 | 10 | 0 | 7 | 0.3 | 0.7 | 0.0 | 1.0 | 1.0 | 0.41 | 0.0 | 0.59 |
| westEast | SumMin | euclidean | strict | strict | 0.6 | 0.43 | 3 | 9 | 1 | 7 | 0.3 | 0.7 | 0.1 | 0.9 | 0.75 | 0.44 | 0.25 | 0.56 |
| westEast | SumMin | euclidean | additive | strict | 0.55 | 0.18 | 1 | 10 | 0 | 9 | 0.1 | 0.9 | 0.0 | 1.0 | 1.0 | 0.47 | 0.0 | 0.53 |
