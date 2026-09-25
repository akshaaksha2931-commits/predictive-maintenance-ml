\# Predictive Maintenance – Machine Failure Prediction



\## 📌 Project Overview



This project is a \*\*Machine Learning-based Predictive Maintenance System\*\* that predicts whether an industrial machine is at risk of failure based on sensor measurements.



The goal is to identify potential machine failures early so that maintenance can be performed before an unexpected breakdown occurs.



The project uses the \*\*Kaggle Playground Series S3E17 – Binary Classification of Machine Failures\*\* dataset and implements a Random Forest classification model.



\---



\## 🎯 Objectives



\* Analyze industrial machine sensor data.

\* Identify patterns associated with machine failures.

\* Handle the highly imbalanced failure dataset.

\* Train a machine learning classification model.

\* Evaluate the model using precision, recall, F1-score and ROC-AUC.

\* Develop a sensor-based deployment model for early-warning prediction.

\* Optimize the probability threshold for failure detection.



\---



\## 📊 Dataset



The dataset contains industrial machine measurements including:



\* Machine Type

\* Air Temperature

\* Process Temperature

\* Rotational Speed

\* Torque

\* Tool Wear

\* Machine Failure

\* Failure-mode indicators



\### Dataset Statistics



\* Training records: \*\*136,429\*\*

\* Test records: \*\*90,954\*\*

\* Missing values: \*\*0\*\*

\* Duplicate records: \*\*0\*\*

\* Machine failures: \*\*2,148\*\*

\* Failure rate: \*\*1.57%\*\*



Because machine failures are relatively rare, accuracy alone is not sufficient for evaluating the model.



\---



\## 🔧 Technologies Used



\* Python

\* Pandas

\* NumPy

\* Scikit-learn

\* Matplotlib

\* Seaborn

\* Joblib

\* Jupyter Notebook



\### Machine Learning Algorithm



\*\*Random Forest Classifier\*\*



Class imbalance was addressed using:



```python

class\_weight="balanced"

```



\---



\## 🔍 Exploratory Data Analysis



The project analyzes:



\* Machine failure distribution

\* Failure rate by machine type

\* Tool wear and failure relationship

\* Feature correlations

\* Failure-mode indicators

\* Feature importance



The analysis showed that \*\*rotational speed, torque and tool wear\*\* were among the most important sensor-based features in the deployment-oriented model.



\---



\## 🤖 Model 1 – Full Feature Benchmark



The initial model used the available machine sensor measurements together with the failure-mode indicators.



\### Results



| Metric    |  Score |

| --------- | -----: |

| Accuracy  | 99.55% |

| Precision | 90.05% |

| Recall    | 80.00% |

| F1 Score  | 84.73% |

| ROC-AUC   | 94.69% |



This model was used as a benchmark to understand the predictive performance when more information is available.



\---



\## 🚀 Model 2 – Deployment-Oriented Model



For a more realistic early-warning system, a second model was developed using only measurements that could be available from machine sensors:



\* Machine Type

\* Air Temperature

\* Process Temperature

\* Rotational Speed

\* Torque

\* Tool Wear



Failure-mode indicators were excluded from this version.



\### Results



| Metric    |  Score |

| --------- | -----: |

| Accuracy  | 98.46% |

| Precision | 51.23% |

| Recall    | 43.72% |

| F1 Score  | 47.18% |

| ROC-AUC   | 90.69% |



Because the dataset is highly imbalanced, \*\*precision, recall, F1-score and ROC-AUC are more informative than accuracy alone.\*\*



\---



\## 🎚️ Threshold Optimization



The default Random Forest classification threshold is 0.50.



Several thresholds were evaluated using the validation dataset:



| Threshold | Precision | Recall |    F1 |

| --------: | --------: | -----: | ----: |

|      0.20 |     0.288 |  0.647 | 0.399 |

|      0.25 |     0.331 |  0.621 | 0.432 |

|         0 |           |        |       |



