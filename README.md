\# Predictive Maintenance ML — Machine Failure Prediction



A machine learning system for \*\*early detection of industrial machine failure risk using sensor data\*\*.



The project uses a Random Forest classifier to analyze machine operating conditions such as temperature, rotational speed, torque, and tool wear. It includes exploratory data analysis, model evaluation, probability-threshold optimization, and a command-line prediction system.



\---



\## 🚀 Project Overview



Unexpected equipment failure can result in production downtime, maintenance costs, and operational losses.



This project explores a \*\*predictive maintenance approach\*\* where machine sensor measurements are used to estimate the probability of failure before it occurs.



\### System Workflow



```text

Machine Sensor Data

&#x20;       ↓

Data Preprocessing

&#x20;       ↓

Exploratory Data Analysis

&#x20;       ↓

Random Forest Model

&#x20;       ↓

Failure Probability

&#x20;       ↓

Optimized Decision Threshold

&#x20;       ↓

Machine Failure Risk Alert

```



\---



\## 🎯 Objectives



\* Analyze industrial machine sensor data.

\* Identify patterns associated with machine failures.

\* Build a machine-learning classification model.

\* Handle severe class imbalance.

\* Evaluate model performance using appropriate classification metrics.

\* Develop a sensor-only deployment model for early-warning prediction.

\* Optimize the classification threshold based on validation performance.

\* Build a command-line prediction system for new machine readings.



\---



\## 📊 Dataset



The project uses the \*\*Kaggle Playground Series S3E17 — Binary Classification of Machine Failures\*\* dataset.



\### Dataset statistics



| Property          |   Value |

| ----------------- | ------: |

| Training records  | 136,429 |

| Test records      |  90,954 |

| Missing values    |       0 |

| Duplicate records |       0 |

| Failure cases     |   2,148 |

| Failure rate      |   1.57% |



The target variable is:



```text

Machine failure

0 → No failure

1 → Failure

```



Because the failure class represents only approximately \*\*1.57%\*\* of the training data, accuracy alone is not sufficient for evaluating the model.



\---



\## 🧰 Technologies



\* \*\*Python\*\*

\* \*\*Pandas\*\*

\* \*\*NumPy\*\*

\* \*\*Scikit-learn\*\*

\* \*\*Matplotlib\*\*

\* \*\*Seaborn\*\*

\* \*\*Joblib\*\*

\* \*\*Jupyter Notebook\*\*



\### Machine Learning Algorithm



\*\*Random Forest Classifier\*\*



Class imbalance was addressed using:



```python

class\_weight="balanced"

```



\---



\## 🔍 Exploratory Data Analysis



The project performs analysis of:



\* Machine failure distribution

\* Failure rate by machine type

\* Tool wear versus failure

\* Sensor correlations

\* Failure-mode indicators

\* Feature importance



\### Key observations



Sensor-based features such as \*\*rotational speed, torque, and tool wear\*\* were among the most influential variables in the deployment-oriented model.



The analysis also showed that machines with higher tool-wear values had higher observed failure rates in the dataset.



These relationships are treated as \*\*associations rather than causal conclusions\*\*.



\---



\## 🤖 Model Development



Two Random Forest models were developed.



\### 1. Full-Feature Benchmark Model



The first model was developed as a benchmark using sensor measurements together with the available failure-mode indicators.



\#### Performance



| Metric    |      Score |

| --------- | ---------: |

| Accuracy  | \*\*99.55%\*\* |

| Precision | \*\*90.05%\*\* |

| Recall    | \*\*80.00%\*\* |

| F1 Score  | \*\*84.73%\*\* |

| ROC-AUC   | \*\*94.69%\*\* |



This model demonstrates the predictive performance when additional failure-related information is available.



\---



\### 2. Sensor-Based Deployment Model



A second model was designed for a more realistic early-warning scenario.



Only measurements that could reasonably come from machine sensors were used:



\* Machine Type

\* Air Temperature

\* Process Temperature

\* Rotational Speed

\* Torque

\* Tool Wear



The failure-mode indicators were excluded.



\#### Performance



| Metric    |      Score |

| --------- | ---------: |

| Accuracy  | \*\*98.46%\*\* |

| Precision | \*\*51.23%\*\* |

| Recall    | \*\*43.72%\*\* |

| F1 Score  | \*\*47.18%\*\* |

| ROC-AUC   | \*\*90.69%\*\* |



The deployment model is the model used by the prediction application.



\---



\## ⚖️ Handling Class Imbalance



Only \*\*1.57%\*\* of the training observations represent machine failures.



Therefore, a model can achieve high accuracy while still missing a significant number of actual failures.



For this reason, the project focuses on:



\* Precision

\* Recall

\* F1 Score

\* ROC-AUC



rather than using accuracy as the primary performance measure.



\---



\## 🎚️ Probability Threshold Optimization



Random Forest normally classifies an observation as positive when its predicted probability reaches the default decision threshold of \*\*0.50\*\*.



For predictive maintenance, the threshold affects the trade-off between:



\* \*\*Precision\*\* — avoiding unnecessary maintenance alerts

\* \*\*Recall\*\* — detecting more actual failures



Several thresholds were evaluated on the validation set.



| Threshold | Precision |    Recall |  F1 Score |

| --------: | --------: | --------: | --------: |

|      0.20 |     0.288 |     0.647 |     0.399 |

|      0.25 |     0.331 |     0.621 |     0.432 |

|      0.30 |     0.369 |     0.591 |     0.454 |

|      0.35 |     0.404 |     0.544 |     0.464 |

|      0.40 |     0.445 |     0.526 |     0.482 |

|  \*\*0.45\*\* | \*\*0.481\*\* | \*\*0.491\*\* | \*\*0.486\*\* |

|      0.50 |     0.508 |     0.444 |     0.474 |



The \*\*0.45 threshold produced the highest F1 score among the tested thresholds\*\*, so it was selected as the operating threshold for this prototype.



```text

Predicted probability ≥ 0.45

&#x20;       ↓

Machine failure risk detected



Predicted probability < 0.45

&#x20;       ↓

No machine failure risk detected

```



\---



\## 🖥️ Prediction System



The project includes a command-line prediction application.



Run:



```bash

python src/predict\_deployment.py

```



The application accepts:



```text

Machine Type

Air Temperature \[K]

Process Temperature \[K]

Rotational Speed \[rpm]

Torque \[Nm]

Tool Wear \[min]

```



It returns:



```text

Predicted failure probability

Failure risk classification

```



\### Example



```text

===== RESULT =====

⚠️ MACHINE FAILURE RISK DETECTED

Predicted failure probability: 46.92%

```



The \*\*46.92% value is the model's predicted probability for the failure class\*\*. It should not be interpreted as a guaranteed real-world probability that the machine will fail.



\---



\## 📈 Project Outputs



The repository contains visualizations generated during analysis and model development:



\* Failure distribution

\* Failure rate by machine type

\* Tool wear versus failure

\* Correlation heatmap

\* Confusion matrix

\* Feature importance



These are available in the \[`outputs/`](outputs/) directory.



\---



\## 📁 Project Structure



```text

predictive-maintenance-ml/

│

├── data/

│   └── Dataset files excluded from Git

│

├── models/

│   ├── deployment\_model.pkl

│   └── machine\_failure\_model.pkl

│

├── outputs/

│   ├── confusion\_matrix.png

│   ├── correlation\_heatmap.png

│   ├── failure\_by\_type.png

│   ├── failure\_distribution.png

│   ├── feature\_importance.png

│   └── tool\_wear\_vs\_failure.png

│

├── src/

│   ├── eda.py

│   ├── load\_data.py

│   ├── model.py

│   ├── model\_deployment.py

│   ├── predict.py

│   ├── predict\_deployment.py

│   ├── threshold\_analysis.py

│   └── visualize.py

│

├── .gitignore

├── README.md

├── requirements.txt

└── venv/                 # Local environment, excluded from Git

```



\---



\## ▶️ Installation \& Usage



\### 1. Clone the repository



```bash

git clone https://github.com/akshaaksha2931-commits/predictive-maintenance-ml.git

cd predictive-maintenance-ml

```



\### 2. Create a virtual environment



```bash

python -m venv venv

```



\### 3. Activate the environment



\#### Windows PowerShell



```powershell

.\\venv\\Scripts\\Activate.ps1

```



\### 4. Install dependencies



```bash

pip install -r requirements.txt

```



\### 5. Run the prediction system



```bash

python src/predict\_deployment.py

```



\---



\## 🧪 Model Development Scripts



The main scripts are:



| Script                  | Purpose                                 |

| ----------------------- | --------------------------------------- |

| `load\_data.py`          | Load and inspect dataset                |

| `eda.py`                | Perform exploratory data analysis       |

| `visualize.py`          | Generate visualizations                 |

| `model.py`              | Train the full-feature benchmark model  |

| `model\_deployment.py`   | Train the sensor-based deployment model |

| `predict.py`            | Run full-feature predictions            |

| `predict\_deployment.py` | Run sensor-based predictions            |

| `threshold\_analysis.py` | Evaluate classification thresholds      |



\---



\## ⚠️ Limitations



This project is a machine-learning prototype and is not intended for direct industrial deployment.



Important limitations include:



1\. The dataset is highly imbalanced.

2\. The models use a random train-validation split rather than time-based validation.

3\. The benchmark model includes failure-mode indicators that may not be available sufficiently early in a real-world maintenance workflow.

4\. The deployment model uses sensor measurements but has lower recall than the full-feature benchmark model.

5\. Real industrial deployment would require validation using real machine data and operational time-series information.

6\. The selected probability threshold should ultimately be determined according to the operational cost of missed failures versus false alarms.



\---



\## 🔮 Future Improvements



Potential improvements include:



\* \[ ] Build a Streamlit monitoring dashboard

\* \[ ] Add real-time sensor input

\* \[ ] Add maintenance alert notifications

\* \[ ] Integrate IoT sensor data

\* \[ ] Perform time-based model validation

\* \[ ] Explore time-series machine learning

\* \[ ] Calibrate predicted probabilities

\* \[ ] Deploy the model using FastAPI or Flask

\* \[ ] Add model monitoring and drift detection

\* \[ ] Validate against real industrial machine data



\---



\## 👩‍💻 Author



\*\*Aksha K.\*\*



B.E. Computer Science and Engineering



\---



\## 📌 Disclaimer



This project is intended for educational and portfolio purposes. Model predictions are statistical estimates generated from the training data and should not be treated as definitive predictions of real-world machine failure.



