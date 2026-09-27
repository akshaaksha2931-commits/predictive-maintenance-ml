# ⚙️ Predictive Maintenance - Machine Failure Prediction

## 📌 Project Overview

Predictive Maintenance is a machine learning project that predicts the probability of machine failure using sensor-based machine parameters.

The system uses a **Random Forest Classifier** trained on historical machine sensor data and provides predictions through an interactive **Streamlit web application**.

The goal is to help identify machines that may be at higher risk of failure so that preventive maintenance can be considered before unexpected breakdowns.

---

## 🎯 Problem Statement

Unexpected machine failures can result in:

- Production downtime
- Increased maintenance costs
- Equipment damage
- Reduced operational efficiency

This project applies machine learning to historical machine sensor data to identify patterns associated with machine failures.

---

## 🚀 Features

- Machine failure prediction using Random Forest
- Failure probability estimation
- Optimized decision threshold
- Interactive Streamlit dashboard
- Sensor parameter input
- Machine risk classification
- Maintenance recommendations
- Feature importance visualization
- Model performance evaluation

---

## 🧠 Machine Learning Model

### Algorithm

**Random Forest Classifier**

Random Forest was selected because it can handle nonlinear relationships between sensor parameters and machine failure while also providing feature importance information.

### Model Configuration

```text
Algorithm: Random Forest Classifier
Number of estimators: 100
Class weighting: Balanced
Random state: 42
Decision threshold: 0.45
```

---

## 📊 Input Parameters

The deployment model uses the following parameters:

| Parameter | Description |
|---|---|
| Machine Type | Type of machine: H, L, or M |
| Air Temperature | Air temperature in Kelvin |
| Process Temperature | Process temperature in Kelvin |
| Rotational Speed | Machine rotational speed in rpm |
| Torque | Machine torque in Nm |
| Tool Wear | Tool usage time in minutes |

---

## 📈 Model Performance

The model was evaluated using multiple classification metrics.

| Metric | Score |
|---|---:|
| Accuracy | 0.9961 |
| Precision | 0.9601 |
| Recall | 0.7837 |
| F1 Score | 0.8630 |
| ROC-AUC | 0.9373 |

> These metrics were obtained during model evaluation on the project's test data.

---

## 🔍 Feature Importance

The Random Forest model provides feature importance values that help identify which machine parameters contribute most strongly to its predictions.

The Streamlit application displays these feature importance values as a chart.

Important sensor variables include:

- Rotational Speed
- Torque
- Tool Wear
- Air Temperature
- Process Temperature

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit dashboard where users can enter machine sensor values and receive a predicted failure probability.

### Application Workflow

```text
Machine Sensor Inputs
        ↓
Data Preprocessing
        ↓
Machine Type Encoding
        ↓
Random Forest Model
        ↓
Failure Probability
        ↓
Threshold-Based Decision
        ↓
Maintenance Recommendation
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Git & GitHub

---

## 📂 Project Structure

```text
PredictiveMaintenance/
│
├── app.py
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   ├── machine_failure_model.pkl
│   └── deployment_model.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── src/
│   ├── eda.py
│   ├── load_data.py
│   ├── model.py
│   ├── model_deployment.py
│   ├── predict.py
│   ├── predict_deployment.py
│   ├── threshold_analysis.py
│   └── visualize.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/akshaaksha2931-commits/predictive-maintenance-machine-failure.git
```

Navigate to the project:

```bash
cd predictive-maintenance-machine-failure
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application

Start the application using:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧪 Example Prediction

Example machine input:

```text
Machine Type: L
Air Temperature: 300 K
Process Temperature: 310 K
Rotational Speed: 1500 rpm
Torque: 40 Nm
Tool Wear: 100 min
```

The application returns:

- Predicted failure probability
- Machine failure risk status
- Failure risk visualization
- Maintenance recommendation
- Sensor input summary
- Feature importance visualization

---

## 📌 Dataset

The project uses a machine failure dataset containing machine sensor measurements and failure labels.

The target variable is:

```text
Machine failure
```

Where:

```text
0 = No machine failure
1 = Machine failure
```

The dataset contains a highly imbalanced target, making metrics such as precision, recall, F1-score and ROC-AUC important when evaluating the model.

---

## 🔮 Future Improvements

Possible future improvements include:

- Real-time sensor data integration
- IoT-based machine monitoring
- Automated maintenance alerts
- Cloud deployment
- Historical prediction tracking
- Advanced anomaly detection
- Model retraining with new sensor data
- Integration with industrial monitoring systems

---

## 👩‍💻 Author

**Aksha K**

Computer Science Engineering Student

---

## ⭐ Project Highlights

This project demonstrates practical knowledge of:

- Machine Learning
- Classification
- Random Forest
- Imbalanced Dataset Handling
- Feature Engineering
- Model Evaluation
- Probability Threshold Optimization
- Data Visualization
- Streamlit Deployment
- Git & GitHub
