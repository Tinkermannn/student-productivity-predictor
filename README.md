# Student Productivity Predictor

## Overview

Student Productivity Predictor is a Streamlit-based web application that estimates a student's productivity score based on academic habits, lifestyle patterns, and focus-related indicators. The prediction is generated using a trained Linear Regression model.

This project was developed as part of the Dasar Analitika Data course project by Group 7.

## Features

- Predicts student productivity on a scale from 0 to 100.
- Uses academic and lifestyle factors as prediction inputs.
- Provides a simple Streamlit interface for entering student data.
- Displays an interpretation of the predicted productivity level.
- Uses saved machine learning artifacts for preprocessing and prediction.

## Prediction Inputs

The application uses the following input variables:

| Variable | Description |
| --- | --- |
| `study_hours_per_day` | Number of study hours per day |
| `sleep_hours` | Number of sleep hours per day |
| `phone_usage_hours` | Number of phone usage hours per day |
| `attendance_percentage` | Student attendance percentage |
| `stress_level` | Stress level score |
| `focus_score` | Focus score |

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- runtime.txt
|-- linear_regression_final.pkl
|-- scaler_final.pkl
|-- selected_features.pkl
|-- Kelompok7_Laporan_Akhir.ipynb
`-- Report
    |-- Laporan_Akhir_Kelompok_7.pdf
    `-- PPT_Laporan_Akhir_Kelompok_7.pdf
```

## File Description

| File | Description |
| --- | --- |
| `app.py` | Main Streamlit application file |
| `requirements.txt` | List of Python dependencies required to run the application |
| `runtime.txt` | Python runtime version used for deployment |
| `linear_regression_final.pkl` | Trained Linear Regression model |
| `scaler_final.pkl` | Scaler used to preprocess input features |
| `selected_features.pkl` | List of selected features used by the model |
| `Kelompok7_Laporan_Akhir.ipynb` | Project notebook containing the analysis and model development process |
| `Report/Laporan_Akhir_Kelompok_7.pdf` | Final project report in PDF format |
| `Report/PPT_Laporan_Akhir_Kelompok_7.pdf` | Final presentation file in PDF format |

## Requirements

The project requires Python 3.11 and the following Python packages:

- Streamlit
- pandas
- NumPy
- scikit-learn
- joblib

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## How to Run the Application

1. Clone or download this repository.
2. Open a terminal in the project directory.
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

5. Open the local Streamlit URL displayed in the terminal.

## Model Artifacts

The application depends on three saved model artifacts:

- `linear_regression_final.pkl`
- `scaler_final.pkl`
- `selected_features.pkl`

These files must remain in the same directory as `app.py` so the application can load the model, scaler, and selected feature list correctly.

## Prediction Output

After the user submits the input values, the application displays:

- The predicted productivity score.
- A short interpretation of the productivity level.

The output score is limited to the range of 0 to 100.

## Deployment Notes

The `runtime.txt` file specifies the Python runtime version:

```text
python-3.11
```

This is useful for deployment platforms that support runtime configuration, such as Streamlit Community Cloud or similar Python application hosting services.

## Authors

Group 7  
Dasar Analitika Data
