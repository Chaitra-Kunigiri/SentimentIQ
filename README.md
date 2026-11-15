# SentimentIQ – NLP Sentiment Analysis Web Application

An interactive Natural Language Processing (NLP) web application that analyzes text and classifies sentiment into Positive, Negative, and Neutral categories using Machine Learning.

SentimentIQ combines text preprocessing, machine learning models, interactive visualizations, batch analysis, historical tracking, and downloadable Excel reports in a Streamlit-based web application.

## Project Overview

SentimentIQ is an end-to-end Natural Language Processing and Machine Learning project developed using Python.

The application analyzes textual data, identifies sentiment patterns, and provides interactive insights through a multi-page dashboard.

It uses TF-IDF for text feature extraction and compares Logistic Regression and Multinomial Naive Bayes classification algorithms.

## Key Features

- **Sentiment Classification:** Classifies text into Positive, Negative, and Neutral categories.
- **Interactive Dashboard:** Displays sentiment distribution and analytical insights.
- **Batch Sentiment Analysis:** Analyzes multiple text records simultaneously.
- **Sentiment Visualization:** Interactive pie charts, bar charts, probability charts, and trend analysis.
- **Word Cloud Generation:** Visualizes frequently occurring words.
- **Prediction Probabilities:** Displays model confidence scores.
- **Analysis History:** Stores previous analysis results using SQLite.
- **Excel Report Generation:** Exports analysis reports with Results and Summary sheets.
- **Model Comparison:** Compares Logistic Regression and Multinomial Naive Bayes.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Development Environment | Jupyter Notebook |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| NLP | NLTK, Regular Expressions |
| Feature Extraction | TF-IDF |
| Machine Learning | Scikit-learn |
| Classification Algorithms | Logistic Regression, Multinomial Naive Bayes |
| Model Persistence | Joblib |
| Web Framework | Streamlit |
| Database | SQLite |
| Excel Reporting | OpenPyXL |

## Machine Learning Workflow

1. Data Inspection
2. Exploratory Data Analysis (EDA)
3. Data Cleaning and Text Preprocessing
4. Text Feature Extraction using TF-IDF
5. Model Training
6. Model Evaluation
7. Hyperparameter Optimization
8. Model Saving and Testing
9. Streamlit Web Application Development

## Model Performance

The project uses an 80:20 training and testing split.

| Metric | Result |
|---|---:|
| Training Samples | 11,220 |
| Testing Samples | 2,806 |
| Logistic Regression Test Accuracy | 76.8% |
| Macro F1-score | 0.7151 |

The models include Logistic Regression and Multinomial Naive Bayes.

The results indicate that Logistic Regression provides a useful baseline for three-class sentiment classification.

## Project Structure

The project contains:

- Eight Jupyter notebooks covering the complete machine learning workflow.
- Python modules for preprocessing and visualization.
- Trained machine learning models.
- TF-IDF vectorizer.
- Model metadata.
- Streamlit web application.
- Requirements file.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Chaitra-Kunigiri/SentimentIQ.git
```

### 2. Navigate to the project directory

```bash
cd SentimentIQ
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access the application.

## Application Screenshots

<img width="1909" height="792" alt="Batch_Analysis" src="https://github.com/user-attachments/assets/63b5fb10-5d45-4ed7-a3d2-ce42ca4bdfd5" />
<img width="1872" height="753" alt="Home_Dashboard" src="https://github.com/user-attachments/assets/b5a9da9d-c12f-48f1-99bf-feccbe364ccd" />
<img width="1859" height="771" alt="Screenshot 2026-11-15 104232" src="https://github.com/user-attachments/assets/8d51f172-d5aa-4f74-8e02-82f2de46f911" />
<img width="1885" height="736" alt="Screenshot 2026-11-15 104253" src="https://github.com/user-attachments/assets/445f739b-5f76-4387-8186-1ffedc6a1455" />
<img width="1896" height="782" alt="Screenshot 2026-11-15 105343" src="https://github.com/user-attachments/assets/8000b353-0249-4db5-a11c-0ade3ee064ec" />
<img width="1848" height="770" alt="Screenshot 2026-11-15 105730" src="https://github.com/user-attachments/assets/4d5323f5-fba3-4edd-8966-5b397b0a3786" />



## Limitations

- Model performance depends on the quality and characteristics of the training dataset.
- Sarcasm, slang, and context-dependent expressions can affect prediction accuracy.
- Prediction probabilities represent model estimates and do not guarantee correctness.
- Performance may vary when applied to text from domains different from the training dataset.

## Future Enhancements

- Experiment with advanced NLP models such as DistilBERT.
- Improve neutral sentiment classification.
- Compare additional machine learning algorithms.
- Implement confidence-based uncertainty detection.
- Explore deployment and real-time sentiment monitoring.

## Author

**Chaitra Kunigiri**

Data Science Student | Aspiring Data Enthusiast

GitHub: https://github.com/Chaitra-Kunigiri
Linkedin: www.linkedin.com/in/chaitra-kunigiri-966499384

---

If you find this project interesting, consider giving the repository a star!
