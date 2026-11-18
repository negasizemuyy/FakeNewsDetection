# Fake News Detection System Using Machine Learning

## 1. Project Overview

The **Fake News Detection System** is a machine learning-based application designed to classify news articles as either fake or real.

The project uses Natural Language Processing (NLP) techniques and machine learning classification algorithms to analyze news article text and predict its category.

It also provides a web-based interface built with Flask, allowing users to submit news articles and receive predictions.

## 2. Project Objectives

- Develop a machine learning system for fake news detection.
- Explore and prepare fake and real news datasets.
- Apply TF-IDF for text feature extraction.
- Implement and compare multiple machine learning algorithms.
- Evaluate model performance using classification metrics.
- Develop a web application for interactive news classification.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preparation |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning and text processing |
| TF-IDF | Text feature extraction |
| Matplotlib | Data visualization library |
| Seaborn | Statistical visualization library |
| Joblib | Model serialization and loading |
| Flask | Web application framework |
| HTML | Frontend structure |
| CSS | Frontend styling |

## 4. Machine Learning Algorithms

The project implements and compares four classification algorithms:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Classifier (Linear SVC)
4. Passive Aggressive Classifier

The models are evaluated using validation data, and their performance is compared using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

The current final implementation uses the Passive Aggressive Classifier for the saved prediction model.

## 5. Text Feature Extraction

The project uses Term Frequency-Inverse Document Frequency (TF-IDF) to convert news article text into numerical features.

The vectorizer is configured with:

- Maximum features: 5,000
- Stop words: None

These numerical representations are used as inputs to the machine learning algorithms.

## 6. Dataset Preparation

The project uses two original datasets:

- `Fake.csv` – Fake news articles.
- `True.csv` – Real news articles.

The preparation process includes:

1. Loading the original datasets.
2. Assigning labels (1 = Fake News, 0 = Real News).
3. Combining both datasets.
4. Removing exact duplicate records.
5. Identifying articles with conflicting labels.
6. Shuffling the combined dataset.
7. Saving the prepared dataset as `news_dataset.csv`.

## 7. Exploratory Data Analysis

The exploratory data analysis examines:

- Distribution of fake and real news articles.
- Percentage distribution of the two classes.
- News article character counts.
- News article word counts.
- Statistical summaries of text length.

## 8. Model Training and Evaluation

The dataset is divided into three subsets:

| Dataset | Percentage |
|---|---:|
| Training | 64% |
| Validation | 16% |
| Testing | 20% |

The training data is used to fit the models, the validation data is used to compare their performance, and the independent testing data is used for final evaluation.

## 9. Web Application

The project includes a Flask-based web application.

Users can:

1. Open the application homepage.
2. Enter or paste a news article.
3. Submit the article for classification.
4. Receive a prediction of FAKE NEWS or REAL NEWS.

The application uses the saved machine learning model and TF-IDF vectorizer to process submitted articles.

## 10. Project Structure

```text

FakeNewsDetection/
│
├── data/
│   ├── raw/
│   │   ├── Fake.csv
│   │   └── True.csv
│   │
│   └── processed/
│       └── news_dataset.csv
│
├── models/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── eda.py
├── explore_dataset.py
├── prepare_dataset.py
├── test_model.py
├── train_model.py
│
├── requirements.txt
├── .gitignore
├── DATASET_LICENSE.md
└── README.md

```

## 11. Installation and Setup

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the project directory

```bash
cd FakeNewsDetection
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 5: Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 12. Running the Project

### Step 1: Prepare the dataset

```bash
python prepare_dataset.py
```

### Step 2: Perform exploratory data analysis

```bash
python eda.py
```

### Step 3: Train the machine learning models

```bash
python train_model.py
```

This generates the trained model and TF-IDF vectorizer inside the `models/` directory.

### Step 4: Run the Flask application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000/
```

## 13. Model Artifacts

The project saves two model artifacts:

- `fake_news_model.pkl` – The trained Passive Aggressive classification model.
- `tfidf_vectorizer.pkl` – The fitted TF-IDF vectorizer.

Both files are stored in the `models/` directory.

## 14. Future Improvements

- Implement additional machine learning and deep learning algorithms.
- Explore advanced NLP techniques.
- Improve model performance through hyperparameter tuning.
- Add data visualization dashboards.
- Improve the user interface and user experience.
- Explore deployment options for public access.

## 15. Author

## Dataset Attribution

This project uses the **Fake and Real News Dataset**, created by **Clément Bisaillon** and published on Kaggle.

- **Dataset Creator:** Clément Bisaillon (`clmentbisaillon`)
- **Original Source:** [Fake and Real News Dataset – Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Dataset License:** CC BY-NC-SA 4.0 (subject to verification).

### Dataset Processing

The original datasets were processed for this project using the following operations:

1. Assigned numerical labels:
   - `1` = Fake News
   - `0` = Real News
2. Combined the fake and real news datasets.
3. Removed exact duplicate records.
4. Identified articles with conflicting labels.
5. Randomly shuffled the combined dataset.
6. Generated the processed dataset (`news_dataset.csv`).

The original dataset creator is not affiliated with or endorsing this project.

For detailed attribution, licensing information, and modification details, see [DATASET_LICENSE.md](DATASET_LICENSE.md).

## 16. Author

**Negasi Zemuy Gerezgiher**

Computer Science and Engineering Graduate

Interested in Artificial Intelligence, Machine Learning, Natural Language Processing, and Data Science.

---

*This project is developed for educational and research purposes. Machine learning predictions should not be treated as definitive verification of the factual accuracy of news articles.*