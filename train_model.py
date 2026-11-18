
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# STEP 1: Load the prepared dataset

news_data = pd.read_csv("data/processed/news_dataset.csv")

# Handle any missing text values
news_data["text"] = news_data["text"].fillna("").astype(str)


# STEP 2: Separate features (X) and target (y)

X = news_data["text"]
y = news_data["label"]


# STEP 3: Split the dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# STEP 4: Display the training and testing results

print("TRAINING AND TESTING DATASET")

print("\nTraining data dimensions:")
print(X_train.shape)

print("\nTesting data dimensions:")
print(X_test.shape)

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())


# STEP 5: Create training and validation datasets

X_train_model, X_val, y_train_model, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

print("\nTRAINING AND VALIDATION DATA")

print("Model training dimensions:", X_train_model.shape)
print("Validation dimensions:", X_val.shape)


# STEP 6: Create the TF-IDF vectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words=None
)


# STEP 7: Fit and transform the model training data

X_train_model_tfidf = vectorizer.fit_transform(X_train_model)


# STEP 8: Transform validation and testing data

X_val_tfidf = vectorizer.transform(X_val)

X_test_tfidf = vectorizer.transform(X_test)


# STEP 9: Display TF-IDF results

print("\nTF-IDF VECTORIZATION RESULTS")

print("\nModel training feature dimensions:")
print(X_train_model_tfidf.shape)

print("\nValidation feature dimensions:")
print(X_val_tfidf.shape)

print("\nTesting feature dimensions:")
print(X_test_tfidf.shape)

print("\nNumber of vocabulary features:")
print(len(vectorizer.get_feature_names_out()))

print("\nFirst 20 vocabulary words:")
print(vectorizer.get_feature_names_out()[:20])


# STEP 10: Create our four machine learning models

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Multinomial Naive Bayes": MultinomialNB(),

    "Linear SVC": LinearSVC(
        max_iter=5000,
        random_state=42
    ),

    "Passive Aggressive": PassiveAggressiveClassifier(
        max_iter=1000,
        random_state=42
    )
}


# STEP 11: Train all four models

for model_name, model in models.items():

    model.fit(X_train_model_tfidf, y_train_model)

    print(f"{model_name} trained successfully!")


# STEP 12: Make predictions using the validation dataset

model_predictions = {}

for model_name, model in models.items():

    predictions = model.predict(X_val_tfidf)

    model_predictions[model_name] = predictions


# STEP 13: Display predictions and actual labels

print("\nMODEL PREDICTIONS")

for model_name, predictions in model_predictions.items():

    print(f"\n{model_name}")

    print("First 10 predictions:")
    print(predictions[:10])

    print("\nFirst 10 actual labels:")
    print(y_val.iloc[:10].values)


# STEP 14: Evaluate each model individually

for model_name, predictions in model_predictions.items():

    print("\n" + "=" * 60)

    print(f"MODEL EVALUATION: {model_name}")

    print("=" * 60)

    # Accuracy
    accuracy = accuracy_score(y_val, predictions)

    print("\nAccuracy:")
    print(f"{accuracy * 100:.2f}%")

    # Classification report
    print("\nClassification Report:")

    print(classification_report(
        y_val,
        predictions,
        target_names=["Real News", "Fake News"],
        zero_division=0
    ))

    # Confusion matrix
    print("\nConfusion Matrix:")

    print(confusion_matrix(y_val, predictions))


# STEP 15: Compare all four models

results = []

for model_name, predictions in model_predictions.items():

    accuracy = accuracy_score(y_val, predictions)

    precision = precision_score(
        y_val,
        predictions,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        pos_label=1,
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })


# STEP 16: Create and display the comparison table

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1-Score",
    ascending=False
)

print("\nMACHINE LEARNING MODEL COMPARISON")

print(results_df.to_string(index=False))


# STEP 17: Visualize model performance

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df,
    x="Model",
    y="F1-Score"
)

plt.title("Comparison of Machine Learning Models")

plt.xlabel("Machine Learning Algorithms")

plt.ylabel("F1-Score")

plt.ylim(0, 1)

plt.xticks(rotation=20)

plt.tight_layout()

plt.show()


# STEP 18: Create a new TF-IDF vectorizer

final_vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words=None
)


# STEP 19: Fit the vectorizer using the full training dataset

X_train_final_tfidf = final_vectorizer.fit_transform(X_train)


# STEP 20: Transform the test dataset

X_test_final_tfidf = final_vectorizer.transform(X_test)


# STEP 21: Create our selected machine learning model

final_model = PassiveAggressiveClassifier(
    max_iter=1000,
    random_state=42
)


# STEP 22: Train the final model

final_model.fit(X_train_final_tfidf, y_train)

print("Final model trained successfully!")


# STEP 23: Make predictions on the test dataset

y_test_predictions = final_model.predict(X_test_final_tfidf)


# STEP 24: Evaluate the final model

print("\nFINAL MODEL EVALUATION ON TEST DATASET")

print("\nAccuracy:")
print(f"{accuracy_score(y_test, y_test_predictions) * 100:.2f}%")


print("\nClassification Report:")
print(classification_report(
    y_test,
    y_test_predictions,
    target_names=["Real News", "Fake News"],
    zero_division=0
))


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_predictions))

# STEP 25: Create models directory

os.makedirs("models", exist_ok=True)


# STEP 26: Save the trained machine learning model

joblib.dump(final_model, "models/fake_news_model.pkl")

print("\nMachine learning model saved successfully!")


# STEP 27: Save the TF-IDF vectorizer

joblib.dump(final_vectorizer, "models/tfidf_vectorizer.pkl")

print("TF-IDF vectorizer saved successfully!")

