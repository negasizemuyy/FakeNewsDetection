import joblib

# STEP 1: Load the saved machine learning model

model = joblib.load("models/fake_news_model.pkl")


# STEP 2: Load the saved TF-IDF vectorizer

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

print("Model and vectorizer loaded successfully!")

# STEP 3: Enter a new news article

news_article = input("\nEnter a news article: ")


# STEP 4: Convert the article into TF-IDF features

article_tfidf = vectorizer.transform([news_article])


# STEP 5: Predict whether the article is fake or real

prediction = model.predict(article_tfidf)


# STEP 6: Display the prediction

if prediction[0] == 1:
    print("\nPrediction: FAKE NEWS")
else:
    print("\nPrediction: REAL NEWS")

    