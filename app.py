import joblib
from flask import Flask, render_template, request

# STEP 1: Initialize Flask application

app = Flask(__name__)


# STEP 2: Load the saved machine learning model

model = joblib.load("models/fake_news_model.pkl")


# STEP 3: Load the saved TF-IDF vectorizer

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# STEP 4: Create the home page route

@app.route("/")
def home():
    return render_template("index.html")


# STEP 5: Create the prediction route

@app.route("/predict", methods=["POST"])
def predict():

    news_article = request.form.get("news_article", "").strip()

    if not news_article:
        return render_template(
            "index.html",
            error="Please enter a news article."
        )

    # Convert article into TF-IDF features
    article_tfidf = vectorizer.transform([news_article])

    # Make prediction
    prediction = model.predict(article_tfidf)

    # Interpret prediction
    if prediction[0] == 1:
        result = "FAKE NEWS"
    else:
        result = "REAL NEWS"

    return render_template(
        "index.html",
        prediction=result,
        article=news_article
    )


# STEP 6: Run the Flask application

if __name__ == "__main__":
    app.run(debug=True)

