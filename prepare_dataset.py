
import pandas as pd

# STEP 1: Load the original datasets
fake_news = pd.read_csv("data/raw/Fake.csv")
real_news = pd.read_csv("data/raw/True.csv")


# STEP 2: Create labels
# 1 represents fake news
# 0 represents real news
fake_news["label"] = 1
real_news["label"] = 0


# STEP 3: Combine both datasets
news_data = pd.concat(
    [fake_news, real_news],
    ignore_index=True
)


# STEP 4: Remove exact duplicate records
news_data = news_data.drop_duplicates()

print("Exact duplicate records removed successfully!")


# STEP 5: Identify identical news content with conflicting labels

conflicting_articles = news_data[
    news_data.duplicated(
        subset=["title", "text"],
        keep=False
    )
]

# Keep only articles whose labels are different
conflicting_articles = conflicting_articles[
    conflicting_articles.groupby(
        ["title", "text"]
    )["label"].transform("nunique") > 1
]

print("\nARTICLES WITH CONFLICTING LABELS:")
print(conflicting_articles[["title", "text", "label"]])

print("\nNumber of conflicting records:")
print(len(conflicting_articles))


# STEP 6: Shuffle the combined dataset
news_data = news_data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# STEP 7: Display the prepared dataset
print("\nPREPARED DATASET")
print(news_data.head())

print("\nDataset dimensions:")
print(news_data.shape)

print("\nLabel distribution:")
print(news_data["label"].value_counts())


# STEP 8: Save the prepared dataset
news_data.to_csv(
    "data/processed/news_dataset.csv",
    index=False
)

print("\nPrepared dataset saved successfully!")
