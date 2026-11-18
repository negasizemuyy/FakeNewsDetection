import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: Load the prepared dataset
news_data = pd.read_csv("data/processed/news_dataset.csv")

# STEP 2: Analyze label distribution

label_counts = news_data["label"].value_counts()

print("\nNUMBER OF FAKE AND REAL NEWS ARTICLES:")
print(label_counts)

# STEP 3: Calculate label percentages

label_percentages = news_data["label"].value_counts(normalize=True) * 100

print("\nPERCENTAGE DISTRIBUTION:")
print(label_percentages)

# STEP 4: Calculate character count

news_data["character_count"] = news_data["text"].str.len()

print("\nCHARACTER COUNT:")
print(news_data[["text", "character_count"]].head())

# STEP 5: Calculate word count

news_data["word_count"] = news_data["text"].str.split().str.len()

print("\nWORD COUNT:")
print(news_data[["text", "word_count"]].head())

# STEP 6: Statistical summary of text length

print("\nTEXT LENGTH STATISTICAL SUMMARY:")
print(news_data[["character_count", "word_count"]].describe())