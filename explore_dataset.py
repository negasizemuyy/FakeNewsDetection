
# Import the pandas library
import pandas as pd


# STEP 1: Load datasets
fake_news = pd.read_csv("data/raw/Fake.csv")
real_news = pd.read_csv("data/raw/True.csv")

# STEP 2: Display first five records
print("FAKE NEWS DATASET")
print(fake_news.head())

print("\nREAL NEWS DATASET")
print(real_news.head())

# STEP 3: Check dimensions
print("\nFAKE NEWS DATASET DIMENSIONS:")
print(fake_news.shape)

print("\nREAL NEWS DATASET DIMENSIONS:")
print(real_news.shape)

# STEP 4: Check column names
print("\nFAKE NEWS COLUMN NAMES:")
print(fake_news.columns)

print("\nREAL NEWS COLUMN NAMES:")
print(real_news.columns)

# STEP 5: Check dataset information
print("\nFAKE NEWS DATASET INFORMATION:")
fake_news.info()

print("\nREAL NEWS DATASET INFORMATION:")
real_news.info()

# STEP 6: Check missing values
print("\nMISSING VALUES IN FAKE NEWS:")
print(fake_news.isnull().sum())

print("\nMISSING VALUES IN REAL NEWS:")
print(real_news.isnull().sum())

# STEP 7: Check duplicate records
print("\nDUPLICATE RECORDS IN FAKE NEWS:")
print(fake_news.duplicated().sum())
print(fake_news[fake_news.duplicated()])

print("\nDUPLICATE RECORDS IN REAL NEWS:")
print(real_news.duplicated().sum())
print(real_news[real_news.duplicated()])