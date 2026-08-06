import pandas as pd
df = pd.read_csv("data/raw/games.csv")
def dataset_summary(df):
    print("\n========== DATASET SUMMARY ==========\n")
    print("ROWS : "+str(df.shape[0]))

    print("COLUMNS : "+str(df.shape[1]))

    print("\nNumeric Columns:")
    print(df.select_dtypes(include=["number"]).shape[1])

    print("\nCategorial Columns : ")
    print(df.select_dtypes(exclude = ["number"]).shape[1])

    memory = round(df.memory_usage().sum() / 1024 / 1024, 2)

    print("\nMemory Usage:")
    print(f"{memory} MB")

def basic_statistics(df):
    print("\n========== BASIC STATISTICS ==========\n")

    numeric_df = df.select_dtypes(include = ["number"])


    print("Mean:")
    print(numeric_df.mean())

    print("\nMedian:")
    print(numeric_df.median())

    print("\nMinimum:")
    print(numeric_df.min())

    print("\nMaximum:")
    print(numeric_df.max())

    print("\nStandard Deviation:")
    print(numeric_df.std())

    print("\nVariance:")
    print(numeric_df.var())

def calculate_percentiles(df):
    print("\n========== PERCENTILES ==========\n")

    numeric_df = df.select_dtypes(include = ["number"])

    print(
        numeric_df.quantile(
            [0.25,0.50,0.75,0.90,0.99]
        )
    )

def unique_values(df):
    print("\n========== UNIQUE VALUES ==========\n")

    categorial_df = df.select_dtypes(exclude = ["number"])

    for column in categorial_df.columns:
        print(column)
        print(categorial_df[column].nunique())
        print()

def correlation_matrix(df):
    print("\n========== CORRELATION MATRIX ==========\n")

    correlation = df.corr(numeric_only=True)

    print(correlation)

    return correlation

def dataset_report(df):

    dataset_summary(df)

    basic_statistics(df)

    calculate_percentiles(df)

    unique_values(df)

    correlation_matrix(df)

dataset_report(df)