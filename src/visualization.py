import matplotlib.pyplot as plt
import seaborn as sns

def histogram(df,column,filename):
    plt.figure(figsize=(8,5))
    sns.histplot(
        data=df,
        x = column
    )
    plt.title(column.replace("_", " ").title())
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Frequency")
    plt.savefig(filename)
    plt.show()
    plt.close()

def boxplot(df,column,filename):
    plt.figure(figsize=(8,5))
    sns.boxplot(
        data=df,
        x = column
    )
    plt.title(column.replace("_", " ").title())
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Frequency")
    plt.savefig(filename)
    plt.show()
    plt.close()

def countplot(df, column, filename):

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x=column
    )

    plt.title(column.replace("_", " ").title())
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Count")

    plt.savefig(filename)

    plt.show()

    plt.close()