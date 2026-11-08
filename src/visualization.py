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

    plt.close()

def scatterplot(df,x_column,y_column,filename):
    plt.figure(figsize=(8,5))

    sns.scatterplot(
        data=df,
        x=x_column,
        y=y_column
    )
    plt.title(
        x_column.replace("_"," ").title()
        + " vs "
        + y_column.replace("_"," ").title()
    )

    plt.xlabel(x_column.replace("_", " ").title())
    plt.ylabel(y_column.replace("_", " ").title())

    plt.savefig(filename)

    plt.close()

def heatmap(df,filename):
    plt.figure(figsize=(8,5))

    corr = df.corr(numeric_only=True)

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Correlation Heatmap")

    plt.savefig(filename)

    plt.close()

def barplot(df, category, value, filename):

    plt.figure(figsize=(8, 5))

    data = df.groupby(category)[value].mean()

    sns.barplot(
        x=data.index,
        y=data.values
    )

    plt.title(
        "Average "
        + value.replace("_", " ").title()
        + " by "
        + category.replace("_", " ").title()
    )

    plt.xlabel(category.replace("_", " ").title())
    plt.ylabel("Average " + value.replace("_", " ").title())

    plt.savefig(filename)

    plt.close()