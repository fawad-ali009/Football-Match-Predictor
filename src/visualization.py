import matplotlib.pyplot as plt
import seaborn as sns

def histogram(df,column,filename):
    plt.figure(figsize=(8,5))
    sns.histplot(
        data=df,
        x = column
    )
    plt.title(column)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.savefig(filename)
    plt.show()
    plt.close()
    