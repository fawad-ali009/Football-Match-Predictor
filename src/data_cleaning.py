
def dataset_info(df):
    print("Shape :  -> "+ str(df.shape))
    print("Columns : Here are the columns\n"+str(df.columns))
    print("Data Types : Below are the data types of all coloumns\n"+str(df.dtypes))
    df.info()
    print(df.describe())

def check_missingValues(df):
    print("Missing Values : \n"+str(df.isnull().sum()))
    print("**********Total Missing Values************")
    print(sum(df.isnull().sum()))

def check_duplicates(df):
    print("************Duplicated Rows****************")
    print(df.duplicated().sum())

def remove_duplicates(df):
    df = df.drop_duplicates()
    return df

def remove_missing_values(df):
    df = df.dropna()
    return df
def save_dataset(df, filename):
    df.to_csv(filename,index=False)

def clean_dataset(df,filename):
    print("\n===================================")
    print("Cleaning Dataset...")
    print("===================================")
    dataset_info(df)
    check_missingValues(df)
    check_duplicates(df)
    df = remove_duplicates(df)
    df = remove_missing_values(df)
    save_dataset(df,filename)
    return df
