import pandas as pd

## opdracht 1

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the hotel bookings Excel file into a pandas DataFrame.
    
    Args:
        filepath (str): Path to the Excel file.
    
    Returns:
        pd.DataFrame: The loaded dataset.
    
    Author: Laurence
    """
    df = pd.read_excel(filepath)
    print(f"Shape: {df.shape}")
    return df

def explore_data(df: pd.DataFrame) -> None:
    """
    Show basic information about the dataset for quick inspection.
    
    Args:
        df (pd.DataFrame): The dataset to explore.
    
    Returns:
        None
    
    Author: Laurence
    """
    print(df.head())
    print(df.info())
    print("missende waardes per kolom:")
    print(df.isnull().sum())

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.
    
    Args:
        df (pd.DataFrame): The dataset with possible duplicates.
    
    Returns:
        pd.DataFrame: Dataset without duplicates.
    
    Author: Laurence
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f" er zijn {before - after} dubbele rijen verwijderd.")
    return df

def delete_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all rows that contain missing (NaN) values.

    Args:
        df (pd.DataFrame): The dataset with possible missing values.

    Returns:
        pd.DataFrame: Dataset with all rows containing NaN removed.

    Author: Laurence
    """
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f" er zijn {before - after} rijen verwijderd die missende waardes bevatten.")
    return df

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in the DataFrame with simple default values 
    depending on the data type.

    Args:
        df (pd.DataFrame): Dataset with possible missing values.

    Returns:
        pd.DataFrame: Dataset with filled values.

    Author: Felix
    """
    for col in df.columns:
        if df[col].dtype == "O":
            df[col] = df[col].fillna("onbekend")
        else:
            df[col] = df[col].fillna(0)
    return df

def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean text columns by stripping spaces and fixing inconsistent casing.

    Args:
        df (pd.DataFrame): Dataset to clean.

    Returns:
        pd.DataFrame: Dataset with formatted text columns.

    Author: Felix
    """
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].apply(lambda x: x.strip().capitalize() if isinstance(x, str) else x)
    return df

def load_and_clean_missing_values(filepath: str) -> pd.DataFrame:
    """
    Main function to load and clean the hotel bookings data.
    
    Steps:
    1. Load the Excel file.
    2. Remove duplicates.
    3. Fill missing values.
    4. Clean text columns.

    Args:
        filepath (str): Path to the Excel file.

    Returns:
        pd.DataFrame: Cleaned dataset.
    
    Author: Laurence
    """
    df = load_data(filepath)
    df = remove_duplicates(df)
    df = fill_missing_values(df)
    df = clean_text_columns(df)
    return df

## opdracht 2

def load_retail_data(filepath: str) -> pd.DataFrame:
    """
    Load the detailedRetail Excel file into a pandas DataFrame.

    Args:
        filepath (str): Path to the Excel file.

    Returns:
        pd.DataFrame: Loaded retail dataset.

    Author: Laurence
    """
    df = pd.read_excel(filepath)
    print(f"Shape: {df.shape}")
    return df

def calculate_sales_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total sales and percentage per product category.

    Args:
        df (pd.DataFrame): Retail dataset containing 'Category' and 'Sales' columns.

    Returns:
        pd.DataFrame: DataFrame with total and percentage of sales per category.

    Author: Laurence
    """
    category_sales = df.groupby("Category", as_index=False)["Sales"].sum()
    total_sales = category_sales["Sales"].sum()
    category_sales["Percentage"] = round((category_sales["Sales"] / total_sales) * 100, 2)
    return category_sales

def calculate_sales_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total sales and percentage per month.

    Args:
        df (pd.DataFrame): Retail dataset containing 'Month' and 'Sales' columns.

    Returns:
        pd.DataFrame: DataFrame with total and percentage of sales per month.

    Author: Laurence
    """
    month_sales = df.groupby("Month", as_index=False)["Sales"].sum()
    total_sales = month_sales["Sales"].sum()
    month_sales["Percentage"] = round((month_sales["Sales"] / total_sales) * 100, 2)
    return month_sales

def calculate_sales_by_manager(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total sales and percentage per sales manager manually.

    Args:
        df (pd.DataFrame): Retail dataset containing 'SalesManager' and 'Sales' columns.

    Returns:
        pd.DataFrame: DataFrame with total and percentage of sales per manager.

    Author: Felix
    """
    if "SalesManager" not in df.columns:
        print("Kolom 'SalesManager' ontbreekt!")
        return pd.DataFrame()
    managers = {}
    for _, row in df.iterrows():
        name = row["SalesManager"]
        sale = row["Sales"]
        if name not in managers:
            managers[name] = 0
        managers[name] += sale
    total = sum(managers.values())
    result = [{"SalesManager": name, "Sales": sales, "Percentage": round((sales/total)*100,2) if total else 0}
              for name, sales in managers.items()]
    return pd.DataFrame(result)

def generate_basic_retail_report(filepath: str) -> dict[str, pd.DataFrame]:
    """
    Main function to load and analyze the retail sales data.
    Calculates totals and percentages per category and month.

    Args:
        filepath (str): Path to 'detailedRetail.xlsx'.

    Returns:
        dict[str, pd.DataFrame]: Dictionary containing category and month reports.

    Author: Laurence
    """
    df = load_retail_data(filepath)
    category_report = calculate_sales_by_category(df)
    month_report = calculate_sales_by_month(df)
    return {
        "category_report": category_report,
        "month_report": month_report
    }

def save_retail_manager_report(filepath: str) -> pd.DataFrame:
    """
    Generate and save sales report per manager.

    Args:
        filepath (str): Path to the Excel file.

    Returns:
        pd.DataFrame: Report with manager sales and percentages.

    Author: Felix
    """
    df = load_retail_data(filepath)
    report = calculate_sales_by_manager(df)
    report.to_excel("reportRetail_managers.xlsx", index=False)
    print("Rapport opgeslagen als reportRetail_managers.xlsx")
    return report

## opdracht 3

def detect_language_simple(tweet: str) -> str:
    """
    Detect the language of a tweet using simple keyword matching.

    Args:
        tweet (str): Tweet text.

    Returns:
        str: Language code ('en', 'nl', or 'unknown').

    Author: Laurence
    """
    if not isinstance(tweet, str) or tweet.strip() == "":
        return "unknown"
    text = tweet.lower()
    if any(word in text for word in ["the", "and", "is"]):
        return "en"
    elif any(word in text for word in ["de", "het", "een"]):
        return "nl"
    else:
        return "unknown"

def add_language_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'language' column to the DataFrame by detecting language of each tweet.

    Args:
        df (pd.DataFrame): DataFrame with a 'Tweet' column.

    Returns:
        pd.DataFrame: DataFrame with an added 'language' column.

    Author: Laurence
    """
    if "Tweet" not in df.columns:
        raise ValueError("The DataFrame must contain a column named 'Tweet'.")
    df["language"] = df["Tweet"].apply(detect_language_simple)
    return df

def main_language_detection(input_path: str) -> pd.DataFrame:
    """
    Detect language of tweets from an Excel file and save result.

    Args:
        input_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: DataFrame with detected language column.

    Author: Laurence
    """
    df = pd.read_excel(input_path)
    df = add_language_column(df)
    df.to_excel("tweets_language_detected.xlsx", index=False)
    return df

def analyze_sentiment_simple(tweet: str) -> str:
    """
    Simple sentiment analysis using keyword matching.

    Args:
        tweet (str): Tweet text.

    Returns:
        str: 'positive', 'negative', or 'neutral'.

    Author: Felix
    """
    if not isinstance(tweet, str) or tweet.strip() == "":
        return "neutral"
    positive_words = ["good", "happy", "love", "great", "goed", "leuk"]
    negative_words = ["bad", "sad", "hate", "terrible", "slecht", "vreselijk"]
    text = tweet.lower()
    if any(word in text for word in positive_words):
        return "positive"
    elif any(word in text for word in negative_words):
        return "negative"
    else:
        return "neutral"

def add_sentiment_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'sentiment' column to the DataFrame based on tweet text.

    Args:
        df (pd.DataFrame): DataFrame with 'Tweet' column.

    Returns:
        pd.DataFrame: DataFrame with added 'sentiment' column.

    Author: Felix
    """
    if "Tweet" not in df.columns:
        raise ValueError("DataFrame must contain 'Tweet' column.")
    df["sentiment"] = df["Tweet"].apply(analyze_sentiment_simple)
    return df

def main_sentiment_detection(input_path: str) -> pd.DataFrame:
    """
    Perform language detection and sentiment analysis on tweets from Excel.

    Args:
        input_path (str): Path to Excel file with tweets.

    Returns:
        pd.DataFrame: DataFrame with 'language' and 'sentiment' columns.

    Author: Felix
    """
    df = pd.read_excel(input_path)
    df = add_language_column(df)
    df = add_sentiment_column(df)
    df.to_excel("tweets_sentiment_detected.xlsx", index=False)
    print("Bestand opgeslagen als tweets_sentiment_detected.xlsx")
    return df
