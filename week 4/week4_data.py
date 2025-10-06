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

def load_and_clean_missing_values(filepath: str) -> pd.DataFrame:
    """
    Main function to load and clean the hotel bookings data (Felix's part).
    
    Steps:
    1. Load the Excel file.
    2. Explore the dataset.
    3. Remove duplicates.
    4. Fill missing values.
    5. Clean text columns.

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
    Load the detailedRetail.xlsx file into a pandas DataFrame.

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

def generate_basic_retail_report(filepath: str) -> dict[str, pd.DataFrame]:
    """
    Main function (Laurence's part) to load and analyze the retail sales data.
    It calculates totals and percentages per category and month.

    Steps:
    1. Load the Excel file.
    2. Calculate total and percentage of sales per category.
    3. Calculate total and percentage of sales per month.

    Args:
        filepath (str): Path to 'detailedRetail.xlsx'.

    Returns:
        dict[str, pd.DataFrame]: Dictionary containing the two summary DataFrames.

    Author: Laurence
    """
    df = load_retail_data(filepath)
    category_report = calculate_sales_by_category(df)
    month_report = calculate_sales_by_month(df)
    return {
        "category_report": category_report,
        "month_report": month_report
    }

## opdracht 3

from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0

def detect_language(tweet: str) -> str:
    """
    Detect the language of a single tweet using the langdetect package.
    
    Parameters
    ----------
    tweet : str
        The tweet text to analyze.
    
    Returns
    -------
    str
        The detected language code (e.g., 'en', 'fr', 'es').
        Returns 'Unknown' if detection fails or the tweet is empty.
    
    Notes
    -----
    Written by Laurence
    """
    try:
        if not isinstance(tweet, str) or tweet.strip() == "":
            return "Unknown"
        return detect(tweet)
    except LangDetectException:
        return "Unknown"
    except Exception:
        return "Unknown"

def add_language_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a new column 'language' to the DataFrame by detecting the language of each tweet.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least a column named 'Tweet'.
    
    Returns
    -------
    pd.DataFrame
        The same DataFrame with an additional column 'language'.
    
    Notes
    -----
    Written by Laurence
    """
    if "Tweet" not in df.columns:
        raise ValueError("The DataFrame must contain a column named 'Tweet'.")
    
    df["language"] = df["Tweet"].apply(detect_language)
    return df

def main_language_detection(input_path: str) -> pd.DataFrame:
    """
    Main function for Exercise 3.1: Language detection.
    Loads the Excel file with tweets, detects their language, and saves the result.
    
    Parameters
    ----------
    input_path : str
        Path to the Excel file containing tweets.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with detected language for each tweet.
    
    Notes
    -----
    Written by Laurence
    """
    df = pd.read_excel(input_path)
    df = add_language_column(df)
    
    # Sla het resultaat op als Excel-bestand
    df.to_excel("tweets_language_detected.xlsx", index=False)
    
    return df

if __name__ == "__main__":
    df_result = main_language_detection("tweets-1.xlsx")
    print(df_result.head())


##opdracht 3.2

# exercise3_2_sentiment_detection_felix.py
from textblob import TextBlob

def analyze_sentiment_english(tweet: str) -> str:
    """
    Analyze the sentiment of an English tweet using the TextBlob library.
    
    Parameters
    ----------
    tweet : str
        The tweet text to analyze.
    
    Returns
    -------
    str
        'positive' if the polarity > 0,
        'negative' if the polarity < 0,
        'neutral' otherwise.
    
    Notes
    -----
    Written by Laurence
    """
    try:
        if not isinstance(tweet, str) or tweet.strip() == "":
            return "neutral"
        polarity = TextBlob(tweet).sentiment.polarity
        if polarity > 0:
            return "positive"
        elif polarity < 0:
            return "negative"
        else:
            return "neutral"
    except Exception:
        return "neutral"

def add_sentiment_column_english(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'sentiment' column to the DataFrame for English tweets only.
    Non-English tweets are labeled as 'Not analyzed'.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns 'Tweet' and 'language'.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with an additional 'sentiment' column.
    
    Notes
    -----
    Written by Laurence
    """
    if "Tweet" not in df.columns or "language" not in df.columns:
        raise ValueError("DataFrame must contain 'Tweet' and 'language' columns.")
    
    df["sentiment"] = df.apply(
        lambda row: analyze_sentiment_english(row["Tweet"]) if row["language"] == "en" else "Not analyzed",
        axis=1,
    )
    return df

def main_sentiment_detection_english(input_path: str) -> pd.DataFrame:
    """
    Main function for Exercise 3.2 (Felix part):
    Performs sentiment detection on English tweets using TextBlob.
    
    Parameters
    ----------
    input_path : str
        Path to the Excel file containing tweets with a 'language' column.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with an additional 'sentiment' column.
    
    Notes
    -----
    Written by Laurence
    """
    df = pd.read_excel(input_path)
    df = add_sentiment_column_english(df)
    df.to_excel("tweets_sentiment_english.xlsx", index=False)
    return df

if __name__ == "__main__":
    d
