import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import config
from config import get_bloomberg_date, classify_period

def main():

    dt_from = config.DEFAULT_DATE_FROM
    dt_to = config.DEFAULT_DATE_TO

    us_cpi = get_bloomberg_date(
            list(config.CPI_TICKERS), 
            dt_from, 
            dt_to, 
            periodicity=config.BLOOMBERG_MONTHLY_PERIODICITY
        )
    
    us_act = get_bloomberg_date(
        list(config.ACT_TICKERS),
        dt_from,
        dt_to,
        periodicity=config.BLOOMBERG_MONTHLY_PERIODICITY
    )

    # Initialize list to store results
    report_data = []

    # Loop through text files
    for filename in sorted(os.listdir(config.FOLDER_PATH)):
        if filename.endswith(".txt"):
            file_path = os.path.join(config.FOLDER_PATH, filename)
            
            # Read file content
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
            
            
            # Extract and parse the date from filename (e.g., 'DDMMYY.txt')
            try:
                date_str = filename.split(".")[0]  # Get the 'DDMMYY' part
                day = int(date_str[:2])
                month = int(date_str[2:4])
                year = int(date_str[4:])
                if year < 50:
                    year += 2000
                else:
                    year += 1900
                
                date = pd.Timestamp(year, month, day)  # Parse date using dateutil
            
            except ValueError as e:
                print(f"Error parsing date for filename {filename}: {e}")
                continue
            # Append results to list
            report_data.append({
                "Date": date,
                "Text": text
            })

    # Convert results into a DataFrame
    df = pd.DataFrame(report_data)
    df["Date"] = pd.to_datetime(df["Date"])  # Ensure Date column is a datetime object
    df['Date'] = df['Date'] + pd.offsets.MonthEnd(0)
    df = df.sort_values("Date")  # Sort by date

    print(df)

    # Preprocess text data
    vectorizer = TfidfVectorizer(stop_words='english')
    text_features = vectorizer.fit_transform(df['Text'])

    # Combine text features with economic data
    combined_features = pd.concat([pd.DataFrame(text_features.toarray()), us_cpi, us_act], axis=1)

    print(combined_features)

    # Create labels (e.g., 0 for low growth/low inflation, 1 for high growth/high inflation, etc.)
    df['label'] = df.apply(lambda row: classify_period(row['inflation'], row['growth']), axis=1)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(combined_features, df['label'], test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()