import os
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import config

def get_beige_book_files():
    """Retrieve and parse Beige Book text files"""
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
                
                date = pd.Timestamp(year, month, day)  # Parse date
            
            except ValueError as e:
                print(f"Error parsing date for filename {filename}: {e}")
                continue
                
            # Append results to list
            report_data.append({
                "Date": date,
                "Text": text,
                "Filename": filename
            })

    # Convert results into a DataFrame
    df = pd.DataFrame(report_data)
    df["Date"] = pd.to_datetime(df["Date"])  # Ensure Date column is datetime
    df['Date'] = df['Date'] + pd.offsets.MonthEnd(0)  # Adjust to month end
    df = df.sort_values("Date")  # Sort by date
    
    return df

def calculate_textblob_sentiment(texts):
    """Calculate TextBlob sentiment scores for the text data"""
    # Function to get TextBlob sentiment
    def get_sentiment(text):
        blob = TextBlob(text)
        return blob.sentiment
    
    # Apply sentiment analysis to each text
    textblob_scores = [get_sentiment(text) for text in texts]
    
    # Convert scores to match the range in original chart (approximately 3-14 scale)
    # Empirically it might need adjusting after seeing initial results
    scaled_scores = [(score * 100) for score in textblob_scores]
    
    # Create a dataframe with scores
    return pd.DataFrame({'textblob_raw_score': textblob_scores, 
                         'textblob_scaled_score': scaled_scores})

def plot_sentiment_over_time(df):
    """Plot sentiment scores over time"""
    plt.figure(figsize=(14, 8))
    
    # Plot the sentiment scores
    plt.plot(df['Date'], df['textblob_scaled_score'], marker='o', markersize=3, linestyle='-', color='blue', label='TextBlob Sentiment')
    
    # Calculate and plot 1-year moving average (8 reports per year)
    if len(df) > 8:
        df['ma_1yr'] = df['textblob_scaled_score'].rolling(window=8).mean()
        plt.plot(df['Date'], df['ma_1yr'], 
                linestyle='-', color='red', linewidth=1.5, label='1-Year MA')
    
    plt.title('US Beige Book Sentiment Score (TextBlob)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sentiment Score (Scaled)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis limits similar to original chart
    plt.ylim(3, 14)
    
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('beige_book_textblob_sentiment.png', dpi=300)
    plt.close()

def main():
    # Get Beige Book text data
    print("Loading Beige Book files...")
    beige_book_df = get_beige_book_files()
    print(f"Loaded {len(beige_book_df)} Beige Book files")
    
    # Calculate TextBlob sentiment scores
    print("Calculating TextBlob sentiment scores...")
    sentiment_scores = calculate_textblob_sentiment(beige_book_df['Text'])
    
    # Combine with original dataframe
    result_df = pd.DataFrame({
        'Date': beige_book_df['Date'],
        'Filename': beige_book_df['Filename'],
        'textblob_raw_score': sentiment_scores['textblob_raw_score'],
        'textblob_scaled_score': sentiment_scores['textblob_scaled_score']
    })
    
    # Save to CSV
    output_file = "beige_book_textblob_scores.csv"
    result_df.to_csv(output_file, index=False)
    print(f"Sentiment scores saved to {output_file}")
    
    # Create visualization
    print("Creating sentiment trend visualization...")
    plot_sentiment_over_time(result_df)
    print("Visualization saved as beige_book_textblob_sentiment.png")
    
    # Show basic statistics
    print("\nTextBlob Score Statistics (Raw -1 to 1 scale):")
    print(result_df['textblob_raw_score'].describe())
    
    print("\nTextBlob Score Statistics (Scaled to chart range):")
    print(result_df['textblob_scaled_score'].describe())
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()