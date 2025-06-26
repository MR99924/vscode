import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
from datetime import datetime

# Define inflationary and expansionary keywords
inflationary_keywords = ["shutdowns", "shutdown", "tie-up", "tie-ups", "disruptions", "disruption", "pressure", "shortage", "cost-push", "controls", "backlogs", 
                         "scrambling", "tight", "strain", "strains", "strained", "shortage", "shortages", "shortfalls", "tight", "tightness", "restricted",
                         "demand-pull", "supply", "strike", "strikes", "striking", "stoppage", "stoppages", "expense", "margin", "tariff", "tariffs",
                         "expenses","escalating", "overheating", "struggle", "climbing", "hikes", "costly", "margins",
                         "pressures","eroding","escalating","retention", "spiral","erodes","escalates","explosive","shortfall",
                         "bottleneck","bottlenecks","abnormal", "erosion", "control", "blockage"]
expansionary_keywords = ["vigorous", "vigor", "expand", "expansion", "favourable", "pickup", "optimistic", "optimism", "improved", "accelerated", "strengthened", "upbeat", "vigor", 
                         "demand", "expansion", "strong", "robust", "boom", "grow", 
                         "expanding","growing","increasing","developing", "solid", "approval", "approvals", "accept", "flourish",
                         "booming","thriving","flourishing","rising output","boost","stimulus", "solid", "strengthened","easing","accommodative",
                         "positive","healthy", "bullish", "firm", "jumpstart", "momentum", "rapid", "strengthened", "improved", "expansion", "turnover"]
deflationary_keywords = ["overstocked", "restrictive", "falling", "deflation", "disinflation", "wage cuts", "surplus","weak", "glut", 
                         "slowing", "fall", "deflate", "deflating", "receding", "caution", "pessimism"]
recessionary_keywords = ["hesitant", "layoffs", "layoff", "resistance", "unemployment", "shortfall", "recession", "recessionary", "unsettled", "cautious", "fear", "concern", "declining", "falling", "lockdown", "COVID",
                         "weak", "deterorating", "worsening", "worsen", "downturn", "depressed", "depression", "depressing", "cutbacks", "delinquencies", "deteriorated"]

def calculate_keyword_score(text, keywords):
    """
    Calculate the frequency of specific keywords in a text, normalized by text length.
    
    Parameters:
        text (str): The text to analyze
        keywords (list): List of keywords to search for
        
    Returns:
        float: Normalized score indicating keyword frequency
    """
    # Convert text to lowercase and split into words
    words = text.lower().split()
    # Count occurrences of each keyword
    keyword_count = sum(words.count(keyword.lower()) for keyword in keywords)
    # Normalize by total word count
    return keyword_count / len(words) if len(words) > 0 else 0

def extract_date_from_filename(filename):
    """
    Extract date information from Bundesbank report filenames.
    Handles both YYYY-MM_extracted_YYYY-MM-monatsbericht-data.txt and
    YYYY-MM_extracted_Monthly-Report---Month-YYYY.txt formats.
    
    Parameters:
        filename (str): The filename to parse
        
    Returns:
        pd.Timestamp or None: Parsed date, or None if parsing failed
    """
    try:
        # Handle old format: YYYY-MM_extracted_YYYY-MM-monatsbericht-data.txt
        if "monatsbericht-data" in filename:
            date_part = filename.split("_extracted_")[0]
            year = int(date_part[:4])
            month = int(date_part[5:7])
            day = 1  # Assume first day of month for monthly reports
            return pd.Timestamp(year, month, day)
            
        # Handle new format: YYYY-MM_extracted_Monthly-Report---Month-YYYY.txt
        elif "Monthly-Report" in filename:
            date_part = filename.split("_extracted_")[0]
            year = int(date_part[:4])
            month = int(date_part[5:7])
            day = 1  # Assume first day of month for monthly reports
            return pd.Timestamp(year, month, day)
            
        # Try to extract YYYY-MM pattern from anywhere in the filename
        else:
            match = re.search(r'(\d{4})-(\d{2})', filename)
            if match:
                year = int(match.group(1))
                month = int(match.group(2))
                day = 1
                return pd.Timestamp(year, month, day)
                
            print(f"Couldn't parse date from filename using standard patterns: {filename}")
            return None
            
    except ValueError as e:
        print(f"Error parsing date for filename {filename}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error processing filename {filename}: {e}")
        return None

def process_text_file(file_path):
    """
    Process a single text file to extract its content and calculate sentiment scores.
    
    Parameters:
        file_path (str): Path to the text file
        
    Returns:
        dict or None: Dictionary with scores and metadata, or None if processing failed
    """
    try:
        filename = os.path.basename(file_path)
        
        # Extract date from filename
        date = extract_date_from_filename(filename)
        if not date:
            return None
            
        # Read file content
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            text = file.read()
            
        # Calculate sentiment scores
        inflationary_score = calculate_keyword_score(text, inflationary_keywords)
        expansionary_score = calculate_keyword_score(text, expansionary_keywords)
        deflationary_score = calculate_keyword_score(text, deflationary_keywords)
        recessionary_score = calculate_keyword_score(text, recessionary_keywords)
        
        # Perform sentiment analysis using TextBlob
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        
        # Return results
        return {
            "Date": date,
            "Filename": filename,
            "Inflationary Score": inflationary_score,
            "Expansionary Score": expansionary_score,
            "Deflationary Score": deflationary_score,
            "Recessionary Score": recessionary_score,
            "Sentiment Score": sentiment_score,
            "Text": text
        }
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def visualize_results(df, output_path):
    """
    Create and save visualizations from the analysis results.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing analysis results
        output_path (str): Directory where plots should be saved
    """
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Created output directory: {output_path}")
        
        # Plot 1: Monthly time series
        plt.figure(figsize=(12, 6))
        plt.plot(df["Date"], df["Sentiment Score"], marker="o", label="Sentiment Score", color="blue")
        plt.title("Bundesbank Report Sentiment Analysis")
        plt.xlabel("Date")
        plt.ylabel("Score")
        plt.legend()
        plt.grid()
        
        monthly_plot_path = os.path.join(output_path, "bundesbank_sentiment_analysis.png")
        plt.savefig(monthly_plot_path)
        print(f"Monthly time series plot saved to: {monthly_plot_path}")
        plt.close()
        
        # Plot 2: Comparison of inflationary and expansionary indicators
        plt.figure(figsize=(12, 6))
        plt.plot(df["Date"], df["Inflationary Score"], marker="o", label="Inflationary Score", color="red")
        plt.plot(df["Date"], df["Expansionary Score"], marker="o", label="Expansionary Score", color="black")
        plt.title("Inflation vs Expansion Indicators in Bundesbank Reports")
        plt.xlabel("Date")
        plt.ylabel("Score")
        plt.legend()
        plt.grid()
        
        comparison_plot_path = os.path.join(output_path, "bundesbank_inflation_recession_comparison.png")
        plt.savefig(comparison_plot_path)
        print(f"Comparison plot saved to: {comparison_plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def analyze_word_frequency(df):
    """
    Analyze the frequency of economic indicator words in the most recent report.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing analysis results
    """
    try:
        if len(df) == 0:
            print("No data available for word frequency analysis.")
            return
            
        # Get the most recent report
        latest_idx = df["Date"].idxmax()
        latest_text = df.loc[latest_idx, "Text"]
        latest_date = df.loc[latest_idx, "Date"].strftime("%Y-%m-%d")
        
        print(f"\nWord frequency analysis for report dated {latest_date}:")
        
        # Count occurrences of each category of keywords
        inflationary_words = [word for word in latest_text.lower().split() if word in inflationary_keywords]
        expansionary_words = [word for word in latest_text.lower().split() if word in expansionary_keywords]
        deflationary_words = [word for word in latest_text.lower().split() if word in deflationary_keywords]
        recessionary_words = [word for word in latest_text.lower().split() if word in recessionary_keywords]
        
        print("Most Frequent Inflationary Words:", Counter(inflationary_words).most_common(10))
        print("Most Frequent Expansionary Words:", Counter(expansionary_words).most_common(10))
        print("Most Frequent Deflationary Words:", Counter(deflationary_words).most_common(5))
        print("Most Frequent Recessionary Words:", Counter(recessionary_words).most_common(5))
        
    except Exception as e:
        print(f"Error analyzing word frequency: {e}")

def analyze_bundesbank_reports(input_path, output_path=None, start_year=None, end_year=None):
    """
    Main function to analyze sentiment in Bundesbank Monthly Reports.
    
    Parameters:
        input_path (str): Directory containing text files to analyze
        output_path (str): Directory where results should be saved (defaults to input_path)
        start_year (int): Optional filter to include only reports from this year onwards
        end_year (int): Optional filter to include only reports up to this year
        
    Returns:
        pd.DataFrame: DataFrame containing analysis results
    """
    # Set default output path if not provided
    if output_path is None:
        output_path = input_path
        
    # Set default end year if not provided
    if end_year is None:
        end_year = datetime.now().year
        
    print(f"Starting analysis of Bundesbank reports from {input_path}")
    print(f"Results will be saved to {output_path}")
    
    if start_year:
        print(f"Including only reports from {start_year} to {end_year}")
    
    # Initialize list to store results
    report_data = []
    
    # Check if directory exists
    if not os.path.exists(input_path):
        print(f"Error: Input directory {input_path} does not exist.")
        return None
    
    # Get list of text files
    file_list = [f for f in os.listdir(input_path) if f.endswith('.txt')]
    if not file_list:
        print(f"No text files found in {input_path}")
        return None
        
    print(f"Found {len(file_list)} text files to process.")
    
    # Process each file
    processed_count = 0
    for filename in sorted(file_list):
        file_path = os.path.join(input_path, filename)
        
        # Process the file
        result = process_text_file(file_path)
        
        # Apply year filter if specified
        if result and start_year:
            file_year = result["Date"].year
            if file_year < start_year or file_year > end_year:
                continue
                
        # Add to results if processing succeeded
        if result:
            report_data.append(result)
            processed_count += 1
            
            # Print progress for every 10 files
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} files...")
    
    print(f"Successfully processed {processed_count} out of {len(file_list)} files.")
    
    # Create DataFrame if we have data
    if report_data:
        df = pd.DataFrame(report_data)
        df["Date"] = pd.to_datetime(df["Date"])  # Ensure Date column is a datetime object
        df = df.sort_values("Date")  # Sort by date
        
        # Save results to CSV
        csv_path = os.path.join(output_path, "bundesbank_sentiment_analysis.csv")
        df[["Date", "Filename", "Inflationary Score", "Expansionary Score", 
            "Deflationary Score", "Recessionary Score", "Sentiment Score"]].to_csv(csv_path, index=False)
        print(f"Analysis results saved to {csv_path}")
        
        # Create yearly averages
        df['Year'] = df['Date'].dt.year
        yearly_avg = df.groupby('Year').mean(numeric_only=True)
        yearly_csv_path = os.path.join(output_path, "bundesbank_yearly_sentiment.csv")
        yearly_avg.to_csv(yearly_csv_path)
        print(f"Yearly averages saved to {yearly_csv_path}")
        
        # Create visualizations
        visualize_results(df, output_path)
        
        # Analyze word frequencies
        analyze_word_frequency(df)
        
        return df
    else:
        print("No data was collected. Check file names and formats.")
        return None

# Execute the analysis if this script is run directly
if __name__ == "__main__":
    # Set the input and output paths
    input_path = r"C:\Users\MR99924\workspace\vscode\Projects\Bundesbank_MonthlyReport\extracted_text"
    output_path = r"C:\Users\MR99924\workspace\vscode\Projects\Bundesbank_MonthlyReport"
    
    # Run the analysis for all available reports
    df = analyze_bundesbank_reports(input_path, output_path)
    
    # You can also run the analysis for a specific time period
    # df = analyze_bundesbank_reports(input_path, output_path, start_year=2015, end_year=2024)