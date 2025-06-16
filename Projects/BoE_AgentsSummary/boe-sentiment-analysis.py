import os
import re
import requests
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import PyPDF2
from io import BytesIO
from datetime import datetime
import time

# Define keyword lists adapted for UK economic terminology
inflationary_keywords = ["pressure", "shortage", "cost-push", "backlogs", "tight", "strain", "strains", 
                         "strained", "shortage", "shortages", "shortfalls", "tight", "tightness", "restricted",
                         "supply", "strike", "strikes", "expense", "margin", "escalating", "overheating", 
                         "struggle", "climbing", "hikes", "costly", "margins", "pressures", "eroding",
                         "retention", "spiral", "erodes", "escalates", "bottleneck", "bottlenecks"]

expansionary_keywords = ["vigorous", "expand", "expansion", "favourable", "pickup", "optimistic", "optimism", 
                         "improved", "accelerated", "strengthened", "upbeat", "demand", "strong", "robust", 
                         "boom", "grow", "expanding", "growing", "increasing", "developing", "solid", "approval", 
                         "accept", "flourish", "booming", "thriving", "flourishing", "boost", "stimulus", 
                         "strengthened", "easing", "accommodative", "positive", "healthy", "bullish", "firm", 
                         "momentum", "rapid"]

deflationary_keywords = ["overstocked", "restrictive", "falling", "deflation", "disinflation", "wage cuts", 
                         "surplus", "weak", "glut", "slowing", "fall", "deflate", "deflating", "receding"]

recessionary_keywords = ["hesitant", "layoffs", "layoff", "resistance", "unemployment", "shortfall", "recession", 
                        "recessionary", "unsettled", "cautious", "fear", "concern", "declining", "falling", 
                        "weak", "deteriorating", "worsening", "worsen", "downturn", "depressed", "depression", 
                        "cutbacks", "delinquencies", "deteriorated", "softer", "softening", "subdued"]

# Helper function to calculate keyword scores
def calculate_keyword_score(text, keywords):
    # Convert text to lowercase and split into words
    words = text.lower().split()
    # Count occurrences of each keyword
    keyword_count = sum(words.count(keyword.lower()) for keyword in keywords)
    # Normalize by total word count
    return keyword_count / len(words) if len(words) > 0 else 0

# Function to download and extract text from PDFs
def download_and_extract_pdf(url):
    """Download PDF from URL and extract text content"""
    try:
        # Add a user agent header to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Download the PDF
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Create a PDF reader object
        pdf_file = BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from all pages
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
        
        return text
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None
    except PyPDF2.errors.PdfReadError as e:
        print(f"Error reading PDF from {url}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error processing {url}: {e}")
        return None

# Function to extract date from URL
def extract_date_from_url(url):
    """Extract the date from BoE URL pattern"""
    try:
        # Dictionary for last day of each month
        month_end_days = {
            1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 
            7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
        }
        
        # Convert month name to number
        month_dict = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        # Check for quarterly report pattern (e.g., "/2019/2019-q1.pdf")
        quarterly_pattern = r"/agents-summary/(\d{4})/\d{4}-(q[1-4])\.pdf"
        quarterly_match = re.search(quarterly_pattern, url)
        
        if quarterly_match:
            year = int(quarterly_match.group(1))
            quarter = quarterly_match.group(2).lower()
            
            # Map quarters to their end months
            quarter_to_month = {'q1': 3, 'q2': 6, 'q3': 9, 'q4': 12}
            month = quarter_to_month.get(quarter)
            
            if month:
                # Adjust February for leap years
                if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
                    day = 29
                else:
                    day = month_end_days[month]
                
                return pd.Timestamp(year, month, day)  # Last day of the quarter
        
        # Check for monthly report pattern (e.g., "/2019/january-2019.pdf")
        monthly_pattern = r"/agents-summary/(\d{4})/([a-zA-Z]+)-\d{4}\.pdf"
        monthly_match = re.search(monthly_pattern, url)
        
        if monthly_match:
            year = int(monthly_match.group(1))
            month_str = monthly_match.group(2)
            
            month = month_dict.get(month_str.lower())
            
            if month:
                # Adjust February for leap years
                if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
                    day = 29
                else:
                    day = month_end_days[month]
                
                return pd.Timestamp(year, month, day)  # Last day of month
        
        # If regex patterns don't match, try alternative approach
        parts = url.split('/')
        filename = parts[-1].split('.')[0]  # Get filename without extension
        
        if '-' in filename:
            # Check if it's a quarterly format (e.g., "2019-q1")
            if 'q' in filename.lower():
                year_str, quarter = filename.split('-')
                year = int(year_str)
                quarter = quarter.lower()
                
                # Map quarters to their end months
                quarter_to_month = {'q1': 3, 'q2': 6, 'q3': 9, 'q4': 12}
                month = quarter_to_month.get(quarter)
                
                if month and year:
                    # Adjust February for leap years
                    if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
                        day = 29
                    else:
                        day = month_end_days[month]
                    
                    return pd.Timestamp(year, month, day)
            else:
                # Try monthly format (e.g., "january-2019")
                month_str, year_str = filename.split('-')
                year = int(year_str)
                
                month = month_dict.get(month_str.lower())
                
                if month and year:
                    # Adjust February for leap years
                    if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
                        day = 29
                    else:
                        day = month_end_days[month]
                    
                    return pd.Timestamp(year, month, day)
        
        # If all parsing attempts fail
        print(f"Unable to extract date from URL: {url}")
        return None
    
    except Exception as e:
        print(f"Error extracting date from {url}: {e}")
        return None

# Function to generate URLs for BoE reports
def generate_urls(start_year=1997, end_year=None):
    """Generate URLs for Bank of England Agents' Summary reports"""
    if end_year is None:
        end_year = datetime.now().year
    
    months = [
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december'
    ]
    
    quarters = ['q1', 'q2', 'q3', 'q4']
    
    urls = []
    
    # Generate monthly report URLs
    for year in range(start_year, end_year + 1):
        for month in months:
            # Skip future months in the current year
            if year == datetime.now().year and months.index(month) + 1 > datetime.now().month:
                continue
                
            url = f"https://www.bankofengland.co.uk/-/media/boe/files/agents-summary/{year}/{month}-{year}.pdf"
            urls.append(url)
    
    # Generate quarterly report URLs
    for year in range(start_year, end_year + 1):
        for quarter in quarters:
            # Skip future quarters in the current year
            current_quarter = (datetime.now().month - 1) // 3 + 1
            if year == datetime.now().year and int(quarter[1]) > current_quarter:
                continue
                
            url = f"https://www.bankofengland.co.uk/-/media/boe/files/agents-summary/{year}/{year}-{quarter}.pdf"
            urls.append(url)
    
    return urls

# Main function to analyze sentiment
def analyze_boe_sentiment(start_year=1997, end_year=None, output_path=None, local_files_path=None):
    """Analyze sentiment in Bank of England Agents' Summary reports"""
    # Generate URLs
    urls = generate_urls(start_year, end_year)
    print(f"Generated {len(urls)} URLs to process")
    
    # Initialize list to store results
    report_data = []
    
    # Create output directory if it doesn't exist
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Process each URL
    for i, url in enumerate(urls):
        print(f"Processing {i+1}/{len(urls)}: {url}")
        
        # Extract date from URL
        date = extract_date_from_url(url)
        if not date:
            continue
        
        # Download and extract text from PDF
        text = download_and_extract_pdf(url)
        if not text:
            continue
        
        # Calculate scores
        inflationary_score = calculate_keyword_score(text, inflationary_keywords)
        expansionary_score = calculate_keyword_score(text, expansionary_keywords)
        deflationary_score = calculate_keyword_score(text, deflationary_keywords)
        recessionary_score = calculate_keyword_score(text, recessionary_keywords)
        
        # Perform sentiment analysis using TextBlob
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        
        # Append results to list
        report_data.append({
            "Date": date,
            "Source": f"BoE Website: {url}",
            "Inflationary Score": inflationary_score,
            "Expansionary Score": expansionary_score,
            "Deflationary Score": deflationary_score,
            "Recessionary Score": recessionary_score,
            "Sentiment Score": sentiment_score,
            "Text": text
        })
        
        # Add a small delay to be polite to the BoE server
        time.sleep(2)
    
    # Process local text files if a path is provided
    if local_files_path and os.path.exists(local_files_path):
        process_local_files(local_files_path, report_data)
    
    # Convert results into a DataFrame
    if report_data:
        df = pd.DataFrame(report_data)
        df["Date"] = pd.to_datetime(df["Date"])  # Ensure Date column is a datetime object
        df = df.sort_values("Date")  # Sort by date
        
        # Save results to CSV
        if output_path:
            csv_path = os.path.join(output_path, "boe_sentiment_analysis.csv")
            df[["Date", "Source", "Inflationary Score", "Expansionary Score", 
                "Deflationary Score", "Recessionary Score", "Sentiment Score"]].to_csv(csv_path, index=False)
            print(f"Analysis results saved to {csv_path}")
        
        # Plot the scores over time
        plt.figure(figsize=(12, 6))
        plt.plot(df["Date"], df["Inflationary Score"], marker="o", label="Inflationary Score", color="red")
        plt.plot(df["Date"], df["Expansionary Score"], marker="o", label="Expansionary Score", color="green")
        plt.plot(df["Date"], df["Sentiment Score"], marker="o", label="Sentiment Score", color="blue")
        #plt.plot(df["Date"], df["Deflationary Score"], marker="o", label="Deflationary Score", color="purple")
        #plt.plot(df["Date"], df["Recessionary Score"], marker="o", label="Recessionary Score", color="black")
        plt.title("Bank of England Sentiment Analysis")
        plt.xlabel("Date")
        plt.ylabel("Score")
        plt.legend()
        plt.grid()
        
        # Save plot
        if output_path:
            plt_path = os.path.join(output_path, "boe_sentiment_plot.png")
            plt.savefig(plt_path)
            print(f"Plot saved to {plt_path}")
        
        plt.show()
        
        # Optional: Analyze frequent inflationary and expansionary words in latest report
        if len(df) > 0:
            latest_text = df.iloc[-1]["Text"]
            inflationary_words = [word for word in latest_text.lower().split() if word in inflationary_keywords]
            expansionary_words = [word for word in latest_text.lower().split() if word in expansionary_keywords]
            
            print("Most Frequent Inflationary Words:", Counter(inflationary_words).most_common(10))
            print("Most Frequent Expansionary Words:", Counter(expansionary_words).most_common(10))
        
        return df
    else:
        print("No data was collected. Check URLs and connectivity.")
        return None

# Function to process local text files
def process_local_files(local_files_path, report_data):
    """Process local text files with DDMMYY naming format"""
    print(f"Processing local text files from: {local_files_path}")
    
    # Get list of text files in the directory
    file_list = [f for f in os.listdir(local_files_path) if f.endswith('.txt') and len(f.split('.')[0]) == 6 and f.split('.')[0].isdigit()]
    
    for filename in file_list:
        try:
            # Extract date from filename (DDMMYY format)
            date_str = filename.split('.')[0]
            
            if len(date_str) != 6:
                print(f"Skipping file with invalid name format: {filename}")
                continue
                
            day = int(date_str[:2])
            month = int(date_str[2:4])
            year = int(date_str[4:])
            
            # Adjust year (assuming 20xx for years < 50, 19xx for years >= 50)
            if year < 50:
                year += 2000
            else:
                year += 1900
                
            # Read file content
            file_path = os.path.join(local_files_path, filename)
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                text = file.read()
                
            # Calculate scores
            inflationary_score = calculate_keyword_score(text, inflationary_keywords)
            expansionary_score = calculate_keyword_score(text, expansionary_keywords)
            deflationary_score = calculate_keyword_score(text, deflationary_keywords)
            recessionary_score = calculate_keyword_score(text, recessionary_keywords)
            
            # Perform sentiment analysis using TextBlob
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity
            
            # Create a date object
            date = pd.Timestamp(year, month, day)
            
            # Append results to list
            report_data.append({
                "Date": date,
                "Source": f"Local file: {filename}",
                "Inflationary Score": inflationary_score,
                "Expansionary Score": expansionary_score,
                "Deflationary Score": deflationary_score,
                "Recessionary Score": recessionary_score,
                "Sentiment Score": sentiment_score,
                "Text": text
            })
            
            print(f"Processed local file: {filename}, Date: {date}")
            
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue
            
    print(f"Completed processing {len(file_list)} local text files")

# Execute the analysis if this script is run directly
if __name__ == "__main__":
    # Set the output path where results will be saved
    
    
    # Set the path to local text files
    local_files_path = "C:\\Users\\MR99924\\workspace\\vscode\\Projects\\BoE_AgentsSummary"
    output_path = local_files_path

    # Run the analysis for a specific period
    current_year = datetime.now().year
    analyze_boe_sentiment(
        start_year=2005, 
        end_year=current_year, 
        output_path=output_path,
        local_files_path=local_files_path
    )