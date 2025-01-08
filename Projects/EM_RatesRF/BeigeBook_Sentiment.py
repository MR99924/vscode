import os 
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# Define inflationary and expansionary keywords
inflationary_keywords = ["shutdowns", "shutdown", "tie-up", "tie-ups", "disruptions", "disruption", "pressure", "shortage", "cost-push", "controls", "backlogs", 
                         "scrambling", "tight", "strain", "strains", "strained", "shortage", "shortages", "shortfalls", "tight", "tightness", "restricted",
                         "demand-pull", "supply", "strike", "strikes", "striking", "stoppage", "stoppages", "expense", "margin", 
                         "expenses","escalating","overheating", "struggle", "climbing","hikes","costly","margins",
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


# Helper function to calculate keyword scores
def calculate_keyword_score(text, keywords):
    # Convert text to lowercase and split into words
    words = text.lower().split()
    # Count occurrences of each keyword
    keyword_count = sum(words.count(keyword.lower()) for keyword in keywords)
    # Normalize by total word count
    return keyword_count / len(words) if len(words) > 0 else 0

# Path to folder containing text files
folder_path = r"C:\Users\MR99924\.spyder-py3\workspace\Projects\US_BeigeBook\Text_files(1974-Present)"

# Initialize list to store results
report_data = []

# Loop through text files
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        
        # Read file content
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        
        # Calculate inflationary and expansionary scores
        inflationary_score = calculate_keyword_score(text, inflationary_keywords)
        expansionary_score = calculate_keyword_score(text, expansionary_keywords)
        deflationary_score = calculate_keyword_score(text, deflationary_keywords)
        recessionary_score = calculate_keyword_score(text, recessionary_keywords)
        
        
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
            "Inflationary Score": inflationary_score,
            "Expansionary Score": expansionary_score,
            "Deflationary Score": deflationary_score,
            "Recessionary Score": recessionary_score,
            "Text": text
        })

# Convert results into a DataFrame
df = pd.DataFrame(report_data)
df["Date"] = pd.to_datetime(df["Date"])  # Ensure Date column is a datetime object
df = df.sort_values("Date")  # Sort by date

# Plot the scores over time
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Inflationary Score"], marker="o", label="Inflationary Score", color="red")
plt.plot(df["Date"], df["Expansionary Score"], marker="o", label="Expansionary Score", color="green")
#plt.plot(df["Date"], df["Deflationary Score"], marker="o", label="Deflationary Score", color="blue")
#plt.plot(df["Date"], df["Recessionary Score"], marker="o", label="Recessionary Score", color="black")
plt.title("Macro Score Sentiment Over Time")
plt.xlabel("Date")
plt.ylabel("Score")
plt.legend()
plt.grid()
plt.show()

# Save results to a CSV file for reference
output_file = r"C:\Users\MR99924\.spyder-py3\workspace\Projects\US_BeigeBook\macroeconomic_sentiment_analysis_dateutil.csv"
df[["Date", "Inflationary Score", "Expansionary Score", "Deflationary Score", "Recessionary Score"]].to_csv(output_file, index=False)
print(f"Analysis results saved to {output_file}")

# Optional: Analyze frequent inflationary and expansionary words
latest_text = df.iloc[-1]["Text"]
inflationary_words = [word for word in latest_text.lower().split() if word in inflationary_keywords]
expansionary_words = [word for word in latest_text.lower().split() if word in expansionary_keywords]
# deflationary_words = [word for word in latest_text.lower().split() if word in deflationary_keywords]
# recessionary_words = [word for word in latest_text.lower().split() if word in recessionary_keywords]

print("Most Frequent Inflationary Words:", Counter(inflationary_words).most_common(10))
print("Most Frequent Expansionary Words:", Counter(expansionary_words).most_common(10))
# print("Most Frequent Deflationary Words:", Counter(deflationary_words).most_common(10))
# print("Most Frequent Recessionary Words:", Counter(recessionary_words).most_common(10))