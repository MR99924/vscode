import os
import re
import pandas as pd
from datetime import datetime
import time

# Import PDF text extraction libraries
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False
    print("Warning: PyPDF2 not available. Install with: pip install PyPDF2")

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("Warning: pdfplumber not available. Install with: pip install pdfplumber")

# Function to extract date from filename pattern
def extract_date_from_filename(filename):
    """Extract date from various Bundesbank report filename patterns"""
    try:
        # Pattern 1a: "YYYY-MM-monatsbericht-data"
        match = re.search(r'(\d{4})-(\d{2})-monatsbericht-data', filename)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            return pd.Timestamp(year, month, 1)
        
        # Pattern 1b: "YYYY-MM-monthly-report-data"
        match = re.search(r'(\d{4})-(\d{2})-monthly-report-data', filename)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            return pd.Timestamp(year, month, 1)
        
        # Pattern 2: "Monthly-Report---Month-YYYY"
        month_names = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 
            'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        
        match = re.search(r'Monthly-Report---([A-Za-z]+)-(\d{4})', filename)
        if match:
            month_name = match.group(1)
            year = int(match.group(2))
            
            # Convert month name to number
            month = month_names.get(month_name, None)
            if month:
                return pd.Timestamp(year, month, 1)
        
        print(f"Could not extract date from filename: {filename}")
        return None
    except Exception as e:
        print(f"Error extracting date from {filename}: {e}")
        return None

# Function to extract text from PDF file (for PDFs with extractable text)
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using direct text extraction (no OCR)"""
    try:
        # Check if the file exists
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            return None
            
        text = ""
        
        # Method 1: Using PyPDF2
        if HAS_PYPDF2:
            try:
                print(f"Extracting text from {pdf_path} using PyPDF2...")
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for i, page in enumerate(reader.pages):
                        print(f"Processing page {i+1}/{len(reader.pages)} of {pdf_path}")
                        page_text = page.extract_text() or ""
                        text += page_text + "\n\n"
                
                if text.strip():  # If we got meaningful text
                    return text
                else:
                    print(f"PyPDF2 extracted empty text from {pdf_path}, trying alternate method")
            except Exception as e:
                print(f"PyPDF2 method failed for {pdf_path}: {e}")
                # Will fall back to alternative method
        
        # Method 2: Using pdfplumber
        if HAS_PDFPLUMBER:
            try:
                print(f"Extracting text from {pdf_path} using pdfplumber...")
                with pdfplumber.open(pdf_path) as pdf:
                    text = ""
                    for i, page in enumerate(pdf.pages):
                        print(f"Processing page {i+1}/{len(pdf.pages)} of {pdf_path}")
                        page_text = page.extract_text() or ""
                        text += page_text + "\n\n"
                
                if text.strip():  # If we got meaningful text
                    return text
                else:
                    print(f"pdfplumber extracted empty text from {pdf_path}")
            except Exception as e:
                print(f"pdfplumber method failed for {pdf_path}: {e}")
        
        # If all methods failed or extracted empty text
        print(f"All text extraction methods failed for {pdf_path}")
        return None
        
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

# Main function to process Bundesbank reports
def process_bundesbank_reports(reports_dir, output_dir=None):
    """Process Bundesbank monthly reports - extract text and save to files"""
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(reports_dir) if f.endswith('.pdf') and 
                ('monatsbericht-data' in f or 'Monthly-Report' in f)]
    print(f"Found {len(pdf_files)} PDF reports to process")
    
    # Initialize list to store results
    report_data = []
    
    # Sort files by date to process chronologically
    dated_files = []
    for pdf_file in pdf_files:
        date = extract_date_from_filename(pdf_file)
        if date:
            dated_files.append((date, pdf_file))
    
    # Sort by date
    dated_files.sort()
    
    # Process each PDF file
    for i, (date, pdf_file) in enumerate(dated_files):
        print(f"Processing {i+1}/{len(dated_files)}: {pdf_file} (Date: {date.strftime('%Y-%m')})")
        
        # Full path to PDF
        pdf_path = os.path.join(reports_dir, pdf_file)
        
        # Extract text from PDF directly (no OCR needed for text-based PDFs)
        text = extract_text_from_pdf(pdf_path)
        if not text or len(text.strip()) == 0:
            print(f"No text extracted from {pdf_file}, skipping")
            continue
        
        # Save extracted text to file
        if output_dir:
            # Create filename with date for better organization
            text_filename = f"{date.strftime('%Y-%m')}_extracted_{pdf_file.replace('.pdf', '.txt')}"
            text_file = os.path.join(output_dir, text_filename)
            
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Text extracted and saved to {text_file}")
        
        # Append basic report info to results
        report_data.append({
            "Date": date,
            "Source": pdf_file,
            "Text File": text_filename if output_dir else None
        })
        
        # Add a small delay between processing files to prevent resource overuse
        time.sleep(0.5)
    
    # Convert results into a DataFrame
    if report_data:
        df = pd.DataFrame(report_data)
        
        # Save index of processed files as CSV
        if output_dir:
            csv_path = os.path.join(output_dir, "bundesbank_reports_index.csv")
            df.to_csv(csv_path, index=False)
            print(f"Index of processed reports saved to {csv_path}")
        
        return df
    else:
        print("No data was collected. Check files and PDF processing.")
        return None

# Execute the processing if this script is run directly
if __name__ == "__main__":
    # Directory containing the Bundesbank PDF reports
    reports_dir = r"C:\Users\MR99924\workspace\vscode\Projects\Bundesbank_MonthlyReport"
    
    # Output directory for extracted text
    output_dir = r"C:\Users\MR99924\workspace\vscode\Projects\Bundesbank_MonthlyReport\extracted_text"
    
    # Run the processing
    process_bundesbank_reports(reports_dir, output_dir)