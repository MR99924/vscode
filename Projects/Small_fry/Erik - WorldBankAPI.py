import requests
import pandas as pd
import matplotlib.pyplot as plt





def get_lorenz_curve_data(country_code="usa", year=2025):
    """
    Fetch Lorenz curve data from the World Bank Poverty and Inequality Platform API.
    """
    # Base URL for the API
    base_url = "https://api.worldbank.org/pip/v1/lorenz-curve"
    
    # Try with only country and year parameters first
    params = {
        'country': country_code,
        'year': year
    }
    
    # Make the request
    response = requests.get(base_url, params=params)
    
    # Check if request was successful
    if response.status_code == 200:
        data = response.json()
        # Convert to DataFrame
        df = pd.DataFrame(data)
        return df
    elif response.status_code == 404:
        # This likely means invalid country/year combination
        print(f"Error: Invalid country or year combination")
        print(f"Response: {response.text}")
        # If we got a list of valid years, display them
        try:
            error_data = response.json()
            if "year" in str(error_data):
                print("Valid years may include:", error_data.get("year", []))
        except:
            pass
        return None
    else:
        print(f"Error: API request failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        
        # Try the alternative approach with default parameters
        print("Trying with default parameters...")
        default_params = {
            'cum_welfare': 0.234,
            'cum_population': 0.234,
            'mean': 1.9,
            'times_mean': 1.9,
            'popshare': 0.234,
            'n_bins': 100,
            'format': 'json'
        }
        
        # Add country and year if provided
        if country_code:
            default_params['country'] = country_code
        if year:
            default_params['year'] = year
        
        response = requests.get(base_url, params=default_params)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            return df
        else:
            print(f"Second attempt failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return None

def plot_lorenz_curve(df):
    """
    Plot Lorenz curve from the returned data.
    
    Parameters:
    - df: DataFrame with Lorenz curve data
    """
    if df is None or df.empty:
        print("No data available to plot")
        return
    
    # Extract cumulative population and welfare columns
    # Actual column names may vary, check the returned data structure
    pop_col = 'cumulative_population' if 'cumulative_population' in df.columns else 'cum_population'
    welfare_col = 'cumulative_welfare' if 'cumulative_welfare' in df.columns else 'cum_welfare'
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot perfect equality line (45-degree line)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Equality')
    
    # Plot Lorenz curve
    plt.plot(df[pop_col], df[welfare_col], 'b-', label='Lorenz Curve')
    
    # Add labels and title
    plt.xlabel('Cumulative Share of Population')
    plt.ylabel('Cumulative Share of Income/Consumption')
    plt.title('Lorenz Curve')
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Show plot
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example: Get Lorenz curve data for Brazil in 2019
    # Replace with your desired country and year
    lorenz_data = get_lorenz_curve_data(country_code="BRA", year="all")
    
    # Print the data
    if lorenz_data is not None:
        print(lorenz_data.head())
    
    # Plot the Lorenz curve
    plot_lorenz_curve(lorenz_data)