import requests
import os
from dotenv import load_dotenv

load_dotenv()

class FMPFetcher:
    """
    A class to fetch quantitative financial data using the Financial Modeling Prep (FMP) API.
    """
    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP API key not provided or found in environment variables.")
        print("FMPFetcher initialized successfully.")

    def _perform_request(self, endpoint, params=None):
        if params is None:
            params = {}
        
        params['apikey'] = self.api_key
        url = f"{self.BASE_URL}{endpoint}"
        print(f"DEBUG: Performing request to URL: {url}")
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            print("DEBUG: Successfully received API response.")
            
            if isinstance(data, dict) and "Error Message" in data:
                print(f"FMP API Error: {data['Error Message']}")
                return None
            
            return data
        except requests.exceptions.RequestException as e:
            print(f"An error occurred during the FMP API request: {e}")
            return None

    def get_analyst_estimates_data(self):
        """
        Fetches analyst estimates for major Indian stocks using the endpoint confirmed
        to work with the user's API key.
        """
        print("Fetching analyst estimates from FMP...")
        
        stocks = {
            "Reliance Industries": "RELIANCE.NS",
            "Tata Consultancy Services": "TCS.NS"
        }
        estimates_data = {}

        for name, symbol in stocks.items():
            # Using the /analyst-estimates endpoint
            data = self._perform_request(f"/analyst-estimates/{symbol}")
            
            if data and isinstance(data, list) and len(data) > 0:
                # Get the most recent estimate
                latest_estimate = data[0]
                estimates_data[name] = {
                    "target_price_high": latest_estimate.get("priceTargetHigh"),
                    "target_price_avg": latest_estimate.get("priceTarget"),
                    "target_price_low": latest_estimate.get("priceTargetLow"),
                    "date": latest_estimate.get("date")
                }
                print(f"DEBUG: Successfully processed estimates for {name}.")
            else:
                estimates_data[name] = {
                    "target_price_high": "N/A",
                    "target_price_avg": "N/A",
                    "target_price_low": "N/A",
                    "date": "N/A"
                }
                print(f"DEBUG: Could not find estimate data for {name} ({symbol}).")
        
        return estimates_data

    def get_dashboard_data(self):
        """
        Orchestrator method to fetch all quantitative data for the dashboard.
        """
        print("Starting quantitative data acquisition from FMP API...")
        dashboard_data = {
            "analyst_estimates": self.get_analyst_estimates_data()
        }
        print("FMP API data acquisition complete.")
        return dashboard_data

# Example of how to use the class
if __name__ == '__main__':
    fetcher = FMPFetcher()
    numeric_data = fetcher.get_dashboard_data()
    
    print("\n--- Analyst Estimates ---")
    print(numeric_data['analyst_estimates'])
