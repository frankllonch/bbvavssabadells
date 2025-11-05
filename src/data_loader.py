import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from pathlib import Path

DATA_PATH = "/Users/frankllonch/Desktop/quattroporte/aprendizado de máquina/bbvavssabadells/data/raw"

def fetch_stock_data(ticker: str, start_date: str = "2024-01-01", end_date: str = "2025-01-01"):
    """Download and clean stock data from Yahoo Finance."""
    print(f"Downloading data for {ticker} from Yahoo Finance...")

    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]  # keep standard columns

    save_path = Path("/Users/frankllonch/Desktop/quattroporte/aprendizado de máquina/bbvavssabadells/data/raw") / f"{ticker}_data.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(save_path, index=False)

    print(f"✅ Saved {ticker} data to {save_path}")
    return data


def update_data(ticker: str):
    """Update existing CSV file with the latest data."""
    file_path = os.path.join(DATA_PATH, f"{ticker}_data.csv")
    
    if not os.path.exists(file_path):
        print(f"No existing data found for {ticker}. Downloading full history...")
        return fetch_stock_data(ticker)
    
    existing_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    last_date = existing_data.index[-1].strftime("%Y-%m-%d")
    new_data = yf.download(ticker, start=last_date)
    
    updated_data = pd.concat([existing_data, new_data]).drop_duplicates()
    updated_data.to_csv(file_path)
    
    print(f"✅ {ticker} data updated successfully.")
    return updated_data