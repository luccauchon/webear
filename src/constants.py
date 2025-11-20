import os

# Determine base finance data directory based on drive availability
if os.path.exists('D:') and os.path.isdir('D:'):
    BASE_UFINANCE_DIR = r"D:\Finance\data\yfinance"
    BASE_FORCAST_DIR =  r"D:\Finance\data\forecast"
else:
    BASE_UFINANCE_DIR = r"C:\Finance\data\yfinance"
    BASE_FORCAST_DIR  = r"C:\Finance\data\forecast"

# Ensure the directory exists (optional, but helpful if you're writing later)
os.makedirs(BASE_UFINANCE_DIR, exist_ok=True)

# Define output filenames using the base directory
FYAHOO__OUTPUTFILENAME       = os.path.join(BASE_UFINANCE_DIR, "snapshot.pkl")
FYAHOO__OUTPUTFILENAME_DAY   = os.path.join(BASE_UFINANCE_DIR, "snapshot_day.pkl")
FYAHOO__OUTPUTFILENAME_WEEK  = os.path.join(BASE_UFINANCE_DIR, "snapshot_week.pkl")
FYAHOO__OUTPUTFILENAME_MONTH = os.path.join(BASE_UFINANCE_DIR, "snapshot_month.pkl")
FYAHOO__OUTPUTFILENAME_QUARTER = os.path.join(BASE_UFINANCE_DIR, "snapshot_quarter.pkl")
FYAHOO__OUTPUTFILENAME_YEAR  = os.path.join(BASE_UFINANCE_DIR, "snapshot_year.pkl")
FYAHOO_TICKER__OUTPUTFILENAME = os.path.join(BASE_UFINANCE_DIR, "snapshot_ticker.pkl")

# Constants
NB_WORKERS = os.cpu_count()

MY_TICKERS = [
    "^XSP", "^GSPC", "^VIX", "SPY", "QQQ", "AAPL", "ADBE", "AFRM", "AMD", "AMZN",
    "ASST", "AVGO", "BAC", "BBAI", "BRK-B", "CLF", "COST", "CRM", "DBRG", "GOOGL",
    "GOOG", "HD", "HIMS", "HOOD", "INTC", "JPM", "LLY", "MA", "META", "MSFT",
    "NFLX", "NVDA", "OPEN", "ORCL", "PINS", "QCOM", "PLTR", "RDDT", "RKT",
    "SOFI", "TSLA", "TSM", "UUUU", "U", "V", "WMT"
]

# Top 10 S&P 500 tickers as of 2025-10-29
TOP10_SP500_TICKERS = ["AMZN", "AAPL", "AVGO", "BRK-B", "GOOG", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]

# Forecast output directory (still assumes D: is preferred)
OUTPUT_DIR_FOURIER_BASED_STOCK_FORECAST = os.path.join(BASE_FORCAST_DIR, r"Fourier_based_stock_forecast")
os.makedirs(OUTPUT_DIR_FOURIER_BASED_STOCK_FORECAST, exist_ok=True)