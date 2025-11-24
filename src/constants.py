import os
try:
    IS_RUNNING_ON_CASIR = True if 2 == int(os.getenv("ENV_EXEC_CODE__WEBEAR")) else -1
except:
    IS_RUNNING_ON_CASIR = False

# Determine base finance data directory based on drive availability
def is_drive_writable(path):
    try:
        test_file = os.path.join(path, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        return True
    except:
        return False

if os.path.exists('D:') and os.path.isdir('D:') and is_drive_writable('D:\\'):
    BASE_YFINANCE_DIR = r"D:\Finance\data\yfinance"
    BASE_FORECAST_DIR = r"D:\Finance\data\forecast"
else:
    BASE_YFINANCE_DIR = r"C:\Finance\data\yfinance"
    BASE_FORECAST_DIR  = r"C:\Finance\data\forecast"
if IS_RUNNING_ON_CASIR:
    BASE_YFINANCE_DIR = r"/gpfs/groups/gc014b/cj3272/experiences/yfinance"
    BASE_FORECAST_DIR  = "/gpfs/groups/gc014b/cj3272/experiences/forecast"
# Ensure the directory exists (optional, but helpful if you're writing later)
os.makedirs(BASE_YFINANCE_DIR, exist_ok=True)

# Define output filenames using the base directory
FYAHOO__OUTPUTFILENAME       = os.path.join(BASE_YFINANCE_DIR, "snapshot.pkl")
FYAHOO__OUTPUTFILENAME_DAY   = os.path.join(BASE_YFINANCE_DIR, "snapshot_day.pkl")
FYAHOO__OUTPUTFILENAME_WEEK  = os.path.join(BASE_YFINANCE_DIR, "snapshot_week.pkl")
FYAHOO__OUTPUTFILENAME_MONTH = os.path.join(BASE_YFINANCE_DIR, "snapshot_month.pkl")
FYAHOO__OUTPUTFILENAME_QUARTER = os.path.join(BASE_YFINANCE_DIR, "snapshot_quarter.pkl")
FYAHOO__OUTPUTFILENAME_YEAR  = os.path.join(BASE_YFINANCE_DIR, "snapshot_year.pkl")
FYAHOO_TICKER__OUTPUTFILENAME = os.path.join(BASE_YFINANCE_DIR, "snapshot_ticker.pkl")

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
OUTPUT_DIR_FOURIER_BASED_STOCK_FORECAST = os.path.join(BASE_FORECAST_DIR, r"Fourier_based_stock_forecast")
os.makedirs(OUTPUT_DIR_FOURIER_BASED_STOCK_FORECAST, exist_ok=True)

OUTPUT_DIR_WAVLET_BASED_STOCK_FORECAST = os.path.join(BASE_FORECAST_DIR, r"Wavlet_based_stock_forecast")
os.makedirs(OUTPUT_DIR_WAVLET_BASED_STOCK_FORECAST, exist_ok=True)