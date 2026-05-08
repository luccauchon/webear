import os
try:
    IS_RUNNING_ON_CASIR = True if 2 == int(os.getenv("ENV_EXEC_CODE__WEBEAR", 0)) else False
except:
    IS_RUNNING_ON_CASIR = False
try:
    IS_RUNNING_ON_LINUX_VMWARE = True if 4 == int(os.getenv("ENV_EXEC_CODE__WEBEAR", 0)) else False
except:
    IS_RUNNING_ON_LINUX_VMWARE = False

BASE_YFINANCE_1MIN_DAILY_SERIALIZER_DIR  = r"D:\Finance\data\daily"
BASE_YFINANCE_30MIN_DAILY_SERIALIZER_DIR = r"D:\Finance\data\daily_30minutes"
BASE_YFINANCE_DIR     = r"C:\Finance\data\yfinance"
BASE_FORECAST_DIR     = r"C:\Finance\data\forecast"
if os.path.exists('D:') and os.path.isdir('D:'):
    BASE_YFINANCE_DIR = r"D:\Finance\data\yfinance"
    BASE_FORECAST_DIR = r"D:\Finance\data\forecast"
if IS_RUNNING_ON_CASIR:
    BASE_YFINANCE_DIR = r"/gpfs/groups/gc014b/cj3272/experiences/yfinance"
    BASE_FORECAST_DIR = "/gpfs/groups/gc014b/cj3272/experiences/forecast"
if IS_RUNNING_ON_LINUX_VMWARE:
    BASE_YFINANCE_DIR = "/home/luccauchon/REALTIME/data/yfinance"
    BASE_FORECAST_DIR = "/home/luccauchon/REALTIME/data/forecast"
if os.getenv("SPECIFIC_BASE_YFINANCE_DIR__WEBEAR") is not None:  # User can override the default directory where to find the data
    BASE_YFINANCE_DIR = os.getenv("SPECIFIC_BASE_YFINANCE_DIR__WEBEAR")

# Ensure the directory exists (optional, but helpful if you're writing later)
os.makedirs(str(BASE_YFINANCE_DIR), exist_ok=True)

# Define output filenames using the base directory
FYAHOO__OUTPUTFILENAME         = os.path.join(str(BASE_YFINANCE_DIR), "snapshot.pkl")
FYAHOO__OUTPUTFILENAME_DAY     = os.path.join(str(BASE_YFINANCE_DIR), "snapshot_day.pkl")
FYAHOO__OUTPUTFILENAME_WEEK    = os.path.join(str(BASE_YFINANCE_DIR), "snapshot_week.pkl")
FYAHOO__OUTPUTFILENAME_MONTH   = os.path.join(str(BASE_YFINANCE_DIR), "snapshot_month.pkl")
FYAHOO__OUTPUTFILENAME_QUARTER = os.path.join(str(BASE_YFINANCE_DIR), "snapshot_quarter.pkl")
FYAHOO__OUTPUTFILENAME_YEAR    = os.path.join(str(BASE_YFINANCE_DIR), "snapshot_year.pkl")
FYAHOO_TICKER__OUTPUTFILENAME  = os.path.join(str(BASE_YFINANCE_DIR), "snapshot_ticker.pkl")
FYAHOO_SPX500__OUTPUTFILENAME  = os.path.join(str(BASE_YFINANCE_DIR), "sp500_daily_data.parquet")
FYAHOO_GITHUB_DIRECTORY        = os.path.join(str(BASE_YFINANCE_DIR), "yfdataset")

# Constants
NB_WORKERS = os.cpu_count()

MY_TICKERS = [
    "^XSP", "^GSPC", "^VIX", "^VVIX", "^SKEW", "^VIX1D", "^VIX9D", "^VIX3M", "^VIX6M", "^VVIX", "^VIX1Y", "SPY", "QQQ",
    "AAPL", "ADBE", "AFRM", "AMD", "AMZN",
    "ASST", "AVGO", "BAC", "BBAI", "BRK-B", "CLF", "COST", "CRM", "DBRG", "GOOGL",
    "GOOG", "HD", "HIMS", "HOOD", "HYG", "INTC", "JPM", "LLY", "MA", "META", "MSFT",
    "^NDX", "NFLX", "NVDA", "OPEN", "ORCL", "PINS", "QCOM", "PLTR", "RDDT", "RKT", "RSP",
    "SOFI", "SNDK", "TSLA", "TSM", "UUUU", "U", "V", "WMT"
]

MY_TICKERS_SMALL_SET = [
    "^XSP", "^GSPC", "^VIX", "^VVIX", "^SKEW", "^VIX1D", "^VIX9D", "^VIX3M", "^VIX6M", "^VVIX", "^VIX1Y", "SPY", "QQQ", "RSP", "^NDX",
]

# Top 10 S&P 500 tickers as of 2025-10-29
TOP10_SP500_TICKERS = ["AMZN", "AAPL", "AVGO", "BRK-B", "GOOG", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]

# Forecast output directory (still assumes D: is preferred)
OUTPUT_DIR_FOURIER_BASED_STOCK_FORECAST = os.path.join(BASE_FORECAST_DIR, r"Fourier_based_stock_forecast")
os.makedirs(OUTPUT_DIR_FOURIER_BASED_STOCK_FORECAST, exist_ok=True)

OUTPUT_DIR_WAVLET_BASED_STOCK_FORECAST = os.path.join(BASE_FORECAST_DIR, r"Wavlet_based_stock_forecast")
os.makedirs(OUTPUT_DIR_WAVLET_BASED_STOCK_FORECAST, exist_ok=True)

# Optional: Load FRED_API_KEY from environment for security
# import os
# FRED_API_KEY = os.getenv('FRED_API_KEY')
FRED_API_KEY = '213742dc08592772cb9502214cdc4397'