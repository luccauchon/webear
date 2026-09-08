import os
import platform
try:
    IS_RUNNING_ON_CASIR = True if 2 == int(os.getenv("ENV_EXEC_CODE__WEBEAR", 0)) else False
except:
    IS_RUNNING_ON_CASIR = False
try:
    IS_RUNNING_ON_LINUX_VMWARE = True if 4 == int(os.getenv("ENV_EXEC_CODE__WEBEAR", 0)) else False
except:
    IS_RUNNING_ON_LINUX_VMWARE = False
try:
    IS_RUNNING_IREQ = False
    if IS_RUNNING_ON_CASIR or IS_RUNNING_ON_LINUX_VMWARE or 8 == int(os.getenv("ENV_EXEC_CODE__WEBEAR", 0)):
        IS_RUNNING_IREQ = True
except:
    IS_RUNNING_IREQ = False
try:
    IS_RUNNING_MAC = False
    if platform.system() == "Darwin":
        IS_RUNNING_MAC = True
except:
    IS_RUNNING_MAC = False
BASE_FINANCE_COMPILED_MODELS             = r"D:\Finance\compiled_models"
BASE_YFINANCE_1MIN_DAILY_SERIALIZER_DIR  = r"D:\Finance\data\daily"
BASE_YFINANCE_30MIN_DAILY_SERIALIZER_DIR = r"D:\Finance\data\daily_30minutes"
BASE_YFINANCE_DIR                        = r"C:\Finance\data\yfinance"
BASE_FORECAST_DIR                        = r"C:\Finance\data\forecast"

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
if IS_RUNNING_MAC:
    BASE_YFINANCE_DIR = r"/Users/luccauchon/WORK/data/yfdataset"
    BASE_FORECAST_DIR = r"/Users/luccauchon/WORK/data"
    BASE_YFINANCE_1MIN_DAILY_SERIALIZER_DIR = r"/Users/luccauchon/WORK/data/daily"
    BASE_YFINANCE_30MIN_DAILY_SERIALIZER_DIR = r"/Users/luccauchon/WORK/data/daily_30minutes"
# Ensure the directory exists (optional, but helpful if you're writing later)
os.makedirs(str(BASE_YFINANCE_DIR), exist_ok=True)
TAURUS_V1_BASE_DIRECTORY                = os.path.join(BASE_FINANCE_COMPILED_MODELS, "taurus", "v1")
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
MOOMOO__PROXY_SPX_FILENAME     = r"D:\Finance\data\moomoo\proxy_for_spx\2026.09.01\df_proxy_spx_moomoo.parquet"
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

#
def GET_EMAILS(dev=False):
    if dev:
        return ("luccauchon@gmail.com", "coalitiondurable@gmail.com")
    # destinataires = ("luccauchon@gmail.com", "luc.vouligny@gmail.com")
    destinataires = ("luccauchon@gmail.com",)
    return destinataires

EMAIL_SENDER_WEBEAR = "luccauchon@gmail.com"
PWD_GOOGLE_API = "thhy qvae fbsb zsbe"

TITLE_WEBEAR = "WEBEAR 1.0"

CLAUSE_NON_RESPONSABILITE = """
Avertissement légal et clause de non-responsabilité (Canada)
À des fins purement éducatives et informatives. Tout le contenu partagé dans ce courriel, incluant les analyses, graphiques, opinions et prédictions concernant le S&P 500 (SPX), des horizons hebdomadaires jusqu'aux stratégies à très court terme (0DTE), est fourni gratuitement et uniquement à titre indicatif et pédagogique.
Absence de conseil en investissement. Les informations contenues dans ce message ne constituent pas, et ne doivent en aucun cas être interprétées comme un conseil en investissement, une recommandation financière personnalisée, ou une sollicitation d'achat ou de vente de titres ou de produits dérivés. L'auteur n'est pas inscrit auprès des autorités canadiennes en valeurs mobilières (telles que l'AMF ou l'OSC) à titre de conseiller en placement ou de gestionnaire de portefeuille.
Risques de pertes importants. Le trading de produits dérivés et d'options à échéance immédiate (0DTE) comporte un niveau de risque extrêmement élevé et ne convient pas à tous les investisseurs. Vous pouvez perdre la totalité, voire plus, de votre capital initial. Les simulations et performances passées ne garantissent pas les résultats futurs.
Utilisation exclusive sur comptes de démonstration (Paper Trading). Les analyses et scénarios présentés dans ce courriel sont conçus pour être appliqués et testés exclusivement sur des comptes de démonstration ou via des simulateurs de marché (Paper Trading). L'auteur encourage vivement ses lecteurs à ne pas engager de capital réel sur la base de ces informations. Tout passage à un environnement de trading réel se fait aux risques et périls de l'utilisateur.
Exclusion totale de responsabilité. En lisant ce courriel, vous reconnaissez que vous êtes le seul responsable de vos décisions financières. L'auteur décline toute responsabilité quant à l'exactitude, l'exhaustivité ou la pertinence des prédictions fournies, et ne pourra être tenu responsable d'aucune perte financière, dommage direct ou indirect, découlant de l'utilisation des informations contenues dans ce message. Il est fortement recommandé de consulter un professionnel de la finance inscrit auprès des autorités réglementaires de votre province avant de prendre toute décision d'investissement.
"""