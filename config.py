"""Configuration constants for GLD ML experiments."""

# Data processing
TRAIN_TEST_SPLIT = 0.8
TRADING_DAYS_PER_YEAR = 252

# Model parameters
RANDOM_STATE = 42
DEFAULT_HORIZONS = [1, 2, 3]

# Backtest parameters
DEFAULT_TRANSACTION_COST = 0.0005
DEFAULT_SLIPPAGE = 0.0005
DEFAULT_LEVERAGE = 1.0

# Signal thresholds
BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45
PROB_THRESHOLD = 0.5

# File paths
MODELS_DIR = "models"
FIGURES_DIR = "figures"
RESULTS_FILE = "results_summary.json"

# Technical indicators
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_WINDOW = 20
BB_STD = 2
ATR_WINDOW = 14