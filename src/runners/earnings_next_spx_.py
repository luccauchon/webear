try:
    from version import sys__name, sys__version
except:
    import sys
    import os
    import pathlib

    # Get the current working directory
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    # print(parent_dir)
    # Add the current directory to sys.path
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version

from constants import TOP10_SP500_TICKERS, FYAHOO_TICKER__OUTPUTFILENAME
from datetime import datetime, timezone
from colorama import init, Fore, Style
import pickle


# Initialize colorama (required on Windows)
init(autoreset=True)

def format_timedelta(td):
    """Return a human-readable string from a timedelta object."""
    if td.total_seconds() < 0:
        return "Earnings already occurred"
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if not parts:
        return "less than a minute"
    return ", ".join(parts)


if __name__ == "__main__":
    now = datetime.now(timezone.utc)  # or your local timezone
    print(f"{Fore.CYAN}{Style.BRIGHT}🚀 Starting Earnings Countdown Tracker...{Style.RESET_ALL}\n")
    with open(FYAHOO_TICKER__OUTPUTFILENAME, 'rb') as f:
        loaded_data = pickle.load(f)

    for ticker in TOP10_SP500_TICKERS:
        print(f"{Fore.YELLOW}🔍 Analyzing earnings surprise for {Style.BRIGHT}{ticker}{Style.RESET_ALL}...")

        # Get stock data
        stock = loaded_data[ticker]

        # Get earnings dates
        earnings_dates = stock['earnings_dates']

        if earnings_dates is None or earnings_dates.empty:
            print(f"  {Fore.RED}⚠️  No upcoming earnings date found.{Style.RESET_ALL}\n")
            continue

        next_earnings = earnings_dates.index[0]  # Most recent/future earnings date

        # Ensure timezone awareness
        if next_earnings.tzinfo is None:
            next_earnings = next_earnings.replace(tzinfo=timezone.utc)

        delta = next_earnings - now
        countdown_str = format_timedelta(delta)

        if delta.total_seconds() < 0:
            status_color = Fore.LIGHTBLACK_EX
            icon = "📅"
        else:
            status_color = Fore.GREEN
            icon = "⏳"

        date_str = next_earnings.strftime('%Y-%m-%d %H:%M %Z')
        print(f"  {status_color}{icon} Next earnings in: {countdown_str} (on {date_str}){Style.RESET_ALL}\n")