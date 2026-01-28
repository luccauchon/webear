try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version

from datetime import datetime, timedelta
from runners.MMI_realtime import main as MMI_realtime


def main(configuration):
    result = MMI_realtime(configuration)
    prediction_date = (result['date'] + timedelta(days=1)).strftime('%Y-%m-%d')
    if configuration.dataset_id == "week":
        prediction_date = (result['date'] + timedelta(weeks=1)).strftime('%Y-%m-%d')
    if result['signal'] == "Choppy":
        lower, upper = result['prices_threshold'][0], result['prices_threshold'][1]
        if configuration.verbose:
            print(f"[{result['signal']}] For the closing price on {prediction_date}, price should be between {lower:0.0f} and {upper:0.0f}. Actual price: {result['prices']:0.0f}.")
    return result

