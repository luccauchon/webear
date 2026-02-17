try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version

from datetime import datetime, timedelta
from runners.MMI_realtime import main as MMI_realtime
from utils import get_weekday_name, next_weekday


def main(configuration):
    result = MMI_realtime(configuration)
    prediction_date = None
    if configuration.dataset_id == "day":
        prediction_date = next_weekday(result['date']).strftime('%Y-%m-%d')
    elif configuration.dataset_id == "week":
        prediction_date = (result['date'] + timedelta(weeks=1)).strftime('%Y-%m-%d')
    elif configuration.dataset_id == "month":
        prediction_date = (result['date'] + timedelta(weeks=4)).strftime('%Y-%m-%d')
    else:
        assert False
    if result['signal'] == "Choppy":
        lower, upper = result['prices_threshold'][0], result['prices_threshold'][1]
        if configuration.verbose:
            if configuration.dataset_id == "day":
                print(f"Last date in data frame: {result['date'].strftime('%Y-%m-%d')}  ({get_weekday_name(result['date'])})")
            else:
                print(f"Last date in data frame: {result['date'].strftime('%Y-%m-%d')}")
            print(f"[{result['signal']}] For the closing price on {prediction_date}, price should be between {lower:0.0f} and {upper:0.0f}. Actual price: {result['prices']:0.0f}.")
    return result

