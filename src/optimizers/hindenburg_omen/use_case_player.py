from pathlib import Path
from optimizers.hindenburg_omen.realtime_backtest_and_hyperparameter_search_optuna import run_realtime_only

def main():
    directory = Path(fr"D:\Temp2\use_case_alpha")
    for file in directory.iterdir():
        if not file.is_file():  # Vérifie que c'est bien un fichier et non un dossier
            continue
        info = run_realtime_only(file, False)
        is_active_now = info["is_active_now"]
        if is_active_now:
            event_direction = info["event_direction"]
            current_count = info["current_count"]
            cluster_threshold = info["cluster_threshold"]
            direction_word = "DROP" if event_direction == "drop" else "SPIKE"
            _tmp_str = f"SIGNAL ACTIVE:  {'YES - PREDICTING ' + direction_word if is_active_now else 'NO - NEUTRAL'}"
            _tmp_str2 = f"Current Count:  {current_count} / {cluster_threshold}"
            print(f"{file.name} {info['last_date'].strftime('%Y-%m-%d')}  {_tmp_str}  {_tmp_str2}")
        #import sys
        #sys.exit()
# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    main()


