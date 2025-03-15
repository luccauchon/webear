from trainers.one_day_ahead_binary_classification_rc1 import main as one_day_ahead_binary_classification_rc1

if __name__ == '__main__':
    margin = [-3, -2.5, -2, -1.5, -1, 0, 1, 1.5, 2, 2.5, 3]
    for a_margin in margin:
        run_id = f"M{a_margin}_"
        one_day_ahead_binary_classification_rc1({"margin": a_margin, "run_id": run_id})