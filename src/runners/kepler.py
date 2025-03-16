from trainers.one_day_ahead_binary_classification_rc1 import main as one_day_ahead_binary_classification_rc1
from datetime import datetime
from loguru import logger
from pathlib import Path
import os
from utils import extract_info_from_filename


if __name__ == '__main__':
    ###########################################################################
    # Select a base directory with "run_id"
    ###########################################################################
    run_id = f'{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}'
    output_dir = []

    ###########################################################################
    # Perform multiple training
    ###########################################################################
    margin = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    # margin = [-3,0]
    for a_margin in margin:
        version = f"M{a_margin}_"
        configuration_for_experience = {"train_margin": a_margin, "test_margin": a_margin, "run_id": run_id, "version": version}
        # # Debug stuff
        # configuration_for_experience.update({'max_iters': 50, 'log_interval': 10})
        # configuration_for_experience.update({"tav_dates": ["2024-01-01", "2025-03-08"]})
        # configuration_for_experience.update({"mes_dates": ["2025-03-09", "2025-03-15"]})
        # Launch training
        results = one_day_ahead_binary_classification_rc1(configuration_for_experience)
        logger.debug(results)
        output_dir.append(Path(results['output_dir']).parent)
    output_dir = list(set(output_dir))
    assert 1 == len(output_dir)
    output_dir = output_dir[0]

    ###########################################################################
    # Select best models
    ###########################################################################
    for one_experience_directory in os.listdir(output_dir):
        for filename in os.listdir(os.path.join(output_dir, one_experience_directory, "checkpoints")):
            if filename.endswith(".pt"):
                info = extract_info_from_filename(filename)
                if info:
                    print(f"Filename: {filename}")
                    print(f"Metric 1 Name: {info['metric1_name']}")
                    print(f"Metric 1 Value: {info['metric1_value']}")
                    print(f"Metric 2 Name: {info['metric2_name']}")
                    print(f"Metric 2 Value: {info['metric2_value']}")
                    print(f"Epoch: {info['epoch']}")
                    print("-" * 50)

    ###########################################################################
    # Do inferences for prediction for tomorrow
    ###########################################################################

