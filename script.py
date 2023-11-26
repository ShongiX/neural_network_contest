import subprocess
import time
from training import format_time


def run_bc_training(class_index, version):
    bc_training_command = f"python bs_training.py --class_index {class_index} --version {version}"
    bc_training_process = subprocess.Popen(bc_training_command.split())
    bc_training_process.wait()


def run_bs_evaluate(version):
    bs_evaluate_command = f"python bs_evaluate.py --version {version}"
    bs_evaluate_process = subprocess.Popen(bs_evaluate_command.split())
    bs_evaluate_process.wait()


if __name__ == "__main__":
    start_time = time.time()

    class_indices = [0, 1, 2]

    for version in [10]:
        for class_index in class_indices:
            run_bc_training(class_index, version)
        run_bs_evaluate(version)

    print(f"All processes have finished. Elapsed time: {format_time(time.time() - start_time)}")
