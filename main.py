import argparse
from tasks.sensitivity import SensitivityScan
from tasks.phi_training import PhiTraining
from tasks.robustness import RobustnessTest
from tasks.failed_transition_pca import FailedTransitionPCA
from tasks.failed_contrastive_pca import FailedContrastivePCA

def main():
    parser = argparse.ArgumentParser(description="ALSI Experiment Runner")
    parser.add_argument("--task", type=str, choices=["sensitivity", "train_phi", "robustness", "failed_linear", "all"], default="all")
    args = parser.parse_args()
    
    tasks = []
    if args.task == "sensitivity" or args.task == "all":
        tasks.append(SensitivityScan())
    if args.task == "failed_linear" or args.task == "all":
        tasks.append(FailedTransitionPCA())
        tasks.append(FailedContrastivePCA())
    if args.task == "train_phi" or args.task == "all":
        tasks.append(PhiTraining())
    if args.task == "robustness" or args.task == "all":
        tasks.append(RobustnessTest())
        
    for task in tasks:
        print(f"\n=== Task: {task.name} ===")
        task.setup()
        task.run()
        task.report()

if __name__ == "__main__":
    main()

