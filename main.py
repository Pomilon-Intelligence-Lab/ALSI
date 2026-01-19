import argparse
from tasks.sensitivity import SensitivityScan
from tasks.phi_training import PhiTraining
from tasks.phi_training_v2 import PhiTrainingV2
from tasks.robustness import RobustnessTest
from tasks.robustness_psi import RobustnessPsiTest
from tasks.robustness_psi_context import RobustnessPsiContextTest
from tasks.ab_test_refusal import RefusalABTest
from tasks.ab_test_refusal_multitarget import RefusalABTestMultiTarget
from tasks.ab_test_comprehensive import ComprehensiveABTest
from tasks.ab_test_v1_vs_v2 import PhiComparisonTest
from tasks.direct_probe_test import DirectProbeTest
from tasks.direct_probe_security_test import DirectProbeSecurityTest
from tasks.direct_probe_correct_token import DirectProbeCorrectToken
from tasks.debug_logits import DebugLogits
from tasks.temperature_scan import TemperatureScan
from tasks.noise_test import NoiseTest
from tasks.cache_alignment_test import CacheAlignmentTest
from tasks.null_test import NullTest
from tasks.failed_transition_pca import FailedTransitionPCA
from tasks.failed_contrastive_pca import FailedContrastivePCA

def main():
    parser = argparse.ArgumentParser(description="ALSI Experiment Runner")
    parser.add_argument("--task", type=str, choices=["sensitivity", "train_phi", "train_phi_v2", "robustness", "robustness_psi", "robustness_psi_context", "ab_test_refusal", "ab_test_refusal_multitarget", "ab_test_comprehensive", "ab_test_v1_vs_v2", "direct_probe_test", "direct_probe_security_test", "direct_probe_correct_token", "debug_logits", "temperature_scan", "noise_test", "cache_alignment_test", "null_test", "failed_linear", "all"], default="all")
    args = parser.parse_args()
    
    tasks = []
    if args.task == "sensitivity" or args.task == "all":
        tasks.append(SensitivityScan())
    if args.task == "failed_linear" or args.task == "all":
        tasks.append(FailedTransitionPCA())
        tasks.append(FailedContrastivePCA())
    if args.task == "train_phi" or args.task == "all":
        tasks.append(PhiTraining())
    if args.task == "train_phi_v2" or args.task == "all":
        tasks.append(PhiTrainingV2())
    if args.task == "robustness" or args.task == "all":
        tasks.append(RobustnessTest())
    if args.task == "robustness_psi" or args.task == "all":
        tasks.append(RobustnessPsiTest())
    if args.task == "robustness_psi_context" or args.task == "all":
        tasks.append(RobustnessPsiContextTest())
    if args.task == "ab_test_refusal":
        tasks.append(RefusalABTest())
    if args.task == "ab_test_refusal_multitarget":
        tasks.append(RefusalABTestMultiTarget())
    if args.task == "ab_test_comprehensive":
        tasks.append(ComprehensiveABTest())
    if args.task == "ab_test_v1_vs_v2":
        tasks.append(PhiComparisonTest())
    if args.task == "direct_probe_test":
        tasks.append(DirectProbeTest())
    if args.task == "direct_probe_security_test":
        tasks.append(DirectProbeSecurityTest())
    if args.task == "direct_probe_correct_token":
        tasks.append(DirectProbeCorrectToken())
    if args.task == "debug_logits":
        tasks.append(DebugLogits())
    if args.task == "temperature_scan":
        tasks.append(TemperatureScan())
    if args.task == "noise_test":
        tasks.append(NoiseTest())
    if args.task == "cache_alignment_test":
        tasks.append(CacheAlignmentTest())
    if args.task == "null_test":
        tasks.append(NullTest())
        
    for task in tasks:
        print(f"\n=== Task: {task.name} ===")
        task.setup()
        task.run()
        task.report()

if __name__ == "__main__":
    main()

