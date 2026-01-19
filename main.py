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
from tasks.cache_fix_verification import CacheFixVerification
from tasks.inspect_cache_metadata import InspectCacheMetadata
from tasks.manual_generate_loop import ManualGenerateLoop
from tasks.manual_injection_test import ManualInjectionTest
from tasks.state_persistence_test import StatePersistenceTest
from tasks.reverse_engineer_cache import ReverseEngineerCache
from tasks.verify_mock_equivalence import VerifyMockEquivalence
from tasks.verify_mamba2cache_optimization import VerifyMamba2CacheOptimization
from tasks.debug_cache_internals import DebugCacheInternals
from tasks.identity_test import IdentityTest
from tasks.inspect_config import InspectConfig
from tasks.final_fix_optimization import FinalFixOptimization
from tasks.inspect_model_structure import InspectModelStructure
from tasks.extract_mamba_source import ExtractMambaSource
from tasks.verify_functional_mamba import VerifyFunctionalMamba
from tasks.functional_optimization import FunctionalOptimization
from tasks.stabilized_alsi import StabilizedALSI
from tasks.null_test import NullTest
from tasks.failed_transition_pca import FailedTransitionPCA
from tasks.failed_contrastive_pca import FailedContrastivePCA

def main():
    parser = argparse.ArgumentParser(description="ALSI Experiment Runner")
    parser.add_argument("--task", type=str, choices=["sensitivity", "train_phi", "train_phi_v2", "robustness", "robustness_psi", "robustness_psi_context", "ab_test_refusal", "ab_test_refusal_multitarget", "ab_test_comprehensive", "ab_test_v1_vs_v2", "direct_probe_test", "direct_probe_security_test", "direct_probe_correct_token", "debug_logits", "temperature_scan", "noise_test", "cache_alignment_test", "cache_fix", "inspect_cache", "manual_generate_loop", "manual_injection_test", "state_persistence_test", "reverse_engineer_cache", "verify_mock_equivalence", "verify_real_opt", "debug_cache", "identity_test", "inspect_config", "final_fix_opt", "inspect_model", "extract_mamba", "verify_functional", "functional_opt", "stabilized_alsi", "null_test", "failed_linear", "all"], default="all")
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
    if args.task == "cache_fix":
        tasks.append(CacheFixVerification())
    if args.task == "inspect_cache":
        tasks.append(InspectCacheMetadata())
    if args.task == "manual_generate_loop":
        tasks.append(ManualGenerateLoop())
    if args.task == "manual_injection_test":
        tasks.append(ManualInjectionTest())
    if args.task == "state_persistence_test":
        tasks.append(StatePersistenceTest())
    if args.task == "reverse_engineer_cache":
        tasks.append(ReverseEngineerCache())
    if args.task == "verify_mock_equivalence":
        tasks.append(VerifyMockEquivalence())
    if args.task == "verify_real_opt":
        tasks.append(VerifyMamba2CacheOptimization())
    if args.task == "debug_cache":
        tasks.append(DebugCacheInternals())
    if args.task == "identity_test":
        tasks.append(IdentityTest())
    if args.task == "inspect_config":
        tasks.append(InspectConfig())
    if args.task == "final_fix_opt":
        tasks.append(FinalFixOptimization())
    if args.task == "inspect_model":
        tasks.append(InspectModelStructure())
    if args.task == "extract_mamba":
        tasks.append(ExtractMambaSource())
    if args.task == "verify_functional":
        tasks.append(VerifyFunctionalMamba())
    if args.task == "functional_opt":
        tasks.append(FunctionalOptimization())
    if args.task == "stabilized_alsi":
        tasks.append(StabilizedALSI())
    if args.task == "null_test":
        tasks.append(NullTest())
        
    for task in tasks:
        print(f"\n=== Task: {task.name} ===")
        task.setup()
        task.run()
        task.report()

if __name__ == "__main__":
    main()

