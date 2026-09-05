import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.generate import SyntheticBenchmarkGenerator
from narrative_gen.evaluation import evaluate_narrative_generator_on_dataset

gen = SyntheticBenchmarkGenerator(seed=42)
df_tx, df_dsp = gen.generate_benchmark(n_total_transactions=5000)

eval_results = evaluate_narrative_generator_on_dataset(df_dsp, sample_size=10)

print(f"Citation rate: {eval_results['citation_resolution_rate']}")
print(f"Grounding rate: {eval_results['sentence_grounding_rate']}")
print(f"Guardrail pass rate: {eval_results['guardrail_pass_rate']}")
print(f"Deliberate detection rate: {eval_results['deliberate_corruption_detection_rate']}")
for k, v in eval_results["corruption_tests"].items():
    print(f"{k}: caught={v['caught']}, detected_by={v['detected_by']}, unresolved={v['unresolved_claims_count']}, guardrail={v['guardrail_violations_count']}")
