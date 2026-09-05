import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from narrative_gen.generator import NarrativeGenerator
from narrative_gen.test_narrative_gen import get_mock_dispute_record
from rubric_scorer.aggregator import RubricScorer
from roi_engine.engine import ROIEngine

record = get_mock_dispute_record()
rubric_scorer = RubricScorer()
roi_engine = ROIEngine()
generator = NarrativeGenerator()

rubric_res = rubric_scorer.score_record(record)
roi_res = roi_engine.evaluate_dispute(record)
packet = generator.generate_packet(record, rubric_result=rubric_res, roi_result=roi_res)

import pprint

print(f"Packet is_valid: {packet.is_valid}")
print(f"Citation 100% faithful: {packet.citation_report.is_100_percent_faithful}")
print(f"Citation resolution rate: {packet.citation_report.citation_resolution_rate}")
print(f"Total claims: {packet.citation_report.total_claims}, Valid claims: {packet.citation_report.valid_claims}")
for uc in packet.citation_report.unresolved_claims:
    print(f"UNRESOLVED: {repr(uc)}")

print(f"Guardrail passed: {packet.guardrail_report.passed}")
print(f"Guardrail details: {repr(packet.guardrail_report.details)}")

