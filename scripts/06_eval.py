"""Generate completeness-first Phase 1 diagnostics from saved artifacts."""
import argparse
import json
import sys
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.cache import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    args=p.parse_args()
    protocol=yaml.safe_load(args.config.read_text())
    gate={'status':'failed','missing_or_failed_gates':['CV, controls, pilot/runtime and parity pending'],
          'feature_set':protocol['feature_set']}
    atomic_json(args.results/'phase_gate.json',gate)
    print(json.dumps(gate,indent=2))
    return 1


if __name__=='__main__':
    raise SystemExit(main())
