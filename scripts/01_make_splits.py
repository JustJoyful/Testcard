"""Generate once, then verify fixed outer folds and inner validation roles."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.splits import freeze_splits

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    print(json.dumps(freeze_splits(args.manifest,args.output,args.seed),indent=2))
