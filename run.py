import argparse
import config
from example_analysis import ExampleAnalysis
from triage_time_analysis import TriageTimeAnalysis

def parse_args():
    ap = argparse.ArgumentParser("run.py")
    ap.add_argument('--feature', '-f', type=int, required=True,
                    help='Which feature to run (0: basic example, 1: first analysis, 2: second analysis, 3: triage time analysis)')
    ap.add_argument('--user', '-u', type=str, required=False, help='Optional user filter')
    ap.add_argument('--label', '-l', type=str, required=False, help='Optional label filter')
    #ap.add_argument('--show-plot', action='store_true', help='Show plots when available')
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config.overwrite_from_args(args)

    if args.feature == 0:
        ExampleAnalysis().run()
    elif args.feature == 1:
        pass  # TODO call first analysis
    elif args.feature == 2:
        pass  # TODO call second analysis
    elif args.feature == 3:
        TriageTimeAnalysis().run()
    else:
        print("Need to specify which feature to run with --feature flag.")
