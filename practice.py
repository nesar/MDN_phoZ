# practicing using booleans between shell scripts and python scripts

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--feature', dest='feature', action='store_true')
parser.set_defaults(feature=False)
args = parser.parse_args()
print(args.feature)