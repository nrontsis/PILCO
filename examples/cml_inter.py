from inverted_pendulum import run
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("output_file")
parser.add_argument("--bf", type=int)
parser.add_argument("--max_action", type=float)
parser.add_argument("--restarts", type=int)
parser.add_argument("--seed", type=int)
parser.add_argument("--T", type=int)
parser.add_argument("--maxiter", type=int)
parser.add_argument("--J", type=int)
parser.add_argument("--N", type=int)
parser.add_argument("--gpu_id")
parser.add_argument("--linear", type=int)
parser.add_argument("--env_id")
args = parser.parse_args()

d = {}
if args.bf:
    d['bf'] = args.bf
if args.max_action:
    d['max_action'] = args.max_action
if args.restarts:
    d['restarts'] = args.restarts
if args.seed:
    d['seed'] = args.seed
if args.T:
    d["T"] = args.T
if args.J:
    d["J"] = args.J
if args.N:
    d["N"] = args.N
if args.maxiter:
    d["maxiter"] = args.maxiter
if args.gpu_id:
    d["gpu_id"] = args.gpu_id
if args.linear:
    d["linear"] = args.linear

if args.env_id:
    X, params = run(args.env_id, **d)
else:
    X, params = run('InvertedDoublePendulumWrapped', **d)

np.save(args.output_file + "_X.npy", X)
np.save(args.output_file + "_d.npy", params)
