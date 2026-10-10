"""Restore the flat run_* files into the runs/ folder layout expected by analyze.py.

usage: python restore_runs.py
"""
import os, glob, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
for f in glob.glob(os.path.join(HERE, "run_*.json")) + glob.glob(os.path.join(HERE, "run_*.jsonl")):
    name = os.path.basename(f)[4:]
    sub, rest = name.split("__", 1) if "__" in name else (None, name)
    dst = os.path.join(HERE, "runs", sub, rest) if sub else os.path.join(HERE, "runs", rest)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(f, dst)
print("restored into", os.path.join(HERE, "runs"))
