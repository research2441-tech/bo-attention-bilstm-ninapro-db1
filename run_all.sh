#!/bin/sh
# Complete pipeline for NinaPro DB1.  Set RAW to the folder holding S{1..27}_A1_E{1,2,3}.mat
# and DB1 to the folder where the windowed subject files are written.
set -e
RAW=${RAW:-raw}
export DB1=${DB1:-db1}
mkdir -p runs figs
python3 prep_db1.py "$RAW" "$DB1"
python3 classical.py runs
python3 tune.py INIT 0 2 runs & python3 tune.py INIT 1 2 runs; wait
python3 tune.py BO runs & python3 tune.py RS runs; wait
python3 panel_ref.py runs
for t in ATTN_BO ATTN_RS ATTN_DEF BILSTM_DEF TCN_DEF ATTN_BO_FULL; do python3 final.py $t runs; done
for t in ATTN_BO ATTN_DEF; do python3 final.py $t runs 3,7,11,15,19,24 1,2; done
for m in ATTN_DEF SVM SVMREP; do python3 leak.py $m runs; done
python3 leak_overlap.py "$RAW" runs
for m in ATTN_BO ATTN_DEF TCN_DEF SVM LDA; do python3 loso.py $m calib runs; done
python3 loso.py ATTN_DEF global runs
python3 probe.py faith runs
python3 probe.py robust runs
python3 efficiency.py runs
python3 analyze.py runs "$DB1" results.json
python3 fig_diagrams.py figs
python3 fig_results.py results.json runs "$DB1" figs
