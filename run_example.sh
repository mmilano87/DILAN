#!/usr/bin/env bash

python ../src/dilan.py \
  --network-1-name IBD \
  --network-1-edges ../data/IBD_edges.txt \
  --network-1-scores ../data/IBD_rewiring_scores.txt \
  --network-2-name T2D \
  --network-2-edges ../data/T2D_edges.txt \
  --network-2-scores ../data/T2D_rewiring_scores.txt \
  --output-dir ../output/IBD_vs_T2D
