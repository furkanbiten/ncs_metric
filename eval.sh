#!/bin/bash

# Print array values in  lines
echo "RUNNING EVALUATION METRICS FOR CMR"

for DATASET in coco f30k;
do
  for METRIC in spice;
  do
    for RECALL in vse_recall;
    do
      for MODEL in VSEPP SCAN VSRN CVSE;
      do
        python metric.py --dataset $DATASET --metric $METRIC --recall_type $RECALL --model_name $MODEL
      done
    done
  done
done
