#!/usr/bin/env bash

python -m bleurt.score \
  -candidate_file=bleurt/test_data/candidates \
  -reference_file=bleurt/test_data/references \
  -bleurt_checkpoint=my_new_bleurt_checkpoint/export/bleurt_best/1599523006
