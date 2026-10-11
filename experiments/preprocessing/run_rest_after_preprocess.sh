#!/bin/bash
# waits for preprocessing to finish, then DeepBrainNet inference on every PASS/WARN subject, then accuracy reports
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf8
while ! grep -q "Wall time" experiments/preprocessing/full_preprocess.log; do sleep 60; done
echo "preprocessing finished: $(date)" > experiments/preprocessing/chain_status.txt
python experiments/preprocessing/infer.py --model dbn --subset all > experiments/preprocessing/dbn_all.log 2>&1
echo "inference finished: $(date)" >> experiments/preprocessing/chain_status.txt
python experiments/preprocessing/accuracy_report.py --pred "G:/My Drive/ML_Project/phase2_corrected/inference/dbn_predictions_all.csv" --tag all_inclusive > experiments/preprocessing/report_inclusive.log 2>&1
python experiments/preprocessing/accuracy_report.py --pred "G:/My Drive/ML_Project/phase2_corrected/inference/dbn_predictions_all.csv" --tag all_strict --strict > experiments/preprocessing/report_strict.log 2>&1
echo "reports finished: $(date)" >> experiments/preprocessing/chain_status.txt
mkdir -p "G:/My Drive/ML_Project/phase2_corrected/reports"
cp experiments/preprocessing/reports/accuracy_*all_* "G:/My Drive/ML_Project/phase2_corrected/reports/"
echo "reports copied to Drive: $(date)" >> experiments/preprocessing/chain_status.txt
