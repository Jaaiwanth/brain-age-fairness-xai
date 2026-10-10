#!/bin/bash
# waits for preprocessing to finish, then DeepBrainNet inference on every PASS/WARN subject, then accuracy reports
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf8
while ! grep -q "Wall time" phase2_corrected/full_preprocess.log; do sleep 60; done
echo "preprocessing finished: $(date)" > phase2_corrected/chain_status.txt
python phase2_corrected/infer.py --model dbn --subset all > phase2_corrected/dbn_all.log 2>&1
echo "inference finished: $(date)" >> phase2_corrected/chain_status.txt
python phase2_corrected/accuracy_report.py --pred "G:/My Drive/ML_Project/phase2_corrected/inference/dbn_predictions_all.csv" --tag all_inclusive > phase2_corrected/report_inclusive.log 2>&1
python phase2_corrected/accuracy_report.py --pred "G:/My Drive/ML_Project/phase2_corrected/inference/dbn_predictions_all.csv" --tag all_strict --strict > phase2_corrected/report_strict.log 2>&1
echo "reports finished: $(date)" >> phase2_corrected/chain_status.txt
mkdir -p "G:/My Drive/ML_Project/phase2_corrected/reports"
cp phase2_corrected/reports/accuracy_*all_* "G:/My Drive/ML_Project/phase2_corrected/reports/"
echo "reports copied to Drive: $(date)" >> phase2_corrected/chain_status.txt
