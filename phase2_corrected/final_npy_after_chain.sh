#!/bin/bash
# after the main chain (preprocess -> inference -> reports) is done, top up the .npy folder with the remaining subjects
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf8
while ! grep -q "reports copied to Drive" phase2_corrected/chain_status.txt 2>/dev/null; do sleep 60; done
python phase2_corrected/to_npy.py > phase2_corrected/to_npy_2.log 2>&1
echo "npy final: $(date)" >> phase2_corrected/chain_status.txt
