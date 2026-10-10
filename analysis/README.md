# analysis/

`sex_disparity.py` compares prediction error between males and females, controlling for age and site. The script itself is valid.

**No results are published from it at the moment.** The output files that were generated earlier
(`analysis/output/sex_disparity_*.png|txt`) were computed from results whose age and sex labels came from a mis-aligned join
(a row-index column had been treated as the IXI ID), so their numbers are **invalid** and have been removed from version control.
See `CASE_STUDY_iAudit_vs_ml_brain.md` for the full explanation and `splits/README.md` for how the labels were rebuilt.

The analysis has not yet been repeated on the corrected labels and the fine-tuned model.
