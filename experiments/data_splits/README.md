# Split definitions

Only identifiers are stored here (no ages, sexes or sites). The IXI images and demographics are **not** redistributed in this repository.

| File | Contents |
|---|---|
| `ixi_splits.csv` | `IXI_ID, split` for the 485 subjects with a usable label: **339 train / 73 validation / 73 test** |
| `excluded_subjects.csv` | the 14 scans without a usable label and the reason (10 scan IDs are absent from `IXI.xls`, 2 have no age, 2 have duplicate rows that disagree on age/sex) |

Provenance: the labels come from the genuine `IXI.xls` joined on `IXI_ID` (see `experiments/preprocessing/label_fix/build_labels.py`; the earlier join used a 0,1,2.. row counter (`Subject_ID`) as the IXI ID, so ages and sexes were attached to the wrong subjects). The split is stratified by site x sex (70/15/15, `random_state=42`)
using the corrected sex labels, with `sklearn.model_selection.train_test_split`.

The split is fixed. The test split was evaluated once, after all training and checkpoint selection (see [`experiments/sfcn/README.md`](../sfcn/README.md)).

To use these definitions, join `ixi_splits.csv` to demographics you obtain yourself from the IXI project
(<https://brain-development.org/ixi-dataset/>, CC BY-SA 3.0, cite the source).
