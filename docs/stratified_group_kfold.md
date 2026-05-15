# Stratified Group K-Fold Split

## Motivation

The original tile-level `StratifiedKFold` split balanced tissue labels well, but it allowed tiles from the same slide to appear in both training and validation partitions. Because tiles from a single whole-slide image are not independent, this can leak slide-specific visual patterns into validation and make validation performance overly optimistic.

To reduce this leakage risk, the splitter now supports `kfold_strategy: stratified_group`, implemented in `split/kfold_split.py`. This mode uses `StratifiedGroupKFold`, stratifying by tissue label while treating `slide_id` as the grouping variable. As a result, all tiles from the same slide are assigned to exactly one validation fold.

The original tile-level `StratifiedKFold` strategy is still available as `kfold_strategy: stratified`. It can be run when slide-level separation is not required, for example for debugging, comparison against older experiments, or workflows where tile-level stratification is intentionally preferred.

## Stratification Target and Rare-Class Protocol

Labels are derived per tile from the `roi_coverage_*` columns:

- `label` is the tissue class with the highest ROI coverage.
- `background` is assigned when a tile has zero ROI coverage.
- `tissue_prop` is the sum of all `roi_coverage_*` values for the tile.

For grouped splitting, the important constraint is no longer only the number of tiles per class. `StratifiedGroupKFold` also needs each stratification class to be represented across enough distinct groups. In this project, a group is a slide, so each retained class must appear in at least `n_folds` distinct `slide_id` values.

The rare-class protocol for `stratified_group` is therefore slide-based:

- The splitter counts the number of distinct slides containing each label.
- Any label present in fewer than `n_folds` slides is considered rare for grouped splitting.
- All tiles with rare labels are dropped before fold assignment.
- A warning lists each dropped label and the number of slides in which it appears.
- If the rare-class filtering would drop every tile, the script raises a `ValueError`.

This differs from the older `stratified` strategy. The tile-level strategy collapses rare tile-count classes into `background` only for stratification. The grouped strategy does not collapse rare classes into `background`, because background can be sparse or filtered upstream and because collapsing would not reliably solve the slide-level group constraint.

## Fold Assignment

`StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=...)` is fitted with:

- `y`: the derived tile label.
- `groups`: the tile `slide_id`.

Each tile is assigned to its validation fold, `fold in [0, n_folds)`. For any fold `k`, the validation set is `fold == k` and the training set is the complement, `fold != k`.

The group constraint means that each slide appears in only one validation fold. Consequently, no tile from a validation slide appears in the corresponding training split.

## Output

The script writes one parquet artifact, `kfold_tiles.parquet`, under the configured `mlflow_artifact_path`.

The output keeps the filtered input tile dataset and adds fold metadata:

| Column | Type | Source |
| --- | --- | --- |
| `tissue_prop` | float | Sum of `roi_coverage_*` columns. |
| `fold` | int8 | Validation fold index in `[0, n_folds)`. |

For `stratified_group`, rare labels may be removed before writing the parquet. When this happens, the logged metric `dropped_rare_class_tiles` records how many tiles were excluded.

Note: labels are derived inside the splitter for stratification and statistics. The current implementation does not add a new `label` column to the output parquet unless such a column is already present in the input dataset.

## Logged Statistics

Per-fold metrics are emitted to MLflow:

- `fold_<k>_train_tiles`: number of tiles outside validation fold `k`.
- `fold_<k>_val_tiles`: number of validation tiles in fold `k`.
- `fold_<k>_val_tile_pct`: fraction of retained tiles assigned to validation fold `k`.
- `fold_<k>_val_slides`: number of distinct validation slides in fold `k`.
- `fold_<k>_val_tissue_prop_mean`: mean tissue coverage in validation fold `k`.
- `fold_<k>_val_tissue_prop_std`: tissue coverage standard deviation in validation fold `k`.
- `fold_size_cv`: coefficient of variation of validation fold sizes.
- `dropped_rare_class_tiles`: number of dropped rare-class tiles, logged only when rare-class filtering removes tiles.

The script also logs a label-distribution table as an MLflow artifact:

- `fold_statistics/label_distribution.json`: fold by original derived label counts.

Unlike the tile-level `stratified` strategy, the `stratified_group` strategy does not log `fold_statistics/stratification_label_distribution.json`, because it uses the original derived labels directly and does not create a separate collapsed stratification-label array.

## Split Statistics

Detailed JSON representations of the metrics are available within the respective MLflow run artifacts.

### Global Metrics

- Total retained tiles: 1,102,086
- Original tiles before rare-class filtering: 1,102,086
- Dropped rare-class tiles: 0
- n_folds: 5
- Random state: 42
- K-fold strategy: `stratified_group`
- Rare labels dropped before splitting: none
- Fold size CV: 0.0517

### Per-Fold Metrics

| Fold | Train tiles | Val tiles | Val % | Val slides | tissue_prop mean +- std |
| --- | --- | --- | --- | --- | --- |
| 0 | 859,277 | 242,809 | 22.03% | 26 | 0.9077 +- 0.2667 |
| 1 | 890,838 | 211,248 | 19.17% | 26 | 0.8809 +- 0.2981 |
| 2 | 884,420 | 217,666 | 19.75% | 27 | 0.8979 +- 0.2797 |
| 3 | 886,197 | 215,889 | 19.59% | 27 | 0.8705 +- 0.3084 |
| 4 | 887,612 | 214,474 | 19.46% | 31 | 0.8812 +- 0.2978 |

### Original Label Distribution per Fold

For `stratified_group`, this table reflects the labels that were retained after rare-class filtering.

| Label | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
| --- | --- | --- | --- | --- | --- |
| background | 4.4941% | 6.0493% | 5.2140% | 6.5742% | 6.1728% |
| Blood | 0.4156% | 0.5027% | 0.5035% | 0.4530% | 0.9059% |
| Connective-Tissue | 3.1514% | 3.4386% | 2.9573% | 3.3703% | 3.8252% |
| Epithelium | 1.2413% | 1.4268% | 1.3948% | 1.3887% | 1.4118% |
| Fat | 10.8517% | 10.8948% | 13.8033% | 12.2832% | 10.8428% |
| Muscle | 14.7194% | 2.9955% | 3.2605% | 2.4929% | 3.3776% |
| Nerve | 1.6153% | 1.8968% | 1.8556% | 1.7856% | 1.8725% |
| Other | 63.5112% | 72.7955% | 71.0111% | 71.6521% | 71.5914% |

### Slide Distribution per Fold

Use this table to document how slides are distributed across validation folds. This is the key leakage-control diagnostic for the grouped split.

| Fold | Val slides | Val slide % | Val tiles | Val tile % |
| --- | --- | --- | --- | --- |
| 0 | 26 | 18.98% | 242,809 | 22.03% |
| 1 | 26 | 18.98% | 211,248 | 19.17% |
| 2 | 27 | 19.71% | 217,666 | 19.75% |
| 3 | 27 | 19.71% | 215,889 | 19.59% |
| 4 | 31 | 22.63% | 214,474 | 19.46% |

### Optional: Label Counts per Fold

Use this table if you want to report absolute counts in addition to percentages.

| Label | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Total |
| --- | --- | --- | --- | --- | --- | --- |
| background | 10,912 | 12,779 | 11,349 | 14,193 | 13,239 | 62,472 |
| Blood | 1,009 | 1,062 | 1,096 | 978 | 1,943 | 6,088 |
| Connective-Tissue | 7,652 | 7,264 | 6,437 | 7,276 | 8,204 | 36,833 |
| Epithelium | 3,014 | 3,014 | 3,036 | 2,998 | 3,028 | 15,090 |
| Fat | 26,349 | 23,015 | 30,045 | 26,518 | 23,255 | 129,182 |
| Muscle | 35,740 | 6,328 | 7,097 | 5,382 | 7,244 | 61,791 |
| Nerve | 3,922 | 4,007 | 4,039 | 3,855 | 4,016 | 19,839 |
| Other | 154,211 | 153,779 | 154,567 | 154,689 | 153,545 | 770,791 |
