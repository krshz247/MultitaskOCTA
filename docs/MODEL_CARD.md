# Model card — joint FAZ / vessel segmentation and Normal / DR / AMD classification on OCTA en-face images

Research code and research weights. **Not a medical device; not validated for clinical use.**

## What it does
One OCTA en-face image (B5 = ILM–OPL projection, 3 mm × 3 mm or 6 mm × 6 mm field of view) goes in. Out come
- the probabilities of Normal / diabetic retinopathy (DR) / age-related macular degeneration (AMD) and the predicted class,
- a mask of the foveal avascular zone (FAZ) and a mask of the large retinal vessels (384 × 384).

Interface: `octa/predict.py` (`load_ensemble`, `predict`, command line `python -m octa.predict predict --weights <dir> --image <file>`; see the module docstring).

## Model
Shared-encoder U-Net-type network with three decoders (segmentation: FAZ / vessel / background; boundary heat map; signed-distance map), ResNeSt50 encoder, plus a second ResNeSt50 classifier that receives the decoder features of all three branches at four depths (the BSDA-Net design, extended to retinal vessels in a Master's thesis, TU München, 2023). Training loss: Dice (FAZ, background) + Dice/soft-clDice (vessel) + MSE of the two auxiliary maps + class-weighted cross-entropy for the classifier; regularised variant (label smoothing 0.1, classifier dropout 0.3, AdamW weight decay 0.01, mask-consistent augmentation). Definition and deviations from the thesis: `docs/METHOD.md`.

The released weights are the five fold models of one 5-fold cross-validation run (`ILM_OPL_large_cv40reg_s1`, regularised joint model, seed 1, 40 epochs, code `bundle-5fcf651`). They were chosen as the first complete Kaggle run, not by test performance; the other runs of the same recipe perform alike (below). A prediction is the average over the five models. Each model carries the normalisation statistics (mean and standard deviation of its training images) in `stats.json`.

## Data
OCTA-500 (Li et al.): 364 scans with a Normal, DR or AMD label (251 / 64 / 49), 195 of them 3 mm × 3 mm and 169 of them 6 mm × 6 mm (AMD: 88 % of the scans are 6 mm, Normal: 64 % are 3 mm); ground truth `GT_FAZ` and `GT_LargeVessel`; the other 136 scans of the release (CNV, CSC, other diagnoses) are not used. Not redistributed here.

## Performance (class-stratified 5-fold cross-validation, every number on scans the model has not seen)
One model per fold, test metrics at the best-validation epoch, mean ± std over the five folds, for the exported run:

| FAZ Dice | vessel Dice | vessel clDice | accuracy | balanced accuracy | macro-F1 | recall Normal / DR / AMD |
|---|---|---|---|---|---|---|
| 0.926 ± 0.019 | 0.785 ± 0.011 | 0.857 ± 0.012 | 0.797 ± 0.030 | 0.755 ± 0.069 | 0.714 ± 0.047 | 0.83 / 0.75 / 0.69 |

Pooled out-of-fold predictions (364 scans, 95 % bootstrap interval): accuracy 0.797 [0.755, 0.838], balanced accuracy 0.758 [0.700, 0.812]. Repeated training of the same recipe (four runs): balanced accuracy 0.747 / 0.758 / 0.789 / 0.771, FAZ Dice 0.934 / 0.926 / 0.888 / 0.884 (3-class argmax; 0.937 / 0.925 / 0.933 / 0.936 with a 0.5 threshold on the FAZ channel: in the last two runs the background channel also claims the FAZ pixels, `analysis/faz_channel_check.txt`) — the run-to-run spread (about ±0.03 pooled, 5–9 points per fold) is as large as most differences between model variants. The metrics above are for single fold models; the five-model average used by `predict()` was not evaluated separately (averaging two or three runs gave +0.02–0.04 balanced accuracy).

## Intended use and limits
- Intended: reproduction, teaching and comparison of methods; a demonstration of the pipeline.
- **Not** for diagnosis or any clinical decision.
- One dataset, one device type, no external validation; 49 AMD scans only, so the AMD figures have wide uncertainty.
- Field of view and class are confounded (AMD scans are mostly 6 mm). A model trained on a common 3 mm field of view performs alike, but both models change their decisions strongly when the outer part of the image is replaced by grey, so they are sensitive to the image frame (`analysis/gradcam_results.txt`).
- Only the superficial B5 projection is used; AMD mainly changes the outer retina, which this input does not show. Probabilities are not calibrated.
- Whether the segmentation tasks improve the classification could not be shown: a classifier trained without segmentation reaches the same accuracy as the joint model within the run-to-run spread, and a joint model trained on shuffled masks is lower on average but not significantly so (`docs/SUMMARY_FOR_SUPERVISORS.md`).

## Weights
`fold0.pt` … `fold4.pt` (322 MB each) and `stats.json`, from the repository's GitHub release; they are not in the git history. Create them from a run with `python -m octa.predict export --run runs/<run> --data Datasets/processed_OCTA500/ILM_OPL_large --out weights/<name>`.
