# Experiment configurations

`python run_experiment.py configs/<file>.yaml [--folds 0 1] [--dry-run] [--set args.num_epochs=2 data.limit=30]`

| file | what it runs |
|---|---|
| `joint_cv40.yaml` | joint RV + FAZ model, thesis loss and hyper-parameters, 5 folds × 40 epochs (the reported results) |
| `joint_thesis.yaml` | same, 100 epochs / classifier joins after epoch 20 (thesis length; needs a GPU) |
| `joint_regularised.yaml` | + mask-consistent augmentation, label smoothing, dropout, AdamW |
| `joint_common_fov.yaml` | common field of view (central 3 mm of every scan), tests the FOV shortcut |
| `classification_baseline.yaml` | classification-only ResNeSt50, same protocol |
| `classification_regularised.yaml` | classification-only with strong augmentation and regularisation |

Sections: `name`, `base` (parent file), `script` (`joint` | `classification`), `paths`, `data` (options of
`Preprocessing/build_processed_octa500.py`), `args` (each key becomes `--key value` of the training script; see `utils.py` for all options,
e.g. `scheduler: onecycle`, `cldice_alpha`, `vessel_bce`, `stratify_fov`). The resolved configuration and the git commit are stored in `<runs_dir>/<name>/config.yaml`.
