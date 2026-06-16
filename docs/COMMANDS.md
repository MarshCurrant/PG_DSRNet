# Commands

All commands assume they are executed from the repository root.

For commands that call both ERRNet and DSRNet, set the Python executables explicitly if you use separate environments:

```bash
export ERRNET_PY=/path/to/dip-errnet/bin/python
export DSRNET_PY=/path/to/dip-dsrnet/bin/python
export TORCH_HOME=weights/torch
export MPLCONFIGDIR=/tmp/matplotlib
```

If the current shell already activates the correct environment, `ERRNET_PY=python` or `DSRNET_PY=python` is enough.

## ERRNet

Prepare ERRNet data:

```bash
conda activate dip-errnet
cd repos/ERRNet
python datasets/prepare_train_data.py
python datasets/prepare_test_data.py
cd ../..
```

Evaluate an ERRNet checkpoint with the native ERRNet script:

```bash
conda activate dip-errnet
cd repos/ERRNet
python test_errnet.py \
  --name errnet \
  --dataset real20 \
  -r \
  --hyper \
  --icnn_path checkpoints/errnet_unaligned_ft/errnet_latest.pt \
  --result_dir ../../outputs/benchmarks/errnet_native \
  --save_subdir real20
cd ../..
```

Train the aligned stage:

```bash
conda activate dip-errnet
cd repos/ERRNet
python train_errnet.py \
  --name errnet_train \
  --hyper
cd ../..
```

Fine-tune on unaligned real pairs:

```bash
conda activate dip-errnet
cd repos/ERRNet
python train_errnet_unaligned.py \
  --name errnet_unaligned_ft \
  --hyper \
  -r \
  --icnn_path checkpoints/errnet_train/errnet_train_latest.pt \
  --unaligned_loss vgg
cd ../..
```

## DSRNet

Prepare the DSRNet bridge layout from the ERRNet processed data:

```bash
conda activate dip-dsrnet
python scripts/prepare_dsrnet_from_errnet.py --force
```

Train DSRNet-L Setting I:

```bash
conda activate dip-dsrnet
cd repos/DSRNet
python train_sirs.py \
  --inet dsrnet_l \
  --model dsrnet_model_sirs \
  --dataset sirs_dataset \
  --loss losses \
  --name dsrnet_l_train_setting_i \
  --lambda_vgg 0.01 \
  --lambda_rec 0.2 \
  --if_align \
  --seed 2018 \
  --base_dir ../../data/processed/dsrnet_official_extract/reflection-removal \
  --nThreads 8
cd ../..
```

Evaluate a DSRNet-L checkpoint with the native DSRNet script:

```bash
conda activate dip-dsrnet
cd repos/DSRNet
python eval_sirs.py \
  --inet dsrnet_l \
  --model dsrnet_model_sirs \
  --dataset sirs_dataset \
  --name dsrnet_l_eval \
  --if_align \
  --resume \
  --weight_path weights/dsrnet_l_epoch18.pt \
  --base_dir ../../data/processed/dsrnet_official_extract/reflection-removal \
  --nThreads 2
cd ../..
```

## PG-DSRNet Ablation

Run the full PG-DSRNet fine-tuning queue:

```bash
ERRNET_PY=/path/to/dip-errnet/bin/python \
DSRNET_PY=/path/to/dip-dsrnet/bin/python \
bash scripts/run_pg_dsrnet_experiment.sh
```

Useful environment overrides:

```bash
PG_BASE_CKPT=repos/DSRNet/checkpoints/dsrnet_l_train_setting_i/weights/20260613-042722/dsrnet_l_train_setting_i_latest.pt
PG_BASE_DIR=data/processed/dsrnet_official_extract/reflection-removal
PG_LR=5e-5
RUN_FREQ=1
RUN_PRIOR=1
RUN_FREQ_PRIOR=1
NTHREADS=8
WANDB_MODE=online
```

## Unified Inference

ERRNet on a custom folder:

```bash
conda activate dip-errnet
python scripts/infer_method.py \
  --method errnet \
  --checkpoint repos/ERRNet/checkpoints/errnet_unaligned_ft/errnet_latest.pt \
  --input_dir data/my_images \
  --output_dir outputs/user/errnet
```

DSRNet or PG-DSRNet on a custom folder:

```bash
conda activate dip-dsrnet
python scripts/infer_method.py \
  --method dsrnet \
  --inet dsrnet_l \
  --checkpoint repos/DSRNet/checkpoints/pg_dsrnet_l_freq_prior_ft/weights/20260616-020451/pg_dsrnet_l_freq_prior_ft_latest.pt \
  --input_dir data/my_images \
  --output_dir outputs/user/pg_dsrnet \
  --max_long_edge 1024
```

## Unified Metrics

```bash
conda activate dip-errnet
python scripts/eval_sirr.py \
  --pred_dir outputs/user/pg_dsrnet \
  --gt_dir data/my_eval/transmission_layer \
  --out_csv outputs/user/pg_dsrnet_metrics.csv \
  --summary_json outputs/user/pg_dsrnet_metrics.json \
  --align crop-min \
  --fail_on_missing
```

## Unified Benchmark Suite

Run official or self-trained checkpoints on CEILNet table2, real20, and SIR2:

```bash
ERRNET_PY=/path/to/dip-errnet/bin/python \
DSRNET_PY=/path/to/dip-dsrnet/bin/python \
python scripts/run_benchmark_suite.py \
  --methods errnet,dsrnet \
  --datasets ceilnet_table2,real20,objects,postcard,wild \
  --errnet_checkpoint repos/ERRNet/checkpoints/errnet_unaligned_ft/errnet_latest.pt \
  --dsrnet_checkpoint repos/DSRNet/checkpoints/pg_dsrnet_l_freq_prior_ft/weights/20260616-020451/pg_dsrnet_l_freq_prior_ft_latest.pt \
  --dsrnet_inet dsrnet_l \
  --output_root outputs/benchmarks/my_run \
  --nThreads 2
```

Recompute metrics from existing predictions only:

```bash
python scripts/run_benchmark_suite.py \
  --methods errnet,dsrnet \
  --datasets ceilnet_table2,real20,objects,postcard,wild \
  --output_root outputs/benchmarks/my_run \
  --skip_infer
```

## Self-Collected Synthetic Reflection

Generate synthetic reflection inputs from clean images:

```bash
conda activate dip-errnet
python scripts/synthesize_self_reflection.py \
  --clean-dir self \
  --output-dir data/custom_synth \
  --seed 2018 \
  --max-long-edge 1024
```

Run the full self-synth benchmark:

```bash
ERRNET_PY=/path/to/dip-errnet/bin/python \
DSRNET_PY=/path/to/dip-dsrnet/bin/python \
bash scripts/run_custom_synth_benchmark.sh
```

## Result Export

```bash
python scripts/export_experiment_results_md.py
python scripts/make_paper_figures.py --output-dir outputs/figures
```
