# Reproduction Commands

All commands assume:

```bash
cd /home/nebula/qqm/DIP
```

## ERRNet Baseline

Prepare data:

```bash
cd repos/ERRNet
MPLCONFIGDIR=/tmp/matplotlib /home/nebula/.conda/envs/dip-errnet/bin/python datasets/prepare_train_data.py
MPLCONFIGDIR=/tmp/matplotlib /home/nebula/.conda/envs/dip-errnet/bin/python datasets/prepare_test_data.py
```

Evaluate official checkpoint with the ERRNet native protocol:

```bash
cd repos/ERRNet
for d in ceilnet_table2 real20 objects postcard wild; do
  TORCH_HOME=/home/nebula/qqm/DIP/weights/torch MPLCONFIGDIR=/tmp/matplotlib /home/nebula/.conda/envs/dip-errnet/bin/python test_errnet.py \
    --name errnet --dataset "$d" -r --hyper \
    --icnn_path checkpoints/errnet/errnet_060_00463920.pt \
    --result_dir ../../outputs/benchmarks/errnet_native \
    --save_subdir "$d"
done
```

Train aligned baseline:

```bash
cd repos/ERRNet
TORCH_HOME=/home/nebula/qqm/DIP/weights/torch MPLCONFIGDIR=/tmp/matplotlib /home/nebula/.conda/envs/dip-errnet/bin/python train_errnet.py \
  --name errnet_train --hyper
```

Fine-tune unaligned stage:

```bash
cd repos/ERRNet
TORCH_HOME=/home/nebula/qqm/DIP/weights/torch MPLCONFIGDIR=/tmp/matplotlib /home/nebula/.conda/envs/dip-errnet/bin/python train_errnet_unaligned.py \
  --name errnet_unaligned_ft --hyper -r \
  --icnn_path checkpoints/errnet/errnet_060_00463920.pt \
  --unaligned_loss vgg
```

Run the long training queue used for self-trained model artifacts:

```bash
cd /home/nebula/qqm/DIP
bash scripts/run_training_queue.sh
```

The queue runs ERRNet aligned training, ERRNet unaligned fine-tuning, DSRNet-L
Setting I training, then self-trained unified benchmarks. Logs are written to
`logs/training/`. To skip the optional ERRNet unaligned fine-tuning, run with
`RUN_ERRNET_UNALIGNED=0 bash scripts/run_training_queue.sh`.

## DSRNet Improved Method

Use the official all-in-one data package for paper reproduction:

```text
data/processed/dsrnet_official_extract/reflection-removal/
```

The bridge below is still useful for unified wrapper comparisons, but do not use
it as the DSRNet paper-protocol `real20_420` source:

```bash
/home/nebula/.conda/envs/dip-dsrnet/bin/python scripts/prepare_dsrnet_from_errnet.py --force
```

Evaluate DSRNet-L epoch18 with the DSRNet native protocol:

```bash
cd repos/DSRNet
TORCH_HOME=/home/nebula/qqm/DIP/weights/torch MPLCONFIGDIR=/tmp/matplotlib /home/nebula/.conda/envs/dip-dsrnet/bin/python eval_sirs.py \
  --inet dsrnet_l --model dsrnet_model_sirs --dataset sirs_dataset \
  --name dsrnet_l_epoch18_official_data_eval --if_align --resume \
  --weight_path /home/nebula/qqm/DIP/repos/DSRNet/weights/dsrnet_l_epoch18.pt \
  --base_dir /home/nebula/qqm/DIP/data/processed/dsrnet_official_extract/reflection-removal \
  --nThreads 2
```

Score saved official outputs with the same bandwise metric convention:

```bash
cd /home/nebula/qqm/DIP
/home/nebula/.conda/envs/dip-errnet/bin/python scripts/score_dsrnet_native_outputs.py \
  --run_dir repos/DSRNet/checkpoints/dsrnet_l_epoch18_official_data_eval/20260612-145852 \
  --out_csv outputs/benchmarks/dsrnet_l_epoch18_official_data_eval/dsrnet_native_metrics.csv \
  --summary_csv outputs/benchmarks/dsrnet_l_epoch18_official_data_eval/dsrnet_native_summary.csv \
  --summary_json outputs/benchmarks/dsrnet_l_epoch18_official_data_eval/dsrnet_native_summary.json
```

Train Setting I:

```bash
cd repos/DSRNet
TORCH_HOME=/home/nebula/qqm/DIP/weights/torch MPLCONFIGDIR=/tmp/matplotlib /home/nebula/.conda/envs/dip-dsrnet/bin/python train_sirs.py \
  --inet dsrnet_l --model dsrnet_model_sirs --dataset sirs_dataset --loss losses \
  --name dsrnet_l_train_setting_i --lambda_vgg 0.01 --lambda_rec 0.2 \
  --if_align --seed 2018 \
  --base_dir /home/nebula/qqm/DIP/data/processed/dsrnet_official_extract/reflection-removal \
  --nThreads 8
```

Use `--nThreads` to control DSRNet training and validation DataLoader workers.

## Unified Inference

ERRNet custom folder:

```bash
/home/nebula/.conda/envs/dip-errnet/bin/python scripts/infer_method.py \
  --method errnet \
  --checkpoint repos/ERRNet/checkpoints/errnet/errnet_060_00463920.pt \
  --input_dir data/custom/reflection \
  --output_dir outputs/errnet/custom
```

DSRNet custom folder:

```bash
/home/nebula/.conda/envs/dip-dsrnet/bin/python scripts/infer_method.py \
  --method dsrnet \
  --checkpoint repos/DSRNet/weights/dsrnet_l_epoch18.pt \
  --inet dsrnet_l \
  --input_dir data/custom/reflection \
  --output_dir outputs/dsrnet/custom
```

For high-resolution folders that may exceed GPU memory, add a bounded long edge and evaluate
against a resized reference:

```bash
/home/nebula/.conda/envs/dip-dsrnet/bin/python scripts/infer_method.py \
  --method dsrnet \
  --checkpoint repos/DSRNet/weights/dsrnet_l_epoch18.pt \
  --inet dsrnet_l \
  --input_dir repos/ERRNet/datasets/processed_data/real20/blended \
  --output_dir outputs/dsrnet/real20 \
  --max_long_edge 512
```

## Unified Metrics

Example:

```bash
/home/nebula/.conda/envs/dip-errnet/bin/python scripts/eval_sirr.py \
  --pred_dir outputs/errnet/CEILNet_table2 \
  --gt_dir repos/ERRNet/datasets/processed_data/testdata_CEILNET_table2 \
  --out_csv outputs/errnet/metrics_ceilnet_table2.csv \
  --summary_json outputs/errnet/metrics_ceilnet_table2.json
```

For DSRNet:

```bash
/home/nebula/.conda/envs/dip-dsrnet/bin/python scripts/eval_sirr.py \
  --pred_dir outputs/dsrnet/real20 \
  --gt_dir repos/ERRNet/datasets/processed_data/real20 \
  --out_csv outputs/dsrnet/metrics_real20.csv \
  --summary_json outputs/dsrnet/metrics_real20.json
```

## Unified Official-Checkpoint Benchmark

Run inference and metrics for ERRNet and DSRNet on CEILNet table2, real20, and SIR2
Objects/Postcard/Wild:

```bash
cd /home/nebula/qqm/DIP
/home/nebula/.conda/envs/dip-errnet/bin/python scripts/run_benchmark_suite.py \
  --methods errnet,dsrnet \
  --datasets ceilnet_table2,real20,objects,postcard,wild \
  --dsrnet_checkpoint repos/DSRNet/weights/dsrnet_l_epoch18.pt \
  --dsrnet_inet dsrnet_l \
  --nThreads 2
```

Recompute metrics from existing predictions only:

```bash
cd /home/nebula/qqm/DIP
/home/nebula/.conda/envs/dip-errnet/bin/python scripts/run_benchmark_suite.py \
  --methods errnet,dsrnet \
  --datasets ceilnet_table2,real20,objects,postcard,wild \
  --dsrnet_checkpoint repos/DSRNet/weights/dsrnet_l_epoch18.pt \
  --dsrnet_inet dsrnet_l \
  --nThreads 2 \
  --skip_infer
```

Outputs are written under:

```text
outputs/benchmarks/official/
  summary.csv
  summary.json
  errnet/<dataset>/predictions/
  errnet/<dataset>/metrics.csv
  dsrnet/<dataset>/predictions/
  dsrnet/<dataset>/metrics.csv
```

The runner uses `--align auto`: DSRNet metrics use `resize-gt`, ERRNet real20 uses
`resize-gt`, and the remaining ERRNet sets use `crop-min`. Both methods use a
512-pixel long-edge cap on real20 to avoid full-resolution GPU memory spikes.

The DSRNet-L epoch18 unified run used for the current report is under:

```text
outputs/benchmarks/unified_dsrnet_l_epoch18/
```
