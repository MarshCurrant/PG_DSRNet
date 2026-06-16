# Environment

The repository uses two environments because ERRNet and DSRNet were released with different dependency assumptions.

## ERRNet

```bash
conda create -n dip-errnet python=3.10 -y
conda activate dip-errnet
pip install -r repos/ERRNet/requirements.txt
```

If you need to install PyTorch manually, choose a build that matches your CUDA version. Example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## DSRNet

```bash
conda create -n dip-dsrnet python=3.9 -y
conda activate dip-dsrnet
pip install -r repos/DSRNet/requirements.txt
```

Again, replace the PyTorch package with the version that matches your machine if necessary.

## Shared Runtime Variables

The scripts accept common environment variables:

```bash
export TORCH_HOME=weights/torch
export MPLCONFIGDIR=/tmp/matplotlib
export ERRNET_PY=/path/to/dip-errnet/bin/python
export DSRNET_PY=/path/to/dip-dsrnet/bin/python
```

When running inside an already activated environment, the Python variables can simply be:

```bash
export ERRNET_PY=python
export DSRNET_PY=python
```

## Smoke Test

Check PyTorch and CUDA:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

Run a lightweight wrapper dry run:

```bash
python scripts/infer_method.py \
  --method dsrnet \
  --inet dsrnet_l \
  --checkpoint repos/DSRNet/checkpoints/pg_dsrnet_l_freq_prior_ft/weights/20260616-020451/pg_dsrnet_l_freq_prior_ft_latest.pt \
  --input_dir data/my_images \
  --output_dir outputs/user/pg_dsrnet \
  --dry_run
```

## Notes

- The DSRNet code includes a PyTorch 2.6+ compatibility change for `torch.load(..., weights_only=False)`.
- The DSRNet small constructor is patched so the released `dsrnet_s_epoch14.pt` can be loaded correctly.
- Long training should be run from a normal terminal with GPU access.
