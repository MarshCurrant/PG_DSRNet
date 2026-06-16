# Environment Notes

## ERRNet

Environment:

```bash
conda create -n dip-errnet python=3.10 -y
/home/nebula/.conda/envs/dip-errnet/bin/python -m pip install \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
conda install -n dip-errnet -y -c conda-forge \
  matplotlib dominate scikit-image tensorboardx pyyaml h5py opencv pillow visdom "setuptools<82"
```

Verified imports:

```text
torch 2.7.0+cu128
torchvision 0.22.0+cu128
cv2 4.12.0
skimage 0.25.2
```

Use `MPLCONFIGDIR=/tmp/matplotlib` for scripts that import matplotlib in this sandbox.
Use `TORCH_HOME=/home/nebula/qqm/DIP/weights/torch` so VGG19 weights are loaded from the project cache.

## DSRNet

Environment:

```bash
conda create -n dip-dsrnet python=3.9 -y
/home/nebula/.conda/envs/dip-dsrnet/bin/python -m pip install \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
conda install -n dip-dsrnet -y -c conda-forge \
  dominate einops kornia opencv pyyaml scikit-image scipy tensorboardx visdom
```

`repos/DSRNet/models/dsrnet_model_sirs.py` has a small PyTorch 2.6+ compatibility patch for `torch.load(..., weights_only=False)`.

Verified imports:

```text
torch 2.7.0+cu128
torchvision 0.22.0+cu128
kornia 0.7.1
einops 0.7.0
cv2 4.9.0
skimage 0.22.0
scipy 1.12.0
```

The upstream `dsrnet_s` constructor was missing `lrm_blk_nums=[2, 4]`; this project patches it so `weights/dsrnet_s_epoch14.pt` loads with all keys matched.

## Shared VGG Cache

Downloaded:

```text
weights/torch/hub/checkpoints/vgg19-dcbb9e9d.pth
```

DSRNet CPU smoke test with `repos/DSRNet/weights/dsrnet_s_epoch14.pt` completed on a temporary 32x32 image and produced `dsrnet_l.png`, `dsrnet_r.png`, `dsrnet_rr.png`, and `m_input.png`.

## GPU Visibility

The RTX 4090 is visible to `nvidia-smi`, but the managed sandbox hides `/dev/nvidia*`, so Python CUDA checks from the sandbox can return `False`. Use a regular terminal for the actual long training and benchmark runs:

```bash
TORCH_HOME=/home/nebula/qqm/DIP/weights/torch python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```
