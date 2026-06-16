# PG-DSRNet: Single Image Reflection Removal

[GitHub Repository](https://github.com/MarshCurrant/PG_DSRNet)

PG-DSRNet is a course project for single image reflection removal. It reproduces the ERRNet baseline, compares it with DSRNet-L, and adds a lightweight prior-guided fine-tuning variant named PG-DSRNet-L.

The project includes:

- unified inference wrappers for ERRNet and DSRNet-style models
- benchmark evaluation with PSNR, SSIM, NCC, and LMSE
- self-collected clean-image reflection synthesis
- result aggregation scripts and paper figures
- CVPR-style paper source

## Repository Layout

```text
repos/ERRNet/                 # ERRNet baseline code
repos/DSRNet/                 # DSRNet and PG-DSRNet code
scripts/                      # inference, evaluation, synthesis, plotting
docs/                         # dataset, command, and weight notes
paper/experiment_results.md   # aggregated experiment tables
paper/cvpr2026_pg_dsrnet/     # paper source and figures
survey/                       # literature survey notes
self/                         # five clean self-collected images
```

## 1. Install Environments

The project uses two Python environments because ERRNet and DSRNet have different dependencies.

ERRNet environment:

```bash
conda create -n dip-errnet python=3.10 -y
conda activate dip-errnet
pip install -r repos/ERRNet/requirements.txt
```

DSRNet environment:

```bash
conda create -n dip-dsrnet python=3.9 -y
conda activate dip-dsrnet
pip install -r repos/DSRNet/requirements.txt
```

Install a PyTorch build matching your CUDA version if the requirements do not match your machine.

## 2. Download Weights

Weights are not stored in Git. Download the shared weight package:

- Baidu Netdisk file: `DIP`
- Link: https://pan.baidu.com/s/151h3CPe1i1OpOMrLGYMfKA?pwd=1234
- Extraction code: `1234`

Recommended final weights:

| Model | Expected path |
| --- | --- |
| ERRNet self-trained | `repos/ERRNet/checkpoints/errnet_unaligned_ft/errnet_latest.pt` |
| DSRNet-L self-trained | `repos/DSRNet/checkpoints/dsrnet_l_train_setting_i/weights/20260613-042722/dsrnet_l_train_setting_i_latest.pt` |
| PG-DSRNet-L freq+prior | `repos/DSRNet/checkpoints/pg_dsrnet_l_freq_prior_ft/weights/20260616-020451/pg_dsrnet_l_freq_prior_ft_latest.pt` |

See `docs/WEIGHTS.md` for file sizes and SHA256 checksums.

## 3. Run Inference on Your Own Images

Put reflection-contaminated input images in a folder, for example:

```text
data/my_images/
  image_001.jpg
  image_002.jpg
```

Run ERRNet:

```bash
conda activate dip-errnet
python scripts/infer_method.py \
  --method errnet \
  --checkpoint repos/ERRNet/checkpoints/errnet_unaligned_ft/errnet_latest.pt \
  --input_dir data/my_images \
  --output_dir outputs/user/errnet
```

Run PG-DSRNet-L:

```bash
conda activate dip-dsrnet
python scripts/infer_method.py \
  --method dsrnet \
  --inet dsrnet_l \
  --checkpoint repos/DSRNet/checkpoints/pg_dsrnet_l_freq_prior_ft/weights/20260616-020451/pg_dsrnet_l_freq_prior_ft_latest.pt \
  --input_dir data/my_images \
  --output_dir outputs/user/pg_dsrnet
```

If your images are very large, add `--max_long_edge 1024` to reduce memory usage.

## 4. Evaluate Paired Results

If clean ground truth images are available, arrange them as:

```text
data/my_eval/
  blended/
    sample_001.png
  transmission_layer/
    sample_001.png
```

Then evaluate predictions:

```bash
conda activate dip-errnet
python scripts/eval_sirr.py \
  --pred_dir outputs/user/pg_dsrnet \
  --gt_dir data/my_eval/transmission_layer \
  --out_csv outputs/user/pg_dsrnet_metrics.csv \
  --summary_json outputs/user/pg_dsrnet_metrics.json \
  --align crop-min
```

The evaluator reports PSNR, SSIM, NCC, and LMSE.

## 5. Reproduce Project Experiments

Prepare the public datasets following `docs/DATASETS.md`, then check the processed layout:

```bash
conda activate dip-errnet
python scripts/check_data_layout.py
```

Generate the five-image self-collected synthetic reflection set:

```bash
conda activate dip-errnet
python scripts/synthesize_self_reflection.py \
  --clean-dir self \
  --output-dir data/custom_synth \
  --seed 2018 \
  --max-long-edge 1024
```

Run the self-collected synthetic benchmark:

```bash
bash scripts/run_custom_synth_benchmark.sh
```

Regenerate aggregated result tables:

```bash
python scripts/export_experiment_results_md.py
```

Regenerate paper figures:

```bash
python scripts/make_paper_figures.py
```

## 6. Results

Aggregated tables are available in:

- `paper/experiment_results.md`
- `outputs/benchmarks/official_protocol_summary.csv`
- `outputs/benchmarks/unified_protocol_summary.csv`
- `outputs/benchmarks/self_trained_summary.csv`
- `outputs/benchmarks/pg_dsrnet_summary.csv`
- `outputs/benchmarks/custom_synth_summary.csv`

Main observations:

- DSRNet-L official epoch18 is strong on SIR2 Objects/Postcard/Wild under the unified evaluator.
- Self-trained DSRNet-L improves over self-trained ERRNet on CEILNet, Objects, Postcard, and Wild, but not on real20.
- PG-DSRNet-L freq+prior improves CEILNet table2 by `+0.866 dB PSNR` over self-trained DSRNet-L.
- PG-DSRNet-L degrades on real20 and most SIR2 subsets, so the paper reports this as a domain-gap limitation rather than hiding it.
- On the five self-collected synthetic reflection images, ERRNet is strongest under the current synthesis protocol.

## 7. Paper

The paper source is in:

```text
paper/cvpr2026_pg_dsrnet/
```

Compile `main.tex` with a standard LaTeX workflow or upload the folder to Overleaf. The paper uses a non-anonymous CVPR-style template and includes the GitHub and Baidu Netdisk links.

## 8. Notes

- Large weights and datasets are intentionally excluded from Git.
- Prediction folders can be large; keep only summaries and selected figures when sharing results.
- For more detailed command examples, see `docs/COMMANDS.md`.
