# Reference Reuse Notes for PG-DSRNet

This note records what was cloned under `repos/reference/` and how each project
should inform the proposed PG-DSRNet improvement. The intent is to reuse ideas
and implementation patterns, not to copy large external modules into DSRNet.

## Clone Snapshot

| Repo | Local path | Remote | Snapshot | Status |
| --- | --- | --- | --- | --- |
| PromptRR | `repos/reference/PromptRR` | `https://github.com/TaoWangzj/PromptRR` | `3cd1829` | README/images only; no released code in this checkout |
| FIRM | `repos/reference/FlexibleReflectionRemoval` | `https://github.com/ShawnChenn/FlexibleReflectionRemoval` | `9cd7141` | Code available; BasicSR/NAFNet/SAM-style mask guidance |
| FUMO | `repos/reference/FUMO` | `https://github.com/Lucious-Desmon/FUMO` | `085d33d` | Code available; prior-modulated diffusion and refinement |
| OpenRR-5k | `repos/reference/OpenRR-5k` | `https://github.com/caijie0620/OpenRR-5k` | `54faba5` | Benchmark README and evaluator |

## Reuse Decision Table

| Reference | Useful logic | PG-DSRNet mapping | Risk | Main experiment? |
| --- | --- | --- | --- | --- |
| PromptRR | LF/HF frequency prompts motivate explicit frequency supervision. The repo states code is still "come soon", so only the paper and README idea is usable. | Add `L_freq` on DSRNet `output_t` vs `target_t` with low-frequency blur and high-frequency residual terms. | Low for simplified loss; high for full diffusion/promptformer. | Yes, simplified loss only |
| FUMO | Implements wavelet high-frequency extraction and combines high-frequency magnitude with a reflection prior using `gate = 1 + beta * prior * hf`. | Use the same prior-plus-HF idea as a loss weight map, not as diffusion feature injection. | Low for loss weighting; high for full Qwen/diffusion/refinement stack. | Yes, simplified prior loss only |
| FIRM | Loads reflection masks and uses mask anchors/cross-attention to guide restoration features. | Future work: replace automatic intensity prior with user/SAM reflection mask guidance. | High because it changes automatic SIRR into interactive/mask-guided SIRR. | No |
| OpenRR-5k | Uses PSNR, SSIM, LPIPS, DISTS, NIQE and separates validation with GT from test without GT. | Report limitation/future benchmark; optionally add LPIPS/NIQE later if dependencies are already available. | Medium due dataset download and new metric dependencies. | No |

## Key Implementation References

### Frequency Prior

- PromptRR motivates LF/HF prompts in `repos/reference/PromptRR/README.md`.
  The current checkout has no training or model code, so do not depend on it.
- FUMO has a compact wavelet-style decomposition:
  `repos/reference/FUMO/wavelet_color_fix.py`.
  The relevant pattern is:
  - apply repeated 3x3 blur with dilations `1,2,4,8,...`;
  - accumulate `image - low_freq` as high frequency;
  - return `(high_freq, low_freq)`.
- PG-DSRNet should implement its own small differentiable blur/decomposition in
  DSRNet losses, preferably with fixed depthwise kernels and no external import.

### Reflection Intensity / Prior Weight

- FUMO expects a `.npy` prior map in `[0,1]`, clips NaN/Inf, resizes it, and
  uses it with high-frequency magnitude.
- FUMO training/inference gates residuals as `1 + beta * prior * hf`, clamped to
  `[1, 1 + beta]`.
- PG-DSRNet should avoid external `.npy` priors for the main experiment. Use
  training targets to generate an automatic prior:
  - synthetic data: `prior = mean(abs(target_r), dim=channel)`;
  - real paired data: `prior = mean(abs(input - target_t), dim=channel)`;
  - normalize per image to `[0,1]`;
  - build `weight = 1 + lambda_prior * prior * hf(input)`.
- Apply the weight only to selected loss terms first, not the network features:
  weighted transmission frequency loss and/or weighted reflection gradient loss.

### Mask-Guided Future Work

- FIRM's dataset code reads `R_binary_masks`, `RSam_masks`, `point_mask`, or
  `point_sam_mask` and returns `r_mask` together with the image pair.
- FIRM's NAFNet variant splits the input into image plus mask channels, creates
  reflection/transmission masks, and uses mask anchors in cross-attention.
- This is useful for a future `Interactive PG-DSRNet`, but should stay out of
  the current automatic benchmark because the user mask changes the task.

### Benchmark / Metrics

- OpenRR-5k's evaluator uses `pyiqa` metrics: PSNR, SSIM, LPIPS, DISTS, NIQE,
  with an 8-pixel border crop.
- Its README describes `train_5000`, `val_300`, and `test_100` without GT.
- For this project, keep the required PSNR/SSIM/NCC/LMSE tables as primary.
  Mention LPIPS/DISTS/NIQE and OpenRR-5k as modern benchmark extensions.

## Recommended PG-DSRNet v1

Implement the smallest defensible variant:

1. Keep DSRNet-L architecture and inference unchanged.
2. Add `L_freq`:
   - `lf(x) = gaussian_or_wavelet_blur(x)`;
   - `hf(x) = x - lf(x)`;
   - `L_freq = L1(lf(output_t), lf(target_t)) + L1(hf(output_t), hf(target_t))`.
3. Add `L_prior` through a soft weight map:
   - `prior = normalize(mean(abs(target_r or input - target_t)))`;
   - `hf_input = normalize(mean(abs(hf(input))))`;
   - `w = clamp(1 + lambda_prior * prior * hf_input, 1, 1 + lambda_prior)`;
   - apply `w` to a reflection/transmission residual loss.
4. Add CLI flags with conservative defaults:
   - `--lambda_freq 0.05`
   - `--lambda_prior 0.10`
   - `--prior_source target_r`
   - `--freq_levels 3`
5. Fine-tune from the completed DSRNet-L Setting I checkpoint first; compare
   against the reproduced DSRNet-L checkpoint before launching a full retrain.

## Implementation Hooks

- PG losses are implemented inside `repos/DSRNet/models/losses.py` as
  `FrequencyLoss` and `PriorWeightedLoss`.
- DSRNet training accepts `--lambda_freq`, `--lambda_prior`, `--prior_source`,
  and `--freq_levels`; all defaults keep vanilla DSRNet behavior.
- `scripts/run_pg_dsrnet_experiment.sh` runs the three ablations from the
  completed DSRNet-L Setting I checkpoint and then calls
  `scripts/summarize_pg_dsrnet.py`.
- `scripts/make_qualitative_grid.py --pg-root outputs/benchmarks/pg_dsrnet_l_freq_prior`
  adds the PG-DSRNet output column once the benchmark predictions exist.

## Risks and Non-Goals

- Do not import FUMO diffusion, Qwen2.5-VL, or refinement training for the main
  project; that would change the compute profile and dependency surface.
- Do not require FIRM masks for the main experiment; it would invalidate the
  automatic-only comparison against ERRNet/DSRNet.
- Do not use OpenRR-5k as a required dataset unless the data are explicitly
  downloaded and the metric dependencies are installed.
- PromptRR currently provides no code in this clone; cite the idea, not a reused
  implementation.
