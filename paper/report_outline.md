# DIP Course Report Outline

## Title

Single Image Reflection Removal: Reproducing ERRNet and Comparing DSRNet

## Abstract

Briefly state the task, datasets, baseline, improved method, metrics, and headline quantitative/qualitative findings.

## 1. Background

- Define single image reflection removal and the transmission/reflection layer decomposition.
- Explain ill-posedness: one input image can map to many valid decompositions.
- Discuss practical difficulty: overlapping structures, similar edges, misalignment in real paired data, and domain gaps between synthetic and real reflection.

## 2. Related Work

Use `survey/reflection_removal_survey.md` as the source material.

- Traditional optimization priors.
- CNN and perceptual-loss methods.
- Misaligned-data learning and ERRNet.
- Cascade/refinement and component-synergy methods.
- Prompt, diffusion, interactive, and benchmark-oriented recent work.

## 3. Methods

### 3.1 Baseline: ERRNet

- Residual backbone with VGG hypercolumn features.
- Channel-wise context and multi-scale spatial context.
- Aligned synthetic training plus real misaligned training support.
- Loss terms for reconstruction, perceptual supervision, and optional adversarial/context losses.

### 3.2 Improved Method: DSRNet

- Joint transmission/reflection decomposition.
- Component synergy between transmission, reflection, and reconstructed mixture.
- Reconstruction consistency and exclusion-like separation objectives.
- Why it is hardware-feasible on one RTX 4090.

### 3.3 Proposed Method: PG-DSRNet

- Keep DSRNet-L architecture unchanged and add training-only priors.
- Frequency prior: supervise low-frequency and high-frequency bands of the
  predicted transmission, inspired by PromptRR and simplified with a fixed
  differentiable blur/residual decomposition.
- Reflection intensity prior: build an automatic spatial weight from
  `target_r` or `input - target_t`, combine it with input high-frequency
  magnitude following the FUMO-style `prior * hf` idea, and weight residual
  supervision in reflection-sensitive regions.
- FIRM-style user/SAM masks are discussed as future interactive guidance, not
  used in the automatic main benchmark.

## 4. Experimental Setup

- Hardware: RTX 4090 49GB, CUDA package `cu128`.
- Environments: `dip-errnet`, `dip-dsrnet`.
- Training data: VOC2012 cropped 224x224 using the repository list (15287 images; course slide states 7643) and Berkeley/Zhang real pairs.
- Test data: CEILNet table2, real20, SIR2 Objects/Postcard/Wild. The custom
  5-photo qualitative set is pending because `data/custom` currently contains no
  input images.
- Metrics: PSNR, SSIM, NCC, LMSE.

## 5. Results

Main unified-protocol table:

| Method | CEILNet PSNR/SSIM | real20 PSNR/SSIM | Objects PSNR/SSIM | Postcard PSNR/SSIM | Wild PSNR/SSIM |
| --- | --- | --- | --- | --- | --- |
| ERRNet official | 27.586 / 0.940 | 23.069 / 0.816 | 24.774 / 0.899 | 21.859 / 0.876 | 24.885 / 0.882 |
| ERRNet reproduced | 27.173 / 0.938 | 22.642 / 0.799 | 24.837 / 0.897 | 20.560 / 0.838 | 23.456 / 0.872 |
| DSRNet-L official epoch18 | 25.444 / 0.920 | 22.515 / 0.779 | 26.285 / 0.922 | 24.724 / 0.912 | 25.809 / 0.918 |
| DSRNet-L reproduced Setting I | 28.342 / 0.948 | 22.222 / 0.790 | 25.679 / 0.912 | 21.477 / 0.905 | 25.857 / 0.916 |
| PG-DSRNet-L freq+prior | pending | pending | pending | pending | pending |

Add NCC/LMSE table separately if space is tight.

PG-DSRNet ablation table to fill after fine-tuning:

| Variant | CEILNet PSNR/SSIM | real20 PSNR/SSIM | Objects PSNR/SSIM | Postcard PSNR/SSIM | Wild PSNR/SSIM |
| --- | --- | --- | --- | --- | --- |
| DSRNet-L reproduced | 28.342 / 0.948 | 22.222 / 0.790 | 25.679 / 0.912 | 21.477 / 0.905 | 25.857 / 0.916 |
| + frequency loss | pending | pending | pending | pending | pending |
| + prior-weighted loss | pending | pending | pending | pending | pending |
| + frequency + prior | pending | pending | pending | pending | pending |

Official-protocol real20 note: DSRNet-L epoch18 on official `real20_420`
achieves 23.882 / 0.816, while ERRNet official native real20 achieves
23.553 / 0.829. This is the table to use when discussing paper-protocol
reproduction; the unified table is for fair local evaluator comparison.

## 6. Qualitative Analysis

- Use `outputs/figures/qualitative/self_trained_summary_grid.png` as the main
  self-trained qualitative figure. It includes one sample each from CEILNet
  table2, real20, Objects, Postcard, and Wild.
- Columns are input, clean GT, ERRNet reproduced output, and DSRNet-L
  reproduced output. Per-dataset single-row figures and the sample manifest are
  in `outputs/figures/qualitative/`.
- Discuss that DSRNet-L reproduced generally preserves cleaner transmission
  layers on CEILNet/SIR2 examples, while real20 remains the local exception in
  the unified metric table.
- The custom 5-photo qualitative set is still pending user-provided images in
  `data/custom`.

## 7. Conclusion

- Summarize reproducibility, performance gap, and practical deployment notes.
- State limitations: dependence on paired GT for metrics, generalization to real social-media images, and training-time constraints.

## Appendix

- Code repository link.
- Weight download link.
- Dataset links and license notes.
- Member contribution statement.
