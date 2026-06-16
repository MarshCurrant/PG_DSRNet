# Single Image Reflection Removal Survey

## 1. Task Definition

Single image reflection removal (SIRR) aims to recover the transmission layer from one image captured through glass. A common image formation view is:

```text
I = T + R * h
```

where `I` is the observed image, `T` is the desired transmission/background layer, `R` is the reflection layer, and `h` represents blur or degradation of the reflection. The problem is ill-posed because many pairs of `T` and `R` can explain the same observation. Real scenes also contain misalignment, ghosting, saturation, low light, and reflections with edges as strong as the background.

## 2. Method Timeline

| Category | Representative Work | Main Idea | Strength | Limitation |
| --- | --- | --- | --- | --- |
| Traditional optimization | Gradient sparsity/layer separation methods | Use priors such as reflection smoothness and transmission edge sharpness | No training data required | Weak on complex real reflections |
| Early CNN | CEILNet / deep image smoothing style methods | Learn direct mapping from reflection image to transmission image | Faster inference, better synthetic performance | Depends heavily on synthetic training distribution |
| Perceptual learning | Zhang et al., CVPR 2018, *Single Image Reflection Separation with Perceptual Losses* | Use perceptual losses and real paired/misaligned data | Better real-image realism than pixel-only losses | Real data are small and alignment remains difficult |
| Misaligned-data enhancement | ERRNet, CVPR 2019 | VGG hypercolumns, channel-wise context, pyramid spatial context, losses for misaligned pairs | Strong course baseline, reproducible, supports real data | Old architecture, limited by small real dataset |
| Cascade/refinement | IBCLN and related cascade networks | Iteratively refine transmission/reflection estimates | Improves decomposition consistency | More stages and training details |
| Component synergy | DSRNet, ICCV 2023 | Jointly predict transmission, reflection, and reconstruction with synergy constraints | Stronger decomposition modeling and practical open-source implementation | Still supervised and data-layout sensitive |
| Prompt/diffusion | PromptRR and related prompt-based works | Use prompt/semantic priors or foundation-style restoration | Can improve robustness on diverse scenes | Often heavier and less directly comparable |
| Interactive guidance | FIRM, AAAI 2025 | User prompt/mask guides reflection removal | Strong for hard real cases with human input | Not a fully automatic baseline-equivalent setting |
| New benchmark | OpenRR-5k and recent benchmark work | Larger real-world evaluation and modern comparison protocol | Better reflects wild usage | May require new data and training recipes |

## 3. Baseline: ERRNet

ERRNet is the required course baseline:

- Paper: *Single Image Reflection Removal Exploiting Misaligned Training Data and Network Enhancements*, CVPR 2019.
- Repository used here: `https://github.com/innerway-xq/ERRNet`, branch `dip26`.
- Main architecture: residual network with VGG hypercolumn features.
- Network enhancements:
  - Channel-wise context for global channel reweighting.
  - Multi-scale spatial context through pyramid pooling.
  - Real-data training support where paired reflection/clean images are not perfectly aligned.
- Why it is a good baseline:
  - It directly matches the course requirement.
  - It has a compact architecture and a known training/evaluation protocol.
  - It reports PSNR, SSIM, NCC, and LMSE on the same benchmark family.

For this project, ERRNet is reproduced first. The official checkpoint is used as a fallback and sanity reference, while a local training run is used to satisfy the reproduction requirement.

## 4. Improved Method: DSRNet

DSRNet is selected as the main improved method:

- Paper: *Single Image Reflection Separation via Component Synergy*, ICCV 2023.
- Repository: `https://github.com/mingcv/DSRNet`.
- Main idea: estimate transmission, reflection, and reconstructed mixture together, then use their component relationship as supervision.
- Difference from ERRNet:
  - ERRNet focuses on enhanced context features and robust learning from misaligned real data.
  - DSRNet explicitly models interaction among separated components.
  - DSRNet provides both transmission and reflection outputs, making qualitative analysis richer.
- Practical reason for selection:
  - Fully automatic SIRR setting.
  - Open-source code and available weights.
  - Compatible with the course test sets and a single RTX 4090.

## 5. Recent Alternatives

### PromptRR

PromptRR represents prompt-oriented reflection removal. Its core value is using richer guidance or learned prompts to improve restoration under varied real-world degradation. It is attractive for a survey section, but its experimental setting can be less direct for a strict ERRNet-vs-method comparison unless the same datasets and full-reference metrics are carefully reproduced.

### FIRM

FIRM is a more recent interactive reflection removal direction. It can exploit user guidance for hard cases, so it may produce strong visual results, but the interaction makes it not fully equivalent to the automatic ERRNet baseline. In this project it is best used as a recent-work discussion point and optional qualitative reference, not as the primary quantitative method.

### OpenRR-5k and New Benchmarks

Recent benchmark work emphasizes that older test sets are small and may not represent modern phone/social-media images. This supports adding five self-collected photos, but full-reference metrics should only be reported when clean paired references are available.

## 6. Innovation Comparison

| Method | Innovation | Data Assumption | Output | Why/Why Not Main Experiment |
| --- | --- | --- | --- | --- |
| ERRNet | Context modules + misaligned real training | Synthetic aligned + small real misaligned | Transmission | Required baseline |
| IBCLN-style cascade | Iterative bidirectional refinement | Paired/synthetic training | Transmission/reflection | Useful comparison, but older than DSRNet |
| DSRNet | Component synergy across T/R/reconstruction | Paired supervised training | Transmission/reflection/reconstruction | Main improved method |
| PromptRR | Prompt-based restoration prior | Method-specific prompts/model recipe | Transmission | Good survey topic, higher setup risk |
| FIRM | Interactive user guidance | User prompt/mask | Transmission | Latest and strong, but not automatic-only |
| OpenRR-5k work | New larger benchmark and stronger training data | New benchmark data | Benchmark-dependent | Useful discussion for future work |

## 7. Experimental Recommendation

Use this priority order:

1. Reproduce ERRNet with official checkpoint, then local training if time permits.
2. Evaluate DSRNet using the included or official checkpoint to establish improved-method numbers.
3. Train or fine-tune DSRNet Setting I on the prepared base layout.
4. Use PromptRR, FIRM, and OpenRR-5k in the survey and innovation comparison instead of overextending the quantitative workload.

## 8. References

- ERRNet: https://arxiv.org/abs/1904.00637
- ERRNet code: https://github.com/innerway-xq/ERRNet
- Zhang et al. perceptual-loss reflection separation: https://arxiv.org/abs/1806.05376
- DSRNet: https://arxiv.org/abs/2308.10027
- DSRNet code: https://github.com/mingcv/DSRNet
- PromptRR: https://arxiv.org/abs/2402.02374
- FIRM: https://arxiv.org/abs/2406.01555
- OpenRR-5k: https://arxiv.org/abs/2506.05482
- Awesome Reflection Removal: https://github.com/ChenyangLEI/awesome-reflection-removal
