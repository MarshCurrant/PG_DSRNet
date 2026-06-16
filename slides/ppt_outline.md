# Presentation Outline

1. Task and Motivation
   - Single image reflection removal
   - Transmission/reflection layer ambiguity

2. Course Requirements
   - Datasets
   - Metrics
   - Baseline and improved method

3. ERRNet Baseline
   - Architecture
   - VGG hypercolumn/context modules
   - Training protocol

4. DSRNet Improved Method
   - Component synergy
   - Transmission/reflection/reconstruction outputs
   - Difference from ERRNet

5. Proposed PG-DSRNet
   - Keep DSRNet-L inference unchanged
   - Frequency prior from low/high-frequency supervision
   - Reflection intensity prior with high-frequency weighting
   - FIRM-style masks as future interactive extension

6. Experimental Setup
   - Hardware and environments
   - Data preparation
   - Evaluation scripts

7. Quantitative Results
   - Main table
   - Self-trained DSRNet-L beats self-trained ERRNet on CEILNet, Objects, Postcard, and Wild
   - real20 exception: official protocol favors DSRNet-L PSNR, unified self-trained table still favors ERRNet
   - PG-DSRNet ablation: DSRNet-L, +frequency, +prior, +frequency+prior
   - Dataset-wise observations

8. Qualitative Results
   - CEILNet/real20/SIR2 examples
   - Use `outputs/figures/qualitative/self_trained_summary_grid.png`
   - After PG run, use `outputs/figures/qualitative/pg_dsrnet_summary_grid.png`
   - Custom photos pending because `data/custom` has no images yet

9. Failure Cases
   - Strong reflection
   - Misalignment
   - Texture loss or oversmoothing

10. Conclusion
   - Baseline reproduction
   - Improved method comparison
   - Lessons learned
