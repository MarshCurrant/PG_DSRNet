# Experiment Results

This document is generated from the current benchmark summary CSV files. It keeps native official protocols, unified evaluator results, self-trained checkpoints, and PG-DSRNet ablations in separate tables so metric protocols are not mixed.

## 1. Result Sources

| Group | File | Meaning |
| --- | --- | --- |
| Official native protocol | outputs/benchmarks/official_protocol_summary.csv | ERRNet native test script and DSRNet native eval protocol; not mixed with unified evaluator. |
| Unified official weights | outputs/benchmarks/unified_protocol_summary.csv | Official ERRNet and DSRNet-L epoch18 evaluated by the shared wrapper/evaluator. |
| Self-trained models | outputs/benchmarks/self_trained_summary.csv | ERRNet self-trained/fine-tuned and DSRNet-L Setting I self-trained checkpoints. |
| PG-DSRNet ablation | outputs/benchmarks/pg_dsrnet_summary.csv | Self-trained DSRNet-L baseline plus frequency, prior, and frequency+prior fine-tuning runs. |
| Self-collected synthetic reflection | outputs/benchmarks/custom_synth_summary.csv | Five self-collected clean photos synthesized with reflection layers and evaluated with full-reference metrics. |
| Per-model summaries | outputs/benchmarks/*/summary.csv and summary.json | Detailed outputs used by the four aggregate CSV files above. |

## 2. Official Protocol Results

ERRNet rows come from the native ERRNet evaluation script. DSRNet rows come from the native DSRNet protocol with official all-in-one data; `real20_420` is intentionally kept distinct from unified `real20`.

| Method | Checkpoint | Protocol | Dataset | Count | PSNR | SSIM | NCC | LMSE | Source | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ERRNet | errnet_060_00463920.pt | ERRNet test_errnet.py native | CEILNet table2 | 100 | 27.877 | 0.9407 | 0.9808 | 0.00480 | runtime log | course ERRNet processed split |
| ERRNet | errnet_060_00463920.pt | ERRNet test_errnet.py native | real20 | 20 | 23.553 | 0.8285 | 0.8877 | 0.02010 | runtime log | course ERRNet processed split |
| DSRNet-L | dsrnet_l_epoch18.pt | DSRNet eval_sirs.py native | real20_420 | 20 | 23.882 | 0.8161 | 0.8995 | 0.01972 | metrics.txt plus saved PNG re-score | official DSRNet all-in-one test/real20_420 |
| ERRNet | errnet_060_00463920.pt | ERRNet test_errnet.py native | Objects | 200 | 24.853 | 0.8980 | 0.9817 | 0.00290 | runtime log | course ERRNet processed split |
| DSRNet-L | dsrnet_l_epoch18.pt | DSRNet eval_sirs.py native | Objects | 200 | 26.440 | 0.9215 | 0.9874 | 0.00238 | metrics.txt plus saved PNG re-score | official DSRNet all-in-one SIR2/SolidObjectDataset |
| ERRNet | errnet_060_00463920.pt | ERRNet test_errnet.py native | Postcard | 179 | 22.071 | 0.8773 | 0.9463 | 0.00440 | runtime log | course ERRNet processed split |
| DSRNet-L | dsrnet_l_epoch18.pt | DSRNet eval_sirs.py native | Postcard | 199 | 24.884 | 0.9102 | 0.9656 | 0.00282 | metrics.txt plus saved PNG re-score | official DSRNet all-in-one SIR2/PostcardDataset |
| ERRNet | errnet_060_00463920.pt | ERRNet test_errnet.py native | Wild | 101 | 25.176 | 0.8861 | 0.9359 | 0.00830 | runtime log | course ERRNet processed split |
| DSRNet-L | dsrnet_l_epoch18.pt | DSRNet eval_sirs.py native | Wild | 55 | 24.771 | 0.8975 | 0.9319 | 0.00717 | metrics.txt plus saved PNG re-score | official DSRNet all-in-one SIR2/WildSceneDataset |

## 3. Unified Protocol: Official Weights

Both methods are evaluated by the same wrapper/evaluator, using the same dataset naming and metric implementation.

| Method | Checkpoint | Protocol | Dataset | Count | PSNR | SSIM | NCC | LMSE | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ERRNet | errnet_060_00463920.pt | unified wrapper plus eval_sirr.py | CEILNet table2 | 100 | 27.586 | 0.9396 | 0.9808 | 0.00491 | outputs/benchmarks/official/summary.csv |
| DSRNet-L | dsrnet_l_epoch18.pt | unified wrapper plus eval_sirr.py | CEILNet table2 | 100 | 25.444 | 0.9195 | 0.9728 | 0.00608 | outputs/benchmarks/unified_dsrnet_l_epoch18/summary.csv |
| ERRNet | errnet_060_00463920.pt | unified wrapper plus eval_sirr.py | real20 | 20 | 23.069 | 0.8157 | 0.8846 | 0.02179 | outputs/benchmarks/official/summary.csv |
| DSRNet-L | dsrnet_l_epoch18.pt | unified wrapper plus eval_sirr.py | real20 | 20 | 22.515 | 0.7789 | 0.8776 | 0.02270 | outputs/benchmarks/unified_dsrnet_l_epoch18/summary.csv |
| ERRNet | errnet_060_00463920.pt | unified wrapper plus eval_sirr.py | Objects | 200 | 24.774 | 0.8986 | 0.9817 | 0.00290 | outputs/benchmarks/official/summary.csv |
| DSRNet-L | dsrnet_l_epoch18.pt | unified wrapper plus eval_sirr.py | Objects | 200 | 26.285 | 0.9215 | 0.9874 | 0.00238 | outputs/benchmarks/unified_dsrnet_l_epoch18/summary.csv |
| ERRNet | errnet_060_00463920.pt | unified wrapper plus eval_sirr.py | Postcard | 179 | 21.859 | 0.8764 | 0.9463 | 0.00442 | outputs/benchmarks/official/summary.csv |
| DSRNet-L | dsrnet_l_epoch18.pt | unified wrapper plus eval_sirr.py | Postcard | 179 | 24.724 | 0.9119 | 0.9663 | 0.00275 | outputs/benchmarks/unified_dsrnet_l_epoch18/summary.csv |
| ERRNet | errnet_060_00463920.pt | unified wrapper plus eval_sirr.py | Wild | 101 | 24.885 | 0.8818 | 0.9358 | 0.00874 | outputs/benchmarks/official/summary.csv |
| DSRNet-L | dsrnet_l_epoch18.pt | unified wrapper plus eval_sirr.py | Wild | 101 | 25.809 | 0.9178 | 0.9516 | 0.00493 | outputs/benchmarks/unified_dsrnet_l_epoch18/summary.csv |

## 4. Self-Trained Results

These rows use our trained checkpoints rather than only the released weights.

| Method | Checkpoint | Training protocol | Dataset | Count | PSNR | SSIM | NCC | LMSE | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ERRNet | repos/ERRNet/checkpoints/errnet_unaligned_ft/errnet_latest.pt | train_errnet.py --name errnet_train --hyper then train_errnet_unaligned.py --name errnet_unaligned_ft | CEILNet table2 | 100 | 27.173 | 0.9377 | 0.9782 | 0.00477 | RUN_ID 20260613-015935; W&B run 4rcq7ojg; outputs/benchmarks/self_trained_errnet/summary.csv |
| DSRNet-L | repos/DSRNet/checkpoints/dsrnet_l_train_setting_i/weights/20260613-042722/dsrnet_l_train_setting_i_latest.pt | train_sirs.py Setting I --lambda_vgg 0.01 --lambda_rec 0.2 --if_align --seed 2018 | CEILNet table2 | 100 | 28.342 | 0.9479 | 0.9859 | 0.00379 | RUN_ID 20260613-015935; W&B run s798u9t7; outputs/benchmarks/self_trained_dsrnet_l/summary.csv |
| ERRNet | repos/ERRNet/checkpoints/errnet_unaligned_ft/errnet_latest.pt | train_errnet.py --name errnet_train --hyper then train_errnet_unaligned.py --name errnet_unaligned_ft | real20 | 20 | 22.642 | 0.7989 | 0.8756 | 0.02701 | RUN_ID 20260613-015935; W&B run 4rcq7ojg; outputs/benchmarks/self_trained_errnet/summary.csv |
| DSRNet-L | repos/DSRNet/checkpoints/dsrnet_l_train_setting_i/weights/20260613-042722/dsrnet_l_train_setting_i_latest.pt | train_sirs.py Setting I --lambda_vgg 0.01 --lambda_rec 0.2 --if_align --seed 2018 | real20 | 20 | 22.222 | 0.7903 | 0.8683 | 0.02312 | RUN_ID 20260613-015935; W&B run s798u9t7; outputs/benchmarks/self_trained_dsrnet_l/summary.csv |
| ERRNet | repos/ERRNet/checkpoints/errnet_unaligned_ft/errnet_latest.pt | train_errnet.py --name errnet_train --hyper then train_errnet_unaligned.py --name errnet_unaligned_ft | Objects | 200 | 24.837 | 0.8967 | 0.9831 | 0.00319 | RUN_ID 20260613-015935; W&B run 4rcq7ojg; outputs/benchmarks/self_trained_errnet/summary.csv |
| DSRNet-L | repos/DSRNet/checkpoints/dsrnet_l_train_setting_i/weights/20260613-042722/dsrnet_l_train_setting_i_latest.pt | train_sirs.py Setting I --lambda_vgg 0.01 --lambda_rec 0.2 --if_align --seed 2018 | Objects | 200 | 25.679 | 0.9119 | 0.9851 | 0.00237 | RUN_ID 20260613-015935; W&B run s798u9t7; outputs/benchmarks/self_trained_dsrnet_l/summary.csv |
| ERRNet | repos/ERRNet/checkpoints/errnet_unaligned_ft/errnet_latest.pt | train_errnet.py --name errnet_train --hyper then train_errnet_unaligned.py --name errnet_unaligned_ft | Postcard | 179 | 20.560 | 0.8379 | 0.9034 | 0.00776 | RUN_ID 20260613-015935; W&B run 4rcq7ojg; outputs/benchmarks/self_trained_errnet/summary.csv |
| DSRNet-L | repos/DSRNet/checkpoints/dsrnet_l_train_setting_i/weights/20260613-042722/dsrnet_l_train_setting_i_latest.pt | train_sirs.py Setting I --lambda_vgg 0.01 --lambda_rec 0.2 --if_align --seed 2018 | Postcard | 179 | 21.477 | 0.9050 | 0.9643 | 0.00312 | RUN_ID 20260613-015935; W&B run s798u9t7; outputs/benchmarks/self_trained_dsrnet_l/summary.csv |
| ERRNet | repos/ERRNet/checkpoints/errnet_unaligned_ft/errnet_latest.pt | train_errnet.py --name errnet_train --hyper then train_errnet_unaligned.py --name errnet_unaligned_ft | Wild | 101 | 23.456 | 0.8723 | 0.9374 | 0.00788 | RUN_ID 20260613-015935; W&B run 4rcq7ojg; outputs/benchmarks/self_trained_errnet/summary.csv |
| DSRNet-L | repos/DSRNet/checkpoints/dsrnet_l_train_setting_i/weights/20260613-042722/dsrnet_l_train_setting_i_latest.pt | train_sirs.py Setting I --lambda_vgg 0.01 --lambda_rec 0.2 --if_align --seed 2018 | Wild | 101 | 25.857 | 0.9156 | 0.9608 | 0.00394 | RUN_ID 20260613-015935; W&B run s798u9t7; outputs/benchmarks/self_trained_dsrnet_l/summary.csv |

## 5. PG-DSRNet Ablation

`DSRNet-L reproduced` is the self-trained DSRNet-L baseline. `freq` and `prior` were fine-tuned to epoch55; `freq+prior` was fine-tuned to epoch60, so this is not a strict same-epoch ablation.

| Variant | Dataset | Count | PSNR | SSIM | NCC | LMSE | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DSRNet-L reproduced | CEILNet table2 | 100 | 28.342 | 0.9479 | 0.9859 | 0.00379 | outputs/benchmarks/self_trained_dsrnet_l/summary.csv |
| DSRNet-L reproduced | real20 | 20 | 22.222 | 0.7903 | 0.8683 | 0.02312 | outputs/benchmarks/self_trained_dsrnet_l/summary.csv |
| DSRNet-L reproduced | Objects | 200 | 25.679 | 0.9119 | 0.9851 | 0.00237 | outputs/benchmarks/self_trained_dsrnet_l/summary.csv |
| DSRNet-L reproduced | Postcard | 179 | 21.477 | 0.9050 | 0.9643 | 0.00312 | outputs/benchmarks/self_trained_dsrnet_l/summary.csv |
| DSRNet-L reproduced | Wild | 101 | 25.857 | 0.9156 | 0.9608 | 0.00394 | outputs/benchmarks/self_trained_dsrnet_l/summary.csv |
| PG-DSRNet-L freq | CEILNet table2 | 100 | 29.080 | 0.9543 | 0.9871 | 0.00345 | outputs/benchmarks/pg_dsrnet_l_freq/summary.csv |
| PG-DSRNet-L freq | real20 | 20 | 21.320 | 0.7770 | 0.8506 | 0.02418 | outputs/benchmarks/pg_dsrnet_l_freq/summary.csv |
| PG-DSRNet-L freq | Objects | 200 | 24.490 | 0.8973 | 0.9846 | 0.00311 | outputs/benchmarks/pg_dsrnet_l_freq/summary.csv |
| PG-DSRNet-L freq | Postcard | 179 | 20.030 | 0.8960 | 0.9609 | 0.00346 | outputs/benchmarks/pg_dsrnet_l_freq/summary.csv |
| PG-DSRNet-L freq | Wild | 101 | 25.256 | 0.9167 | 0.9601 | 0.00367 | outputs/benchmarks/pg_dsrnet_l_freq/summary.csv |
| PG-DSRNet-L prior | CEILNet table2 | 100 | 29.002 | 0.9535 | 0.9873 | 0.00349 | outputs/benchmarks/pg_dsrnet_l_prior/summary.csv |
| PG-DSRNet-L prior | real20 | 20 | 21.506 | 0.7798 | 0.8571 | 0.02400 | outputs/benchmarks/pg_dsrnet_l_prior/summary.csv |
| PG-DSRNet-L prior | Objects | 200 | 24.593 | 0.8989 | 0.9848 | 0.00306 | outputs/benchmarks/pg_dsrnet_l_prior/summary.csv |
| PG-DSRNet-L prior | Postcard | 179 | 20.101 | 0.8976 | 0.9611 | 0.00343 | outputs/benchmarks/pg_dsrnet_l_prior/summary.csv |
| PG-DSRNet-L prior | Wild | 101 | 25.474 | 0.9194 | 0.9606 | 0.00366 | outputs/benchmarks/pg_dsrnet_l_prior/summary.csv |
| PG-DSRNet-L freq+prior | CEILNet table2 | 100 | 29.208 | 0.9548 | 0.9874 | 0.00340 | outputs/benchmarks/pg_dsrnet_l_freq_prior/summary.csv |
| PG-DSRNet-L freq+prior | real20 | 20 | 21.369 | 0.7733 | 0.8431 | 0.02502 | outputs/benchmarks/pg_dsrnet_l_freq_prior/summary.csv |
| PG-DSRNet-L freq+prior | Objects | 200 | 24.546 | 0.8987 | 0.9846 | 0.00306 | outputs/benchmarks/pg_dsrnet_l_freq_prior/summary.csv |
| PG-DSRNet-L freq+prior | Postcard | 179 | 20.801 | 0.9039 | 0.9604 | 0.00328 | outputs/benchmarks/pg_dsrnet_l_freq_prior/summary.csv |
| PG-DSRNet-L freq+prior | Wild | 101 | 25.473 | 0.9181 | 0.9611 | 0.00369 | outputs/benchmarks/pg_dsrnet_l_freq_prior/summary.csv |

## 6. Delta vs Self-Trained DSRNet-L

Positive deltas mean the PG variant is higher than `DSRNet-L reproduced`; for LMSE, lower is better, so negative deltas are favorable.

| Variant | Dataset | ΔPSNR | ΔSSIM | ΔNCC | ΔLMSE |
| --- | --- | --- | --- | --- | --- |
| PG-DSRNet-L freq | CEILNet table2 | +0.738 | +0.0064 | +0.0012 | -0.00034 |
| PG-DSRNet-L freq | real20 | -0.902 | -0.0133 | -0.0177 | +0.00105 |
| PG-DSRNet-L freq | Objects | -1.189 | -0.0145 | -0.0005 | +0.00075 |
| PG-DSRNet-L freq | Postcard | -1.447 | -0.0091 | -0.0034 | +0.00034 |
| PG-DSRNet-L freq | Wild | -0.601 | +0.0011 | -0.0007 | -0.00028 |
| PG-DSRNet-L prior | CEILNet table2 | +0.660 | +0.0056 | +0.0014 | -0.00030 |
| PG-DSRNet-L prior | real20 | -0.716 | -0.0104 | -0.0112 | +0.00088 |
| PG-DSRNet-L prior | Objects | -1.086 | -0.0130 | -0.0003 | +0.00070 |
| PG-DSRNet-L prior | Postcard | -1.376 | -0.0075 | -0.0032 | +0.00031 |
| PG-DSRNet-L prior | Wild | -0.383 | +0.0039 | -0.0002 | -0.00028 |
| PG-DSRNet-L freq+prior | CEILNet table2 | +0.866 | +0.0069 | +0.0015 | -0.00039 |
| PG-DSRNet-L freq+prior | real20 | -0.853 | -0.0170 | -0.0252 | +0.00190 |
| PG-DSRNet-L freq+prior | Objects | -1.133 | -0.0132 | -0.0004 | +0.00070 |
| PG-DSRNet-L freq+prior | Postcard | -0.676 | -0.0011 | -0.0039 | +0.00016 |
| PG-DSRNet-L freq+prior | Wild | -0.384 | +0.0025 | +0.0003 | -0.00025 |

## 7. Self-Collected Synthetic Reflection Results

The five self-collected photos in `self/` are treated as clean transmission images. Reflection inputs are synthesized with a fixed seed and evaluated with full-reference metrics.

| Method | Dataset | Count | PSNR | SSIM | NCC | LMSE | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ERRNet self-trained | Self-synth | 5 | 23.979 | 0.9616 | 0.9847 | 0.00163 | outputs/benchmarks/custom_synth/errnet/metrics.csv |
| DSRNet-L self-trained | Self-synth | 5 | 14.040 | 0.3928 | 0.7256 | 0.08835 | outputs/benchmarks/custom_synth/dsrnet_l/metrics.csv |
| PG-DSRNet-L freq+prior | Self-synth | 5 | 14.066 | 0.3929 | 0.7288 | 0.08685 | outputs/benchmarks/custom_synth/pg_dsrnet_l_freq_prior/metrics.csv |

## 8. Key Findings

- PG-DSRNet improves CEILNet synthetic most clearly. The `freq+prior` run improves by +0.866 PSNR over the self-trained DSRNet-L baseline on CEILNet table2.
- On real20 and most SIR2 subsets, the current PG variants degrade relative to self-trained DSRNet-L. The `freq+prior` real20 delta is -0.853 PSNR.
- On the self-collected synthetic set, ERRNet is much stronger than DSRNet-style models under the current synthesis protocol; PG-DSRNet is only slightly above DSRNet-L. This should be discussed as a self-synthesis/domain-gap result, not as broad real-world superiority.
- The official DSRNet-L epoch18 checkpoint remains stronger than the self-trained DSRNet-L baseline on several real/SIR2 rows under the unified evaluator, while the self-trained DSRNet-L baseline is strongest on CEILNet among the baseline rows.
- Because `freq+prior` trained to epoch60 while `freq` and `prior` trained to epoch55, the PG table should be reported as completed ablation evidence, not definitive same-budget causal proof.
