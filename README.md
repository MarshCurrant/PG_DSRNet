# PG-DSRNet

PG-DSRNet 是一个用于单张图像反射去除（Single Image Reflection Removal, SIRR）的开源实验代码库。仓库包含 ERRNet 基线、DSRNet-L 复现代码，以及在 DSRNet-L 上加入轻量频率先验和反射强度先验训练损失的 PG-DSRNet-L。

仓库目标是让使用者能够：

- 对自己的反光图像运行 ERRNet / DSRNet / PG-DSRNet 推理；
- 在成对数据上计算 PSNR、SSIM、NCC、LMSE；
- 复现实验汇总表和自采 clean 图像的合成反光评测；
- 查看最终权重说明和实验结果摘要。

## 目录结构

```text
repos/ERRNet/                 # ERRNet 基线代码
repos/DSRNet/                 # DSRNet 与 PG-DSRNet 代码
scripts/                      # 推理、评测、数据合成、实验汇总脚本
docs/                         # 数据、环境、命令、权重和结果说明
outputs/benchmarks/           # 指标汇总文件；大规模预测图默认不提交
data/                         # 数据目录；原始数据和生成数据默认不提交
```

论文成稿、课程汇报模板、调研草稿和个人图片不属于开源代码主体，默认不纳入 Git 提交。

## 1. 环境安装

建议分别创建 ERRNet 和 DSRNet 环境，因为两个项目依赖版本不同。

ERRNet 环境：

```bash
conda create -n dip-errnet python=3.10 -y
conda activate dip-errnet
pip install -r repos/ERRNet/requirements.txt
```

DSRNet 环境：

```bash
conda create -n dip-dsrnet python=3.9 -y
conda activate dip-dsrnet
pip install -r repos/DSRNet/requirements.txt
```

如果 requirements 中的 PyTorch 版本不适合你的 CUDA，请按机器环境单独安装匹配版本的 PyTorch。

## 2. 下载权重

模型权重不存入 Git。请下载共享权重包：

- 文件名：`DIP`
- 链接：https://pan.baidu.com/s/151h3CPe1i1OpOMrLGYMfKA?pwd=1234
- 提取码：`1234`

推荐放置路径如下：

| 模型 | 推荐路径 |
| --- | --- |
| ERRNet self-trained | `repos/ERRNet/checkpoints/errnet_unaligned_ft/errnet_latest.pt` |
| DSRNet-L self-trained | `repos/DSRNet/checkpoints/dsrnet_l_train_setting_i/weights/20260613-042722/dsrnet_l_train_setting_i_latest.pt` |
| PG-DSRNet-L freq+prior | `repos/DSRNet/checkpoints/pg_dsrnet_l_freq_prior_ft/weights/20260616-020451/pg_dsrnet_l_freq_prior_ft_latest.pt` |

文件大小和 SHA256 校验值见 [docs/WEIGHTS.md](docs/WEIGHTS.md)。

## 3. 对自己的图片运行推理

把待处理图片放入任意目录，例如：

```text
data/my_images/
  image_001.jpg
  image_002.jpg
```

运行 ERRNet：

```bash
conda activate dip-errnet
python scripts/infer_method.py \
  --method errnet \
  --checkpoint repos/ERRNet/checkpoints/errnet_unaligned_ft/errnet_latest.pt \
  --input_dir data/my_images \
  --output_dir outputs/user/errnet
```

运行 PG-DSRNet-L：

```bash
conda activate dip-dsrnet
python scripts/infer_method.py \
  --method dsrnet \
  --inet dsrnet_l \
  --checkpoint repos/DSRNet/checkpoints/pg_dsrnet_l_freq_prior_ft/weights/20260616-020451/pg_dsrnet_l_freq_prior_ft_latest.pt \
  --input_dir data/my_images \
  --output_dir outputs/user/pg_dsrnet
```

如果输入图像分辨率较大，可以加入 `--max_long_edge 1024` 限制最长边，降低显存占用。

## 4. 成对数据评测

如果有反光输入和 clean ground truth，请按如下结构准备：

```text
data/my_eval/
  blended/
    sample_001.png
  transmission_layer/
    sample_001.png
```

评测预测结果：

```bash
conda activate dip-errnet
python scripts/eval_sirr.py \
  --pred_dir outputs/user/pg_dsrnet \
  --gt_dir data/my_eval/transmission_layer \
  --out_csv outputs/user/pg_dsrnet_metrics.csv \
  --summary_json outputs/user/pg_dsrnet_metrics.json \
  --align crop-min
```

评测指标包括 PSNR、SSIM、NCC 和 LMSE。

## 5. 数据准备

公开数据集的推荐目录结构见 [docs/DATASETS.md](docs/DATASETS.md)。完成数据下载和预处理后，可以检查布局：

```bash
conda activate dip-errnet
python scripts/check_data_layout.py
```

DSRNet 训练和官方协议复现需要 DSRNet all-in-one 数据包。统一 wrapper 比较可以使用 `scripts/prepare_dsrnet_from_errnet.py` 从 ERRNet 数据布局创建桥接目录：

```bash
conda activate dip-dsrnet
python scripts/prepare_dsrnet_from_errnet.py --force
```

## 6. 训练与消融

训练 ERRNet / DSRNet-L 自训练基线：

```bash
ERRNET_PY=/path/to/dip-errnet/bin/python \
DSRNET_PY=/path/to/dip-dsrnet/bin/python \
bash scripts/run_training_queue.sh
```

从 DSRNet-L 自训练权重继续 fine-tune PG-DSRNet-L，并运行频率先验、反射先验、两者结合的消融：

```bash
ERRNET_PY=/path/to/dip-errnet/bin/python \
DSRNET_PY=/path/to/dip-dsrnet/bin/python \
PG_BASE_CKPT=repos/DSRNet/checkpoints/dsrnet_l_train_setting_i/weights/20260613-042722/dsrnet_l_train_setting_i_latest.pt \
bash scripts/run_pg_dsrnet_experiment.sh
```

如果当前 shell 已经激活了正确环境，也可以把 `ERRNET_PY` 或 `DSRNET_PY` 设置为 `python`。

## 7. 自采 clean 图像合成反光

如果你有 clean 图片但没有真实反光 paired GT，可以用现有反射源合成可评测数据。把 clean 图片放到 `self/` 或自定义目录，然后运行：

```bash
conda activate dip-errnet
python scripts/synthesize_self_reflection.py \
  --clean-dir self \
  --output-dir data/custom_synth \
  --seed 2018 \
  --max-long-edge 1024
```

完整运行三种方法的自采合成评测：

```bash
ERRNET_PY=/path/to/dip-errnet/bin/python \
DSRNET_PY=/path/to/dip-dsrnet/bin/python \
bash scripts/run_custom_synth_benchmark.sh
```

## 8. 结果汇总

重新生成 Markdown 结果表：

```bash
python scripts/export_experiment_results_md.py
```

默认输出：

```text
docs/EXPERIMENT_RESULTS.md
```

主要 CSV/JSON 汇总位于：

```text
outputs/benchmarks/official_protocol_summary.csv
outputs/benchmarks/unified_protocol_summary.csv
outputs/benchmarks/self_trained_summary.csv
outputs/benchmarks/pg_dsrnet_summary.csv
outputs/benchmarks/custom_synth_summary.csv
```

当前结果摘要：

- PG-DSRNet-L freq+prior 在 CEILNet table2 上相比自训练 DSRNet-L 提升 `+0.866 dB PSNR`；
- 在 real20 和多数 SIR2 子集上，PG-DSRNet-L 相比自训练 DSRNet-L 有退化，说明该轻量先验存在 domain gap；
- 在五张 self-synth 合成评测图上，ERRNet 在当前合成协议下表现最好。

## 9. 常用文档

- [docs/DATASETS.md](docs/DATASETS.md)：数据集目录结构和下载来源
- [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)：环境安装说明
- [docs/COMMANDS.md](docs/COMMANDS.md)：更完整的训练和评测命令
- [docs/WEIGHTS.md](docs/WEIGHTS.md)：权重下载、大小和 SHA256
- [docs/EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md)：实验结果汇总

## 10. 注意事项

- 大模型权重、原始数据集、训练日志和大规模预测图默认不提交到 Git。
- `repos/ERRNet` 和 `repos/DSRNet` 在本仓库中应作为普通源码目录提交，而不是嵌套 git 仓库或 submodule。
- 参考项目仅用于方法调研，不作为本仓库运行依赖。
