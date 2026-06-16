# PG-DSRNet 图像反射去除课程项目

本仓库用于复现实验和课程论文提交，主题为单图反射去除（Single Image Reflection Removal）。

- 课程 baseline：ERRNet，复现官方权重与自训练/微调权重
- 改进基线：DSRNet-L，复现官方 epoch18 与 Setting I 自训练权重
- 本项目方法：PG-DSRNet-L，在 DSRNet-L 上加入训练期 frequency loss 与 reflection-prior weighted loss
- 评测指标：PSNR、SSIM、NCC、LMSE
- 测试集：CEILNet table2、real20、SIR2 Objects/Postcard/Wild、五张自采 clean 图合成反光测试

## 目录结构

```text
repos/ERRNet/                 # ERRNet baseline 代码
repos/DSRNet/                 # DSRNet 与 PG-DSRNet 修改代码
scripts/                      # 数据准备、推理、评测、作图脚本
paper/experiment_results.md   # 当前所有实验结果 Markdown 汇总
paper/cvpr2026_pg_dsrnet/     # CVPR 模板课程论文源码与图片
survey/reflection_removal_survey.md
docs/WEIGHTS.md               # 需要单独上传的权重说明
self/                         # 五张自采 clean 图
```

## 环境

本项目使用两个 conda 环境：

```bash
conda activate dip-errnet
conda activate dip-dsrnet
```

在本机运行官方脚本时建议设置：

```bash
export TORCH_HOME=/home/nebula/qqm/DIP/weights/torch
export MPLCONFIGDIR=/tmp/matplotlib
```

## 数据与评测

检查课程测试集布局：

```bash
/home/nebula/.conda/envs/dip-errnet/bin/python scripts/check_data_layout.py
```

重新导出汇总结果：

```bash
./scripts/export_experiment_results_md.py
```

生成五张自采 clean 图的合成反光数据：

```bash
/home/nebula/.conda/envs/dip-errnet/bin/python scripts/synthesize_self_reflection.py \
  --clean-dir self \
  --output-dir data/custom_synth \
  --seed 2018 \
  --max-long-edge 1024
```

运行自采合成反光三方法评测：

```bash
bash scripts/run_custom_synth_benchmark.sh
```

生成论文图：

```bash
./scripts/make_paper_figures.py
```

## 主要结果

完整结果见：

- `paper/experiment_results.md`
- `outputs/benchmarks/official_protocol_summary.csv`
- `outputs/benchmarks/unified_protocol_summary.csv`
- `outputs/benchmarks/self_trained_summary.csv`
- `outputs/benchmarks/pg_dsrnet_summary.csv`
- `outputs/benchmarks/custom_synth_summary.csv`

核心结论：

- 官方 DSRNet-L epoch18 在 DSRNet native `real20_420` 协议下 PSNR 高于本地 ERRNet native real20。
- 统一 evaluator 下，自训练 DSRNet-L 在 CEILNet、Objects、Postcard、Wild 上优于自训练 ERRNet，但 real20 仍落后。
- PG-DSRNet-L freq+prior 在 CEILNet table2 上相对自训练 DSRNet-L 提升 `+0.866 dB PSNR`。
- PG-DSRNet-L 在 real20 和多数 SIR2 子集下降，应作为轻量 prior 的局限诚实报告。
- 五张自采 clean 图合成反光测试中，ERRNet 对当前合成协议最稳，DSRNet/PG-DSRNet 表现较弱。

## 论文

论文源码位于：

```text
paper/cvpr2026_pg_dsrnet/
```

本地当前未安装 LaTeX 工具链，建议将该目录上传到 Overleaf 编译 PDF。作者姓名、学号、邮箱和成员贡献仍保留 `TODO_*` 占位符，提交前需要替换。

## 权重

权重文件不提交到 GitHub。请按 `docs/WEIGHTS.md` 上传到百度网盘或 OneDrive，并在论文与 README 中替换链接占位符。

权重链接占位：

- 百度网盘：`TODO_BAIDU_LINK`
- OneDrive：`TODO_ONEDRIVE_LINK`

## GitHub 提交准备

当前目录之前不是有效 git worktree。准备提交时执行：

```bash
git init
git remote add origin https://github.com/MarshCurrant/PG_DSRNet.git
git status --ignored
```

确认 `.pt/.pth` 权重、raw/processed datasets、logs、大量 prediction 图片没有进入 git 后，再执行 commit 和 push。
