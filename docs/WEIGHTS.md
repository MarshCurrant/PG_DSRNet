# 权重提交说明

GitHub 仓库不直接提交 `.pt/.pth` 权重文件。请将以下最终权重上传到百度网盘或 OneDrive，并在论文与 README 中替换链接占位符。

## 必交/建议提交权重

| 用途 | 建议上传文件名 | 本地路径 | 大小 | SHA256 |
| --- | --- | --- | --- | --- |
| ERRNet 自训练/unaligned fine-tune | `errnet_self_trained_latest.pt` | `repos/ERRNet/checkpoints/errnet_unaligned_ft/errnet_latest.pt` | 331M | `7f168e224a612951564cc9f5fad756de27578f91b84dff3f77fd4a5a3d84e2d3` |
| DSRNet-L 自训练 Setting I | `dsrnet_l_setting_i_self_trained_latest.pt` | `repos/DSRNet/checkpoints/dsrnet_l_train_setting_i/weights/20260613-042722/dsrnet_l_train_setting_i_latest.pt` | 1.4G | `bc474166864b6894584773829971d1efcbccbe873edc27132fc4943d56e6a194` |
| PG-DSRNet-L freq+prior | `pg_dsrnet_l_freq_prior_latest.pt` | `repos/DSRNet/checkpoints/pg_dsrnet_l_freq_prior_ft/weights/20260616-020451/pg_dsrnet_l_freq_prior_ft_latest.pt` | 1.4G | `7dbbe30d542c827c7ba6269ba573d8630e1f301f16b512d7595d7bff46d342b0` |

## 复现参考权重

以下官方权重用于复现实验和 sanity check，不作为本项目自训练成果主权重：

- ERRNet 官方权重：`repos/ERRNet/checkpoints/errnet/errnet_060_00463920.pt`
- DSRNet-L 官方 epoch18：`repos/DSRNet/weights/dsrnet_l_epoch18.pt`
- DSRNet-S 官方 epoch14 smoke test：`repos/DSRNet/weights/dsrnet_s_epoch14.pt`

## 链接占位

- 百度网盘：`TODO_BAIDU_LINK`
- OneDrive：`TODO_ONEDRIVE_LINK`
- 提取码：`TODO_CODE`

## 上传后检查

1. 下载网盘中的文件到临时目录。
2. 重新计算 SHA256，与上表一致。
3. 在论文和 README 中填入链接。
4. 保持 `.gitignore` 中的 `*.pt`、`*.pth` 规则，避免权重误提交。
