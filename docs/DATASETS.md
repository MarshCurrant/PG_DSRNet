# Dataset Layout

Datasets are not stored in Git. Download them separately and place them under the expected directories below.

## ERRNet Layout

ERRNet expects raw data under:

```text
repos/ERRNet/datasets/raw_data/
```

Expected structure:

```text
raw_data/
  VOCdevkit/
    VOC2012/
      JPEGImages/
  CEILNet/
    testdata_reflection_synthetic_table2/
  real89/
    blended/
    transmission_layer/
  robustsirr_test_dataset/
    real20/
      blended/
      transmission_layer/
    SIR2/
      PostcardDataset/
      SolidObjectDataset/
      WildSceneDataset/
  Dataset/
    DSLR/
      unaligned_train250/
```

Prepare ERRNet processed data:

```bash
conda activate dip-errnet
cd repos/ERRNet
python datasets/prepare_train_data.py
python datasets/prepare_test_data.py
cd ../..
```

Check the processed layout:

```bash
conda activate dip-errnet
python scripts/check_data_layout.py
```

## DSRNet Layout

For native DSRNet reproduction, use the official all-in-one reflection-removal data package and extract it to:

```text
data/processed/dsrnet_official_extract/reflection-removal/
```

Expected important subdirectories:

```text
reflection-removal/
  train/VOCdevkit/VOC2012/PNGImages
  train/real
  train/nature
  test/real20_420
  test/SIR2/SolidObjectDataset
  test/SIR2/PostcardDataset
  test/SIR2/WildSceneDataset
```

For unified wrapper comparison, a bridge layout can be created from ERRNet processed data:

```bash
conda activate dip-dsrnet
python scripts/prepare_dsrnet_from_errnet.py --force
```

The bridge creates:

```text
data/processed/dsrnet_base/
  train/VOCdevkit/VOC2012/PNGImages
  train/real
  test/CEILNet_table2
  test/real20_420
  test/SIR2/PostcardDataset
  test/SIR2/SolidObjectDataset
  test/SIR2/WildSceneDataset
```

The bridge uses symlinks by default. Add `--copy` only when the target filesystem does not preserve symlinks.

Important: the bridge-created `test/real20_420` is for unified wrapper comparison only. For the DSRNet native paper protocol, use the official all-in-one `test/real20_420`.

## Dataset Sources

- VOC2012: https://dataset.bj.bcebos.com/voc/VOCtrainval_11-May-2012.tar
- Berkeley/Zhang real reflection data: https://arxiv.org/abs/1806.05376
- SIR2 project page: https://sir2data.github.io/
- DSRNet all-in-one data: use the official DSRNet release link from the upstream project.

## Custom Images

For full-reference evaluation, prepare paired inputs and clean targets:

```text
data/my_eval/
  blended/
    sample_001.png
  transmission_layer/
    sample_001.png
```

For reflection-only images without clean targets, run inference and report qualitative results only.

For clean self-collected photos, `scripts/synthesize_self_reflection.py` can synthesize reflection-contaminated inputs and produce paired evaluation data:

```bash
python scripts/synthesize_self_reflection.py \
  --clean-dir self \
  --output-dir data/custom_synth
```
