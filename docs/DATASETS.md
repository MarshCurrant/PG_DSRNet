# Dataset Layout

The course branch of ERRNet expects all raw downloads under:

```text
repos/ERRNet/datasets/raw_data/
```

Expected layout:

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

## Sources

- VOC2012: https://dataset.bj.bcebos.com/voc/VOCtrainval_11-May-2012.tar
- Berkeley/Zhang real data: https://arxiv.org/abs/1806.05376
- Course ERRNet package: https://drive.google.com/drive/folders/1_tN6JDlAmKZTgaqniQep1YJXmbFwGav7?usp=drive_link
- Course Baidu package: https://pan.baidu.com/s/1MWb4eT18ySjogKVlcfPozg?pwd=egv2
- DSRNet all-in-one data: https://drive.google.com/file/d/1hFZItZAzAt-LnfNj-2phBRwqplDUasQy/view?usp=sharing
- SIR2 project page: https://sir2data.github.io/

## Prepare Processed Data

Note: the current ERRNet/DSRNet code lists contain 15287 VOC crop filenames. This exceeds the 7643-image count stated in the course slides, so the checker uses the repository count while the report should mention this implementation detail.

ERRNet:

```bash
cd /home/nebula/qqm/DIP/repos/ERRNet
MPLCONFIGDIR=/tmp/matplotlib /home/nebula/.conda/envs/dip-errnet/bin/python datasets/prepare_train_data.py
MPLCONFIGDIR=/tmp/matplotlib /home/nebula/.conda/envs/dip-errnet/bin/python datasets/prepare_test_data.py
```

Count check:

```bash
cd /home/nebula/qqm/DIP
/home/nebula/.conda/envs/dip-errnet/bin/python scripts/check_data_layout.py
```

DSRNet layout bridge:

```bash
cd /home/nebula/qqm/DIP
/home/nebula/.conda/envs/dip-dsrnet/bin/python scripts/prepare_dsrnet_from_errnet.py --force
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

By default it uses symlinks. Add `--copy` only if the target machine cannot preserve symlinks.

DSRNet official all-in-one data has been extracted to:

```text
data/processed/dsrnet_official_extract/reflection-removal/
  train/VOCdevkit/VOC2012/PNGImages      # 17125 PNG crops
  train/real                             # 89 paired real training images
  train/nature                           # 200 paired nature images, for Setting II only if used
  test/real20_420                        # 20 official 420-wide real20 pairs
  test/SIR2/SolidObjectDataset           # 200 pairs
  test/SIR2/PostcardDataset              # 199 pairs
  test/SIR2/WildSceneDataset             # 55 pairs
```

For DSRNet paper-protocol reproduction, use this official all-in-one path as
`--base_dir`. The bridge-created `data/processed/dsrnet_base/test/real20_420`
points to the ERRNet processed real20 data and should only be used for unified
wrapper comparisons.

## Custom Photos

For the five self-collected examples, prefer paired capture:

```text
data/custom/reflection/
  001.png
  ...
data/custom/clean/
  001.png
  ...
```

If clean references are not available, use:

```text
data/custom/reflection_only/
  001.png
  ...
```

and report qualitative results only.
