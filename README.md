# GL261 Brain Tumor Segmentation Benchmark

First segmentation benchmark on the GL261 mouse brain ultrasound dataset.

## Dataset

The GL261 dataset (Dorosti et al., *Scientific Data*, 2025) contains **1,856
annotated B-mode ultrasound images** from **12 C57BL/6 mice** (7 tumor-bearing,
5 non-tumor mice), acquired at **30 MHz** with a custom 64-element phased
array. Five annotators plus two expert reviewers produced consensus ground-truth
masks. Inter-rater Dice is 0.88--0.90.

- **Source:** Figshare, DOI [10.6084/m9.figshare.27237894](https://doi.org/10.6084/m9.figshare.27237894)
- **Paper:** Dorosti et al., *High-Resolution Ultrasound Data for AI-Based Segmentation in Mouse Brain Tumor.* Scientific Data 12, 1322 (2025).
  DOI [10.1038/s41597-025-05619-z](https://doi.org/10.1038/s41597-025-05619-z)

The original data descriptor provides no model baselines.

## Quickstart

```bash
pip install -e .

# Download dataset from Figshare
python download.py

# Prepare data (parse metadata, binarize masks, create nnU-Net layout)
python prepare.py

# Train SMP UNet with EfficientNet-B4 encoder
python train.py --encoder efficientnet-b4 --epochs 300 --aug-preset medical_v1 \
    --lr 1e-3 --encoder-lr-mult 0.1 --seed 42

# Evaluate (Dice, IoU, HD95, Surface Dice)
python evaluate.py --pred-dir checkpoints/predictions --model-name "EfficientNet-B4"
```

## Results

All results on the v2.1 validation set (352 images: 246 tumor-positive,
106 tumor-free). Models are fully automatic unless noted.

| Rank | Model | Encoder | Pretraining | T-Dice | O-Dice | FP/106 |
|-----:|-------|---------|-------------|-------:|-------:|-------:|
| 1 | UltraSam MMDet (mixed) | ViT-B | US-43d (282K US) | **0.837** | **0.849** | 13 |
| 2 | SMP UNet | ConvNeXt-V2 Tiny | FCMAE + IN-22k | 0.748 | 0.781 | 15 |
| 3 | SMP UNet | EfficientNet-B4 | ImageNet-1k | 0.726 | 0.755 | 19 |
| 4 | USFM FPN | ViT-B | 2M clinical US | 0.668 | 0.703 | 23 |
| 5 | nnU-Net 2D | From scratch | None | 0.652 | 0.725 | 11 |
| 6 | SMP UNet | ResNet-34 | ImageNet-1k | 0.628 | 0.666 | 26 |

**T-Dice** = Dice averaged over tumor-positive images only (primary metric).
**O-Dice** = Dice averaged over all images, including tumor-free (TN=1.0, FP=0.0).
**FP/106** = false positive count on 106 tumor-free images.

### Observations

- **Pretrained encoders help.** Two of three ImageNet-pretrained SMP UNets
  outperform nnU-Net 2D trained from scratch (0.652), though ResNet-34 (0.628) does not.
- **Dataset mixing.** UltraSam MMDet trained on GL261 + BraTioUS (human
  brain tumor US) reaches 0.837 T-Dice with 13 FP.
- **US-specific pretraining is mixed.** UltraSam (282K clinical US) is the best
  encoder overall, but USFM (2M clinical US, self-supervised) underperforms
  ImageNet baselines. The 30 MHz preclinical domain gap is one possible
  explanation, though our tests are too limited to confirm this.
- **Seed variance is large.** ConvNeXt-V2 Tiny across 3 seeds: 0.596, 0.628,
  0.748 (std=0.065). Single-seed rankings should be treated with caution.

## Reproducing Other Models

### UltraSam MMDet (T-Dice=0.837)

Requires [UltraSam](https://github.com/openmedlab/UltraSam) repo + MMDetection.
Key config: Mask2Former decoder, 1024px, COCO-format annotations. See upstream
repo for training recipe.

### nnU-Net (T-Dice=0.652)

```bash
pip install nnunetv2
export nnUNet_raw="data/processed/nnunet"
export nnUNet_preprocessed="data/processed/nnunet_preprocessed"
export nnUNet_results="checkpoints/nnunet"
nnUNetv2_plan_and_preprocess -d 501
nnUNetv2_train 501 2d 0
```

## Evaluation Protocol

**Split (v2.1):** Mouse-level 2-way split. Train = 9 mice / 1,504 images.
Val = 3 mice / 352 images (M04 tumor in-vivo, M09 non-tumor in-vivo, M10
tumor ex-vivo). No image-level leakage.

**Primary metric: T-Dice.** Per-image Dice averaged over tumor-positive images.
This follows the nnU-Net / Metrics Reloaded convention for foreground-only
evaluation. All masks binarized at threshold 0.5.

**Secondary metric: O-Dice.** Per-image Dice averaged over all 352 images.
True negatives score 1.0; false positives score 0.0.

**Detection indicator: FP/106.** Count of tumor-free images where the model
predicts a non-empty mask.

## Citation

If you use this benchmark, please cite the original dataset paper:

> *High-Resolution Ultrasound Data for AI-Based Segmentation in Mouse Brain Tumor.*
> Dorosti et al., Scientific Data 12, 1322 (2025).
> DOI [10.1038/s41597-025-05619-z](https://doi.org/10.1038/s41597-025-05619-z)

## License

MIT
