# Rice Leaf Disease Detection Using CNN

Classifies rice leaf images into three diseases: bacterial leaf blight, brown spot and leaf smut.

## Dataset
119 images (40 bacterial leaf blight, 40 brown spot, 39 leaf smut):
https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1001-RiceLeaf.zip

The images are not included in this repository.

## What's in the notebook
- Data analysis (class balance, image sizes, duplicates, colors, background)
- Preprocessing (resize to 448×128, stratified train/validation/test split)
- Baseline CNN
- Augmentation study
- Transfer learning (MobileNetV2, EfficientNetB0)
- Model comparison on a test set and with 5-fold cross-validation
- Challenges report and conclusion

## Results

| Model | Cross-validation accuracy (119 images) | Test accuracy (18 images) |
|---|---|---|
| MobileNetV2 | 0.889 | 0.722 |
| EfficientNetB0 | 0.899 | 0.778 |

My own baseline CNN scored about 0.87 on validation.

**Recommended model: MobileNetV2.** Accuracy is tied with EfficientNetB0 (a one-image difference), and MobileNetV2 is about 30% smaller and faster per image.

## Limits
- Small dataset, so scores are rough estimates (I would expect about 85 to 90%).
- Some photos may show the same leaf, which could make scores slightly high.
- Image size and background differ between classes, and I did not check whether the model relies on them.

## How to run
1. Download the dataset and set `DATA_DIR` in the notebook to your folder.
2. Install: `pip install tensorflow scikit-learn pandas matplotlib seaborn pillow scipy`
3. Run the notebook from top to bottom (the full run takes about an hour on a CPU).
