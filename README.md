# Image Augmentor: Large-Scale Dataset Expansion

A high-throughput image augmentation engine designed to generate an **arbitrary number of variations (100+)** for every source image. This tool is optimized for machine learning tasks where dataset diversity and model robustness are critical.


## Description

Unlike standard augmentation libraries, this engine allows users to specify an exact number of variations to generate per image. It uses a multi-stage pipeline—combining spatial transformations, custom intensity mapping (HEQ), and frequency filters—to ensure that even when generating 100+ variations from a single source, the resulting data remains diverse and valuable for training.

## Key Features

* **User-Defined Scaling**: Generate exactly the number of variations you need (from 1 to 100+).
* **Class-Aware Processing**: Automatically detects and preserves subfolder structures.
* **Multi-Resolution Output**: Process images into multiple dimensions (e.g., 32, 48, 64, 96) simultaneously.
* **Advanced Pipeline**: Integrates custom Histogram Equalization and automated filtering for high-volume requests.

## Implemented Transformations

* **Geometric Flips**: Left-right and up-down mirroring.
* **Standard Rotations**: Fixed angles at 90°, 180°, and 270°.
* **Perturbed Rotations**: Fine-grained rotations at $\pm 5^\circ$ using bicubic interpolation.
* **Scaling (Crop-and-Zoom)**: Dimension-specific cropping followed by resizing to the target dimension.
* **Intensity Normalization**: Custom Histogram Equalization (`histeq`) to normalize contrast.
* **Filtering**:
* **Median Filtering**: For noise reduction and smoothing.
* **Gaussian Blur & Sharpening**: Automatically triggered when requested variation count $n > 72$.
* **Realistic Artifacts**: Variations are saved with JPEG quality (60) and BMP formats to simulate different data sources.

## Dataset Structure

The script expects a root dataset folder containing subfolders for each class. This hierarchy is preserved in the output directory.

```text
Your_Dataset_Name/
├── Class_A/ 
│   ├── img01.jpg
│   └── img02.jpg
└── Class_B/ 
    ├── img03.jpg
    └── img04.jpg

```

## Installation

```bash
git clone https://github.com/YaserGholizade/image_augmenter.git
cd image_augmenter
pip install -r requirements.txt

```

## Usage

Run the script by defining your input path, target dimensions, and the number of variations per image.

```bash
python main.py --data ./path/to/dataset --out ./path/to/output --vars 100 --dims 32 64

```
## Dataset Used

The dataset used in the associated research paper (the paper is available at https://arxiv.org/html/2606.26207v1)  is available at the following link:

 https://drive.google.com/drive/folders/12m_l-SrigRYjSTMOStH4epzLw_P1HJP0?usp=drive_link

Please download the dataset from the link above and place it in the appropriate directory before running experiments. 

The repository includes the following datasets:
### 1. Resynth Dataset (Reorganized Version)
**License:** CC BY-NC-ND 4.0

**Disclaimer:** This distribution contains a reorganized version of the Resynth dataset. All visual content remains identical to the source; modifications are strictly limited to directory structure and file mapping for technical optimization. In compliance with the "NoDerivatives" clause, no alterations have been made to the images. Users must adhere to the original license terms, which prohibit commercial use and the distribution of modified content. The authors of this reorganization claim no ownership over the underlying data.

- Bongini, P., Molinari, V., Costanzo, A., Tondi, B., & Barni, M. (2025). Training-free Source Attribution of AI-generated Images via Resynthesis. arXiv preprint arXiv:2510.24278.

### 2. VegSeed Dataset (Reorganized Version)
**License:** CC BY 4.0

**Disclaimer:** This dataset is a restructured version of the VegSeed dataset, originally hosted on Mendeley Data. While the image content is unchanged, the directory structure has been modified for improved workflow compatibility. This version is distributed under the Creative Commons Attribution 4.0 International license, allowing for sharing and adaptation provided appropriate credit is given to the original creators.

- Ferdaus, Md Hasanul; Ohona, Syeda Raisha Abedin; Prito, Rizvee Hassan; Ahmed, Masud (2025), “VegSeedsBD: A Comprehensive Image Dataset of Vegetable Seeds.”, Mendeley Data, V1, doi: 10.17632/dtpzbwwpm7.1

### 3. Food-30 Dataset (Revised Version)
**License:** CC BY-NC-SA

**Disclaimer:** This is a revised distribution of the Food-101 dataset. This version has been modified by reducing the scope to 30 classes and filtering images based on specific resolution and aspect ratio criteria. Under the "ShareAlike" clause of the CC BY-NC-SA license, this derivative work is distributed under the same terms as the original. Any further use must remain non-commercial, and proper attribution must be maintained. The authors of this revision claim no ownership over the original image content.

- Bossard, L., Guillaumin, M., & Van Gool, L. (2014, September). Food-101–mining discriminative components with random forests. In European conference on computer vision (pp. 446-461). Cham: Springer International Publishing.

### Example:

If you have a dataset folder named `PIZZA` with class subfolders (e.g., `Class1` and `Class2`):

```bash
python main.py --data ./input/PIZZA --out ./output/PIZZA --dims 32 48 --vars 10

```

* **Process**: Every image in every subfolder of `PIZZA` will get 10 unique variations.
* **Output**: The `./output/PIZZA` directory will mirror your input structure, creating `Dim32` and `Dim48` folders inside each class directory.

