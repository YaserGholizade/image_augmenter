# Image Augmentor: Large-Scale Dataset Expansion

A high-throughput image augmentation engine designed to generate an **arbitrary number of variations (100+)** for every source image. This tool is optimized for machine learning tasks where dataset diversity and model robustness are critical.

## 📝 Description

Unlike standard augmentation libraries, this engine allows users to specify an exact number of variations to generate per image. It uses a multi-stage pipeline—combining spatial transformations, custom intensity mapping (HEQ), and frequency filters—to ensure that even when generating 100+ variations from a single source, the resulting data remains diverse and valuable for training.

## 🚀 Key Features

* **User-Defined Scaling**: Generate exactly the number of variations you need (from 1 to 100+).
* **Class-Aware Processing**: Automatically detects and preserves subfolder structures.
* **Multi-Resolution Output**: Process images into multiple dimensions (e.g., 32, 48, 64, 96) simultaneously.
* **Advanced Pipeline**: Integrates custom Histogram Equalization and automated filtering for high-volume requests.

## 🛠️ Implemented Transformations

* **Geometric Flips**: Left-right and up-down mirroring.
* **Standard Rotations**: Fixed angles at 90°, 180°, and 270°.
* **Perturbed Rotations**: Fine-grained rotations at $\pm 5^\circ$ using bicubic interpolation.
* **Scaling (Crop-and-Zoom)**: Dimension-specific cropping followed by resizing to the target dimension.
* **Intensity Normalization**: Custom Histogram Equalization (`histeq`) to normalize contrast.
* **Filtering**:
* **Median Filtering**: For noise reduction and smoothing.
* **Gaussian Blur & Sharpening**: Automatically triggered when requested variation count $n > 72$.
* **Realistic Artifacts**: Variations are saved with JPEG quality (60) and BMP formats to simulate different data sources.

## 📂 Dataset Structure

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

## ⚙️ Installation

```bash
git clone https://github.com/YaserGholizade/image_augmenter.git
cd image_augmenter
pip install -r requirements.txt

```

## 💻 Usage

Run the script by defining your input path, target dimensions, and the number of variations per image.

```bash
python main.py --data ./path/to/dataset --out ./path/to/output --vars 100 --dims 32 64

```

### Example:

If you have a dataset folder named `PIZZA` with class subfolders (e.g., `Class1` and `Class2`):

```bash
python main.py --data ./input/PIZZA --out ./output/PIZZA --dims 32 48 --vars 10

```

* **Process**: Every image in every subfolder of `PIZZA` will get 10 unique variations.
* **Output**: The `./output/PIZZA` directory will mirror your input structure, creating `Dim32` and `Dim48` folders inside each class directory.

