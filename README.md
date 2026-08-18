# VA-DETECTOR — Oriented Object Detection with YOLO

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-OBB-111111?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![DOTA](https://img.shields.io/badge/Dataset-DOTA%20v2-222222?style=flat-square)

Computer vision project for **oriented object detection (OBB)** of vehicles and aircraft in high-resolution aerial imagery using **YOLO**.

The project implements an end-to-end workflow covering **image preprocessing, dataset tiling, YOLO OBB training, validation, metric analysis, confidence-threshold optimization and real-time inference**.

> Developed as an academic project associated with **Pontifícia Universidade Católica de Campinas (PUC-Campinas)**.

---

## Overview

Detecting objects in aerial and satellite images introduces challenges that are less common in conventional object detection. Objects can appear at arbitrary orientations, different scales and, in the case of small vehicles, occupy only a small number of pixels.

To address the orientation problem, VA-DETECTOR uses **Oriented Bounding Boxes**, allowing each detection to represent the object's rotation instead of restricting the bounding box to the horizontal and vertical axes.

The selected classes in the project are:

- **Plane**
- **Large Vehicle**
- **Small Vehicle**

The training pipeline was implemented in Python with **Ultralytics YOLO and PyTorch**, using GPU acceleration.

---

## Project Pipeline

```text
DOTA v2
   │
   ▼
High-resolution aerial images
   │
   ▼
Image tiling / cropping
   │
   ├── 1024 × 1024 tiles
   └── OBB annotations preserved
   │
   ▼
YOLO OBB training
   │
   ▼
Validation & metric analysis
   │
   ├── Precision
   ├── Recall
   ├── mAP@0.5
   ├── mAP@0.5:0.95
   ├── Confusion matrix
   └── F1-Confidence curve
   │
   ▼
Confidence threshold analysis
   │
   ▼
Inference
   ├── Images
   └── Webcam / real-time video
```

---

## Dataset & Preprocessing

The project uses **DOTA v2 (Dataset for Object Detection in Aerial Images)**.

The original images are very large, so the project uses an image-tiling strategy to make them suitable for model training. The training configuration uses **1024×1024 input images**, with a batch size of 4 and 100 training epochs. See the experiment configuration in [`run_exp_2/exp_2_tiles/args.yaml`](run_exp_2/exp_2_tiles/args.yaml).

A custom preprocessing workflow was used to generate smaller image regions while keeping the corresponding oriented annotations aligned with the image tiles.

This step is especially important for aerial detection because aggressive resizing of the original images could make small objects even harder to detect.

---

## Oriented Bounding Boxes

A conventional detector predicts axis-aligned bounding boxes:

```text
┌────────────────────┐
│       object       │
└────────────────────┘
```

With OBB detection, the box can rotate to better follow the object's orientation:

```text
       ╱──────────╲
      ╱   object   ╲
      ╲            ╱
       ╲──────────╱
```

This representation is particularly useful for aerial images, where aircraft and vehicles may appear at arbitrary angles.

---

## Training Configuration

The main experiment was trained with the following configuration:

| Parameter | Value |
|---|---|
| Task | OBB detection |
| Image size | 1024 × 1024 |
| Epochs | 100 |
| Batch size | 4 |
| Device | CUDA / GPU (`device: 0`) |
| Optimizer | Auto |
| Pretrained | Yes |
| Mixed precision | AMP enabled |
| Workers | 4 |
| IoU threshold | 0.70 |
| Initial learning rate | 0.01 |
| Weight decay | 0.0005 |

The complete experiment configuration is available in [`run_exp_2/exp_2_tiles/args.yaml`](run_exp_2/exp_2_tiles/args.yaml).

---

## Results

The final training log records the performance of the model throughout **100 epochs**. At epoch 100, the experiment reached:

| Metric | Result |
|---|---:|
| Precision | **0.798** |
| Recall | **0.711** |
| mAP@0.5 | **0.779** |
| mAP@0.5:0.95 | **0.617** |

These values come directly from the experiment's `results.csv`.

### Best mAP@0.5

The highest recorded **mAP@0.5 was 0.77947 at epoch 83**, with an mAP@0.5:0.95 of **0.61651**.

The project analysis also identified a particularly strong result for the **plane** class, reaching **0.951 mAP@0.5**, while **small vehicles** were the most difficult class, with **0.683 mAP@0.5**.

The lower performance on small vehicles is consistent with the visual limitation of very small targets in aerial imagery: fewer pixels are available to represent shape and appearance, making localization and classification more difficult.

---

## Confidence Threshold Optimization

The project did not rely only on the default inference threshold. A **F1-Confidence curve** was analyzed to determine an operating point that provides a better balance between Precision and Recall.

The selected operating point was:

```text
Confidence threshold: 0.282
Maximum F1-Score:      0.75
```

The experiment artifacts include the F1, Precision and Recall curves used for this analysis.

---

## Inference

The repository includes a dedicated real-time inference script, [`real_time_det.py`](real_time_det.py), which loads the trained `best.pt` weights, opens the default webcam, performs YOLO OBB inference at `imgsz=1024` on GPU device 0, and renders the oriented polygons on the video stream. The application exits when `ESC` is pressed.

### Run real-time detection

After installing the dependencies and ensuring the trained weights are present at:

```text
run_exp_2/exp_2_tiles/weights/best.pt
```

run:

```bash
python real_time_det.py
```

> The current script is configured for a local webcam (`VideoCapture(0)`) and GPU device `0`.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sam0929/Computer_Vision-VA_DETECTOR.git
cd Computer_Vision-VA_DETECTOR
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The repository pins the main deep-learning components to **PyTorch 2.9.0**, **TorchVision 0.24.0** and **Ultralytics 8.3.227**, together with OpenCV, NumPy, Jupyter and related dependencies.

### 4. Run the notebook

The main development and analysis workflow is stored in:

```text
main.ipynb
```

Open it with Jupyter Notebook or JupyterLab and execute the cells in sequence.

---

## Repository Structure

```text
Computer_Vision-VA_DETECTOR/
│
├── main.ipynb                         # Main notebook / analysis workflow
├── real_time_det.py                   # Real-time OBB inference
├── requirements.txt                   # Python dependencies
├── README.md
│
├── run_exp_2/
│   └── exp_2_tiles/
│       ├── args.yaml                  # Training configuration
│       ├── results.csv                # Epoch-by-epoch metrics
│       ├── results.png                # Training curves
│       ├── confusion_matrix.png       # Confusion matrix
│       ├── confusion_matrix_normalized.png
│       ├── BoxF1_curve.png
│       ├── BoxP_curve.png
│       ├── BoxR_curve.png
│       ├── BoxPR_curve.png
│       ├── labels.jpg
│       ├── train_batch*.jpg
│       ├── val_batch*_labels.jpg
│       ├── val_batch*_pred.jpg
│       └── weights/
│           ├── best.pt
│           └── last.pt
│
└── runs/
    └── obb/
        └── predict/                   # Example prediction outputs
```

The repository stores the trained model weights and generated evaluation artifacts, making the experiment results inspectable directly from GitHub.

---

## Technologies

### Programming & Numerical Computing

- Python
- NumPy
- Jupyter Notebook

### Computer Vision & Deep Learning

- Ultralytics YOLO
- YOLO OBB
- PyTorch
- OpenCV
- Oriented Bounding Boxes
- Deep Learning
- Computer Vision

### Hardware Acceleration

- NVIDIA GPU
- CUDA / PyTorch GPU execution
- Automatic Mixed Precision (AMP)

---

## Technical Challenges

### High-resolution aerial imagery

The original DOTA imagery is significantly larger than the model input size. Tiling the images into **1024×1024** regions makes training computationally practical while preserving fine details that are important for small objects.

### Small object detection

Small vehicles were the most challenging class in the project, reaching **0.683 mAP@0.5**, which indicates a clear opportunity for future model and data improvements.

### Choosing an inference threshold

The model's operating point was investigated through the F1-Confidence relationship rather than relying exclusively on a fixed default threshold. The selected threshold of **0.282** produced the maximum F1-Score of **0.75** according to the project analysis.

---

## Evaluation Artifacts

The repository includes the main artifacts used to inspect model behavior:

- Training and validation curves
- Precision-Recall curve
- Precision-Confidence curve
- Recall-Confidence curve
- F1-Confidence curve
- Confusion matrix
- Normalized confusion matrix
- Ground-truth and prediction examples
- Epoch-by-epoch metrics in CSV format

These files are available under [`run_exp_2/exp_2_tiles/`](run_exp_2/exp_2_tiles/).

---

## Future Improvements

Possible next steps include:

- Improving detection of small vehicles
- Experimenting with larger or different YOLO OBB architectures
- Exploring higher input resolutions where hardware permits
- Testing additional augmentation strategies
- Performing broader hyperparameter optimization
- Evaluating inference speed and latency more systematically
- Improving the real-time visualization with class names and confidence scores
- Packaging the trained detector for easier deployment

---

## Author

**Samuel Vanini**  
Computer Engineering student at **PUC-Campinas, Brazil**.

Interested in **computer vision, machine learning, software engineering, backend development, mobile development and embedded systems**.

- GitHub: https://github.com/Sam0929
- LinkedIn: https://www.linkedin.com/in/samuel-vanini-851a4a207/

---

## License

This repository is intended primarily for academic and portfolio purposes.
