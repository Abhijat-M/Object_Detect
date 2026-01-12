
# Task 1: Object Detection Pipeline (Pascal VOC)

## Overview

This task implements a complete **end-to-end object detection pipeline** using deep learning.
The objective of Task 1 is to design, train, and run an object detector on a standard annotated dataset, demonstrating correct **system architecture, data handling, training, and inference**.

The focus of this task is **pipeline construction and execution**, not benchmark-level performance comparison.

---

## Objective

The goals of Task 1 are to:

* Build a modular object detection system
* Load and parse Pascal VOC–style annotations
* Train a deep learning–based detector
* Perform inference on images or video
* Demonstrate correct end-to-end execution

This task emphasizes **engineering correctness and structure** rather than quantitative evaluation metrics.

---

## Dataset

* **Dataset format:** Pascal VOC
* **Annotation type:** XML bounding boxes
* **Dataset location:** `PASCAL_VOC/`

Each image has a corresponding XML file containing:

* Object class names
* Bounding box coordinates (`xmin`, `ymin`, `xmax`, `ymax`)

The dataset is loaded and parsed programmatically using a custom dataset loader.

---

## Object Classes

The detector is trained on the following object categories:

* Person
* Car
* Dog
* Bicycle

A background class is handled internally by the detection framework.

---

## Model Architecture

The object detector is based on **Faster R-CNN**, a widely used two-stage object detection architecture.

### Components

* **Backbone:** Custom convolutional neural network for feature extraction
* **Region Proposal Network (RPN):**

  * Generates candidate object regions
  * Uses multi-scale anchors
* **Detection Head:**

  * Classifies proposed regions
  * Refines bounding box coordinates

This architecture provides strong localization performance and serves as a reusable foundation for later tasks.

---

## Training Configuration

* **Framework:** PyTorch
* **Optimizer:** Stochastic Gradient Descent (SGD)
* **Learning rate:** 0.005
* **Momentum:** 0.9
* **Weight decay:** 0.0005
* **Batch size:** 2
* **Epochs:** 40

Model checkpoints are saved during training for reuse during inference.

---

## Inference

After training, the model can perform inference on images or video streams.

During inference, the system outputs:

* Bounding boxes for detected objects
* Predicted class labels
* Confidence scores

Non-maximum suppression (NMS) is applied internally to remove duplicate detections.

---

## Project Structure

```
Object_Detect/
├── checkpoints/          # Saved model checkpoints
├── configs/              # Configuration files
├── data/                 # Dataset loading utilities
├── engine/               # Training logic
├── inference/            # Inference scripts (image/video)
├── models/               # Model and backbone definitions
├── PASCAL_VOC/           # Pascal VOC dataset
├── requirements.txt      # Python dependencies
├── train.py              # Training entry point
└── README.md
```

This modular structure ensures:

* Clean separation of concerns
* Reusability across tasks
* Maintainability and scalability

---

## Execution

All scripts are executed from the project root.

### Training

```
python train.py
```

### Inference (video or image)

```
python inference/video_infer.py
```

---

## Evaluation Note

Task 1 focuses on **building and validating a working object detection pipeline**.
Formal quantitative evaluation metrics such as mean Average Precision (mAP) were **not emphasized** in this task.

Quantitative evaluation is introduced in **Task 2**, where defect detection reliability and validation performance are critical.

---

## Key Takeaways

* Implemented a complete object detection system from dataset to inference
* Demonstrated correct parsing of Pascal VOC annotations
* Built a reusable Faster R-CNN–based detection pipeline
* Established a strong foundation for domain-specific detection tasks

---

## Conclusion

Task 1 demonstrates the successful implementation of a structured object detection pipeline using deep learning. The system is modular, extensible, and suitable as a base for more specialized applications, such as manufacturing quality inspection explored in subsequent tasks.
