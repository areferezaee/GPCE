# GPEC: Efficient Pre-LLM Gaussian Process Embedding Correction for Cardiac Video Caption Generation

## Overview

Multimodal large language models (MLLMs) have demonstrated promising capabilities for video understanding and caption generation. However, their performance may degrade when applied to specialized medical imaging domains such as echocardiography.

This repository contains the implementation of **Gaussian Process Embedding Correction (GPEC)**, a modular and computationally efficient **pre-LLM error-correction method** designed to improve the visual representations used by **VideoChat2** for cardiac ultrasound caption generation.

GPEC is inserted between the visual projection stage and the language model. Instead of fine-tuning the original multimodal backbone, GPEC learns a residual correction that moves the projected visual representation toward an annotation-guided target representation while keeping the original VideoChat2 components frozen.

## Method

The proposed GPEC pipeline consists of the following main stages:

1. **Structured annotation processing**
   Structured video annotations are transformed into qualitative attributes.

2. **Reference caption construction**
   The qualitative attributes are used to instantiate a fixed-format reference caption.

3. **Embedding-space target construction**
   The reference caption is mapped into the language-model embedding space to define an annotation-guided target representation.

4. **Residual correction learning**
   GPEC learns a residual correction between the projected visual representation and the target representation.

5. **Gaussian Process modeling**
   The correction function is modeled using a **Sparse Variational Gaussian Process (SVGP)** with:

   * inducing points,
   * natural-parameter variational updates, and
   * a block-wise linear kernel formulation.

6. **Pre-LLM integration**
   The learned correction is applied between the visual projection stage and the language model, while the original VideoChat2 components remain frozen during both training and inference.

### Conceptual Pipeline

```text
Cardiac Ultrasound Video
          │
          ▼
      VideoChat2
          │
          ▼
   Visual Projection
          │
          ▼
        GPEC
   (Residual Correction)
          │
          ▼
 Language Model Embedding
          │
          ▼
 Caption Generation
```

## Evaluation

GPEC is evaluated using complementary measures at multiple levels, including:

* **Representation-level evaluation**
* **Caption-level evaluation**
* **Content-oriented evaluation**
* **Execution-time evaluation**

The experiments compare the original **VideoChat2** configuration with **VideoChat2 + GPEC** under the same input and reference conditions.

The reported experiments show improvements in caption similarity and content alignment after applying the proposed pre-LLM correction. In the evaluated experimental setting, GPEC adds **less than 0.05 seconds of inference-time overhead per video**.

## Repository Structure

```text
GPCE/
├── CardioModule.py
├── utils.py
│
├── Error_Estimator/
│   ├── core_gp_model.py
│   ├── core_model.py
│   ├── err_est.py
│   └── runner.py
│
├── create_GT_captions/
│   ├── main_get_GT.py
│   └── qualitative_captions.csv
│
├── final_caption_evaluation/
│   └── comp_metrics.py
│
└── representation_eng/
    ├── __init__.py
    ├── cardio.py
    └── cardio_loader.py
```

## Installation

Installation instructions will be added here.

## Usage

Usage examples and training/inference commands will be added here.

## Data

Details about the cardiac ultrasound dataset, annotation format, preprocessing, and evaluation splits will be added here.

## Results

Detailed quantitative results and comparison tables will be added here.

## Citation

If you use this repository or the GPEC method in your research, please cite:

```bibtex
@article{gpec,
  title   = {GPEC: Efficient Pre-LLM Gaussian Process Embedding Correction for Cardiac Video Caption Generation},
  author  = {Author Names},
  journal = {Venue},
  year    = {Year}
}
```

## License

License information will be added here.
