# GPEC: Efficient Pre-LLM Gaussian Process Embedding Correction for Cardiac Video Caption Generation

## Overview

Multimodal large language models (MLLMs) have demonstrated promising capabilities for video understanding and caption generation. However, their performance may degrade when applied to specialized medical imaging domains such as echocardiography.

This repository contains the implementation of **Gaussian Process Embedding Correction (GPEC)**, a modular and computationally efficient **pre-LLM error-correction method** designed to improve the visual representations used by **VideoChat2** for cardiac ultrasound caption generation.

GPEC is inserted between the visual projection stage and the language model. Instead of fine-tuning the original multimodal backbone, GPEC learns a residual correction that moves the projected visual representation toward an annotation-guided target representation while keeping the original VideoChat2 components frozen.

## Method

The proposed GPEC pipeline consists of the following main stages:

1. Structured video annotations are transformed into qualitative attributes.
2. The qualitative attributes are used to construct a fixed-format reference caption.
3. The reference caption is mapped into the language-model embedding space to define an annotation-guided target representation.
4. GPEC learns a residual correction between the projected visual representation and the target representation.
5. The correction function is modeled using a **Sparse Variational Gaussian Process (SVGP)** with inducing points, natural-parameter variational updates, and a block-wise linear kernel formulation.
6. The learned correction is applied between the visual projection stage and the language model while the original VideoChat2 components remain frozen during training and inference.

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
   Residual Correction
          │
          ▼
 Language Model Embedding
          │
          ▼
 Caption Generation
```

## Repository Structure

```text
GPCE/
├── CardioModule.py
├── utils.py
├── requirements.txt
├── README.md
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

The implementation was developed and tested in a Python 3.12.3 virtual environment.

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The repository requires a CUDA-enabled PyTorch environment for GPU-based execution.

## Usage

### 1. VideoChat2 Video Encoding

For each input video, VideoChat2 is used to extract the projected visual representation.

```python
vid_path = "<VIDEO_PATH>"
vname = "<VIDEO_NAME>"

num_frame = 16
resolution = 224

vid, msg = load_video(
    vid_path,
    num_segments=num_frame,
    return_msg=True,
    resolution=resolution,
    norm="vit"
)

new_pos_emb = get_sinusoid_encoding_table(
    n_position=(resolution // 16) ** 2 * num_frame,
    cur_frame=num_frame
)

model.vision_encoder.encoder.pos_embed = new_pos_emb

print(msg)

TC, H, W = vid.shape
video = vid.reshape(1, TC // 3, 3, H, W).to("cuda:0")

img_list = []

phi_proj_o, _, feat1, feat2, Qformero = model.encode_img(
    video,
    "Watch the video and answer the question."
)

temporal_embedimag = feat1
img_embeds = feat2

img_list.append(phi_proj_o)
```

Here, `phi_proj_o` denotes the projected visual representation used as the input representation before GPEC correction.

### 2. GPEC Test Configuration

The GPEC test configuration can be executed with:

```bash
python3 CardioModule.py \
    --dataset='Cardio' \
    --natural-lr=.9 \
    --num-inducing-points=11 \
    --train_strategy='Joint' \
    --outputscale=1. \
    --Only_TEST='Y'
```

### 3. Applying the Learned Embedding Correction

The learned correction is loaded and added to the original projected visual representation.

```python
import time
import torch

start = time.perf_counter()

deltzbar = "<CORRECTION_PATH>"

deltz = torch.load(deltzbar)
deltz = deltz.to("cuda")

alpha = 1.0

z_corrected = phi_proj_o + alpha * deltz

img_list3 = []
img_list3.append(z_corrected)

end = time.perf_counter()

print(f"Time: {end - start:.6f} seconds")
```

The corrected representation is defined as:

```text
z_corrected = z + α Δz
```

where `z` is the original projected visual representation, `Δz` is the learned correction, and `α` controls the correction strength.

### 4. Caption Generation

The corrected representation is then passed to VideoChat2 for caption generation.

```python
chat = EasyDict({
    "system": "",
    "roles": ("<|user|>\n", "<|end|>\n<|assistant|>"),
    "messages": [],
    "sep": ""
})

ask(
    f"<Video><VideoHere></Video> {msg} Describe the process.",
    chat
)

llm_message = answer(
    conv=chat,
    model=model,
    do_sample=False,
    img_list=img_list3,
    max_new_tokens=96,
    print_res=True
)[0]

print(llm_message)
```

## Evaluation Protocol

GPEC is evaluated by comparing the original **VideoChat2** representation with the **VideoChat2 + GPEC** corrected representation under the same input and generation conditions.

The following settings are kept consistent between the baseline and corrected configurations:

* input video,
* number of sampled frames,
* video resolution,
* prompt,
* generation configuration,
* reference conditions.

The main difference between the two configurations is the application of the learned embedding correction.

### Neutral Prompt Design

The evaluation uses a neutral, open-ended prompt so that the language prompt does not provide task-specific information to the model.

The prompt should not explicitly mention:

* cardiac anatomy,
* echocardiography,
* ultrasound,
* a specific clinical process,
* a predefined content category, or
* the expected answer content.

For example, domain-specific prompts such as:

```text
Describe the cardiac process.
```

are avoided.

Instead, a generic prompt such as:

```text
Describe the content. or Describe the process.
```

is used so that the generated description is driven primarily by the visual representation.

The same prompt is used for both **VideoChat2** and **VideoChat2 + GPEC** to maintain a controlled comparison.

## Evaluation

The proposed method is evaluated using complementary measures at multiple levels:

* representation-level evaluation,
* caption-level evaluation,
* content-oriented evaluation,
* execution-time evaluation.

The experiments compare the original VideoChat2 model with the proposed VideoChat2 + GPEC configuration under the same input and reference conditions.

The reported experiments show improvements in caption similarity and content alignment after applying the proposed pre-LLM correction.

In the evaluated experimental setting, GPEC introduces less than **0.05 seconds of inference-time overhead per video**.

## Data

The experiments use cardiac ultrasound videos together with structured annotations and reference captions.

Dataset preparation details, annotation format, data splits, and local data paths will be documented here.

Paths to local or private storage locations are intentionally omitted from the repository.

## Results

Detailed quantitative results, evaluation tables, and qualitative examples will be added here.

## Citation

If you use this repository or the GPEC method in your research, please cite:

```bibtex
@article{gpec,
  title   = {GPEC: Efficient Pre-LLM Gaussian Process Embedding Correction for Cardiac Video Caption Generation},
  author  = {Arefeh Rezaei},
  journal = {Venue},
  year    = {2026}
}
```


