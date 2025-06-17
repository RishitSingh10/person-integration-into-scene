# Seamless Person Integration into Scene

This project implements a step-by-step pipeline to seamlessly integrate a person into a background scene with photorealistic quality. The process involves background removal, shadow detection, light estimation, color matching, and final blending. This repository includes both scripts and demo notebooks for visualization and testing.

## Table of Contents

* [Project Objectives](#project-objectives)
* [Features](#features)
* [Installation](#installation)
* [Project Structure](#project-structure)
* [Usage](#usage)
* [Models](#models)
* [Tasks Breakdown](#tasks-breakdown)
* [Contributing](#contributing)
* [License](#license)

## Project Objectives

* Remove background from a person's image using multiple models
* Analyze shadows in the background
* Estimate lighting conditions
* Match lighting and color of the person to the scene
* Composite the person into the background photorealistically

## Features

* Support for multiple state-of-the-art background removal models:

  * [U²-Net](https://github.com/xuebinqin/U-2-Net)
  * [SAM (Segment Anything Model)](https://github.com/facebookresearch/segment-anything)
  * [RMBG (BRIA-RMBG)](https://huggingface.co/briaai/RMBG-1.4)
  * [BiRefNet](https://github.com/yujheliu/BiRefNet)
* Optional person localization via DETR before background removal

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/RishitSingh10/seamless-person-integration.git
cd seamless-person-integration
```

2. **Set up the environment using [uv](https://github.com/astral-sh/uv):**
   If you don't have `uv` installed, follow the [official instructions](https://github.com/astral-sh/uv#installation).

3. **Install dependencies:**

```bash
uv sync
uv pip install git+https://github.com/facebookresearch/segment-anything.git
```

4. **Download the SAM model (if using SAM):**

* Model: `vit_h`
* [Download Link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
* Place the file in the `models/` directory

> **Note**: `vit_h` requires 12–16 GB of VRAM. You may consider using `vit_b` or `vit_l` with corresponding checkpoints if using a lower-end GPU.

5. **Set up BiRefNet (if using BiRefNet):**

Clone the BiRefNet repository into the project directory:

```bash
git clone https://github.com/yujheliu/BiRefNet.git
```

## Project Structure

```
.
├── data/
│   ├── processed/         # Output directory for processed images
│   └── raw/               # Input images placed here
├── models/                # Model checkpoints
│   └── sam_vit_h_4b8939.pth
├── scripts/
│   └── remove_bg.py       # Main background removal script
│   ├── birefnet_infer.py  # BiRefNet utility
│   └── person_detector.py # DETR-based person detection
├── BiRefNet/ # Module to use for BiRefNet
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock
```

## Usage

### Background Removal

1. Place your image (e.g. `person.jpeg`) in the `data/raw/` directory.

2. Run the background removal script:

#### Using U²-Net (default):

```bash
python scripts/remove_bg.py person.jpeg --model isnet-general-use
```

#### Using SAM:

```bash
python scripts/remove_bg.py person.jpeg --model sam
```

#### Using RMBG:

```bash
python scripts/remove_bg.py person.jpeg --model rmbg
```

#### Using BiRefNet:

```bash
python scripts/remove_bg.py person.jpeg --model birefnet
```

#### Enable Person Localization (Optional):

For SAM, RMBG, U²-Net, and BiRefNet models, you can enable DETR-based person localization:

```bash
python scripts/remove_bg.py person.jpeg --model sam --localize_person
```

> Note: Some models perform better with tightly cropped input around the person. Enabling localization helps with this.

3. The processed image with a transparent background will be saved in `data/processed/`:

* For `u2net`/`isnet-general-use`: `person.png`
* For `sam`: `person_sam.png`
* For `rmbg`: `person_rmbg.png`
* For `birefnet`: `person_subject.png`

## Tasks Breakdown

### Task 1: Capturing and Preparing the Person's Image

#### Step 1: Capture a High-Quality Image

* Use a high-resolution, front-facing photo with even lighting.
* Save it as `person.jpeg` in `data/raw/`.

#### Step 2: Remove the Background

* Choose a background removal model based on your needs. Recommended to use BiRefNet for best results.
* Enable `--localize_person` for better cropping using DETR.

#### Step 3: Continue with Shadow and Lighting Analysis *(coming soon)*

## Contributing

Contributions are welcome! Please feel free to submit issues, improvements, or pull requests.
