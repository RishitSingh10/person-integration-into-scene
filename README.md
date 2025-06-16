# Seamless Person Integration into Scene

This project implements a step-by-step pipeline to seamlessly integrate a person into a background scene with photorealistic quality. The process involves background removal, shadow detection, light estimation, color matching, and final blending. This repository includes both scripts and demo notebooks for visualization and testing.

## Table of Contents
- [Project Objectives](#project-objectives)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Tasks Breakdown](#tasks-breakdown)
- [Contributing](#contributing)
- [License](#license)

## Project Objectives

- Remove background from a person's image
- Analyze shadows in the background
- Determine light direction (indoor and outdoor)
- Match lighting and color of the person to the scene
- Composite the person seamlessly into the background


## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/RishitSingh10/seamless-person-integration.git
cd seamless-person-integration
```

2. **Set up the environment using [uv](https://github.com/astral-sh/uv):**
If you don't have `uv` installed, follow the [official instructions](https://github.com/astral-sh/uv#installation).

Create and activate a virtual environment:
```bash
uv venv .venv
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
uv sync
uv pip install git+https://github.com/facebookresearch/segment-anything.git
```

4. **Download the SAM model:**
- Model: vit_h
- [Download Link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- Place the downloaded file in the `models/` directory

## Project Structure

```
.
├── data/
│   ├── processed/
│   │   └── person_u2net.png
│   └── raw/
│       └── person.jpeg
├── models/
│   └── sam_vit_h_4b8939.pth
├── scripts/
│   └── remove_bg.py
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock
```

## Usage

### Background Removal

1. Place your input image in the `data/raw/` directory (e.g., `person.jpeg`).
2. Run the script using one of the following commands:

**Using SAM model:**
```bash
python scripts/remove_bg.py person.jpeg --model sam
```

**Using U2Net model:**
```bash
python scripts/remove_bg.py person.jpeg --model isnet-general-use
```

The processed image will be saved in the `data/processed/` directory with a transparent background (e.g., `person_u2net.png`).

## Tasks Breakdown

### Task 1: Capturing and Preparing the Person's Image

#### Step 1: Capture a High-Quality Image
- Use any high-resolution front-view image of a person with even lighting
- Place the image in `data/raw/person.jpeg`

#### Step 2: Remove the Background
- Use the provided script to remove the background
- Choose between SAM or U2Net models based on your needs
- The output will be saved with a transparent background in `data/processed/`





