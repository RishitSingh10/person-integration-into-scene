# Seamless Person Integration into Scene

This project implements a step-by-step pipeline to seamlessly integrate a person into a background scene with photorealistic quality. The process involves background removal, shadow detection, light estimation, color matching, and final blending. This repository includes both scripts and demo notebooks for visualization and testing.

---

## Project Objectives

- Remove background from a person's image.
- Analyze shadows in the background.
- Determine light direction (indoor and outdoor).
- Match lighting and color of the person to the scene.
- Composite the person seamlessly into the background.

---

## Tasks Breakdown

### Task 1: Capturing and Preparing the Person's Image

#### Step 1: Capture a High-Quality Image
- Use any high-resolution front-view image of a person with even lighting.
- Place the image in `data/raw/person.jpg`.

#### Step 2: Remove the Background

Script: `scripts/remove_bg.py`