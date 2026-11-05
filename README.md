# PillScope

[![CI](https://github.com/Ethan-Thomas-Mathew/PillScope/actions/workflows/ci.yml/badge.svg)](https://github.com/Ethan-Thomas-Mathew/PillScope/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

Identify a pill from a photo using classical computer vision (OpenCV, no machine learning).
Load a photo, and the app finds the pill, measures its shape and colour, and looks it up in a database of 251 known pills by matching SIFT keypoints. It also tells you when a pill is probably *not* in the database.

**[Pipeline walkthrough](https://ethan-thomas-mathew.github.io/PillScope/)**: step through what each stage sees, with saved results (instant, no server). **[Try it live](https://huggingface.co/spaces/Ethan-Thomas-Mathew/pillscope)**: upload your own pill photo and get a real answer from the actual pipeline, running on the spot.

![Identify tab](docs/screenshots/identify.png)

This project combines eight classical computer-vision techniques - segmentation, filtering, sampling, colour, frequency analysis, edge detection, feature matching - into one pipeline and one application. Each technique is now one stage of the pipeline, and the standalone building blocks are still available, side by side with their own sliders, in the **Explore** tab.

## How it works

```
photo
  |  find the pill: threshold the difference from the background (or grow the background
  |  from the corners), keep the biggest blob
  |  measure the outline: shape name, circularity, aspect ratio
  |  colour signature: base colour (HSV / LAB) + coloured speckles
  |  crop, grayscale, median denoise, CLAHE contrast, Canny edges
  |  colour histogram, FFT texture profile, HOG + SIFT keypoints
  v
compare with every pill in the database  ->  rank the best matches  ->  verdict
```

The deciding clue is **SIFT**: keypoints are matched with Lowe's ratio test, then only matches that agree on one rotation + scale + shift (RANSAC) are counted. Right matches have 29+ consistent keypoints, pills that are not in the database have about 3, so the app says
*Confident match* (20+), *Possible match* (10 to 19) or *No reliable match* (under 10).
Colour, FFT texture and HOG are cheaper clues that add to the score and break ties.

| Module | Used for |
|---|---|
| `basic_ops.py` | grayscale, geometry, arithmetic (Explore tab) |
| `spatial.py` | convolution vs correlation (Explore tab) |
| `sampling.py` | sampling and quantization (Explore tab) |
| `color.py` | colour signature and histogram (pipeline) |
| `frequency.py` | FFT texture profile (pipeline), ideal filters (Explore) |
| `edges.py` | denoise, CLAHE, Canny (pipeline), Sobel/LoG (Explore) |
| `segmentation.py` | pill outline and shape (pipeline), region growing (Explore) |
| `features.py` | SIFT matching and HOG (pipeline), ORB comparison (Explore) |
| `database.py` | CSV database, validation checks, feature cache |

`spatial.py` and `sampling.py` aren't part of identification itself; they live in the Explore tab.

## Results

Tested on the 251 pills in `data/gallery` (`python scripts/evaluate.py --source <pill_images folder>`):

| Test | Top-1 | Top-5 |
|---|---|---|
| Same photo | 100.0% | 100.0% |
| Rotated 30 / 45 / 180 degrees | 99.6% / 99.2% / 99.6% | 99.6% |
| Shrunk to 70% | 99.6% | 99.6% |
| Gaussian noise | 99.6% | 100.0% |
| Darker | 100.0% | 100.0% |
| **Real photos rotated 30 degrees (227 from the dataset)** | **99.1%** | **100.0%** |

About 0.08 seconds per photo. Of the two misses at 45 degrees, one looks like the same pill stored twice under different numbers (74 matching keypoints), the other is a real miss (it gets flagged as "no reliable match").

**Read this before quoting the numbers.** These tests check whether a pill the database already knows can be recognised again after rotation, shrinking, noise or dimming. They do **not** show that the app can identify pills it has never seen, because the dataset has no imprint, dose or maker information, only photos. Only 75 of the 227 `(N)r30.jpg` photos carry the same number as the original they were made from, so the right answer for that test is found by comparing pixels, not by trusting the file name.

## Quick start

Python 3.9 or newer. Tkinter ships with the Windows installer.

```
pip install -r requirements.txt
python run.py
```

On first start the app computes features for the database (a few seconds). Pick a sample from the drop-down, or use **Open image...**. Samples in `data/samples`:

- `sample_*_rotated30.jpg`: known pills photographed at a different angle
- `sample_100_rotated45_noisy_dark.jpg`: a known pill, rotated, noisy and dim
- `sample_unknown_pill252_not_in_database.jpg`: a pill that was left out of the database on purpose, so the answer should be "no reliable match"

Without a window:

```
python scripts/identify.py data/samples/sample_042_rotated30.jpg --report outputs/report.png
```

Rebuild the database from the dataset folder (the files named `12.jpg` are the originals):

```
python scripts/build_database.py --source "C:/path/to/pill_images"
python scripts/evaluate.py --source "C:/path/to/pill_images"
```

## Web demo

`app.py` is a small [Gradio](https://gradio.app) front end over the same `pillscope` package the desktop
app and CLI use — there's no separate "demo" implementation to keep in sync. It ships as a
[Hugging Face Space](https://huggingface.co/spaces/Ethan-Thomas-Mathew/pillscope): pick one of the sample
photos or upload your own, and it runs the real pipeline (segmentation, colour, texture, HOG, SIFT
matching) against the 251-pill database and returns a real answer, not a saved one.

Run it locally the same way it runs on the Space:

```
pip install -r requirements.txt
python app.py
```

To deploy your own copy: create a (free) [Hugging Face Space](https://huggingface.co/new-space) with
SDK **Gradio**, then push this repo to it and copy `SPACE_README.md`'s content into the Space's
`README.md` (that file carries the Space's configuration, which this repo's own `README.md` doesn't
need):

```
git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
git push space main:main
```

## The app

| Tab | What it does |
|---|---|
| **Identify** | Photo to answer, with every pipeline stage shown, the top 5 matches, and a picture of the matching keypoints. "Save report picture" exports a summary. |
| **Explore** | One panel per technique, with sliders: basic operations, filters, sampling, colour spaces, Fourier, edges, segmentation (click to place a seed), SIFT/ORB/HOG. |
| **Database** | Browse, edit and delete pills, add a pill from a photo (shape and colour are measured for you), import a folder, and run the checks from the original `pillDatabase.py` (missing fields, colour that disagrees with the image, duplicate imprints). |

![Features panel](docs/screenshots/explore_features.png)

## Project layout

```
run.py                     start the app
pillscope/
  pipeline.py              analyze() and identify(): the whole pipeline
  basic_ops.py spatial.py sampling.py color.py frequency.py edges.py segmentation.py features.py
  database.py              CSV + feature cache
  report.py                one-picture summary
  gui/                     Tkinter app (identify_tab, panels, database_tab, app)
scripts/                   build_database.py, evaluate.py, identify.py
data/                      pill_database.csv, gallery/ (reference photos), samples/
docs/screenshots/
```

## Limits

- Matching works on the speckle pattern, so it suits pills that are visibly different from each other. Pills that differ only by a printed imprint need an imprint reader (OCR), which is not included.
- The pill is expected on a plain, light background with one pill per photo.
- The imprint, dose and maker columns of the database are empty. Fill them in the Database tab.
- Some images in the source dataset are truncated JPEGs. OpenCV prints "Corrupt JPEG data" warnings for them; it is harmless.

## Ideas for next steps

- Add imprint reading (OCR) so pills with the same look can be told apart.
- Detect and split several pills in one photo (connected components + watershed).
- Use a spatial index (for example FLANN) when the database grows past a few thousand pills.
