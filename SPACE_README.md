---
title: PillScope
emoji: 💊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# PillScope — live demo

Identify a pill from a photo using classical computer vision (OpenCV, SIFT keypoint matching —
no machine learning). Pick a sample photo or upload your own, and it's matched against a database
of 251 known pills, live, by the exact pipeline in the
[GitHub repo](https://github.com/Ethan-Thomas-Mathew/PillScope).

Nothing here is precomputed: every photo you try is run through segmentation, colour, FFT texture,
HOG and SIFT matching on the spot. Uploading anything that isn't one of the 251 pills should
correctly come back as *no reliable match* — see the repo's
[Limits section](https://github.com/Ethan-Thomas-Mathew/PillScope#limits) for what the pipeline can
and can't do.
