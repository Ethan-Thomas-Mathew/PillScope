"""Web demo: the real PillScope pipeline, running live, no desktop app needed.

Pick one of the sample photos to see it work instantly, or upload your own pill photo
(plain, light background, one pill) and get a real answer from the same code the
desktop app uses - this is not a replay of saved results.

    python app.py            # local dev server
    (deployed as a Hugging Face Space: see README.md)
"""
import cv2
import gradio as gr

from pillscope import pipeline, report
from pillscope.config import SAMPLES_DIR, Settings
from pillscope.database import PillDatabase
from pillscope.io_utils import imread

SETTINGS = Settings()
DB = PillDatabase()
if not DB.rows:
    raise SystemExit("data/pill_database.csv is empty - this demo needs the pill gallery shipped with it.")
print(f"Preparing features for {len(DB.rows)} pills (first run only; cached after)...")
DB.ensure_features(SETTINGS)
print("Ready.")

SAMPLES = sorted(SAMPLES_DIR.glob("*.jpg"))
VERDICT_COLOR = {"match": "green", "possible": "orange", "unknown": "red"}


def _rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def identify_photo(image):
    """image: RGB numpy array from Gradio. Returns everything the UI displays."""
    if image is None:
        return None, "Pick a sample photo or upload one to begin.", None, [], None

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    a = pipeline.analyze(img_bgr, SETTINGS)
    matches = pipeline.identify(a, DB, SETTINGS)
    kind, sentence = pipeline.verdict(matches)

    verdict_md = f"<span style='color:{VERDICT_COLOR[kind]}; font-size:1.2em'><b>{sentence}</b></span>"
    stages = [
        (_rgb(report.mask_overlay(a)), "1. Pill outline"),
        (a.gray, "2. Preprocessed"),
        (a.edge_map, "3. Canny edges"),
        (report.keypoint_view(a), "4. SIFT keypoints"),
    ]
    table = [[i + 1, m.row["name"], f"{m.score * 100:.1f}", m.inliers, m.row["shape"], m.row["colorName"]]
             for i, m in enumerate(matches)]

    match_view = None
    if matches:
        ref_img = imread(DB.image_path(matches[0].row))
        if ref_img is not None:
            ref = pipeline.analyze(ref_img, SETTINGS)
            match_view = pipeline.draw_match_view(a, ref, SETTINGS.sift_ratio)

    return stages, verdict_md, match_view, table, "\n".join(report.describe(a))


with gr.Blocks(title="PillScope demo") as demo:
    gr.Markdown(
        "# PillScope\n"
        "Identify a pill from a photo using classical computer vision (OpenCV, SIFT - no machine learning). "
        f"Matched against a database of {len(DB.rows)} known pills. "
        "[Code on GitHub](https://github.com/Ethan-Thomas-Mathew/PillScope) · "
        "[Pipeline walkthrough](https://ethan-thomas-mathew.github.io/PillScope/)"
    )
    with gr.Row():
        with gr.Column(scale=1):
            photo = gr.Image(label="Pill photo", type="numpy", height=320)
            gr.Examples(
                examples=[[str(p)] for p in SAMPLES],
                inputs=photo,
                label="Or try a sample photo",
                examples_per_page=len(SAMPLES) or 1,
            )
            run_btn = gr.Button("Identify pill", variant="primary")
            details = gr.Textbox(label="Pill properties", lines=7, interactive=False)
        with gr.Column(scale=2):
            verdict = gr.Markdown("Pick a sample photo or upload one to begin.")
            stage_gallery = gr.Gallery(label="What the pipeline sees", columns=4, height=180)
            results = gr.Dataframe(
                headers=["#", "Pill", "Score", "SIFT matches", "Shape", "Colour"],
                label="Top matches",
            )
            match_view = gr.Image(label="Your photo (left) vs best match (right) - green lines are matching keypoints")

    run_btn.click(identify_photo, inputs=photo, outputs=[stage_gallery, verdict, match_view, results, details])
    photo.change(identify_photo, inputs=photo, outputs=[stage_gallery, verdict, match_view, results, details])

    gr.Markdown(
        "**About this demo.** Every photo here is run through the real pipeline (segmentation, colour, "
        "FFT texture, HOG, SIFT keypoint matching) on each request - nothing is precomputed or faked. "
        "The database only has the 251 pills from the project's training set, so an upload of anything else "
        "should correctly come back as *no reliable match*. See the "
        "[README](https://github.com/Ethan-Thomas-Mathew/PillScope#limits) for what the pipeline does and doesn't do."
    )


if __name__ == "__main__":
    demo.launch()
