# Week 3 — Vape Image Moderation (ResNet-18 vs ViT-B/16)

Flags marketplace listing photos that show an e-vaporiser (vape) so they can be blocked or sent to human review, using transfer learning with a recall-tuned decision threshold.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tantg88888/sit-ai-practice-2026/blob/main/week3-vape-image-moderation/notebook.ipynb)

**Solo project.** It extends SIT Lab 3 (DNN Workflow Part 2).

## Run the demo

1. Click **Open in Colab** above.
2. Go to **Runtime → Change runtime type** and pick **T4 GPU**. CPU also works, just more slowly.
3. Choose **Runtime → Run all**. Leave `TRAIN = False` in the first code cell.
4. After about 3–5 minutes, a public Gradio link appears in the last code cell. Upload a photo and the app shows:
   - P(vape) from both top models;
   - an **ALLOW** or **BLOCK (send to review)** decision at each model's tuned threshold.

Demo mode needs no API keys. It downloads the two trained checkpoints, `manifest.json` and `metrics.json` from the Hugging Face Hub repo [`tantg88888/vape-image-moderation`](https://huggingface.co/tantg88888/vape-image-moderation).

## Results

> **Not trained yet.** Run `TRAIN = True` once (see below), and these numbers will be filled in from `metrics.json`. The notebook also shows the table in demo mode. No numbers have been recorded so far.

Test set (held out; each model scored once):

| Config | Val acc | Test acc | Vape precision | Vape recall | Vape F1 |
|---|---|---|---|---|---|
| resnet_full | | | | | |
| resnet_head | | | | | |
| vit_full | | | | | |
| vit_head | | | | | |

The champion is chosen on validation only. Its test results at the default and tuned thresholds:

| Threshold | Accuracy | Vape precision | Vape recall | Confusion [tn fp fn tp] |
|---|---|---|---|---|
| 0.50 | | | | |
| tuned (target recall ≥ 0.99 on val) | | | | |

**Why v2 exists.** In v1, every configuration scored about 100%, because the negatives were easy 160-px Imagenette photos. v2 adds harder negatives so the comparison means something. Accuracy is therefore expected to come in below v1's ~100%.

## How it works

- **Data.**
  - Vape photos (positives) come from a Roboflow Universe dataset.
  - Negatives are a roughly 50/50 mix:
    - everyday objects from Imagenette-320;
    - vape lookalikes from Open Images (pens, flashlights, mobile phones, remote controls, microphones), the hard negatives.
  - Up to 800 images per class, with a seeded, stratified 70/15/15 split.
- **Models.**
  - ResNet-18 (torchvision) and ViT-B/16 (`google/vit-base-patch16-224`, Hugging Face), both pretrained on ImageNet.
  - The new 2-class head is trained either by **full fine-tune** or **head-only** (frozen backbone), giving 4 runs.
  - Training uses augmentation (crop, flip, rotation, colour jitter) and a class-weighted cross-entropy loss.
- **Selection without test leakage.** The top two configurations are ranked by validation accuracy, with ties broken by validation vape recall. The test set is used only for reporting.
- **Moderation threshold.** For each saved model, the P(vape) threshold is tuned on the validation set. It is the most precise threshold that still keeps vape recall at or above 99%. Missing a vape matters more than a false alarm that a reviewer can clear.
- **Evaluation.** Accuracy and vape precision, recall and F1, plus confusion matrices, ROC and PR curves, and a gallery of misclassified test images.
- **Experiment tracking.** Optional Weights & Biases logging, used only when a `WANDB_API_KEY` secret exists.

## Reproduce training / publish the model (one-time)

1. In Colab, open the 🔑 **Secrets** panel and add these, with notebook access enabled:
   - `ROBOFLOW_API_KEY`: a free Roboflow key. It is needed to download the vape dataset.
   - `HF_TOKEN`: a Hugging Face token with **write** access. It is needed to publish.
   - `WANDB_API_KEY` (optional): turns on wandb logging.
2. Set `TRAIN = True` in the first code cell. Choose a **T4 GPU** and click **Run all**. On a T4, a full run takes roughly 20–30 minutes; the ViT full fine-tune is the slow part.
3. **Step 11b** uploads these files to `tantg88888/vape-image-moderation`, creating the repo if needed:
   - `<top-1 config>.pt` and `<top-2 config>.pt`: weights plus metadata (backbone, strategy, normalisation, class names and the tuned threshold);
   - `manifest.json`: the top-2 names, thresholds and test accuracy;
   - `metrics.json`: the test-set table, the champion's results at 0.5 and at the tuned threshold, and dataset sizes.
4. Copy the numbers from `metrics.json` (or the printed tables) into the **Results** section above. Then set `TRAIN = False` again for the demo.

**Example images (optional).** Add 3–4 of your own photos, or CC-licensed ones, to a `samples/` folder next to this README, for example `vape_1.jpg`, `vape_2.jpg`, `pen_1.jpg` and `phone_1.jpg`. The demo finds them through the GitHub API and shows them as clickable `gr.Examples`. If the folder is missing, it skips them.

## Data

- **Vape images:** Roboflow Universe [`vape-dataset/vape-dataset`](https://universe.roboflow.com/vape-dataset/vape-dataset), version 1. It needs a Roboflow API key, and the licence is the one shown on the dataset page. Images are downloaded at run time and are **not** redistributed here.
- **Everyday negatives:** [Imagenette-320](https://github.com/fastai/imagenette) (fast.ai), a subset of ImageNet.
- **Hard negatives:** [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) validation images, downloaded with FiftyOne. Annotations are CC BY 4.0; the images are listed as CC BY 2.0.
- **Pretrained weights:** torchvision ResNet-18 (ImageNet) and `google/vit-base-patch16-224` (Apache-2.0).

## Limitations

- No results have been recorded yet. Every number has to come from the first `TRAIN = True` run.
- The dataset is small (at most 800 vape images, all from one Roboflow source). Roboflow sets can contain near-duplicate frames, and there is no deduplication step, so test scores may be optimistic.
- "Not vape" covers only 10 Imagenette categories plus 5 lookalike classes. Real listing photos are far more varied.
- The 99% recall target is met on the validation set. Recall on new data can be lower.
- Each config trains for only 5 epochs, and there is no ablation that measures how much the augmentation helps.
- It is a single-label, whole-image classifier: it does not localise the vape or explain its decision (for example, no Grad-CAM).
