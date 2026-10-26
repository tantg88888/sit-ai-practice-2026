# Week 4 · SmartCart: photo of groceries → priced receipt

Upload one photo of a grocery counter and get back an itemised receipt in SGD. Items the model is unsure about are flagged for a human instead of being guessed.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tantg88888/sit-ai-practice-2026/blob/main/week4-smartcart-vision/notebook.ipynb)

**Area:** Computer vision · **Format:** 4-person team project · SIT Graduate Certificate in AI Practice, 2026

## Run the demo
1. Click **Open in Colab** above.
2. Use the preset **T4 GPU** runtime. A CPU runtime also works for the demo, just slower.
3. Click **Runtime → Run all**. Leave `TRAIN = False` in the settings cell.
4. After about 3–5 minutes, a public Gradio link appears in the last cell. Upload a photo or click a sample scene, press **Scan basket**, edit any line, then press **Recalculate**.

Demo mode skips training. It downloads the trained artifacts from the Hugging Face Hub (`tantg101/smartcart-vision`). The charts and tables saved from the original training run stay visible in the notebook.

## Results
All numbers below were printed in the notebook during the recorded training run, which ran on a CPU runtime.

| What | Result |
|---|---|
| Data | 3,232 images, 81 product classes (40 per class cap). Stratified split 2,262 / 485 / 485. 0 near-duplicate pairs with conflicting labels |
| Detector (YOLO11n, 20 epochs, synthetic val: 20 scenes / 67 boxes) | mAP50-95 **0.924**, mAP50 0.978, precision 0.99, recall 0.925 |
| Detector on 24 hard stress-test scenes (real-photo backgrounds, occlusion) | best F1 **0.763** at conf 0.10 (P 0.935, R 0.644). Lowest threshold tried |
| Recognition (linear head on DINOv2-small features) | validation accuracy **0.939** (970 images). **200 / 223** detector crops named correctly |
| Day 4 retrain (all dataset images, 30% held out) | val acc 0.952 with no augmentation, 0.952 with classical augmentation only, 0.950 with classical + cGAN. Mean lift on the 8 weakest classes −0.007 |
| ONNX export of the head | max PyTorch-vs-ONNX difference 1.9e-6. Size 122.0 KB fp32 → 31.6 KB int8 (3.9× smaller). 0.029 ms (fp32) vs 0.038 ms (int8) per call, batch of 8 on CPU |

The cGAN result is a negative one: synthetic GAN images did not beat classical augmentation at this data scale. The app ships the classical-augmentation head.

## How it works
- **Free box labels.** Product photos are pasted onto 640×640 canvases. The paste positions become YOLO labels, and YOLO11n is fine-tuned on a single class, `product`.
- **Detector operating point.** A stress-test set with photo backgrounds, rotation, lighting jitter and up to 30% overlap is used to sweep the confidence threshold. The best-F1 value (0.10) is saved in the bundle manifest.
- **Recognition.** Each crop goes through frozen DINOv2-small, giving 384-d features. A linear head classifies them into 81 classes. The head is exported to ONNX (fp32 and int8) and served with ONNX Runtime.
- **Checkout logic.** Below a confidence threshold, an item is marked UNKNOWN, flagged for a human and charged 0. The threshold is **0.45, a fixed default rather than a tuned value**. Prices are an invented catalog in SGD, with about 20% of items on a 10% promo. The receipt is editable.
- **Explainability.** Grad-CAM on the ViT backbone (256 patch tokens reshaped to a 16×16 grid) is shown alongside the top-3 most similar catalog images.

## Reproduce training / publish the model
This is a one-time step. It creates the Hugging Face repo that the demo downloads.
1. Create a Hugging Face access token with **write** permission. In Colab, open 🔑 **Secrets** and add it as `HF_TOKEN`, with notebook access enabled.
2. In the settings cell, set `TRAIN = True`, then run **Runtime → Run all**. The original notebook estimates about 45–60 minutes on a T4. For a quick smoke run, set `SC_FAST` to `'1'` in the same cell.
3. The **Publish to Hugging Face** cell uploads these files to `tantg101/smartcart-vision`: `detector.pt`, `head.onnx`, `head.int8.onnx`, `head_v2.pt`, `catalog_prices.csv`, `gallery_index.npy`, `gallery_meta.csv` (class names only), `manifest.json` (class list + detector confidence), `sample_scene.jpg`, and `hard_000.jpg`–`hard_002.jpg`.
4. Set `TRAIN = False` again before the demo.

## Data
- [GroceryStoreDataset](https://github.com/marcusklasson/GroceryStoreDataset) (Klasson et al.): photos of fruit, vegetables and packaged goods. It is cloned at run time. Check its licence before reusing its images. The sample scenes on Hugging Face are composites of these images.
- Prices and discounts are invented for the demo. No real store or partner data is used.

## Limitations
- The detector was trained and tuned only on synthetic collages. Real counter photos, with shadows, hands and packaging glare, will be harder.
- The detector threshold of 0.10 was the lowest value in the sweep, so the true best point may be lower. The UNKNOWN threshold (0.45) has not been tuned.
- The 81 classes are the dataset's products only. Anything else is either forced into the nearest class or flagged as UNKNOWN.
- The validation sets are small: 20 synthetic detector scenes and a few test images per weak class. Lift numbers per class are noisy.
- The training run was on CPU, so the timings are CPU timings for the tiny head only, not for the full pipeline.

## Stack
YOLO11 (Ultralytics) · DINOv2 (timm) · PyTorch · ONNX Runtime · Grad-CAM · Gradio · Hugging Face Hub
