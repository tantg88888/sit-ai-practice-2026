# Week 8 · SENTRY-10: acoustic safety monitor

Listens to a (fictional) industrial site, classifies safety-relevant sounds with a fine-tuned Audio Spectrogram Transformer, grades each one by context (zone, time of night, recent events), and writes and speaks the supervisor alert, all inside a live Gradio app.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tantg88888/sit-ai-practice-2026/blob/main/week8-acoustic-safety-monitor/B_live_demo.ipynb) **← live demo (Notebook B)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tantg88888/sit-ai-practice-2026/blob/main/week8-acoustic-safety-monitor/A_train_and_evaluate.ipynb) **← training and evaluation (Notebook A)**

**Format:** 4-person team project (Team 10), Speech Processing and Multimodal AI, Week 8, SIT Graduate Certificate in AI Practice, 2026
**My role:** <TODO: fill in>

## Run the demo
1. Open **`B_live_demo.ipynb`** in Colab (badge above) and pick **Runtime → Change runtime type → T4 GPU**. CPU also works, but more slowly.
2. Optional: add a Colab secret `GROQ_API_KEY` (key icon in the sidebar, with notebook access switched on). With the key, the LLM writes each alert live (MODE `LIVE`). Without it, the notebook runs in MODE `SAFE`: the model and the severity rules still run live, and the alert text comes from the cache that Notebook A built.
3. **Runtime → Run all.** The model and demo files download from the Hugging Face Hub (`tantg88888/sentry10-ast-esc50`, about 350 MB). After about **3–5 minutes**, a public Gradio link appears under the last cell.
4. Open the link **in a new tab**. The browser won't give the microphone to a page inside the Colab frame. Press **🔊 test the PA**, then **▶ NEXT** through the 8-beat night shift. You can also make the sound the NEXT button points at into the microphone, or drop an audio file into the upload box.

## Results
All figures below were printed by the notebooks. They come from the run recorded in `B_live_demo.ipynb` (the metrics summary and the incident report), at confidence gate τ = 0.45.

| Test | Result |
|---|---|
| In-domain accuracy, ESC-50 fold 5 (held out) | **0.981** |
| Linear probe on the frozen AST backbone (baseline) | **0.994**, higher than the fine-tune |
| False-alarm rate on 240 impulsive clips never seen in training (fireworks, can opening, door knock, mouse click, clapping, typing) | **30.8%** |
| Recall through a simulated cheap-microphone channel (band-limit, reverb, 10 dB noise, MP3) | **56.2%** |
| Cross-source recall on a different recording library | **60.0%** (n = 5) |
| Scripted demo night: correct classifications | **8/8** |

What the numbers say:
- **Fine-tuning didn't beat the baseline.** On this small test fold, a logistic regression on the frozen AudioSet features scored slightly higher than the fine-tuned model. Most of the accuracy comes from AST's pretraining.
- **In-domain accuracy overstates real-world performance.** Recall falls to 56% through a degraded microphone and to 60% on a different recording source. About a third of unseen impulsive sounds still raise a false alarm at τ = 0.45.
- **The context layer changes the decision, not the label.** In the demo, the same crying-baby clip is scored **INFO at 22:31** (suppressed in the childcare annexe) and **HIGH at 01:17** (the suppression is revoked while an incident is open).

## How it works
- **Classifier:** `MIT/ast-finetuned-audioset-10-10-0.4593` with a new 11-way head: 10 safety classes (glass breaking, siren, clock alarm, crying baby, chainsaw, hand saw, car horn, engine, vacuum cleaner, footsteps) plus `background_other`. Training uses ESC-50 folds 1–4, SpecAugment and discriminative learning rates (backbone 3e-5, head 3e-4), for 6 epochs. Fold 5 is held out.
- **Honest evaluation:** a linear-probe baseline, 6 impulsive categories held out of training entirely to measure false alarms, a simulated degraded channel, an optional cross-source test, and a τ sweep that trades recall against false alarms.
- **Context layer:** hand-written, auditable rules. Severity depends on the zone (Machine Hall, Chemical Store, Childcare Annexe, Perimeter), with +1 out of hours. Events within a 5-minute window are correlated, and only a cascade across zones can declare CRITICAL. A siren is treated as help arriving and never escalates.
- **Alert and voice:** a Groq LLM (Llama 3.3 70B, with fallback models) receives only the event list, never audio, and writes a two-sentence alert. gTTS speaks it on the public address at **HIGH and CRITICAL only**. Cached text and pre-rendered MP3s are used when the network or key is missing.
- **Gradio app:** a beat player (NEXT, back, replay, restart, playback speed). A streaming microphone has its own live spectrogram, a top-3 confidence panel and interlocks that stop the system hearing its own PA. There is also a file-upload fallback. After the demo, the notebook draws an incident card and an incident report from the log.

## Reproduce training / publish the model
This is a one-time step. The demo only needs the published Hub repo.
1. In Colab, add a secret **`HF_TOKEN`** with **write** access. `GROQ_API_KEY` is optional; without it, template alert text is cached.
2. Open **`A_train_and_evaluate.ipynb`** on a **T4 GPU**. Check that `TRAIN = True` (the default) and that `HF_USER` in the first code cell is your username.
3. **Runtime → Run all** (about 25–35 minutes). This downloads ESC-50, fine-tunes the model, evaluates it, builds the demo night, caches the alerts and renders the PA audio. The **Step 8b** cell then pushes everything to `HF_REPO = tantg88888/sentry10-ast-esc50`:
   - repo root: `model.safetensors`, `config.json` (with the `id2label` class names), `preprocessor_config.json`, `README.md`
   - `artifacts/demo_cache.json` (metrics, events and cached alerts, with local paths removed)
   - `artifacts/demo_audio/ev_00…ev_07_*.wav` (8 ESC-50 fold-5 clips), `artifacts/demo_audio/pa/pa_*.mp3` (pre-rendered announcements)
   - `artifacts/figures/*.png`
4. Notebook B then downloads the repo with `snapshot_download` on every Run all.

## Data
- **[ESC-50](https://github.com/karolpiczak/ESC-50)** (K. J. Piczak, 2015). There are 2,000 five-second environmental clips; 11 classes are used here, and Notebook A downloads the dataset automatically. Licence: **CC BY-NC 3.0**. The 8 demo clips on the Hugging Face Hub are ESC-50 clips and are for non-commercial use only.
- **AST backbone:** `MIT/ast-finetuned-audioset-10-10-0.4593`, pretrained on AudioSet, from the Hugging Face Hub.
- The optional cross-source test uses a course-provided sample library. That library isn't redistributed: it isn't in this repo or on the Hub. Without it, Notebook A skips that test.
- "Tuas Bay Works, Block 7" and its four zones are a fictional case-study site.

## Limitations
- The test set is small: 40 clips per class in ESC-50, and the cross-source figure rests on only 5 clips. The confidence intervals are wide.
- The fine-tune doesn't beat a linear probe in-domain, and performance drops sharply on other microphones and recording sources.
- A false-alarm rate of 30.8% on unseen impulsive sounds is too high for real use without a stricter gate or more negative training data.
- The model assigns one label per 5-second window, so overlapping events aren't handled. `clock_alarm` stands in for a fire-alarm sounder.
- The severity rules are written by hand, not learned, and the system has never been tested on a real factory floor.
- The demo is a student prototype and not a safety system.
