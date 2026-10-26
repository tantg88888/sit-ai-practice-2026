# Week 5 · ElderShield closed-book Q&A: SFT then DPO on Llama-3.2-3B

Fine-tunes a small open LLM to answer questions about Singapore's ElderShield scheme from memory, decline questions it was never taught, and reply in a warm tone. It also measures honestly where this worked and where it did not.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tantg88888/sit-ai-practice-2026/blob/main/week5-llm-finetune-abstention/notebook.ipynb)

**Team:** 4-person team project, SIT Graduate Certificate in AI Practice (2026).

## Run the demo
1. Click **Open in Colab** above.
2. Go to **Runtime → Change runtime type** and choose **T4 GPU**.
3. Go to **Runtime → Run all**. `TRAIN = False` is the default.
4. After about 5–7 minutes, a public Gradio link appears at the bottom. Pick a sample question to see the base model (v0) and the fine-tuned model (v2) answer side by side.

Demo mode installs Unsloth and loads the 4-bit base model together with the trained LoRA adapter from Hugging Face (`tantg101/eldershield-llama3.2-3b-lora`). It skips training and evaluation and shows the scores recorded from the training run.

## Results
The model was scored on 62 held-out probes, worded differently from the training questions: 30 recall, 16 unanswerable and 16 general-knowledge. All numbers come from one training run with a single seed (3407).

| Version | Recall (taught facts) | Abstain (untaught questions) | General knowledge | Warm tone |
|---|---|---|---|---|
| v0: base model | 37% | 75% | 94% | 23% |
| v1: + SFT | 67% | 69% | 94% | 100% |
| v2: + DPO (final) | 63% | 75% | 88% | 100% |

- **Recall improved: 37% → 63%.** SFT did most of the work (67%), and DPO cost one probe.
- **Warm tone improved: 23% → 100%.** Tone was taught inside SFT.
- **Abstention did not improve: 75% → 75%.** SFT made the model slightly over-eager (69%). The DPO round only won that back, so v2 still guesses on 1 in 4 unanswerable questions, the same as the base model.
- **General knowledge dipped slightly: 94% → 88%** (one more probe missed).
- The final model got 17 of 62 probes wrong.
- **Verdict:** iterate, don't deploy. This is a prototype and is not ready for unsupervised use.

## How it works
- **Model:** `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` with one QLoRA adapter (rank 16, all attention and MLP projections), trained with Unsloth.
- **Data:** 26 ElderShield facts, expanded into 159 question paraphrases with warm answers that still contain the exact fact. There are also 26 "I don't have that information…" rows, built from a separate pool of made-up questions that is checked to have no overlap with the eval set, plus 8 general-knowledge rehearsal rows. That makes 193 SFT rows.
- **Round 1, SFT:** 90 steps, trained on the response tokens only. This teaches the facts, the tone and the habit of abstaining.
- **Round 2, DPO:** 78 preference pairs over 60 steps, with beta 0.3 to keep the update gentle. The pairs prefer refusing over made-up answers, the true fact over refusing or a wrong fact, and a warm answer over a terse one.
- **Evaluation:** greedy decoding. Recall and general knowledge are scored by keyword match. Abstention means the reply contains the refusal phrase. Tone is a proxy: at least 12 words and a warm phrase.
- **App:** Gradio. v0 is produced by temporarily disabling the LoRA adapter (`model.disable_adapter()`), so a single loaded model can show both versions.

## Reproduce training / publish the model
You only need to do this once, to create the Hugging Face repo that demo mode loads from.
1. In Colab, open the key icon (**Secrets**) and add `HF_TOKEN`, a Hugging Face token with **write** access. Turn on notebook access.
2. Set `TRAIN = True` in the first code cell, choose a T4 GPU (or better), and run **Runtime → Run all**.
3. The notebook evaluates v0, trains with SFT, evaluates v1, trains with DPO and evaluates v2. A push cell then uploads `eldershield_v2/` to `tantg101/eldershield-llama3.2-3b-lora` as a public repo. It contains:
   - `adapter_config.json` and `adapter_model.safetensors` (the LoRA adapter)
   - the tokenizer files (`tokenizer.json`, `tokenizer_config.json` and related files, including the Llama-3.1 chat template)
   - a PEFT model-card `README.md`
4. Set `TRAIN = False` again before committing the notebook.

The recorded run used an NVIDIA L4: SFT took about 140 s and DPO about 117 s. A T4 will be slower. The notebook installs `unsloth` and `gradio` without pinned versions. The recorded run used unsloth 2026.7.2, transformers 5.5.0, trl 0.24.0, peft 0.19.1 and gradio 6.19.0. If a newer release breaks the notebook, pin those versions.

## Data
- The ElderShield facts were written by the team from public information about the scheme. They are embedded in the notebook.
- The unanswerable questions, the made-up "fake facts" used for refusal training, and the general-knowledge questions were all written by the team.
- No personal or partner data is used.
- The base model is Meta Llama 3.2 (Llama 3.2 Community License), via Unsloth's 4-bit build.

## Limitations
- The main goal, abstaining on untaught questions, did not improve over the base model (75% → 75%).
- Results come from one seed and 62 probes. A single probe moves recall by about 3 points and abstain or general by about 6 points.
- Scoring uses keyword matching plus a simple warmth heuristic. That is cheap and transparent, but it can be fooled. For example, the tone proxy rewards a template phrase.
- Closed-book only, with no retrieval. Facts are frozen at training time. This is an educational demo and not financial advice.
