# Week 7 · Four-modality RAG assistant (PDF, scanned PDF, audio, video)

> *Built for an SIT industrial-attachment case study. Not affiliated with or endorsed by SMF; all data is synthetic.*

A grant-advisory assistant that answers questions about Singapore enterprise support schemes. It retrieves from text PDFs, scanned PDFs (OCR), audio and video (Whisper ASR), and answers with checked `[S#]` citations or abstains.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tantg88888/sit-ai-practice-2026/blob/main/week7-multimodal-rag/notebook.ipynb)

**Team:** 4-person team project, SIT Graduate Certificate in AI Practice, 2026.
**My role:** evaluation and testing. I built the 38-question test set, the retrieval and answer-quality metrics, the modality-leak audit and the failure analysis.

## Run the demo
1. Open the notebook in Colab using the badge above.
2. Go to **Runtime → Change runtime type → T4 GPU**.
3. Make sure the Colab secret `HF_TOKEN` is set (key icon in the sidebar) with notebook access enabled. The demo bundle lives in a **private** Hugging Face dataset.
4. Choose **Runtime → Run all** with `BUILD = False`. After about 4–6 minutes a public Gradio link appears in the last cell.

The app has a question box, a `top_k` slider and toggles for hybrid search, rerank and the similarity floor.
It shows the answer, a citation check and a sources table (title, page or timestamp, modality, excerpt).
Without access to the private dataset, the loader cell stops with an explanation. The recorded outputs remain readable in the notebook.

## Results
These numbers are from the recorded build run: 38 questions, with retrieval metrics on the 29 that have a relevant chunk, `top_k = 5`.

| Configuration | Hit@5 | P@5 | R@5 | MRR | Retrieval latency |
|---|---|---|---|---|---|
| Baseline (dense only) | 0.966 | 0.455 | 0.548 | 0.798 | 7.0 ms |
| + Hybrid BM25/dense (alpha 0.8) | 0.966 | 0.497 | 0.577 | 0.863 | 7.3 ms |
| + Cross-encoder rerank (on top of hybrid) | **1.000** | 0.510 | **0.618** | **0.917** | 39.5 ms |
| Hybrid + rewrite + similarity floor 0.4 | 0.828 | 0.589 | 0.456 | 0.770 | – |
| All combined | 0.862 | 0.607 | 0.494 | 0.802 | – |

- **Baseline per-question outcome:** 20 PASS, 8 PARTIAL, 10 FAIL.
- **Abstention:** at baseline, correct behaviour on the 9 must-refuse, must-caveat or must-clarify questions was **0%**, because the baseline never abstained.
  The similarity floor raised it to 33.3% and raised `must_not_contain` from 0.625 to 0.875 (1.000 combined), at the cost of hit rate.
- **Chunking sweep:** 180/40 words had the best MRR of the four sizes tested (0.798, against 0.737–0.777 for the others).

## How it works
- **Ingestion.** There are 57 source units: 13 text PDFs via PyMuPDF, 2 image-only PDFs via Tesseract OCR (220 dpi, psm 3), and 4 MP3s and 2 MP4s via faster-whisper `tiny.en` with segment timestamps.
- **Chunks.** The units become 156 chunks (180 words, 40 overlap) that never cross a page or clip. Each chunk carries the source, page or timestamp, modality and extraction method.
- **Retrieval.** MiniLM-L6-v2 embeddings feed a FAISS `IndexFlatIP`, fused with BM25 (min-max normalised, alpha 0.8). A `ms-marco-MiniLM-L-6` cross-encoder reranks a 20-candidate pool, and a cosine floor triggers abstention.
- **Generation.** Qwen2.5-Instruct generates the answer locally. The recorded metrics used 0.5B; the demo uses 1.5B in fp16 on a T4. Prompt rules ban figures not in the evidence and treat retrieved text as untrusted, and citation labels are validated in code.
- **Evaluation.** The test set has 38 questions. 12 of them can only be answered from OCR, audio or video, and a leak check proves those 12 cannot be answered from text. Metrics are Hit, P@k, R@k and MRR, plus `must_contain`, `must_not_contain`, citation validity and expected-behaviour scoring (answer, refuse, caveat or clarify).

## Reproduce the build / publish the demo bundle
This is done once, by the owner only, because it needs the raw knowledge base.
1. Add a Colab secret `HF_TOKEN` with **write** access.
2. In the settings cell, set `BUILD = True`. Point `PROJECT_ROOT` at the folder that holds `knowledge_base/` and `eval/`; with `USE_DRIVE = True` it can sit in your Google Drive.
3. Choose **Runtime → Run all**. This takes about 15–30 minutes and covers OCR, ASR, indexing, the evaluation and all experiments.
4. The section 14 push cell creates the **private** dataset `tantg101/smf-rag-demo-assets` and uploads:
   - `artifacts/chunks.json`, `chunk_embeddings.npy`, `faiss_index.bin`, `config.json`
   - `results/experiment_comparison.csv`, `experiment_comparison.png`, `chunk_size_sweep.csv`
   - `eval/test_questions.csv`

## Data
The knowledge base is team-authored advisory notes on Singapore enterprise support schemes, plus **synthetic** SMF-style material: illustrative figures in the scanned PDFs, synthetic-voice audio clips, and case-study videos about fictional companies.
It was prepared for an industrial-attachment case study and is **not published** in this repository. The demo reads a prebuilt bundle from a private Hugging Face dataset.
The public models are all-MiniLM-L6-v2, ms-marco-MiniLM-L-6-v2, Qwen2.5-Instruct and faster-whisper `tiny.en`.

## Limitations
- **Abstention was the weakest result.** The baseline abstained correctly 0% of the time. The best abstention setting reached only 33.3%, and it cost hit rate (0.966 → 0.828).
- **The improvements did not stack.** Rerank alone gave MRR 0.917, but the combined configuration gave 0.802.
- **The recorded answers were weak.** They came from Qwen2.5-0.5B, which produced no citations on the worked examples and invented a membership fee on an unanswerable question. The demo uses 1.5B and the best retrieval config, but answer quality was not re-evaluated with it.
- **Relevance is approximate.** It is judged by phrase matching, not human judgement.
- **OCR loses chart labels** (question F01), and `tiny.en` ASR on synthetic speech is an easy test.
- **156 chunks is not a scale test.**
