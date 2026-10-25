# Applied AI projects: SIT Graduate Certificate in AI Practice (2026)

Eight weeks of full-time applied AI training at the Singapore Institute of Technology with the SIT × NVIDIA AI Centre, with a hands-on project every week. Every project runs in Google Colab, and most open a **live Gradio demo**: click the Colab badge in the project's README, then **Runtime → Run all**.

| Week | Project | Area | Headline result | Live demo |
|---|---|---|---|---|
| 1 | [HDB resale price prediction](week1-hdb-resale-price-prediction/) | ML foundations | R² 0.903 on a 2025–26 hold-out | ✅ CPU, ~4 min |
| 2 | [Real-time taxi availability forecasting (Singapore)](week2-realtime-taxi-forecasting/) | Data engineering & MLOps | R² ≈ 0.75, MAE ≈ 20 taxis | — |
| 3 | [Vape image moderation for marketplace listings](week3-vape-image-moderation/) | Deep learning, transfer learning | ResNet-18 vs ViT-B/16, recall-targeted threshold | ✅ T4, ~4 min |
| 4 | [SmartCart: photo of groceries → priced receipt](week4-smartcart-vision/) | Computer vision | YOLO11n mAP50-95 0.924 · ONNX int8 3.9× smaller | ✅ T4, ~4 min |
| 5 | [ElderShield closed-book LLM (QLoRA SFT + DPO)](week5-llm-finetune-abstention/) | LLM fine-tuning | Fact recall 37% → 63%, warm tone 23% → 100% | ✅ T4, ~6 min |
| 6 | [Red-teaming and hardening a RAG chatbot](week6-rag-red-teaming/) | AI security & governance | Attack success 83% → 0% | ✅ T4, ~7 min |
| 7 | [Four-modality RAG assistant (PDF, scanned PDF, audio, video)](week7-multimodal-rag/) | RAG & agentic AI | MRR 0.798 → 0.917 | ✅ T4, ~5 min* |
| 8 | [SENTRY-10: acoustic safety monitor for factory zones](week8-acoustic-safety-monitor/) | Speech & audio AI | Held-out accuracy 0.981 | ✅ T4, ~4 min |

\* Week 7's knowledge base belongs to an industry-partner case study and is kept private, so its demo runs only from the author's Colab.

Trained models are hosted on Hugging Face ([huggingface.co/tantg88888](https://huggingface.co/tantg88888)), so the demos load them in seconds instead of retraining. Set `TRAIN = True` in a notebook to reproduce the full training run.

**NVIDIA Deep Learning Institute certificates:** Building AI Agents with Multimodal Models · Adding New Knowledge to LLMs · Generative AI with Diffusion Models

> Most projects were team projects; each README says so and describes my part. Data that belongs to industry partners is not published; only public datasets or synthetic examples are included.
