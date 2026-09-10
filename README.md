# Mistral 7B Finance Fine-Tuning using QLoRA

Fine-tuned Mistral 7B Instruct on a financial Q&A dataset using QLoRA (Quantized Low-Rank Adaptation) to improve the model's responses on finance-related queries.

---

## Problem Statement

General purpose LLMs like Mistral 7B have broad knowledge but lack depth in specific domains like finance. Fine-tuning on domain-specific data improves response quality, factual accuracy, and terminology usage for financial queries without retraining the entire model.

---

## Approach

Instead of full fine-tuning which requires 40-80GB VRAM, this project uses QLoRA:
- The base model is loaded in **4-bit quantization** (bitsandbytes) reducing memory from ~14GB to ~5GB
- **LoRA adapters** are injected into attention and MLP layers - only 0.57% of parameters are trained (41M out of 7.2B)
- **Unsloth** is used for 2x faster training on free Colab T4 GPU

This makes fine-tuning a 7B model possible on a free Google Colab T4 (15GB VRAM).

---

## Tech Stack

- Model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- Dataset: `gbharti/finance-alpaca` (68,912 finance Q&A pairs, used 10,000)
- Libraries: Unsloth, HuggingFace TRL, PEFT, bitsandbytes, datasets
- Hardware: Google Colab Free Tier - Tesla T4 (15GB VRAM)
- Training time: ~16 minutes for 100 steps

---

## LoRA Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| LoRA rank (r) | 16 | Balance between capacity and memory |
| LoRA alpha | 16 | Scaling factor equal to rank |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | All attention + MLP layers |
| Dropout | 0 | Unsloth recommendation for speed |
| Trainable params | 41,943,040 (0.57%) | Only adapters trained, base frozen |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 2 |
| Gradient accumulation | 4 (effective batch size = 8) |
| Max steps | 100 |
| Learning rate | 2e-4 |
| Optimizer | AdamW 8-bit |
| LR Scheduler | Linear |
| Precision | float16 |

---

## Training Loss

| Step | Loss |
|------|------|
| 10 | 2.8286 |
| 20 | 2.3895 |
| 30 | 2.3814 |
| 40 | 2.2795 |
| 50 | 2.2917 |
| 60 | 2.3193 |
| 70 | 2.2375 |
| 80 | 2.2571 |
| 90 | 2.2200 |
| 100 | 2.2279 |

Loss decreased from 2.83 to 2.23 over 100 steps confirming the model is learning the finance domain.

---

## Sample Outputs

**Q: What is the difference between a mutual fund and an ETF?**

A: The main difference is that ETFs are traded on an exchange, while mutual funds are not. This means you can buy or sell an ETF at any time during the trading day, while mutual funds only trade at end of day. ETFs also have a bid/ask spread, are generally more tax-efficient, and tend to be cheaper than mutual funds.

---

**Q: What is a bull market?**

A: A bull market is a market where the prices of securities are rising. The term comes from the fact that a bull attacks by thrusting its horns upward. The opposite is a bear market where prices are falling.

---

**Q: What is the risk of investing in bonds?**

A: The primary risk is that the issuer defaults on the bond - failing to pay interest or principal. This default risk is measured by credit ratings from agencies like Moody's, S&P, and Fitch using scales from AAA (lowest risk) to D (default).

---

## Limitations

- Only 100 training steps (0.08 epochs) - full fine-tuning would require 1-3 complete epochs
- No held-out test set evaluation - loss alone does not measure response quality
- No baseline comparison against the unmodified Mistral 7B on same questions
- Small subset used (10,000 of 68,912 available examples)
- No finance-specific benchmark evaluation (e.g. FinQA, FiNER)

---

## Future Improvements

- Train for full 1-3 epochs on complete dataset
- Evaluate using finance-specific benchmarks
- Compare base vs fine-tuned model responses systematically
- Experiment with different LoRA ranks (8, 32, 64) to find optimal trade-off
- Try larger models like Llama 3.1 8B with same approach
- Push final model to HuggingFace Hub for public use

---

## How to Run

Open `fine_tuning_model.ipynb` in Google Colab with T4 GPU runtime and run cells sequentially.
