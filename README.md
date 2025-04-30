# 🧠 Clinical-Grade Large Language Model for Diagnostic Decision Support

## Overview

This project develops a **domain-specific, clinically aligned Large Language Model (LLM)** optimized for factual correctness, safety, and clinical utility. Using **Reinforcement Learning with AI Feedback (RLAIF)** and gold-standard datasets such as **PubMedQA**, we fine-tune a biomedical LLM to support diagnostic reasoning and reduce preventable diagnostic errors in real-world healthcare workflows.

> 📍 **Impact**: In the Chicago metropolitan area alone, which accounts for ~3.5% of U.S. healthcare spending, the deployment of this model could reduce over **$700M/year in avoidable diagnostic costs**, while enhancing patient safety and outcomes.

---

## 🧬 Motivation

Diagnostic errors cost the U.S. healthcare system an estimated **$20 billion annually**. More than **30% of LLM-generated medical responses** still contain factual inaccuracies, largely due to:

- Fragmented medical knowledge bases  
- Lack of real-time clinical context  
- Poor alignment with clinician needs

This project addresses these challenges by:

- Aligning model reasoning with clinical expectations through RLAIF  
- Training on validated medical QA datasets  
- Benchmarking against leading clinical LLMs (e.g., Med-PaLM 2, Nuance DAX)

---

## 📊 Project Goals

- ✅ **Develop** a factual, safe, and aligned medical LLM  
- ✅ **Optimize** diagnostic QA performance on datasets like PubMedQA  
- ✅ **Evaluate** against USMLE-style benchmarks  
- ✅ **Quantify** real-world financial and clinical impact  
- 🚧 **Prepare** the model for integration into EHRs and telehealth systems  

---

## 🧪 Datasets

| Dataset           | Type         | Purpose                            |
|------------------|--------------|------------------------------------|
| **PubMedQA**      | QA Dataset   | Factual QA, biomedical grounding   |
| **MedQA-USMLE**   | QA Dataset   | Clinical exam-style reasoning      |
| **HealthSearchQA**| IR + QA      | Long-form inference modeling       |
| MIMIC-III (optional) | EHR       | EHR grounding for future RAG       |

---

## 🧠 Training Methodology

### 1. Supervised Fine-Tuning (SFT)
- Base model: `LLaMA-3-OpenBioLLM`
- Tokenizer + causal LM head
- Optimizer: AdamW with linear learning rate decay
- Mixed precision (fp16) for training efficiency

### 2. RLAIF: Reinforcement Learning with AI Feedback
- Custom reward model scoring:
  - Factual correctness (vs PubMedQA answer)
  - Clinical reasoning transparency
  - Medical safety constraints
- PPO (Proximal Policy Optimization) algorithm for reward optimization
- Reward shaping: KL penalty, reward clipping

### 3. Evaluation
- `BERTScore` vs ground truth
- `Accuracy`, `Exact Match` on PubMedQA and USMLE
- Optional: Human preference testing via simulated clinicians

---

## 📦 Tech Stack

- **Transformers**: Hugging Face (`transformers`, `trl`)
- **RLHF/RLAIF**: PPO, custom reward models
- **LLMs**: BioGPT, LLaMA-3 (via HF or vLLM)
- **Quantization**: BitsAndBytes 4-bit (NF4)
- **Infrastructure**: AWS EC2 A100, CUDA
- **Tracking**: Weights & Biases

---

## 🧭 Future Work

- 🔄 Integrate **retrieval-augmented generation (RAG)** for EHR-based grounding  
- 🏥 Optimize for **hospital deployment** via model distillation  
- 🧪 Launch **clinician-in-the-loop evaluation trials**  
- 📝 Submit to **ACL Clinical NLP / JAMIA** for publication  


---

## 📬 Contact

📧 easonwang@uchicago.edu 
