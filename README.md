
# 🧠 Text Summarization using Transformer Models

## 📘 Project Title & Description
**Text Summarization using Transformer Models (T5 and BART)**  
This project focuses on generating concise and meaningful summaries from long news articles. Using state-of-the-art transformer-based models such as **T5-small** and **BART**, it aims to produce human-like summaries from the **CNN/DailyMail dataset**.

---

## 🚩 Problem Statement
In today’s information-driven world, people often face difficulty reading lengthy news articles or documents. The goal of this project is to automatically summarize large pieces of text into shorter, informative versions without losing key information or context.

---

## 🧾 Dataset Source
We use the **CNN/DailyMail dataset**, a well-known benchmark for text summarization tasks.  
- **Source:** [Hugging Face Dataset - CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail)
- **Dataset Type:** News articles and human-written summaries  
- **Size:** ~300k samples  

---

## 🧠 Model Details
We experimented with and fine-tuned the following pre-trained transformer models:
1. **T5-small** — Text-to-text transformer model fine-tuned for summarization.
2. **BART-base** — Sequence-to-sequence transformer model pre-trained with denoising objectives.

**Frameworks Used:**
- PyTorch  
- Hugging Face Transformers  
- Datasets Library  

**Evaluation Metrics:**
- ROUGE-1  
- ROUGE-2  
- ROUGE-L  

---

## 🚀 Deployment
**Deployment Link:** [🔗 View Demo](#) *(Add your Hugging Face Space / Streamlit / Gradio link here once deployed)*

Example:  
`https://huggingface.co/spaces/ahmedhassan/text-summarizer`  

---

## ✍️ Medium Article
For a detailed walkthrough of this project:  
**Medium Link:** [🔗 Read Article](#) *[(Add your Medium article link here)](https://medium.com/@techwithahmedhassan/text-summarization-using-transformer-models-t5-pegasus-d6ff1124887b)*

---

## ⚙️ Instructions to Run Locally

### 1. Clone this repository
```bash
git clone https://github.com/ahmedhassan/text-summarization.git
cd text-summarization

2. Create a virtual environment
python -m venv env
source env/bin/activate   # (Linux/Mac)
env\Scripts\activate      # (Windows)

3. Install dependencies
pip install -r requirements.txt

4. Run the notebook or script
python train_t5.py
# or
python app.py

5. (Optional) Run the Gradio/Streamlit app
streamlit run app.py

🧩 License

This project is licensed under the MIT License – feel free to use, modify, and share with attribution.

📦 Other Required Stuff

Python Version: 3.10 or above

GPU Recommended: Yes (for fine-tuning transformers)

Libraries:

transformers

datasets

torch

rouge-score


Author

Ahmed Hassan
Data Science & AI Engineer
Portfolio
 | LinkedIn
 | GitHub
=======
# Text-Summarizer
>>>>>>> 7052632fcba52e2b0af48bb7777abee212159e32
