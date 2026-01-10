# 📚 Narrative Consistency Verification System  
## Kharagpur Data Science Hackathon 2026 — Track A

This project implements a **Pathway-based Retrieval-Augmented Generation (RAG) system** to verify whether a character’s backstory or claim is **consistent or contradictory** with the original book text, as required in **Track A** of the Kharagpur Data Science Hackathon 2026.

The project is designed to be submitted as a **ZIP file (offline submission)** and does **not require GitHub**.

---

## 🧠 System Architecture

Book Text (.txt files)  
→ `app_server.py` (Pathway Vector Store + Embeddings)  
→ HTTP API (Port 8000)  
→ `app_client.py` (Multi-Agent Reasoning Pipeline)  
→ `results.csv` (Final Output)

---

## 📂 Project Structure

.
├── data/  
│   ├── In search of the castways.txt  
│   ├── The Count of Monte Cristo.txt  
│    
│  
├── test.csv  
├── results.csv  
├── app_server.py  
├── app_client.py  
├── requirements.txt  
├── Dockerfile  
├── .env  
└── README.md  

---

## ⚙️ Requirements

- Python 3.12.1
- LLM Used (Groq API)
- Optional: Docker
- Using Pathway's Local Vectorstore

---

## 📦 Installation (ZIP-based)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🔐 Environment Setup

Create `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

DO NOT include API keys in the ZIP.

---

## 🚀 Running the Project

### Step 1: Start Server
```bash
python app_server.py
```

WAIT **7–8 minutes** for chunking & embedding. You will see an non loading message in terminal

Wait for:
```
🚀 Augmented Vector Store running on 0.0.0.0:8000
```

### Step 2: Run Client
```bash
python app_client.py
```

---

## 📄 Output

Generated file:
```
results.csv
```

---

## 🧠 LLM Models Used

- Constraint Extraction: qwen2.5-7b-instruct
- Query Generation: llama-3.1-8b-instant
- Reasoning & Decision: llama-3.3-70b-versatile

---

## ⚠️ Notes

- Always start server before client
- Wait for indexing to complete
- Handle Groq rate limits carefully

---

## 🏁 Final Execution Order

```bash
python app_server.py
(wait 7–8 minutes)
python app_client.py
```

- This is the complete running mechanism of the code for this hackthon.
- If you have any doubt about code files , please visit this repo --> https://github.com/Arjun-E-Naik/KSDH26

 **Kharagpur Data Science Hackathon 2026 — Track A**.
