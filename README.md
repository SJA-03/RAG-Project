# 📚 Context-Aware Academic Assistant (RAG System)

> **멀티 PDF 기반 Hybrid Retrieval + Reranker를 적용한 End-to-End RAG 시스템**

<img width="1418" height="734" alt="스크린샷 2026-04-29 오후 6 01 18" src="https://github.com/user-attachments/assets/18c51667-238a-445c-8181-f7abc62d502c" />


---

## 🚀 Overview

이 프로젝트는 여러 학습 자료(PDF)를 기반으로 질문에 답변하는 **Context-aware Retrieval-Augmented Generation (RAG) 시스템**입니다.

단순한 LLM 응답이 아닌,
**검색 → 재정렬 → 근거 기반 답변 생성**까지 포함한 전체 파이프라인을 직접 설계 및 구현했습니다.

---

## ✨ Key Features

### 🔍 1. Multi-PDF Ingestion

* 여러 PDF 업로드 지원 (Streamlit UI)
* pdfplumber 기반 텍스트 추출
* source metadata 포함 (출처 추적 가능)

---

### 🧩 2. Advanced Retrieval Pipeline

#### Dense Retrieval (FAISS)

* SentenceTransformer 기반 embedding
* semantic similarity 검색

#### Sparse Retrieval (BM25)

* keyword 기반 검색
* exact match 보완

#### Hybrid Retrieval ⭐

* Dense + Sparse 결합
* score normalization + weighted fusion

```text
final_score = α * dense + (1 - α) * sparse
```

---

### 🧠 3. Reranker (Cross-Encoder)

* `cross-encoder/ms-marco-MiniLM-L-6-v2`
* query-document relevance 직접 학습된 모델 사용
* 검색 결과 재정렬

---

### 📊 4. Evaluation Pipeline

* Hit@1 / Hit@3 / Hit@5
* MRR (Mean Reciprocal Rank)
* Retrieval 성능 정량 평가

---

### 🤖 5. LLM Answer Generation

* OpenAI API 기반 답변 생성
* Retrieved context 기반 답변
* hallucination 방지 프롬프트 적용

---

### 🖥️ 6. Streamlit UI

* PDF 업로드 기반 인터랙티브 UI
* 검색 방식 선택 (Hybrid / Reranker)
* 답변 + 근거(context) + score 시각화

---

## 🏗️ Architecture

```text
User Query
   ↓
PDF Upload / Data
   ↓
Text Extraction (pdfplumber)
   ↓
Chunking
   ↓
Embedding (SentenceTransformer)
   ↓
FAISS (Dense)
   + BM25 (Sparse)
   ↓
Hybrid Retrieval
   ↓
Reranker (Cross Encoder)
   ↓
LLM (Answer Generation)
   ↓
Final Answer + Context
```

---

## 📈 Evaluation Results

| Method   | Hit@1 | Hit@3 | Hit@5 | MRR  |
| -------- | ----- | ----- | ----- | ---- |
| FAISS    | 0.67  | 1.00  | 1.00  | 0.83 |
| BM25     | 1.00  | 1.00  | 1.00  | 1.00 |
| Hybrid   | 1.00  | 1.00  | 1.00  | 1.00 |
| Reranker | 0.67  | 1.00  | 1.00  | 0.83 |

> Hybrid retrieval을 통해 성능 개선을 확인했고,
> Reranker는 데이터셋에 따라 성능이 달라질 수 있음을 확인했습니다.

---

## 🛠️ Tech Stack

* **Python**
* **pdfplumber** (PDF parsing)
* **SentenceTransformers** (Embedding)
* **FAISS** (Vector Search)
* **rank-bm25** (Sparse Retrieval)
* **Cross-Encoder** (Reranker)
* **OpenAI API** (LLM)
* **Streamlit** (UI)

---

## ⚙️ Installation

```bash
git clone https://github.com/your-repo/RAG-Project.git
cd RAG-Project

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## 🔑 Environment Setup

```bash
export OPENAI_API_KEY="your_api_key"
```

---

## ▶️ Run

### CLI 실행

```bash
python src/main.py
```

---

### Streamlit UI 실행

```bash
streamlit run app/streamlit_app.py
```

---

## 📌 Future Work

* Multi-domain dataset 확장
* Query routing (domain-aware retrieval)
* Reranker fine-tuning
* Caching / latency optimization
* Production deployment (Cloud)

---

## 💡 Key Insight

> Dense retrieval만으로는 정확한 ranking이 어려울 수 있으며,
> Sparse retrieval과 결합한 Hybrid approach가 더 안정적인 성능을 보였습니다.

---

## 👨‍💻 Author

* AI Engineer Portfolio Project
* Designed & Implemented end-to-end RAG pipeline

---
