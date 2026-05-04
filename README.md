# 📚 Context-Aware RAG System

> **Multi-PDF 기반 Hybrid Retrieval + Reranker + Query Routing + FastAPI Backend를 적용한 End-to-End RAG 시스템**

---

## 🚀 Overview

이 프로젝트는 여러 도메인의 PDF 문서를 기반으로 질문에 답변하는
**Context-aware Retrieval-Augmented Generation (RAG) 시스템**입니다.

단순한 LLM 응답을 넘어서:

* 검색 품질 개선 (Hybrid + Reranker)
* 정량 평가 (Hit@K, MRR)
* 도메인 라우팅 (Query Routing)
* 서비스 구조 (FastAPI + Streamlit 분리)

까지 포함한 **실서비스 수준의 AI 시스템**을 구현했습니다.

---

## ✨ Key Features

### 📂 Multi-PDF Ingestion

* 여러 PDF 업로드 지원
* pdfplumber 기반 텍스트 추출
* chunk 단위 분할 + source metadata

---

### 🔍 Hybrid Retrieval ⭐

* FAISS (Dense) + BM25 (Sparse)
* score normalization 기반 결합

```text
final_score = α * dense + (1 - α) * sparse
```

---

### 🧠 Reranker (Cross-Encoder)

* query-document relevance 직접 평가
* retrieval 결과 재정렬
* ranking quality 개선

---

### 🧭 Query Routing ⭐

* query 기반 도메인 자동 추정
* 해당 domain 문서만 검색

```text
Query → Domain Detection → Filtered Retrieval
```

---

### 📊 Evaluation Pipeline

* Hit@1 / Hit@3 / Hit@5
* MRR
* retrieval 성능 정량 분석

---

### 🤖 LLM Answer Generation

* OpenAI API 기반
* retrieved context 기반 답변 생성
* hallucination 최소화

---

### 🖥️ Frontend (Streamlit)

* PDF 업로드
* 검색 방식 선택 (Hybrid / Reranker)
* Query Routing ON/OFF
* 답변 + 근거 + score 시각화

---

### ⚙️ Backend (FastAPI) ⭐

* REST API 기반 RAG 서비스
* `/rag/query` 엔드포인트
* multipart PDF 업로드 지원
* UI와 완전 분리된 구조

---

## 🏗️ Architecture

```text
User (Streamlit UI)
   ↓
FastAPI Backend
   ↓
Query Routing
   ↓
Domain Filtered Retrieval
   ↓
FAISS + BM25
   ↓
Hybrid Retrieval
   ↓
Reranker
   ↓
LLM
   ↓
Final Answer + Context
```

---

## 📈 Evaluation Results

| Method   | Hit@1    | Hit@3    | Hit@5    | MRR      |
| -------- | -------- | -------- | -------- | -------- |
| FAISS    | 0.43     | 1.00     | 1.00     | 0.67     |
| BM25     | 0.50     | 0.64     | 0.79     | 0.59     |
| Hybrid   | 0.64     | 0.93     | 1.00     | 0.77     |
| Reranker | **0.79** | **1.00** | **1.00** | **0.88** |

> Reranker가 Hit@1 기준 가장 큰 성능 개선을 보였습니다.

---

## 🛠️ Tech Stack

* Python
* pdfplumber
* SentenceTransformers
* FAISS
* rank-bm25
* Cross-Encoder
* OpenAI API
* FastAPI
* Streamlit

---

## ⚙️ Installation

```bash
git clone https://github.com/SJA-03/Context-Aware-RAG-System.git
cd Context-Aware-RAG-System

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## 🔑 Environment Setup

```bash
export OPENAI_API_KEY="your_api_key"
```

또는 `.env` 사용:

```text
OPENAI_API_KEY=your_api_key
```

---

## ▶️ Run

### 1️⃣ Backend (FastAPI)

```bash
uvicorn api.main:app --reload
```

헬스 체크:

```bash
curl http://127.0.0.1:8000/health
```

---

### 2️⃣ Frontend (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

---

### 3️⃣ API Docs

```text
http://127.0.0.1:8000/docs
```

---

## 📌 Future Work

* Embedding-based Query Routing
* Reranker fine-tuning
* Multi-domain dataset 확장
* Caching / latency optimization
* Cloud deployment

---

## 🧠 Key Insight

* Dense retrieval은 의미 이해에 강하지만 ranking이 불안정
* Sparse retrieval은 정확하지만 일반화가 약함
* Hybrid retrieval이 안정적인 성능 제공
* Reranker가 ranking 품질을 크게 개선
* Query Routing으로 검색 공간을 줄여 효율 향상

---

## 👨‍💻 Author

* AI Engineer Portfolio Project
* End-to-End RAG System Design & Implementation

---