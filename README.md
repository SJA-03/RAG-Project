# 📚 Context-Aware RAG System

> **Multi-PDF 기반 Hybrid Retrieval + Reranker + Query Routing을 적용한 End-to-End RAG 시스템**

<img width="1418" height="734" alt="스크린샷 2026-04-29 오후 6 01 18" src="https://github.com/user-attachments/assets/18c51667-238a-445c-8181-f7abc62d502c" />

---

## 🚀 Overview

이 프로젝트는 여러 도메인의 학습 자료(PDF)를 기반으로 질문에 답변하는 **Context-aware Retrieval-Augmented Generation (RAG) 시스템**입니다.

단순한 LLM 응답을 넘어서,
**검색 품질 개선 → 정량 평가 → 도메인 라우팅 → 답변 생성**까지 포함한 전체 파이프라인을 설계 및 구현했습니다.

---

## ✨ Key Features

### 📂 1. Multi-PDF Ingestion

* 여러 PDF 업로드 지원 (Streamlit UI)
* pdfplumber 기반 텍스트 추출
* chunk metadata에 source 정보 포함
* 다양한 도메인(OS, ML 등) 확장 가능

---

### 🔍 2. Advanced Retrieval Pipeline

#### 🔹 Dense Retrieval (FAISS)

* SentenceTransformer 기반 embedding
* semantic similarity 검색

#### 🔹 Sparse Retrieval (BM25)

* keyword 기반 검색
* exact match 보완

#### 🔹 Hybrid Retrieval ⭐

```text
final_score = α * dense + (1 - α) * sparse
```

* Dense + Sparse 결합
* score normalization 적용
* 안정적인 retrieval 성능 확보

---

### 🧠 3. Reranker (Cross-Encoder)

* `cross-encoder/ms-marco-MiniLM-L-6-v2`
* query-document relevance 직접 학습된 모델
* retrieval 결과 재정렬

---

### 🧭 4. Query Routing ⭐

* rule-based lightweight router
* query 기반 도메인 자동 추정 (OS, ML 등)
* 해당 도메인 chunk만 우선 검색

```text
Query → Domain Detection → Filtered Retrieval
```

---

### 📊 5. Evaluation Pipeline

* Hit@1 / Hit@3 / Hit@5
* MRR (Mean Reciprocal Rank)
* Retrieval 성능 정량 분석

---

### 🤖 6. LLM Answer Generation

* OpenAI API 기반 답변 생성
* retrieved context 기반 응답
* hallucination 방지 프롬프트 적용

---

### 🖥️ 7. Streamlit UI

* PDF 업로드 기반 인터랙티브 UI
* 검색 방식 선택 (Hybrid / Reranker)
* Query Routing 활성화 옵션
* 답변 + 근거(context + source) + score 시각화

---

## 🏗️ Architecture

```text
User Query
   ↓
Query Routing
   ↓
Domain Filtered Retrieval
   ↓
FAISS (Dense) + BM25 (Sparse)
   ↓
Hybrid Retrieval
   ↓
Reranker
   ↓
LLM (Answer Generation)
   ↓
Final Answer + Context + Source
```

---

## 📈 Evaluation Results

| Method   | Hit@1    | Hit@3    | Hit@5    | MRR      |
| -------- | -------- | -------- | -------- | -------- |
| FAISS    | 0.43     | 1.00     | 1.00     | 0.67     |
| BM25     | 0.50     | 0.64     | 0.79     | 0.59     |
| Hybrid   | 0.64     | 0.93     | 1.00     | 0.77     |
| Reranker | **0.79** | **1.00** | **1.00** | **0.88** |

> Hybrid retrieval을 통해 안정적인 성능 개선을 달성했고,
> Cross-Encoder reranker가 Hit@1 기준 가장 높은 성능을 보였습니다.

---

## 🧠 Key Insights

* Dense retrieval만으로는 ranking 정확도가 부족함
* Sparse retrieval은 특정 query에 강하지만 일반화 성능이 제한적임
* Hybrid retrieval이 안정적인 성능 개선을 제공
* Reranker는 ranking quality를 크게 향상
* Query Routing을 통해 검색 공간을 줄이고 relevance 향상 가능

---

## 🛠️ Tech Stack

* Python
* pdfplumber
* SentenceTransformers
* FAISS
* rank-bm25
* Cross-Encoder
* OpenAI API
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

* Embedding-based Query Routing (semantic routing)
* Reranker fine-tuning
* Multi-domain dataset 확장
* Latency optimization (caching)
* Cloud deployment (Streamlit / HF Spaces)

---

## 👨‍💻 Author

* AI Engineer Portfolio Project
* End-to-End RAG system design & implementation

---