# 📚 Context-Aware Academic Assistant

## 🧠 Overview

This project is a **multi-domain Retrieval-Augmented Generation (RAG) system** designed to answer questions based on personal academic materials such as lecture PDFs and notes.

Unlike simple RAG implementations, this project focuses on **improving retrieval performance through systematic experimentation**, including chunking strategies, hybrid search, and reranking.

> 🎯 Goal: Build a system that not only answers questions, but also **retrieves the most relevant context reliably and explainably**.

---

## 🚀 Key Features

### 1. Multi-domain Document Understanding

* Supports multiple subjects (e.g., OS, Data Structures, ML, DL)
* Organized document ingestion pipeline
* Enables domain-aware retrieval

### 2. Hybrid Retrieval System

* Combines:

  * Dense retrieval (Embeddings + FAISS)
  * Sparse retrieval (BM25)
* Improves recall and relevance compared to single-method retrieval

### 3. Query Routing

* Automatically identifies the relevant subject/domain from user query
* Reduces search space and improves accuracy

### 4. Chunking Strategy Optimization

* Experiments with:

  * Fixed-size chunking
  * Semantic/paragraph-based chunking
* Evaluates impact on retrieval performance

### 5. Answer Generation with Citation

* LLM generates answers using retrieved context
* Provides **source grounding** for transparency

### 6. Evaluation Pipeline (Core Contribution)

* Custom QA dataset built from lecture materials
* Metrics:

  * Retrieval Hit Rate
  * Context Coverage
  * Answer Correctness

---

## 🏗️ System Architecture

```
[PDF / Notes]
      ↓
[Ingestion Pipeline]
      ↓
[Chunking]
      ↓
[Embedding]
      ↓
[Vector DB (FAISS)] + [BM25 Index]
      ↓
[Query Routing]
      ↓
[Hybrid Retrieval]
      ↓
[Reranker (Fine-tuned)]
      ↓
[LLM Answer + Citation]
```

---

## ⚙️ Tech Stack

| Component     | Technology                            |
| ------------- | ------------------------------------- |
| Language      | Python                                |
| Embedding     | BGE-small (Sentence Transformers)     |
| Vector DB     | FAISS                                 |
| Sparse Search | BM25 (rank-bm25)                      |
| LLM           | OpenAI API / Ollama (Llama3, Mistral) |
| Backend       | FastAPI                               |
| Frontend      | Streamlit                             |
| Evaluation    | Custom pipeline                       |

---

## 📊 Experiments & Improvements

### 1. Chunking Strategy Comparison

* Fixed-size vs semantic chunking
* Impact on retrieval accuracy

### 2. Retrieval Method Comparison

* Embedding-only
* BM25-only
* Hybrid (Dense + Sparse)

### 3. Top-K Sensitivity Analysis

* Evaluated performance across different k values

### 4. Query Routing Impact

* Domain filtering vs global search

---

## 🔬 Fine-tuning (Advanced)

To further improve retrieval quality:

### Reranker Model

* Input: (Query, Document Chunk)
* Output: Relevance Score

Used to:

* Re-rank retrieved candidates
* Improve final answer quality

---

## 📂 Project Structure

```
rag-project/
│
├── data/                # Raw PDFs
├── processed/           # Extracted text
├── embeddings/          # Vector data
│
├── src/
│   ├── ingestion/       # PDF parsing
│   ├── chunking/        # Chunk logic
│   ├── embedding/       # Embedding generation
│   ├── retrieval/       # BM25 + FAISS
│   ├── routing/         # Query classification
│   ├── reranker/        # Fine-tuned model
│   ├── llm/             # Answer generation
│   └── evaluation/      # Metrics & testing
│
├── api/                 # FastAPI server
├── app/                 # Streamlit UI
└── README.md
```

---

## 📈 Results (Example)

| Method           | Hit Rate | Answer Accuracy |
| ---------------- | -------- | --------------- |
| Embedding Only   | 0.62     | 0.58            |
| BM25 Only        | 0.55     | 0.52            |
| Hybrid Retrieval | 0.71     | 0.66            |
| + Reranker       | **0.78** | **0.72**        |

---

## 🎯 Key Learnings

* Retrieval quality is the **core bottleneck** in RAG systems
* Chunking strategy significantly affects performance
* Hybrid retrieval consistently outperforms single methods
* Evaluation is critical — without it, improvements are unclear

---

## 🔮 Future Work

* Personalized retrieval based on user learning history
* Better semantic chunking using structure-aware parsing
* Multi-turn conversational memory
* Lightweight fine-tuned LLM for domain-specific answering

---

## 💡 Why This Project Matters

This project goes beyond building a simple chatbot.
It demonstrates:

* Understanding of **retrieval systems**
* Ability to **design experiments and evaluate performance**
* Experience with **real-world AI system pipelines**

> This reflects the practical skills required for an **AI Engineer**, not just model usage.

---

# 📚 Context-Aware Academic Assistant(Korean ver)

### (멀티 도메인 RAG 기반 학습 어시스턴트)

---

## 🧠 프로젝트 개요 (Overview)

본 프로젝트는 강의 자료(PDF), 개인 노트 등을 기반으로 질문에 답변하는
**멀티 도메인 Retrieval-Augmented Generation(RAG) 시스템**입니다.

단순한 RAG 구현을 넘어,
**검색(Retrieval) 성능 개선을 중심으로 실험 및 최적화**를 수행한 프로젝트입니다.

> 🎯 목표:
> 단순히 답변을 생성하는 시스템이 아니라,
> **정확한 문맥을 검색하고 근거 기반으로 답변하는 시스템**을 구축하는 것

---

## 🚀 핵심 기능 (Key Features)

### 1️⃣ 멀티 도메인 문서 이해

* 운영체제, 자료구조, 기계학습 등 다양한 과목 지원
* 과목별 문서 구조화
* 도메인 기반 검색 가능

---

### 2️⃣ Hybrid Retrieval (핵심 기능)

* Dense Retrieval (Embedding + FAISS)
* Sparse Retrieval (BM25)

👉 두 방식을 결합하여 검색 성능 향상

---

### 3️⃣ Query Routing

* 질문을 분석하여 관련 과목(도메인) 자동 분류
* 불필요한 검색 범위 축소 → 정확도 향상

---

### 4️⃣ Chunking 전략 실험

* Fixed-size chunking
* 문단 기반 chunking

👉 chunking 방식에 따른 성능 비교 및 최적화

---

### 5️⃣ 근거 기반 답변 생성

* LLM이 검색된 문서를 기반으로 답변 생성
* 출처(citation) 제공

---

### 6️⃣ 평가 시스템 (핵심 차별화)

* 직접 구축한 QA 데이터셋 기반 평가
* 성능을 정량적으로 측정

---

## 🏗️ 시스템 아키텍처

```id="lqtsbm"
[강의 PDF / 노트]
        ↓
[텍스트 추출 및 전처리]
        ↓
[Chunking]
        ↓
[Embedding 생성]
        ↓
[Vector DB (FAISS)] + [BM25 Index]
        ↓
[Query Routing]
        ↓
[Hybrid Retrieval]
        ↓
[Reranker (Fine-tuning)]
        ↓
[LLM 답변 생성 + 출처 제공]
```

---

## ⚙️ 기술 스택 (Tech Stack)

| 구성 요소     | 기술                  |
| --------- | ------------------- |
| 언어        | Python              |
| Embedding | BGE-small           |
| Vector DB | FAISS               |
| 검색        | BM25                |
| LLM       | OpenAI API / Ollama |
| Backend   | FastAPI             |
| Frontend  | Streamlit           |
| 평가        | 자체 평가 파이프라인         |

---

## 📊 실험 및 성능 개선 (Experiments)

### ✔️ 1. Chunking 전략 비교

* Fixed-size vs 문단 기반
* Retrieval 정확도 영향 분석

---

### ✔️ 2. Retrieval 방식 비교

* Embedding 기반
* BM25 기반
* Hybrid 방식

👉 Hybrid 방식이 가장 높은 성능

---

### ✔️ 3. Top-K 실험

* k 값에 따른 성능 변화 분석

---

### ✔️ 4. Query Routing 효과 분석

* 전체 검색 vs 도메인 제한 검색 비교

---

## 🔬 Fine-tuning (고도화)

### ✔️ Reranker 모델

* 입력: (질문, 문서 chunk)
* 출력: relevance score

👉 검색 결과 재정렬 → 최종 성능 향상

---

## 📂 프로젝트 구조

```id="c1rzdg"
rag-project/
│
├── data/                # 원본 PDF
├── processed/           # 텍스트 변환
├── embeddings/          # 벡터 데이터
│
├── src/
│   ├── ingestion/       # PDF 처리
│   ├── chunking/        # chunk 생성
│   ├── embedding/       # 임베딩 생성
│   ├── retrieval/       # 검색 로직
│   ├── routing/         # 도메인 분류
│   ├── reranker/        # 파인튜닝 모델
│   ├── llm/             # 답변 생성
│   └── evaluation/      # 평가
│
├── api/                 # FastAPI 서버
├── app/                 # Streamlit UI
└── README.md
```

---

## 📈 성능 결과 (예시)

| 방법             | Hit Rate | 정답 정확도   |
| -------------- | -------- | -------- |
| Embedding only | 0.62     | 0.58     |
| BM25 only      | 0.55     | 0.52     |
| Hybrid         | 0.71     | 0.66     |
| + Reranker     | **0.78** | **0.72** |

---

## 🎯 핵심 학습 내용 (Key Learnings)

* RAG 시스템에서 가장 중요한 요소는 **Retrieval 성능**
* Chunking 방식이 결과에 큰 영향을 미침
* Hybrid Retrieval이 단일 방식보다 우수
* 평가 시스템 없이는 개선 여부 판단 불가

---

## 🔮 향후 개선 방향 (Future Work)

* 사용자 기반 개인화 검색
* 구조 기반 semantic chunking
* 멀티턴 대화 기능
* 경량화된 도메인 특화 모델

---

## 💡 프로젝트 의의 (Why This Project Matters)

본 프로젝트는 단순한 챗봇 구현이 아니라,
다음과 같은 AI 엔지니어 역량을 보여주는 데 목적이 있습니다:

* 검색 시스템(Retrieval) 이해
* 실험 설계 및 성능 개선 능력
* 실제 AI 서비스 구조 설계 경험

> **“단순히 모델을 사용하는 것이 아니라, 성능을 개선할 수 있는 AI 엔지니어”를 목표로 한 프로젝트**

---