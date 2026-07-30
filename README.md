# Nikky Mperito_chatbot
All  three code  file 

requirement files 

All text file (raw data ) of Mperito 


# 🤖 Mperito AI Chatbot API

A Retrieval-Augmented Generation (RAG) chatbot built using:

- FastAPI
- LangChain
- FAISS Vector Database
- HuggingFace Embeddings
- Groq LLM (Llama 3.3 70B)
- Python

The chatbot retrieves relevant information from a local FAISS vector database and answers user queries using a Large Language Model.

---

# Project Architecture

```
User
   │
   ▼
FastAPI
   │
   ▼
Pydantic Validation
   │
   ▼
Retrieve Relevant Chunks
(FAISS)
   │
   ▼
Prompt Creation
   │
   ▼
Groq LLM
   │
   ▼
JSON Response
```

---

# Features

- FastAPI REST API
- RAG (Retrieval Augmented Generation)
- HuggingFace Embeddings
- FAISS Vector Search
- Groq LLM Integration
- Automatic API Documentation
- JSON Response
- Easy Deployment

---

# Project Structure

```
project/

│
├── question_answer_fastapi.py
├── .env
├── requirements.txt
├── vectormoretxtfastapi/
│
├── README.md
│
└── data/
```

---

# Prerequisites

Install

- Python 3.10+
- Git
- VS Code (Recommended)

---

# Step 1 : Clone Project

```bash
git clone <repository-url>
```

Move inside project

```bash
cd chatbotdemo
```

---

# Step 2 : Create Virtual Environment

Windows

```bash
python -m venv .venv
```

Activate

```bash
.venv\Scripts\activate
```

---

# Step 3 : Install Requirements

```bash
pip install -r requirements.txt
```

If requirements file is not available

```bash
pip install fastapi
pip install uvicorn
pip install langchain
pip install langchain-community
pip install langchain-groq
pip install sentence-transformers
pip install faiss-cpu
pip install python-dotenv
```

---

# Step 4 : Create .env File

```
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

---

# Step 5 : Load Vector Database

Project expects

```
vectormoretxtfastapi/
```

inside the project directory.

Example

```
project

│
├── vectormoretxtfastapi
│      index.faiss
│      index.pkl
```

---

# Step 6 : Run FastAPI

```bash
uvicorn question_answer_fastapi:app --reload
```

Server starts

```
http://127.0.0.1:8000
```

---

# Step 7 : Swagger Documentation

Open

```
http://127.0.0.1:8000/docs
```

Interactive API documentation.

---

# Available APIs

## Home API

### GET /

Returns

```json
{
    "message":"Welcome to Mperito Chatbot API",
    "hello":"hello how can i help you"
}
```

---

## Chat API

### POST /chat

Request

```json
{
    "user_query":"What is Mperito?"
}
```

Response

```json
{
    "question":"What is Mperito?",
    "answer":"..."
}
```

---

## Sum API

### POST /sum

Request

```json
{
    "num1":10,
    "num2":20
}
```

Response

```json
{
    "num1":10,
    "num2":20,
    "sum":30
}
```

---

# API Flow

```
Client

↓

POST /chat

↓

FastAPI

↓

Pydantic

↓

Similarity Search

↓

FAISS

↓

Top 3 Chunks

↓

Prompt

↓

Groq LLM

↓

Answer

↓

JSON Response
```

---

# Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| FastAPI | REST API Framework |
| Uvicorn | ASGI Server |
| Pydantic | Request Validation |
| LangChain | RAG Framework |
| FAISS | Vector Database |
| HuggingFace | Embedding Model |
| Groq | Large Language Model |

---

# Request Validation

Request Model

```python
class UserQuery(BaseModel):
    user_query:str
```

FastAPI automatically validates incoming JSON.

---

# Embedding Model

```
sentence-transformers/all-MiniLM-L6-v2
```

Used for generating vector embeddings.

---

# LLM

```
llama-3.3-70b-versatile
```

Provided by Groq.

---

# Retrieval Process

```
User Query

↓

Embedding

↓

Similarity Search

↓

Relevant Chunks

↓

Prompt

↓

LLM

↓

Answer
```

---

# HTTP Methods

| Method | Purpose |
|---------|----------|
| GET | Read Data |
| POST | Create/Send Data |
| PUT | Update Complete Data |
| PATCH | Update Partial Data |
| DELETE | Delete Data |

---

# HTTP Status Codes

| Code | Meaning |
|------|----------|
|200|Success|
|201|Created|
|400|Bad Request|
|401|Unauthorized|
|403|Forbidden|
|404|Not Found|
|500|Internal Server Error|

---

# Testing

Swagger

```
http://127.0.0.1:8000/docs
```

or

Postman

```
POST

http://127.0.0.1:8000/chat
```

Body

```json
{
    "user_query":"What services does Mperito provide?"
}
```

---

# Future Improvements

- Authentication (JWT)
- Docker
- CI/CD Pipeline
- Database Integration
- Logging
- Unit Testing
- Kubernetes Deployment
- Caching
- Streaming Responses

---

# Author

Nikky Gupta

Data Science / Generative AI Project

---
