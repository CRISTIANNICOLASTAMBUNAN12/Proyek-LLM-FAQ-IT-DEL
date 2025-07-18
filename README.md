# IT Del FAQ Chatbot 🤖

Sistem chatbot FAQ berbasis LLM (Large Language Model) untuk Institut Teknologi Del yang menggunakan teknologi RAG (Retrieval-Augmented Generation) dengan model Qwen untuk memberikan jawaban yang akurat dan contextual.

## ✨ Fitur Utama

- **🧠 RAG-based Architecture**: Menggunakan FAISS vector store untuk pencarian dokumen yang relevan
- **🚀 Model Qwen 1.5**: Implementasi model language Qwen 1.5-0.5B-Chat untuk generasi jawaban
- **📄 PDF Processing**: Otomatis memproses dokumen PDF FAQ menjadi knowledge base
- **🔍 Semantic Search**: Pencarian semantik menggunakan embeddings multilingual
- **⚡ FastAPI Backend**: API yang cepat dan scalable
- **🎨 Modern UI**: Interface web yang responsif dan modern
- **🔧 Error Handling**: Sistem fallback yang robust untuk handling error
- **💬 Real-time Chat**: Interface chat real-time dengan typing indicator

## 🏗️ Arsitektur Sistem

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Document  │────│  Text Splitter  │────│  Vector Store   │
│   (FAQ Data)    │    │  (Chunking)     │    │   (FAISS)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │────│  RAG Pipeline   │────│   Qwen Model    │
│                 │    │  (Retrieval)    │    │  (Generation)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │────│  FastAPI Server │────│   Response      │
│   (Frontend)    │    │   (Backend)     │    │  Processing     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Teknologi yang Digunakan

### Backend
- **Python 3.8+**
- **LangChain**: Framework untuk aplikasi LLM
- **Transformers**: Library Hugging Face untuk model AI
- **FAISS**: Vector similarity search
- **FastAPI**: Web framework untuk API
- **PyPDF**: PDF processing
- **HuggingFace Embeddings**: Sentence embeddings

### Frontend
- **HTML5 & CSS3**
- **Vanilla JavaScript**
- **Modern UI Design**
- **Responsive Layout**

### Model & Embeddings
- **Qwen/Qwen1.5-0.5B-Chat**: Model generasi teks
- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2**: Embeddings model

## 📦 Instalasi

### Prerequisites
- Python 3.8 atau lebih tinggi
- Git
- CUDA-compatible GPU (opsional, untuk performa lebih baik)

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/itdel-faq-chatbot.git
cd itdel-faq-chatbot
```

### 2. Buat Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Konfigurasi Environment
Buat file `.env` atau sesuaikan path di `Config` class:
```python
# Sesuaikan path berikut di Config class
HF_HOME = "path/to/your/hf_cache"
TEMP_DIR = "path/to/your/temp"
STORAGE_DIR = "path/to/your/storage"
PDF_PATH = "path/to/your/faq.pdf"
FAISS_DIR = "path/to/your/faiss_index"
```

### 5. Siapkan Dokumen FAQ
- Place your PDF FAQ document in the specified `PDF_PATH`
- Pastikan PDF berisi informasi FAQ yang ingin dijadikan knowledge base

## 🚀 Menjalankan Aplikasi

### 1. Jalankan Backend Server
```bash
python main.py
```

Server akan berjalan di `http://localhost:8000`

### 2. Akses Web Interface
- Buka `index.html` di browser
- Atau setup web server untuk serving static files

## Screenshot Tampilan System

<img width="1085" height="944" alt="image" src="https://github.com/user-attachments/assets/4dd6f0a2-8e55-41da-9c06-61063124f5bb" />

<img width="1075" height="953" alt="image" src="https://github.com/user-attachments/assets/c9cffc74-8146-425a-90d3-adca0dabf854" />



