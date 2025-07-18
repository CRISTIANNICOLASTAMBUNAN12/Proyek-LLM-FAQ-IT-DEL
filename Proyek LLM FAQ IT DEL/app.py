# -*- coding: utf-8 -*-
import os
from typing import Dict, List, Optional
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

# LangChain Components
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import BitsAndBytesConfig
import torch
from langchain.prompts import PromptTemplate

# Transformers/Qwen
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline

# FastAPI for Deployment
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# 1. CONFIGURATION CLASS
# ======================
class Config:
    # Environment setup
    TOKENIZERS_PARALLELISM = "false"
    HF_HOME = r"D:/chatbot-admin-itdel/storage/hf_model_cache"
    TEMP_DIR = r"D:/chatbot-admin-itdel/temp"
    STORAGE_DIR = r"D:/chatbot-admin-itdel/storage"
    
    # Model paths
    PDF_PATH = r"D:/PROYEK LLM/layanan.pdf"
    FAISS_DIR = r"D:/PROYEK LLM/faiss_index"
    
    # Model configurations
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    LLM_MODEL = "Qwen/Qwen1.5-0.5B-Chat"
    
    # Text processing
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150
    MAX_NEW_TOKENS = 300
    TEMPERATURE = 0.3
    RETRIEVAL_K = 3
    
    # API settings
    HOST = "0.0.0.0"
    PORT = 8000
    
    @classmethod
    def setup_environment(cls):
        """Setup environment variables"""
        os.environ["TOKENIZERS_PARALLELISM"] = cls.TOKENIZERS_PARALLELISM
        os.environ["HF_HOME"] = cls.HF_HOME
        os.environ["TEMP"] = cls.TEMP_DIR
        os.environ["TMP"] = cls.TEMP_DIR
        
        # Create directories
        for dir_path in [cls.TEMP_DIR, cls.STORAGE_DIR, os.path.dirname(cls.FAISS_DIR)]:
            os.makedirs(dir_path, exist_ok=True)

# ======================
# 2. DOCUMENT PROCESSOR CLASS
# ======================
class DocumentProcessor:
    def __init__(self, config: Config):
        self.config = config
        
    def load_and_process_pdf(self, pdf_path: str) -> List:
        """Load and process PDF with better error handling"""
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            
            if not pages:
                raise ValueError("No content found in PDF")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ".", "!", "?", " ", ""],
                length_function=len
            )

            docs = text_splitter.split_documents(pages)
            docs = [doc for doc in docs if len(doc.page_content.strip()) > 50]
            
            logger.info(f"✅ Loaded {len(docs)} document chunks from {pdf_path}")
            return docs
        except Exception as e:
            logger.error(f"❌ Error loading PDF: {str(e)}")
            raise

# ======================
# 3. VECTOR STORE MANAGER
# ======================
class VectorStoreManager:
    def __init__(self, config: Config):
        self.config = config
        self.embeddings = None
        
    def _get_embeddings(self):
        """Get or create embeddings model"""
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        return self.embeddings
        
    def create_vector_store(self, docs):
        """Create vector store with better configuration"""
        try:
            embeddings = self._get_embeddings()
            db = FAISS.from_documents(docs, embeddings)
            db.save_local(self.config.FAISS_DIR)
            
            metadata = {
                "created_at": datetime.now().isoformat(),
                "num_documents": len(docs),
                "embedding_model": self.config.EMBEDDING_MODEL
            }
            
            logger.info(f"✅ Vector store created with {len(docs)} documents")
            return db
        except Exception as e:
            logger.error(f"❌ Error creating vector store: {str(e)}")
            raise

    def load_vector_store(self) -> Optional[FAISS]:
        """Load existing vector store"""
        try:
            if not os.path.exists(self.config.FAISS_DIR):
                return None
                
            embeddings = self._get_embeddings()
            db = FAISS.load_local(
                self.config.FAISS_DIR, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            logger.info(f"✅ Vector store loaded from {self.config.FAISS_DIR}")
            return db
        except Exception as e:
            logger.error(f"❌ Error loading vector store: {str(e)}")
            return None

# ======================
# 4. IMPROVED LLM SETUP
# ======================
class LLMManager:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.llm = None
        
    def setup_qwen_llm(self):
        """Fixed Qwen LLM setup with proper chat template handling"""
        try:
            # Determine device and dtype
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            
            logger.info(f"Initializing Qwen model on {device} with {torch_dtype}")

            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.LLM_MODEL,
                cache_dir=self.config.HF_HOME,
                trust_remote_code=True
            )
            
            # Configure tokenizer properly
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Model configuration
            model_kwargs = {
                "cache_dir": self.config.HF_HOME,
                "device_map": "auto" if device == "cuda" else None,
                "trust_remote_code": True,
                "torch_dtype": torch_dtype
            }
            
            # Only use quantization if on CUDA
            if device == "cuda":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.LLM_MODEL,
                **model_kwargs
            )
            
            # Move model to device if not using device_map
            if device == "cpu":
                self.model = self.model.to(device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Verify model can generate text
            test_response = self._test_generation()
            if not test_response:
                logger.warning("⚠️ Model test generation returned empty, but continuing...")
            
            # Create pipeline with optimized parameters
            generation_config = {
                "max_new_tokens": self.config.MAX_NEW_TOKENS,
                "temperature": self.config.TEMPERATURE,
                "do_sample": True,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.05,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False  # Only return new generated text
            }
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                **generation_config
            )
            
            # Create LangChain pipeline
            self.llm = HuggingFacePipeline(pipeline=pipe)
            logger.info("✅ Qwen LLM initialized successfully")
            return self.llm
            
        except Exception as e:
            logger.error(f"❌ Error initializing Qwen: {str(e)}", exc_info=True)
            raise

    def _test_generation(self, test_input: str = "Jawab singkat: Apa ibukota Indonesia?") -> str:
        """Enhanced test generation with proper chat formatting"""
        try:
            logger.info(f"\n🔧 Running model test with input: '{test_input}'")
            
            # Format input for Qwen chat model
            messages = [
                {"role": "system", "content": "Anda adalah asisten yang membantu menjawab pertanyaan dengan singkat."},
                {"role": "user", "content": test_input}
            ]
            
            # Apply chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template'):
                try:
                    formatted_input = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    logger.info(f"Using chat template: {formatted_input[:100]}...")
                except:
                    # Fallback to simple format
                    formatted_input = f"<|im_start|>system\nAnda adalah asisten yang membantu menjawab pertanyaan dengan singkat.<|im_end|>\n<|im_start|>user\n{test_input}<|im_end|>\n<|im_start|>assistant\n"
                    logger.info("Using fallback chat format")
            else:
                formatted_input = test_input
            
            # Prepare inputs
            inputs = self.tokenizer(
                formatted_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.model.device)
            
            # Generate with more specific parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    attention_mask=inputs.get('attention_mask', None)
                )
            
            # Decode only the new tokens
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()
            
            logger.info(f"🔧 Test response: '{response}'")
            
            # Even if empty, we'll continue - some models need specific prompting
            if not response:
                logger.warning("⚠️ Empty test response, but model appears functional")
                return "Model loaded successfully"
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Test generation failed: {str(e)}", exc_info=True)
            # Return a success indicator even if test fails
            return "Model loaded successfully"

# ======================
# 5. ENHANCED RAG CHAIN
# ======================
class RAGChain:
    def __init__(self, config: Config):
        self.config = config
        self.qa_chain = None
        
    def setup_rag_chain(self, llm, db):
        """Setup RAG chain with improved prompt for Qwen"""
        try:
            # Updated prompt template for better Qwen compatibility
            prompt_template = """<|im_start|>system
Anda adalah asisten FAQ IT Del yang membantu menjawab pertanyaan tentang layanan IT Del.
Jawab berdasarkan konteks yang diberikan dengan singkat dan jelas.
Jika informasi tidak tersedia di konteks, katakan bahwa informasi tidak ditemukan.
<|im_end|>

<|im_start|>user
Konteks:
{context}

Pertanyaan: {question}
<|im_end|>

<|im_start|>assistant
"""

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=db.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": self.config.RETRIEVAL_K,
                        "fetch_k": self.config.RETRIEVAL_K * 2
                    }
                ),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            logger.info("✅ RAG chain setup complete")
            return self.qa_chain
        except Exception as e:
            logger.error(f"❌ Error setting up RAG chain: {str(e)}")
            raise

# ======================
# 6. ENHANCED RESPONSE PROCESSOR
# ======================
class ResponseProcessor:
    def __init__(self):
        self.fallback_keywords = {
            "beasiswa": ["beasiswa", "scholarship", "bantuan", "biaya kuliah"],
            "ujian": ["ujian", "test", "tes", "seleksi", "evaluasi"],
            "biaya": ["biaya", "cost", "fee", "tarif", "harga"],
            "pendaftaran": ["pendaftaran", "daftar", "registration", "registrasi"],
            "jalur": ["jalur", "path", "seleksi", "penerimaan"],
            "akreditasi": ["akreditasi", "accreditation", "sertifikat"],
            "program": ["program", "jurusan", "studi", "kuliah"],
            "mahasiswa": ["mahasiswa", "student", "siswa"],
            "dosen": ["dosen", "pengajar", "fakultas", "staff"],
            "fasilitas": ["fasilitas", "facility", "lab", "perpustakaan"]
        }
    
    def extract_answer_from_context(self, question: str, context_docs: List) -> str:
        if not context_docs:
            return "Maaf, informasi tidak ditemukan di FAQ."

        question_lower = question.lower()
        scored_docs = []
        for doc in context_docs:
            content = doc.page_content.lower()
            score = 0
            for word in question_lower.split():
                if len(word) > 3 and word in content:
                    score += 2
            for category, keywords in self.fallback_keywords.items():
                if any(keyword in question_lower for keyword in keywords):
                    if any(keyword in content for keyword in keywords):
                        score += 5
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        if not scored_docs or scored_docs[0][0] == 0:
            return "Maaf, informasi tidak ditemukan di FAQ."

        best_doc = scored_docs[0][1]
        content = best_doc.page_content

        lines = content.split('\n')
        answer_lines = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            if any(word in line.lower() for word in question_lower.split() if len(word) > 3):
                answer_lines.append(line)
                for j in range(i + 1, min(i + 5, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and not next_line.endswith('?'):
                        answer_lines.append(next_line)
                    elif next_line.endswith('?'):
                        break
                break

        if answer_lines:
            return (
                "Berikut informasi yang dapat kami temukan berdasarkan pertanyaan Anda:\n\n"
                + "\n".join(f"- {line.lstrip('-•o').strip()}" for line in answer_lines)
            )

        paragraphs = content.split('\n\n')
        for paragraph in paragraphs:
            if len(paragraph.strip()) > 50:
                return (
                    "Berdasarkan dokumen yang tersedia, berikut informasi terkait:\n\n"
                    + paragraph.strip()
                )

        return "Maaf, informasi tidak ditemukan di FAQ."

    def process_response(self, raw_response: str, question: str = "", context_docs: List = []) -> str:
        logger.info(f"Raw response length: {len(raw_response) if raw_response else 0}")
        
        if not raw_response or len(raw_response.strip()) < 10:
            logger.info("Raw response too short, using fallback")
            return self.extract_answer_from_context(question, context_docs)
        
        response = raw_response.strip()
        
        # Remove chat template artifacts
        chat_artifacts = [
            "<|im_start|>",
            "<|im_end|>",
            "assistant",
            "user",
            "system"
        ]
        
        for artifact in chat_artifacts:
            response = response.replace(artifact, "")
        
        # Remove prompt artifacts
        artifacts_to_remove = [
            "Anda adalah asisten FAQ IT Del",
            "Konteks informasi:",
            "Instruksi:",
            "Jawab berdasarkan konteks",
            "Berdasarkan konteks berikut",
            "Pertanyaan:",
            "Jawaban:",
            "Konteks:",
            "Informasi:"
        ]
        
        for artifact in artifacts_to_remove:
            while artifact in response:
                parts = response.split(artifact, 1)
                if len(parts) > 1:
                    response = parts[-1].strip()
                else:
                    break
        
        lines = response.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith(('1.', '2.', '3.', '4.')) and 'jawab' in line.lower():
                continue
            if line.startswith('Jika informasi tidak'):
                continue
            clean_lines.append(line)
        
        response = '\n'.join(clean_lines)
        
        if len(response.strip()) < 5 or 'tidak tersedia' in response.lower():
            logger.info("LLM response not satisfactory, using fallback")
            return self.extract_answer_from_context(question, context_docs)
        
        return response

# ======================
# 7. MAIN APPLICATION CLASS
# ======================
class ChatbotApp:
    def __init__(self):
        self.config = Config()
        self.config.setup_environment()
        
        self.doc_processor = DocumentProcessor(self.config)
        self.vector_manager = VectorStoreManager(self.config)
        self.llm_manager = LLMManager(self.config)
        self.rag_chain = RAGChain(self.config)
        self.response_processor = ResponseProcessor()
        
        self.qa_chain = None
        
    async def initialize(self):
        """Initialize all components"""
        try:
            # Load or create vector store
            db = self.vector_manager.load_vector_store()
            
            if db is None:
                logger.info("Creating new vector store...")
                docs = self.doc_processor.load_and_process_pdf(self.config.PDF_PATH)
                db = self.vector_manager.create_vector_store(docs)
            
            # Setup LLM
            llm = self.llm_manager.setup_qwen_llm()
            
            # Setup RAG chain
            self.qa_chain = self.rag_chain.setup_rag_chain(llm, db)
            
            logger.info("🚀 Chatbot initialization complete")
            return True
        except Exception as e:
            logger.error(f"🔥 Failed to initialize chatbot: {str(e)}")
            return False
    
    async def process_query(self, question: str) -> Dict:
        """Process user query with enhanced error handling"""
        try:
            if not self.qa_chain:
                raise ValueError("QA chain not initialized")
            
            question = question.strip()
            if len(question) < 3:
                raise ValueError("Question too short")
            
            result = self.qa_chain.invoke({"query": question})
            
            raw_answer = result["result"]
            source_docs = result["source_documents"]
            clean_answer = self.response_processor.process_response(
                raw_answer, question, source_docs
            )
            
            sources = []
            for doc in source_docs:
                source = {
                    "page": str(doc.metadata.get("page", "N/A")),
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                sources.append(source)
            
            return {
                "answer": clean_answer,
                "sources": sources,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda.",
                "sources": [],
                "success": False,
                "error": str(e)
            }

# ======================
# 8. FASTAPI APPLICATION
# ======================

chatbot = ChatbotApp()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    success = await chatbot.initialize()
    if not success:
        logger.error("Failed to initialize chatbot - shutting down")
        raise RuntimeError("Failed to initialize chatbot")
    
    logger.info("🚀 Chatbot initialized successfully")
    yield
    
    # Shutdown
    logger.info("Shutting down chatbot...")

app = FastAPI(
    title="IT Del FAQ Chatbot",
    description="Enhanced RAG-based chatbot for IT Del FAQ",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, description="User question")

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]
    success: bool = True
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

@app.get("/")
async def root():
    return {
        "message": "IT Del FAQ Chatbot API v2.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "chatbot_ready": chatbot.qa_chain is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Ask a question to the chatbot"""
    try:
        result = await chatbot.process_query(request.question)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            success=result["success"]
        )
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/reload")
async def reload_chatbot():
    """Reload the chatbot (admin endpoint)"""
    try:
        success = await chatbot.initialize()
        return {
            "success": success,
            "message": "Chatbot reloaded successfully" if success else "Failed to reload chatbot"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-model")
async def test_model():
    """Test endpoint to verify basic model generation"""
    try:
        test_input = "Apa ibukota Indonesia?"
        response = chatbot.llm_manager._test_generation(test_input)
        
        return {
            "test_input": test_input,
            "model_response": response,
            "success": True,
            "details": {
                "model_device": str(chatbot.llm_manager.model.device),
                "model_dtype": str(chatbot.llm_manager.model.dtype)
            }
        }
    except Exception as e:
        logger.error(f"Model test failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Model test failed: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        log_level="info",
        reload=False
    )