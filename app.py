import os
import time
import logging
from typing import List, Optional
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field
import pinecone
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
load_dotenv()

# Configuration
class Config:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV")
    INDEX_NAME = "medical-chatbot"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "deepseek-r1:1.5b"
    LLM_TEMPERATURE = 0.3
    MAX_RETRIES = 3
    RETRY_DELAY = 2

# Initialize Pinecone with retry logic
def initialize_pinecone() -> Pinecone:
    for attempt in range(Config.MAX_RETRIES):
        try:
            pc = Pinecone(api_key=Config.PINECONE_API_KEY)
            # Verify connection by listing indexes
            pc.list_indexes()
            return pc
        except Exception as e:
            logger.warning(f"Pinecone initialization attempt {attempt + 1} failed: {str(e)}")
            if attempt < Config.MAX_RETRIES - 1:
                time.sleep(Config.RETRY_DELAY)
    raise ConnectionError("Failed to initialize Pinecone after multiple attempts")

try:
    pc = initialize_pinecone()
    index = pc.Index(Config.INDEX_NAME)
except Exception as e:
    logger.error(f"Failed to connect to Pinecone index: {str(e)}")
    raise

# Initialize embeddings with retry logic
def initialize_embeddings():
    for attempt in range(Config.MAX_RETRIES):
        try:
            return HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        except Exception as e:
            logger.warning(f"Embedding initialization attempt {attempt + 1} failed: {str(e)}")
            if attempt < Config.MAX_RETRIES - 1:
                time.sleep(Config.RETRY_DELAY)
    raise RuntimeError("Failed to initialize embeddings after multiple attempts")

embeddings = initialize_embeddings()

# Enhanced Retriever with better error handling
class PineconeRetriever(BaseRetriever, BaseModel):
    index: any
    embeddings: any
    k: int = Field(default=3, description="Number of documents to retrieve")

    def _get_relevant_documents(self, query: str) -> List[Document]:
        try:
            vector = self.embeddings.embed_query(query)
            results = self.index.query(
                vector=vector,
                top_k=self.k,
                include_metadata=True,
                namespace="medical"
            )
            return [
                Document(
                    page_content=match['metadata']['text'],
                    metadata={
                        'source': match['metadata'].get('source', 'Unknown'),
                        'confidence': match.get('score', 0.0)
                    }
                )
                for match in results.get('matches', [])
                if match.get('metadata')
            ]
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            return []

retriever = PineconeRetriever(index=index, embeddings=embeddings)

# Enhanced prompt template with medical disclaimer
PROMPT_TEMPLATE = """You are MediBot, a medical information assistant. Provide answers in this EXACT format:

**Answer:** [Direct one-sentence answer to the question]\n\n

- [Bullet 1 from context]\n
- [Bullet 2 from context]\n
- [Bullet 3 from context]\n\n

**When to see a doctor:** [Specific guidance from context]\n\n

**Note:** Always consult a healthcare professional for medical advice.

Rules you MUST follow:
1. NEVER show your thinking process
2. Use ONLY the provided context
3. Answer in maximum 5 sentences total
4. Use simple language (8th grade level)
5. If context doesn't contain answer, say "I don't have enough medical information about this"

Context: {context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# Initialize LLM with health check
def initialize_llm():
    for attempt in range(Config.MAX_RETRIES):
        try:
            llm = OllamaLLM(
                model=Config.LLM_MODEL,
                temperature=Config.LLM_TEMPERATURE,
                top_p=0.9,
                repeat_penalty=1.1,
                num_ctx=2048
            )
            # Test with a simple prompt
            llm("Test connection")
            return llm
        except Exception as e:
            logger.warning(f"LLM initialization attempt {attempt + 1} failed: {str(e)}")
            if attempt < Config.MAX_RETRIES - 1:
                time.sleep(Config.RETRY_DELAY)
    raise RuntimeError("Failed to initialize LLM after multiple attempts")

llm = initialize_llm()

# Build QA chain with fallback
try:
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PROMPT,
            "verbose": False
        }
    )
except Exception as e:
    logger.error(f"QA chain initialization failed: {str(e)}")
    raise

# Response processing pipeline
def process_response(response: str) -> str:
    """Clean and format the bot's response"""
    # Ensure standard formatting
    if not response.startswith("**Answer:**"):
        response = f"**Answer:** {response}"
    
    # Ensure disclaimer is present
    if "**Note:**" not in response:
        response += "\n\n**Note:** This information is not a substitute for professional medical advice."
    
    # Clean up any odd formatting
    response = response.replace("[Direct Answer]", "**Answer:**")\
                      .replace("[When to consult a doctor]", "**When to see a doctor:**")
    
    return response.strip()

@app.route("/")
def index_route():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    try:
        user_input = request.form.get("msg", "").strip()
        if not user_input or len(user_input) < 3:
            return jsonify({
                "error": "Please provide a valid health question (at least 3 characters)"
            }), 400
        
        logger.info(f"Processing query: '{user_input}'")
        
        # Execute with retry
        for attempt in range(Config.MAX_RETRIES):
            try:
                result = qa({"query": user_input})
                response = process_response(result["result"])
                logger.info(f"Response generated (length: {len(response)})")
                return response
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(Config.RETRY_DELAY)
        
        return "I'm currently unable to process medical questions. Please try again later.", 503
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return "An error occurred while processing your request. Our team has been notified.", 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)