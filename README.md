# Medical Chatbot with Generative AI

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-orange.svg)
![Ollama](https://img.shields.io/badge/Ollama-LLM-purple.svg)

A professional medical chatbot that provides accurate health information using Retrieval-Augmented Generation (RAG) with Pinecone vector database and Ollama LLM.

---

## 🚀 Features

- 🏥 **Medical Q&A** – Answers health-related questions with vetted information  
- 🔍 **Context-Aware** – Retrieves relevant medical context before generating responses  
- ⚡ **Fast Responses** – Optimized pipeline for quick answers  
- 🛡️ **Safety-First** – Always recommends consulting doctors for serious concerns  
- 💬 **Web Interface** – Clean, responsive chat UI  

---

## 🧰 Tech Stack

- **Backend**: Python, Flask  
- **AI Models**:  
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2`  
  - LLM: Ollama `deepseek-r1:1.5b`  
- **Vector Database**: Pinecone  
- **Frontend**: HTML5, CSS3, JavaScript  

---

## 🔧 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/lakshya-hidau/AI-Medical-Chatbot.git
cd medical-chatbot
```

2. **Create and activate a virtual environment:**
```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.env\Scriptsctivate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp .env.example .env
```
Edit `.env` and add your:
- Pinecone API key  
- Ollama configuration  

---

## ▶️ Usage

**Start the Flask server:**
```bash
python app.py
```

**Access the chatbot:**  
[http://localhost:8080](http://localhost:8080)

---

## ⚙️ Configuration

You can customize parameters in `app.py`:
```python
class Config:
    PINECONE_API_KEY = "your_api_key"
    INDEX_NAME = "medical-chatbot"
    LLM_MODEL = "deepseek-r1:1.5b"
    TEMPERATURE = 0.3  # For more factual responses
```

---

## 📁 Project Structure

```
medical-chatbot/
├── app.py               # Main application
├── templates/
│   └── chat.html        # Chat interface
├── static/
│   └── style.css        # Custom styles
├── .env.example         # Environment template
├── requirements.txt     # Dependencies
└── README.md            # This file
```

---

## 🤝 Contributing

1. Fork the project  
2. Create your feature branch: `git checkout -b feature/AmazingFeature`  
3. Commit your changes: `git commit -m 'Add some amazing feature'`  
4. Push to the branch: `git push origin feature/AmazingFeature`  
5. Open a Pull Request  

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## ⚠️ Disclaimer

This chatbot provides general health information only.  
It is **not** a substitute for professional medical advice, diagnosis, or treatment.  
Always consult a qualified healthcare provider with any questions about a medical condition.
