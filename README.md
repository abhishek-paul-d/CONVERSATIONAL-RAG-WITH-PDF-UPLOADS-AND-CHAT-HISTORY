# Conversational RAG with PDF Upload

A sophisticated **Retrieval-Augmented Generation (RAG)** application that enables users to upload PDF documents and engage in intelligent, context-aware conversations about their content. Built with Streamlit, LangChain, and powered by cutting-edge AI technologies.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![LangChain](https://img.shields.io/badge/langchain-v0.1+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🚀 Features

- **📄 Multi-PDF Upload**: Upload and process multiple PDF documents simultaneously
- **🤖 Conversational AI**: Chat naturally with your documents using state-of-the-art LLMs
- **🧠 Memory-Enabled**: Maintains conversation history and context across interactions
- **⚡ Ultra-Fast Inference**: Powered by Groq's LPU architecture for sub-second responses
- **🔍 Semantic Search**: Advanced vector-based document retrieval using HuggingFace embeddings
- **👥 Multi-Session Support**: Handle multiple conversation sessions independently
- **🎯 Context-Aware**: Follow-up questions automatically reference previous conversation context

## 🏗️ Architecture Overview

```
PDF Upload → Text Extraction → Chunking → Embeddings → Vector Store
                                                           ↓
User Query → History-Aware Retriever → Document Retrieval → Context Assembly
                                                           ↓
                    Response ← LLM Processing ← Prompt Engineering
```

### Core Components

- **Frontend**: Streamlit web interface
- **Document Processing**: PyPDFLoader + RecursiveCharacterTextSplitter
- **Embeddings**: HuggingFace all-MiniLM-L6-v2 model
- **Vector Database**: Chroma for similarity search
- **LLM**: Groq-hosted Llama 3.3 70B model
- **Orchestration**: LangChain framework
- **Memory**: Session-based chat history management

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Groq API key (sign up at [console.groq.com](https://console.groq.com))
- HuggingFace account (optional, for embedding model access)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/conversational-rag-pdf.git
   cd conversational-rag-pdf
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional)
   ```bash
   # Create .env file
   echo "HF_TOKEN=your_huggingface_token_here" > .env
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the application**
   - Open your browser to `http://localhost:8501`
   - Enter your Groq API key in the interface
   - Upload PDF files and start chatting!

## 📦 Dependencies

Create a `requirements.txt` file with the following dependencies:

```txt
streamlit>=1.28.0
langchain>=0.1.0
langchain-community>=0.0.20
langchain-chroma>=0.1.0
langchain-core>=0.1.23
langchain-groq>=0.1.0
langchain-huggingface>=0.0.1
langchain-text-splitters>=0.0.1
python-dotenv>=1.0.0
pypdf>=3.0.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `HF_TOKEN` | HuggingFace API token for embedding model access | Optional |

### Model Configuration

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **LLM**: `llama-3.3-70b-versatile` via Groq
- **Chunk Size**: 5000 characters with 500 character overlap
- **Vector Store**: Chroma (in-memory)

## 💡 Usage

### Basic Workflow

1. **Launch Application**
   ```bash
   streamlit run app.py
   ```

2. **Configure API Access**
   - Enter your Groq API key in the password field
   - Optionally set a custom session ID

3. **Upload Documents**
   - Click "Choose a PDF File" 
   - Select one or multiple PDF files
   - Wait for processing completion

4. **Start Chatting**
   - Type questions about your documents
   - Ask follow-up questions naturally
   - View conversation history

### Example Interactions

```
👤 User: "What is the main topic of this document?"
🤖 Assistant: "Based on the uploaded PDF, the main topic is..."

👤 User: "Can you summarize the key findings?"
🤖 Assistant: "The key findings from the document include..."

👤 User: "What about the methodology section?"
🤖 Assistant: "Regarding the methodology you mentioned earlier..."
```

### Session Management

- **Session ID**: Set custom session identifiers to maintain separate conversations
- **Chat History**: View complete conversation history in the interface
- **Multi-User**: Different sessions maintain independent conversation contexts

## 🔧 Technical Details

### Document Processing Pipeline

1. **PDF Upload**: Files uploaded through Streamlit interface
2. **Temporary Storage**: Files written to temporary location for processing
3. **Text Extraction**: PyPDFLoader extracts text content
4. **Text Chunking**: RecursiveCharacterTextSplitter creates overlapping segments
5. **Embedding Generation**: HuggingFace model creates vector representations
6. **Vector Storage**: Chroma stores embeddings for similarity search

### Conversational RAG Flow

1. **Query Processing**: User input collected via Streamlit
2. **Context Reformulation**: History-aware retriever creates standalone queries
3. **Document Retrieval**: Vector similarity search finds relevant content
4. **Context Assembly**: Retrieved documents combined with conversation history
5. **Response Generation**: LLM generates contextually appropriate response
6. **Memory Update**: Conversation history updated with new exchange

### Key LangChain Components

- **`create_history_aware_retriever`**: Contextualizes queries using chat history
- **`create_stuff_documents_chain`**: Combines documents for LLM processing
- **`create_retrieval_chain`**: Orchestrates retrieval and generation
- **`RunnableWithMessageHistory`**: Manages conversation memory


## 🙏 Acknowledgments

- **LangChain**: For the excellent orchestration framework
- **Streamlit**: For the intuitive web app framework
- **Groq**: For ultra-fast LLM inference
- **HuggingFace**: For state-of-the-art embedding models
- **Chroma**: For efficient vector database capabilities

## 📊 Project Stats

- **Language**: Python 3.10
- **Framework**: Streamlit + LangChain
- **Database**: Chroma Vector DB
- **AI Models**: Llama 3.3 70B + MiniLM-L6-v2
- **Deployment**: Local development server

## 🔗 Related Projects

- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Groq API Documentation](https://console.groq.com/docs)
- [Chroma Vector Database](https://www.trychroma.com/)
