# Medical AI Agent with RAG Integration

## 🎯 Project Overview

This project combines an **AI Agent (using CrewAI)** with a **RAG (Retrieval-Augmented Generation) system** to create a comprehensive medical information assistant. The agent can query a medical knowledge base to provide accurate, evidence-based responses to health-related questions.

### Pattern Implementation
This implements **Pattern 2**: An AI Agent that interfaces with a RAG application using CrewAI framework.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Medical AI Agent System                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐     ┌─────────────────────────────┐   │
│  │   CrewAI Agent  │────▶│    RAG Tools (Custom)       │   │
│  │                 │     │  - medical_knowledge_query   │   │
│  │  - Medical      │     │  - medical_document_search   │   │
│  │    Assistant    │     └─────────────┬───────────────┘   │
│  │  - Researcher   │                   │                    │
│  │  - Educator     │                   ▼                    │
│  └─────────────────┘     ┌─────────────────────────────┐   │
│          │               │     RAG Pipeline             │   │
│          ▼               │  - Document Chunking         │   │
│  ┌─────────────────┐     │  - Embedding (HuggingFace)   │   │
│  │  Google Gemini  │     │  - Vector Store (Chroma)     │   │
│  │   (LLM Engine)  │     │  - Similarity Search         │   │
│  └─────────────────┘     └─────────────┬───────────────┘   │
│                                        │                    │
│                                        ▼                    │
│                          ┌─────────────────────────────┐   │
│                          │   Medical Knowledge Base     │   │
│                          │  - MedQA (Licensing Exams)   │   │
│                          │  - MedDialog (Conversations) │   │
│                          │  - HealthSearchQA            │   │
│                          │  - LiveQA (Consumer Health)  │   │
│                          └─────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
medical_agent/
├── agent_main.py          # Main agent implementation with CrewAI
├── rag_tools.py           # Custom CrewAI tools for RAG integration
├── rag_pipeline.py        # RAG system implementation
├── data_loader.py         # Medical dataset loading and processing
├── config.py              # Configuration management
├── logger.py              # Logging setup
├── demo.py                # Demonstration script
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
├── README.md              # This file
└── data/
    ├── raw/               # Source datasets
    ├── processed/         # Preprocessed data
    └── vector_store/      # Chroma/FAISS index
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd medical_agent
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

Get your Google API key from: https://makersuite.google.com/app/apikey

### 3. Prepare Data (Optional)

If you have processed medical data, place it in `data/processed/combined_medical_data.json`.
Otherwise, the system will attempt to load datasets from HuggingFace on first run.

### 4. Run the Agent

```bash
# Interactive mode
python agent_main.py --interactive

# Single question
python agent_main.py --question "What are the symptoms of diabetes?"

# Research mode
python agent_main.py --research "Hypertension treatment options"

# Run demo
python demo.py
```

## 🛠️ Components

### 1. CrewAI Agents

- **Medical Assistant**: Primary agent for answering medical questions
- **Medical Researcher**: Conducts in-depth research on medical topics
- **Medical Educator**: Explains medical concepts in simple terms

### 2. RAG Tools

- **medical_knowledge_query**: Queries the RAG system with natural language questions
- **medical_document_search**: Searches for relevant medical documents

### 3. RAG Pipeline

- **Document Processing**: Chunks documents for efficient retrieval
- **Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: Chroma DB for fast similarity search
- **LLM Integration**: Google Gemini for response generation

### 4. Medical Knowledge Base

The system includes data from:
- **MedQA**: Medical licensing exam questions
- **MedDialog**: Doctor-patient conversations
- **HealthSearchQA**: Health search queries
- **LiveQA**: Consumer health questions

## 📋 Usage Examples

### Interactive Session

```python
from agent_main import run_interactive_session
run_interactive_session()
```

### Programmatic Usage

```python
from agent_main import run_medical_assistant, run_medical_research

# Ask a question
answer = run_medical_assistant("What causes high blood pressure?")
print(answer)

# Research a topic
report = run_medical_research("Diabetes management")
print(report)
```

### Using RAG Tools Directly

```python
from rag_tools import medical_query_tool, medical_search_tool

# Query with LLM response
result = medical_query_tool._run("What is hypertension?")
print(result)

# Search without LLM
documents = medical_search_tool._run("diabetes symptoms", num_results=5)
print(documents)
```

## ⚙️ Configuration

Key configuration options in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key | Required |
| `AGENT_MODEL` | Gemini model to use | `gemini-2.5-flash` |
| `EMBEDDING_MODEL` | HuggingFace embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `VECTOR_STORE_TYPE` | Vector store type | `chroma` |
| `TOP_K_RESULTS` | Number of documents to retrieve | `5` |
| `AGENT_VERBOSE` | Enable verbose agent output | `true` |

## 🔍 How It Works

1. **User Query**: User asks a medical question
2. **Agent Processing**: CrewAI agent receives the query
3. **RAG Tool Invocation**: Agent uses RAG tools to search knowledge base
4. **Document Retrieval**: Vector store returns relevant documents
5. **Context Assembly**: Retrieved documents form the context
6. **LLM Generation**: Gemini generates response using context
7. **Response Delivery**: Agent returns formatted answer with sources

## ⚠️ Disclaimer

This system is for **educational and informational purposes only**. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

## 📊 Technologies Used

- **CrewAI**: AI agent orchestration framework
- **LangChain**: LLM application framework
- **Google Gemini**: Large Language Model
- **Chroma**: Vector database
- **HuggingFace**: Embeddings and datasets
- **Python**: Core programming language

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📜 License

MIT License - See LICENSE file for details.
