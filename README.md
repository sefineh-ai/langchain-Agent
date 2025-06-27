# 🎓 SFSU Scholarship AI Agent

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.16-green.svg?style=for-the-badge)](https://www.langchain.com/)
[![AstraDB](https://img.shields.io/badge/AstraDB-Vector%20Store-purple.svg?style=for-the-badge)](https://www.datastax.com/products/datastax-astra)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg?style=for-the-badge)](https://openai.com/)
[![Google Search](https://img.shields.io/badge/Google%20Search-API-yellow.svg?style=for-the-badge)](https://developers.google.com/custom-search)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> **An intelligent AI agent designed to help students find scholarship opportunities at San Francisco University** - Leveraging LangChain, AstraDB Vector Store, and Google Search to provide comprehensive scholarship matching and information retrieval.

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Data Sources](#-data-sources)
- [Contributing](#-contributing)
- [Support](#-support)
- [License](#-license)

## 🎯 About the Project

The **SFSU Scholarship AI Agent** is a sophisticated AI-powered system designed to assist students worldwide in finding and accessing scholarship opportunities at San Francisco University. The agent combines local knowledge with real-time web search capabilities to provide comprehensive, up-to-date scholarship information.

### Mission Statement

> *"To democratize access to scholarship information by providing an intelligent, conversational interface that helps students navigate the complex landscape of university funding opportunities."*

### Key Capabilities

- **🔍 Intelligent Search**: Combines vector similarity search with Google web search
- **📚 Comprehensive Database**: Access to both structured and unstructured scholarship data
- **🌐 Real-time Updates**: Live web search for the latest scholarship information
- **💬 Natural Language**: Conversational interface for easy interaction
- **🎯 Personalized Matching**: Context-aware responses based on student queries

## ✨ Features

### Core Functionality
- **Scholarship Discovery**: Find relevant scholarships based on student criteria
- **Information Retrieval**: Access detailed scholarship requirements and deadlines
- **Real-time Updates**: Get the latest information through web search integration
- **Conversational Interface**: Natural language interaction with the AI agent
- **Data Processing**: Handles both structured (CSV) and unstructured (text) data

### Advanced Capabilities
- **Vector Similarity Search**: Semantic search through scholarship database
- **Web Search Integration**: Google Search API for additional information
- **Data Cleaning**: Automated preprocessing of scholarship data
- **Memory Management**: Chat history for contextual conversations
- **Error Handling**: Robust error management and fallback mechanisms

### User Experience
- **Interactive CLI**: Command-line interface for easy interaction
- **Contextual Responses**: AI remembers conversation history
- **Comprehensive Answers**: Combines local data with web search results
- **Fast Response Times**: Optimized vector search and caching

## 🏗️ Architecture

The system follows a modular architecture with clear separation of concerns:

```
langchain-Agent/
├── main.py                 # Application entry point
├── agent/
│   ├── agent.py           # Core AI agent implementation
│   ├── store.py           # Vector store management
│   └── web_page_loader.py # Web content processing
├── data/
│   ├── structured_scholarship_data.csv    # Structured scholarship data
│   ├── unstructured_scholarship_data.txt  # Unstructured scholarship data
│   └── scholarship_pages.csv              # Additional scholarship pages
└── requirements.txt       # Python dependencies
```

### System Components

1. **SFResourceMatching Class**: Main agent orchestrator
2. **VectorStoreManager**: AstraDB vector store operations
3. **ScholarshipDataProcessor**: Data preprocessing and cleaning
4. **GoogleSearchAPIWrapper**: Web search integration
5. **AgentExecutor**: LangChain agent execution engine

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **LangChain 0.2.16**: AI agent framework
- **OpenAI GPT-4**: Large language model for natural language processing
- **AstraDB Vector Store**: Vector database for semantic search
- **Google Search API**: Real-time web search capabilities

### Key Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| `langchain` | 0.2.16 | AI agent framework |
| `langchain-astradb` | 0.3.3 | AstraDB vector store integration |
| `langchain-openai` | 0.1.23 | OpenAI model integration |
| `langchain-community` | 0.2.16 | Community tools and utilities |
| `python-dotenv` | Latest | Environment variable management |

### Data Processing
- **Pandas**: Structured data manipulation
- **RecursiveCharacterTextSplitter**: Text chunking for vector storage
- **OpenAI Embeddings**: Text embedding generation

## 🚀 Installation

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher**
- **Git** for cloning the repository
- **API Keys** for required services (see Configuration section)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/sefineh-ai/langchain-Agent.git
cd langchain-Agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run the application
python main.py

# Or run the data processor
python agent/store.py
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# AstraDB Configuration
ASTRA_DB_APPLICATION_TOKEN=your_astra_db_token_here
ASTRA_DB_API_ENDPOINT=your_astra_db_endpoint_here
ASTRA_DB_NAMESPACE=your_namespace_here

# Google Search Configuration
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_custom_search_engine_id_here

# Optional: Serper API (alternative search)
SERPER_API_KEY=your_serper_api_key_here
```

### API Key Setup

1. **OpenAI API Key**
   - Visit [OpenAI Platform](https://platform.openai.com/)
   - Create an account and generate an API key
   - Add the key to your `.env` file

2. **AstraDB Configuration**
   - Sign up at [DataStax Astra](https://www.datastax.com/products/datastax-astra)
   - Create a new database and get your credentials
   - Configure the vector store collection

3. **Google Search API**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Enable Custom Search API
   - Create API credentials and Custom Search Engine

## 📖 Usage

### Basic Usage

```bash
# Start the application
python main.py

# Follow the prompts:
# Do you have any questions? (y/n): y
# What is your question?
# ---: What scholarships are available for international students?
```

### Example Interactions

```python
# Example questions you can ask:
"What scholarships are available for computer science students?"
"How do I apply for the Presidential Scholarship?"
"What are the requirements for international student scholarships?"
"Are there any scholarships for graduate students?"
"What is the deadline for fall semester scholarships?"
```

### Programmatic Usage

```python
from agent.agent import SFResourceMatching

# Initialize the agent
agent = SFResourceMatching()

# Process and store data
agent.process_and_store_documents(text_data)
agent.process_and_store_structured_data(csv_file_path)

# Query the system
results = agent.query_vectorstore("international student scholarships")
print(results)
```

## 📚 API Reference

### SFResourceMatching Class

#### Core Methods

```python
class SFResourceMatching:
    def __init__(self) -> None:
        """Initialize the scholarship agent with configurations."""
        
    def process_and_store_documents(self, text_data: str) -> None:
        """Process and store unstructured text data in vector store."""
        
    def process_and_store_structured_data(self, csv_file_path: str) -> None:
        """Process and store structured CSV data in vector store."""
        
    def query_vectorstore(self, query: str, top_k: int = 5) -> List[Document]:
        """Search the vector store for relevant documents."""
        
    def create_tools(self) -> List[Tool]:
        """Create tools for the AI agent (Google Search, etc.)."""
        
    def create_agent(self, llm_with_tools) -> Agent:
        """Create the LangChain agent with tools."""
```

#### Data Processing Methods

```python
def clean_data(self, data: str) -> str:
    """Clean and preprocess text data."""
    
def read_text_data(self, file_path: str) -> str:
    """Read text data from file."""
    
def read_csv(self, csv_file_path: str) -> str:
    """Read and process CSV data."""
```

### VectorStoreManager Class

```python
class VectorStoreManager:
    def __init__(self) -> None:
        """Initialize vector store manager."""
        
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to vector store."""
        
    def similarity_search(self, query: str) -> List[Document]:
        """Perform similarity search."""
```

## 📊 Data Sources

### Structured Data (`structured_scholarship_data.csv`)
- Scholarship names and descriptions
- Eligibility requirements
- Award amounts and deadlines
- Application procedures
- Contact information

### Unstructured Data (`unstructured_scholarship_data.txt`)
- Detailed scholarship descriptions
- Application guidelines
- Frequently asked questions
- Success stories and testimonials
- Policy documents

### Web Search Integration
- Real-time scholarship updates
- Additional university resources
- External scholarship opportunities
- News and announcements

## 🤝 Contributing

We welcome contributions to improve the SFSU Scholarship AI Agent! Please follow these guidelines:

### Development Setup

1. **Fork the Repository**
   ```bash
   git clone https://github.com/your-username/langchain-Agent.git
   cd langchain-Agent
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/improvement
   ```

3. **Make Changes**
   - Follow PEP 8 style guidelines
   - Add type hints to new functions
   - Include docstrings for new methods
   - Update tests if applicable

4. **Test Your Changes**
   ```bash
   # Run the application
   python main.py
   
   # Test data processing
   python agent/store.py
   ```

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "Add feature: description of changes"
   git push origin feature/improvement
   ```

6. **Create Pull Request**
   - Provide clear description of changes
   - Include any new dependencies
   - Update documentation if needed

### Contribution Guidelines

- **Code Quality**: Follow Python best practices and PEP 8
- **Documentation**: Add docstrings and update README as needed
- **Testing**: Ensure your changes don't break existing functionality
- **Data Privacy**: Don't commit sensitive data or API keys
- **Performance**: Consider the impact on response times

## 📞 Support

### Getting Help

- **Documentation**: Check this README and code comments
- **Issues**: [GitHub Issues](https://github.com/sefineh-ai/langchain-Agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sefineh-ai/langchain-Agent/discussions)

### Common Issues

#### API Key Errors
```bash
# Ensure all required API keys are set in .env file
OPENAI_API_KEY=your_key_here
ASTRA_DB_APPLICATION_TOKEN=your_token_here
GOOGLE_API_KEY=your_key_here
```

#### Vector Store Connection Issues
```bash
# Check AstraDB configuration
ASTRA_DB_API_ENDPOINT=https://your-database-id-us-east1.apps.astra.datastax.com
ASTRA_DB_NAMESPACE=your_namespace
```

#### Data Processing Errors
```bash
# Ensure data files exist
ls data/structured_scholarship_data.csv
ls data/unstructured_scholarship_data.txt
```

### Community Resources

- **LangChain Documentation**: [langchain.com](https://www.langchain.com/)
- **AstraDB Documentation**: [docs.datastax.com](https://docs.datastax.com/)
- **OpenAI API Documentation**: [platform.openai.com](https://platform.openai.com/docs)
- **Google Search API**: [developers.google.com](https://developers.google.com/custom-search)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

- **San Francisco University** - for providing scholarship data and support
- **LangChain Community** - for the excellent AI agent framework
- **DataStax Astra** - for the vector database infrastructure
- **OpenAI** - for the GPT-4 language model
- **Google** - for the search API capabilities

---

**Made with ❤️ for Students**

*Empowering students worldwide to find and access scholarship opportunities through intelligent AI assistance*
