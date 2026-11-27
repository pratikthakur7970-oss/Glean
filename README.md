# 🚀 RAG Docker Azure Pipeline

### AI-Powered Retrieval-Augmented Generation (RAG) Application

An end-to-end **Retrieval-Augmented Generation (RAG) Pipeline** built using Python, FastAPI, LangChain, Vector Databases, Docker, and Microsoft Azure.

This project demonstrates how to build a modular RAG application that processes documents, generates embeddings, stores vector representations, retrieves relevant information, and generates context-aware AI responses.

---

## 🌟 Key Features

- 📄 **Document Loading:** Load and process PDF documents.
- ✂️ **Text Splitting:** Split documents into manageable chunks with configurable overlap.
- 🧠 **Embeddings:** Convert text into vector representations.
- 🗄️ **Vector Database:** Store and retrieve document embeddings.
- 🔍 **Semantic Search:** Retrieve relevant document chunks based on user queries.
- 🤖 **AI Response Generation:** Generate context-aware responses using retrieved information.
- ⚡ **FastAPI:** Expose the RAG pipeline through API endpoints.
- 🐳 **Docker:** Containerize the application for consistent deployment.
- ☁️ **Azure Deployment:** Docker-based deployment configuration for Microsoft Azure.

## 🏗️ Project Architecture

```text
User Query
    |
    v
FastAPI Application
    |
    v
RAG Pipeline
    |
    v
Query Embeddings
    |
    v
Vector Database
    |
    v
Document Retrieval
    |
    v
Context + User Query
    |
    v
LLM Response Generation
    |
    v
Final Answer
```

## 📁 Project Structure

```text
RAG_Docker_Azure_Pipeline/
│
├── data/                 # Document storage
├── embeddings/           # Embedding modules
├── generators/           # LLM response generation
├── loaders/              # Document loading modules
├── splitters/            # Text splitting modules
├── static/               # Static assets
├── templates/            # HTML templates
├── vectorstores/         # Vector database modules
│
├── config.py             # Application configuration
├── main.py               # FastAPI application entry point
├── pipeline.py           # RAG pipeline implementation
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
└── README.md             # Project documentation
```

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| FastAPI | Backend API |
| LangChain | RAG orchestration |
| Embeddings | Text vectorization |
| Vector Database | Vector storage and retrieval |
| LLM | AI response generation |
| Docker | Containerization |
| Microsoft Azure | Cloud deployment |
| Uvicorn | ASGI server |

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

```bash
cd RAG_Docker_Azure_Pipeline
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root and add the API keys and configuration variables required by `config.py`.

**Important:** Never upload your `.env` file or expose API keys on GitHub.

### 5. Run the Application

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open the API documentation:

```text
http://localhost:8000/docs
```

## 🐳 Docker Deployment

### 1. Build Docker Image

```bash
docker build -t rag-azure-app .
```

### 2. Run Docker Container

```bash
docker run -d -p 8000:8000 --env-file .env --name rag-container rag-azure-app
```

### 3. Check Running Containers

```bash
docker ps
```

### 4. View Container Logs

```bash
docker logs rag-container
```

### 5. Access Application

```text
http://localhost:8000/docs
```

## ☁️ Microsoft Azure Deployment

The project includes a Dockerfile configured for container-based deployment.

Deployment workflow:

1. Build the Docker image.
2. Test the container locally.
3. Push the image to Azure Container Registry (ACR).
4. Deploy the container using Azure Container Apps or Azure App Service.
5. Configure environment variables and required secrets.
6. Access the deployed application through its Azure endpoint.

## 🎯 Learning Outcomes

Through this project, I explored:

- Understanding Retrieval-Augmented Generation.
- Building modular Python applications using OOP concepts.
- Working with document processing and text chunking.
- Implementing embeddings and vector search.
- Developing APIs using FastAPI.
- Containerizing applications using Docker.
- Understanding cloud deployment workflows with Microsoft Azure.

## 🔮 Future Improvements

- Add authentication and user management.
- Support additional document formats.
- Improve retrieval accuracy.
- Add chat history and conversational memory.
- Implement monitoring and logging.
- Add automated CI/CD deployment.

## 👨‍💻 Author

**Pratik Thakur**

Aspiring Data Scientist | Python Developer | Generative AI Enthusiast

Interested in Machine Learning, Generative AI, RAG Systems, and Cloud Technologies.

---

⭐ If you find this project interesting, consider giving it a star!

**Built with ❤️ using Python, AI & Cloud Technologies.**
