# BuildABot - Agentic Chatbot Platform

A customizable platform for building and managing AI agents using FastAPI and OpenAI.

## Features

- 🤖 **Customizable Agents**: Create and manage multiple AI agents with different personalities and capabilities
- 💬 **Real-time Chat**: Interactive chat interface for communicating with agents
- 🔧 **Agent Management**: Full CRUD operations for agents (Create, Read, Update, Delete)
- 🎯 **Conversation Tracking**: Maintain conversation history across sessions
- 🚀 **FastAPI Backend**: High-performance API with automatic documentation
- 🔌 **OpenAI Integration**: Powered by OpenAI's GPT models

## Quick Start

### 1. Setup Environment

```bash
# Clone or create the project directory
cd buildabot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy the example environment file
cp env.example .env

# Edit .env and add your API keys
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
QDRANT_URL=https://your-cluster-id.us-east-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here
```

### 3. Setup Qdrant Cloud

#### Option A: Qdrant Cloud (Recommended)
1. **Sign up** at [cloud.qdrant.io](https://cloud.qdrant.io)
2. **Create a cluster** and get your URL and API key
3. **Update .env** with your Qdrant Cloud credentials:
   ```bash
   QDRANT_URL=https://your-cluster-id.us-east-1-0.aws.cloud.qdrant.io:6333
   QDRANT_API_KEY=your_qdrant_api_key_here
   ```
4. **Setup collection**:
   ```bash
   python setup_qdrant.py
   ```

#### Option B: Local Qdrant (Development)
```bash
# Start Qdrant using Docker
docker-compose up -d

# Wait for Qdrant to start (about 30 seconds)
# Then setup the collection
python setup_qdrant.py
```

### 4. Run the Application

```bash
# Start the FastAPI server
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 5. Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Chat
- `POST /chat` - Send a message to an agent
- `GET /conversations/{conversation_id}` - Get conversation history

### Agents
- `GET /agents` - List all agents
- `GET /agents/{agent_id}` - Get specific agent
- `POST /agents` - Create new agent
- `PUT /agents/{agent_id}` - Update agent
- `DELETE /agents/{agent_id}` - Delete agent

### Tools & Pipelines
- `GET /tools` - List available tools
- `GET /pipelines` - List all pipelines
- `POST /pipelines` - Create new pipeline
- `GET /pipelines/{id}` - Get specific pipeline
- `PUT /pipelines/{id}` - Update pipeline
- `DELETE /pipelines/{id}` - Delete pipeline
- `POST /pipelines/{id}/execute` - Execute pipeline

### Vector Database
- `POST /vector-db/add` - Add document to knowledge base
- `GET /vector-db/search` - Search knowledge base
- `GET /vector-db/list` - List all documents

### System
- `GET /` - API information
- `GET /health` - Health check

## Usage Examples

### 1. Basic Chat

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, how are you?",
    "agent_id": "default"
  }'
```

### 2. Create Custom Agent

```bash
curl -X POST "http://localhost:8000/agents" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Code Assistant",
    "description": "An AI assistant specialized in programming help",
    "system_prompt": "You are a helpful programming assistant. Provide clear code examples and explanations.",
    "model": "gpt-3.5-turbo",
    "temperature": 0.3,
    "max_tokens": 1500
  }'
```

### 3. Chat with Custom Agent

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I create a Python function?",
    "agent_id": "agent_20231201_143022"
  }'
```

## Agent Configuration

Each agent can be customized with:

- **Name**: Display name for the agent
- **Description**: Brief description of the agent's purpose
- **System Prompt**: The core personality and behavior instructions
- **Model**: OpenAI model to use (gpt-3.5-turbo, gpt-4, etc.)
- **Temperature**: Creativity level (0.0 to 1.0)
- **Max Tokens**: Maximum response length

## Project Structure

```
buildabot/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── env.example         # Environment variables template
├── README.md           # This file
└── venv/               # Virtual environment
```

## Development

### Adding New Features

1. **Agent Types**: Extend the Agent model to support different agent types
2. **Memory Systems**: Add persistent memory for agents
3. **Tools Integration**: Allow agents to use external tools and APIs
4. **Multi-agent Conversations**: Support conversations between multiple agents
5. **Web Interface**: Add a frontend web interface

### Database Integration

Currently using in-memory storage. For production, integrate with:
- PostgreSQL with SQLAlchemy
- MongoDB with Motor
- Redis for caching

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details
