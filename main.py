from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from datetime import datetime
from agents.preset_agents import PRESET_AGENTS, get_preset_agent_config, get_all_preset_names
from agents.agent_router import AgentRouter
from agents.agent_pipeline import AgentPipeline, PipelineStep, PipelineExecutor
from tools import BaseTool, WebSearchTool, VectorDBTool
from tools.base_tool import ToolConfig
from db import init_db, get_session, AgentORM
import tiktoken
import uuid

# Load environment variables
load_dotenv()

# Knowledge Base Processing Functions
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end == len(text):
            break
            
        start = end - overlap
    
    return chunks

async def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Create OpenAI embeddings for text chunks using the latest model"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-large",  # Latest and most powerful embedding model
            input=texts,
            dimensions=1536  # Optional: reduce dimensions for faster queries (default is 3072)
        )
        return [embedding.embedding for embedding in response.data]
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return []

async def store_knowledge_in_qdrant(agent_id: str, documents: List[Dict[str, Any]], vector_db_tool):
    """Process documents, create embeddings, and store in Qdrant"""
    try:
        all_chunks = []
        all_metadata = []
        
        for doc in documents:
            title = doc.get('title', 'Untitled')
            content = doc.get('content', '')
            
            # Chunk the document
            chunks = chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'agent_id': agent_id,
                    'document_title': title,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'timestamp': datetime.now().isoformat()
                })
        
        if not all_chunks:
            return True
        
        # Create embeddings
        embeddings = await create_embeddings(all_chunks)
        
        if not embeddings:
            return False
        
        # Store in Qdrant
        for chunk, embedding, metadata in zip(all_chunks, embeddings, all_metadata):
            # Use the vector DB tool to store the chunk
            point_id = str(uuid.uuid4())
            
            # Store the embedding with metadata using the vector DB tool
            result = await vector_db_tool._add_document(
                title=metadata['document_title'],
                content=chunk,
                agent_id=agent_id
            )
        
        return True
        
    except Exception as e:
        print(f"Error storing knowledge in Qdrant: {e}")
        return False

# Initialize FastAPI app
app = FastAPI(
    title="BuildABot - Agentic Chatbot Platform",
    description="A customizable platform for building and managing AI agents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize Agent Router for intelligent routing
agent_router = AgentRouter(openai_client)

# Initialize available tools
available_tools = {
    "web_search": WebSearchTool(ToolConfig(
        name="web_search",
        description="Search the web for current information",
        parameters={
            "search_depth": "basic",
            "include_answer": True,
            "max_results": 5
        }
    )),
    "vector_db": VectorDBTool(ToolConfig(
        name="vector_db",
        description="Search through knowledge base using semantic similarity with Qdrant Cloud",
        parameters={
            "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
            "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
            "collection_name": "knowledge_base",
            "vector_size": 1536,  # OpenAI embedding size
            "max_results": 5
        }
    ))
}

# Initialize pipeline executor
pipeline_executor = PipelineExecutor(available_tools)

# Pydantic models
class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str

class ChatRequest(BaseModel):
    message: str
    agent_id: Optional[str] = None
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    conversation_id: str
    timestamp: str
    agent_reasoning: Optional[str] = None
    confidence: Optional[float] = None
    sources: Optional[List[Dict[str, Any]]] = None  # URLs and sources used

class Agent(BaseModel):
    id: str
    name: str
    description: str
    system_prompt: str
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    pipeline_id: Optional[str] = None  # Legacy reference to agent pipeline
    workflow_json: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str

class AgentCreateRequest(BaseModel):
    name: str
    description: str
    system_prompt: str
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    pipeline_id: Optional[str] = None
    workflow_json: Optional[Dict[str, Any]] = None
    knowledge_documents: Optional[List[Dict[str, Any]]] = None

class AgentUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    pipeline_id: Optional[str] = None
    workflow_json: Optional[Dict[str, Any]] = None

# In-memory storage for conversations and transient pipelines
conversations_storage: Dict[str, List[ChatMessage]] = {}
pipelines_storage: Dict[str, AgentPipeline] = {}

# Initialize database tables
init_db()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return {
        "message": "Welcome to BuildABot - Agentic Chatbot Platform",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "agents": "/agents",
            "health": "/health",
            "web_interface": "/static/index.html",
            "manage_agents": "/static/index.html#agents"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint that processes messages using OpenAI with intelligent agent routing
    """
    try:
        # Load agents from DB
        with get_session() as session:
            db_agents = session.query(AgentORM).filter(AgentORM.is_deleted == False).all()
        # Check if any agents exist
        if not db_agents:
            # No agents created yet - use raw LLM response
            agent_id = None
            agent_name = "Raw LLM"
            agent_reasoning = "No custom agents created yet - using raw LLM response"
            confidence = 1.0
            
            # Get or create conversation
            conversation_id = request.conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if conversation_id not in conversations_storage:
                conversations_storage[conversation_id] = []
            
            # Add user message to conversation
            conversations_storage[conversation_id].append(
                ChatMessage(role="user", content=request.message)
            )
            
            # Prepare messages for OpenAI (no system prompt - raw LLM)
            messages = [{"role": msg.role, "content": msg.content} for msg in conversations_storage[conversation_id]]
            
            # Call OpenAI API with default settings
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            assistant_response = response.choices[0].message.content
            
            # Add assistant response to conversation
            conversations_storage[conversation_id].append(
                ChatMessage(role="assistant", content=assistant_response)
            )
            
            return ChatResponse(
                response=assistant_response,
                agent_id=agent_id,
                agent_name=agent_name,
                conversation_id=conversation_id,
                timestamp=datetime.now().isoformat(),
                agent_reasoning=agent_reasoning,
                confidence=confidence
            )
        
        # Map DB agents to simple structures for router
        available_agents = [
            {
                "id": a.id,
                "name": a.name,
                "description": a.description
            }
            for a in db_agents
        ]

        # Intelligent agent selection (only if agents exist)
        if request.agent_id:
            # User specified an agent
            agent_id = request.agent_id
            agent_row = next((a for a in db_agents if a.id == agent_id), None)
            if not agent_row:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
            agent_name = agent_row.name
            agent_reasoning = f"Using user-selected agent: {agent_row.name}"
            confidence = 1.0
        else:
            # Intelligent routing - analyze query to select best agent
            if available_agents:
                # Use AI to analyze query and select best agent
                agent_id, analysis = agent_router.get_best_agent(request.message, available_agents)
                
                if agent_id:
                    agent_row = next((a for a in db_agents if a.id == agent_id), None)
                    agent_name = agent_row.name if agent_row else "Unknown Agent"
                    agent_reasoning = analysis.get("reasoning", f"Intelligently selected: {agent_name}")
                    confidence = analysis.get("confidence", 0.8)
                else:
                    # Use raw LLM for simple/general queries
                    agent_id = None
                    agent_name = "Raw LLM"
                    agent_reasoning = analysis.get("reasoning", "Simple query - using raw LLM for better response")
                    confidence = analysis.get("confidence", 0.9)
                    agent_row = None
            else:
                # This should not happen as we check above, but just in case
                agent_id = None
                agent_name = "Raw LLM"
                agent_reasoning = "No agents available - using raw LLM response"
                confidence = 1.0
                agent_row = None
        
        # Get or create conversation
        conversation_id = request.conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if conversation_id not in conversations_storage:
            conversations_storage[conversation_id] = []
        
        # Add user message to conversation
        conversations_storage[conversation_id].append(
            ChatMessage(role="user", content=request.message)
        )
        
        # Check if agent has a pipeline and execute it
        sources = []
        knowledge_base_content = []
        pipeline_context = {'agent_id': agent_id} if agent_id else {}
        
        # If agent has a saved workflow, execute it
        if agent_row and agent_row.workflow_json and isinstance(agent_row.workflow_json, dict):
            steps_json = agent_row.workflow_json.get("steps", [])
            steps: List[PipelineStep] = []
            for i, s in enumerate(steps_json):
                # Map frontend field names to backend field names
                tool_name = s.get("tool_name") or s.get("tool")  # Support both formats
                step_id = str(s.get("step_id") or s.get("id", f"step_{i}"))
                
                steps.append(PipelineStep(
                    step_id=step_id,
                    tool_name=tool_name,
                    tool_config=s.get("tool_config", s.get("config", s.get("parameters", {}))),
                    input_mapping=s.get("input_mapping", {}),
                    output_mapping=s.get("output_mapping", {}),
                    condition=s.get("condition"),
                    order=int(s.get("order", i))
                ))

            if steps:
                pipeline = AgentPipeline(
                    id=f"wf_{agent_row.id}",
                    name=f"Workflow for {agent_row.name}",
                    description="Agent workflow",
                    steps=steps,
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat(),
                )
                print(f"🔍 Executing pipeline for agent {agent_row.name} with context: {pipeline_context}")
                pipeline_result = await pipeline_executor.execute_pipeline(
                    pipeline, request.message, pipeline_context
                )
                print(f"📋 Pipeline result: {pipeline_result}")
                
                # Extract search results and sources from pipeline
                knowledge_base_content = []
                for step_result in pipeline_result.get("results", []):
                    print(f"🔧 Step result: {step_result}")
                    if step_result.get("success") and step_result.get("metadata", {}).get("sources"):
                        sources.extend(step_result["metadata"]["sources"])
                        print(f"📚 Added sources: {step_result['metadata']['sources']}")
                    
                    # Extract knowledge base content from vector search results
                    if step_result.get("tool_name") == "vector_db" and step_result.get("success"):
                        step_data = step_result.get("data", {})
                        documents = step_data.get("documents", [])
                        for doc in documents:
                            knowledge_base_content.append({
                                "title": doc.get("title", ""),
                                "content": doc.get("content", ""),
                                "score": doc.get("score", 0)
                            })
                        print(f"📖 Extracted {len(documents)} knowledge base documents")
                
                pipeline_context = pipeline_result.get("final_context", {})
        
        # Prepare messages for OpenAI
        if agent_row:
            # Use agent's system prompt and settings
            system_prompt = agent_row.system_prompt
            
            # Add knowledge base content to system prompt
            if knowledge_base_content:
                system_prompt += "\n\n=== KNOWLEDGE BASE CONTEXT ===\n"
                system_prompt += "Use the following information from the knowledge base to answer the user's question. "
                system_prompt += "Reference the sources when relevant:\n\n"
                
                for i, doc in enumerate(knowledge_base_content, 1):
                    system_prompt += f"Document {i}: {doc['title']}\n"
                    system_prompt += f"Content: {doc['content']}\n"
                    system_prompt += f"Relevance Score: {doc['score']:.3f}\n\n"
                
                print(f"📚 Added {len(knowledge_base_content)} knowledge base documents to LLM context")
            
            # If pipeline was executed, add context to system prompt
            if pipeline_context:
                context_info = "\n\nAdditional context from tools:\n"
                for key, value in pipeline_context.items():
                    context_info += f"- {key}: {value}\n"
                system_prompt += context_info
            
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend([{"role": msg.role, "content": msg.content} for msg in conversations_storage[conversation_id]])
            
            # Call OpenAI API with agent settings
            response = openai_client.chat.completions.create(
                model=agent_row.model,
                messages=messages,
                temperature=agent_row.temperature,
                max_tokens=agent_row.max_tokens
            )
        else:
            # Fallback to raw LLM (should not happen but just in case)
            messages = [{"role": msg.role, "content": msg.content} for msg in conversations_storage[conversation_id]]
            
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
        
        assistant_response = response.choices[0].message.content
        
        # Add assistant response to conversation
        conversations_storage[conversation_id].append(
            ChatMessage(role="assistant", content=assistant_response)
        )
        
        # Update last_used_at if an agent was used
        if agent_row:
            with get_session() as session:
                db_agent = session.get(AgentORM, agent_row.id)
                if db_agent:
                    db_agent.last_used_at = datetime.utcnow()

        return ChatResponse(
            response=assistant_response,
            agent_id=agent_id,
            agent_name=agent_name,
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat(),
            agent_reasoning=agent_reasoning,
            confidence=confidence,
            sources=sources if sources else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.get("/agents", response_model=List[Agent])
async def get_agents():
    """Get all available agents from DB"""
    with get_session() as session:
        rows = session.query(AgentORM).filter(AgentORM.is_deleted == False).all()
    return [
        Agent(
            id=r.id,
            name=r.name,
            description=r.description,
            system_prompt=r.system_prompt,
            model=r.model,
            temperature=r.temperature,
            max_tokens=r.max_tokens,
            pipeline_id=r.pipeline_id,
            workflow_json=r.workflow_json,
            created_at=r.created_at.isoformat(),
            updated_at=r.updated_at.isoformat(),
        )
        for r in rows
    ]

@app.get("/agents/{agent_id}", response_model=Agent)
async def get_agent(agent_id: str):
    """Get a specific agent by ID"""
    with get_session() as session:
        r = session.get(AgentORM, agent_id)
    if not r or r.is_deleted:
        raise HTTPException(status_code=404, detail="Agent not found")
    return Agent(
        id=r.id,
        name=r.name,
        description=r.description,
        system_prompt=r.system_prompt,
        model=r.model,
        temperature=r.temperature,
        max_tokens=r.max_tokens,
        pipeline_id=r.pipeline_id,
        workflow_json=r.workflow_json,
        created_at=r.created_at.isoformat(),
        updated_at=r.updated_at.isoformat(),
    )

@app.get("/agents/{agent_id}/details")
async def get_agent_details(agent_id: str):
    """Get detailed agent information including pipeline"""
    with get_session() as session:
        r = session.get(AgentORM, agent_id)
    if not r or r.is_deleted:
        raise HTTPException(status_code=404, detail="Agent not found")
    agent = Agent(
        id=r.id,
        name=r.name,
        description=r.description,
        system_prompt=r.system_prompt,
        model=r.model,
        temperature=r.temperature,
        max_tokens=r.max_tokens,
        pipeline_id=r.pipeline_id,
        workflow_json=r.workflow_json,
        created_at=r.created_at.isoformat(),
        updated_at=r.updated_at.isoformat(),
    )
    
    # Get pipeline details if agent has one
    pipeline_details = None
    if agent.pipeline_id:
        pipeline = pipelines_storage.get(agent.pipeline_id)
        if pipeline:
            pipeline_details = {
                "id": pipeline.id,
                "name": pipeline.name,
                "description": pipeline.description,
                "steps": [
                    {
                        "step_id": step.step_id,
                        "tool_name": step.tool_name,
                        "order": step.order
                    }
                    for step in pipeline.steps
                ]
            }
    
    return {
        "agent": agent,
        "pipeline": pipeline_details
    }

@app.post("/agents", response_model=Agent)
async def create_agent(request: AgentCreateRequest):
    """Create a new agent in DB"""
    agent_id = f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create agent in database first
    with get_session() as session:
        row = AgentORM(
            id=agent_id,
            name=request.name,
            description=request.description,
            system_prompt=request.system_prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            pipeline_id=request.pipeline_id,
            workflow_json=request.workflow_json,
        )
        session.add(row)
    
    # Process knowledge documents if provided
    if request.knowledge_documents and len(request.knowledge_documents) > 0:
        try:
            # Initialize vector DB tool for storing embeddings
            vector_db_tool = available_tools["vector_db"]
            
            # Store knowledge documents in Qdrant
            success = await store_knowledge_in_qdrant(
                agent_id=agent_id,
                documents=request.knowledge_documents,
                vector_db_tool=vector_db_tool
            )
            
            if not success:
                print(f"Warning: Failed to store knowledge documents for agent {agent_id}")
        except Exception as e:
            print(f"Error processing knowledge documents: {e}")
    
    return Agent(
        id=agent_id,
        name=request.name,
        description=request.description,
        system_prompt=request.system_prompt,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        pipeline_id=request.pipeline_id,
        workflow_json=request.workflow_json,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )

@app.put("/agents/{agent_id}", response_model=Agent)
async def update_agent(agent_id: str, request: AgentUpdateRequest):
    """Update an existing agent in DB"""
    with get_session() as session:
        row = session.get(AgentORM, agent_id)
        if not row or row.is_deleted:
            raise HTTPException(status_code=404, detail="Agent not found")
        if request.name is not None:
            row.name = request.name
        if request.description is not None:
            row.description = request.description
        if request.system_prompt is not None:
            row.system_prompt = request.system_prompt
        if request.model is not None:
            row.model = request.model
        if request.temperature is not None:
            row.temperature = request.temperature
        if request.max_tokens is not None:
            row.max_tokens = request.max_tokens
        if request.pipeline_id is not None:
            row.pipeline_id = request.pipeline_id
        if request.workflow_json is not None:
            row.workflow_json = request.workflow_json
    return await get_agent(agent_id)

@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str):
    """Soft delete an agent in DB"""
    if agent_id == "default":
        raise HTTPException(status_code=400, detail="Cannot delete default agent")
    with get_session() as session:
        row = session.get(AgentORM, agent_id)
        if not row or row.is_deleted:
            raise HTTPException(status_code=404, detail="Agent not found")
        row.is_deleted = True
    return {"message": f"Agent {agent_id} deleted successfully"}

@app.get("/agents/{agent_id}/workflow")
async def get_agent_workflow(agent_id: str):
    with get_session() as session:
        row = session.get(AgentORM, agent_id)
    if not row or row.is_deleted:
        raise HTTPException(status_code=404, detail="Agent not found")
    return row.workflow_json or {"steps": []}

@app.put("/agents/{agent_id}/workflow")
async def update_agent_workflow(agent_id: str, workflow: Dict[str, Any]):
    with get_session() as session:
        row = session.get(AgentORM, agent_id)
        if not row or row.is_deleted:
            raise HTTPException(status_code=404, detail="Agent not found")
        row.workflow_json = workflow
    return {"status": "ok"}

@app.get("/agents/last-used", response_model=Optional[Agent])
async def get_last_used_agent():
    """Return the most recently used agent, if any"""
    with get_session() as session:
        row = (
            session.query(AgentORM)
            .filter(AgentORM.is_deleted == False, AgentORM.last_used_at.isnot(None))
            .order_by(AgentORM.last_used_at.desc())
            .first()
        )
    if not row:
        return None
    return Agent(
        id=row.id,
        name=row.name,
        description=row.description,
        system_prompt=row.system_prompt,
        model=row.model,
        temperature=row.temperature,
        max_tokens=row.max_tokens,
        pipeline_id=row.pipeline_id,
        workflow_json=row.workflow_json,
        created_at=row.created_at.isoformat(),
        updated_at=row.updated_at.isoformat(),
    )

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    conversation = conversations_storage.get(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation_id": conversation_id, "messages": conversation}

@app.get("/preset-agents")
async def get_preset_agents():
    """Get all available preset agent configurations"""
    return {
        "presets": PRESET_AGENTS,
        "descriptions": {name: config["description"] for name, config in PRESET_AGENTS.items()}
    }


@app.post("/agents/from-preset/{preset_name}")
async def create_agent_from_preset(preset_name: str):
    """Create a new agent from a preset configuration"""
    try:
        config = get_preset_agent_config(preset_name)
        agent_id = f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Persist directly to DB
        with get_session() as session:
            row = AgentORM(
                id=agent_id,
                name=config["name"],
                description=config["description"],
                system_prompt=config["system_prompt"],
                model=config["model"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
            )
            session.add(row)
        
        return await get_agent(agent_id)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# Pipeline endpoints
@app.get("/tools")
async def get_available_tools():
    """Get all available tools"""
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "schema": tool.get_schema()
            }
            for tool in available_tools.values()
        ]
    }

@app.get("/pipelines")
async def get_pipelines():
    """Get all agent pipelines"""
    return list(pipelines_storage.values())

@app.get("/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    """Get a specific pipeline"""
    pipeline = pipelines_storage.get(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return pipeline

@app.post("/pipelines")
async def create_pipeline(pipeline_data: dict):
    """Create a new agent pipeline"""
    pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Validate pipeline data
    required_fields = ["name", "description", "steps"]
    for field in required_fields:
        if field not in pipeline_data:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
    
    # Create pipeline steps
    steps = []
    for i, step_data in enumerate(pipeline_data["steps"]):
        step = PipelineStep(
            step_id=step_data.get("step_id", f"step_{i}"),
            tool_name=step_data["tool_name"],
            tool_config=step_data.get("tool_config", {}),
            input_mapping=step_data.get("input_mapping", {}),
            output_mapping=step_data.get("output_mapping", {}),
            condition=step_data.get("condition"),
            order=step_data.get("order", i)
        )
        steps.append(step)
    
    pipeline = AgentPipeline(
        id=pipeline_id,
        name=pipeline_data["name"],
        description=pipeline_data["description"],
        steps=steps,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )
    
    pipelines_storage[pipeline_id] = pipeline
    return pipeline

@app.put("/pipelines/{pipeline_id}")
async def update_pipeline(pipeline_id: str, pipeline_data: dict):
    """Update an existing pipeline"""
    pipeline = pipelines_storage.get(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    # Update pipeline fields
    if "name" in pipeline_data:
        pipeline.name = pipeline_data["name"]
    if "description" in pipeline_data:
        pipeline.description = pipeline_data["description"]
    if "steps" in pipeline_data:
        # Recreate steps
        steps = []
        for i, step_data in enumerate(pipeline_data["steps"]):
            step = PipelineStep(
                step_id=step_data.get("step_id", f"step_{i}"),
                tool_name=step_data["tool_name"],
                tool_config=step_data.get("tool_config", {}),
                input_mapping=step_data.get("input_mapping", {}),
                output_mapping=step_data.get("output_mapping", {}),
                condition=step_data.get("condition"),
                order=step_data.get("order", i)
            )
            steps.append(step)
        pipeline.steps = steps
    
    pipeline.updated_at = datetime.now().isoformat()
    return pipeline

@app.delete("/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    """Delete a pipeline"""
    if pipeline_id not in pipelines_storage:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    del pipelines_storage[pipeline_id]
    return {"message": f"Pipeline {pipeline_id} deleted successfully"}

@app.post("/pipelines/{pipeline_id}/execute")
async def execute_pipeline(pipeline_id: str, request_data: dict):
    """Execute a pipeline"""
    pipeline = pipelines_storage.get(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    query = request_data.get("query", "")
    context = request_data.get("context", {})
    
    result = await pipeline_executor.execute_pipeline(pipeline, query, context)
    return result

# Vector DB management endpoints
@app.post("/vector-db/add")
async def add_to_vector_db(request_data: dict):
    """Add a document to the vector database"""
    title = request_data.get("title", "")
    content = request_data.get("content", "")
    
    if not title and not content:
        raise HTTPException(status_code=400, detail="Title or content is required")
    
    agent_id = request_data.get("agent_id")
    vector_tool = available_tools["vector_db"]
    result = await vector_tool.execute(
        title or content,
        context={"operation": "add", "title": title, "content": content, "agent_id": agent_id}
    )
    
    if result.success:
        return result.data
    else:
        raise HTTPException(status_code=500, detail=result.error)

@app.post("/vector-db/search")
async def search_vector_db(request_data: dict):
    """Search the vector database"""
    query = request_data.get("query", "")
    limit = request_data.get("limit", 5)
    
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required")
    
    agent_id = request_data.get("agent_id")
    vector_tool = available_tools["vector_db"]
    result = await vector_tool.execute(query, context={"operation": "search", "limit": limit, "agent_id": agent_id})
    
    if result.success:
        return result.data
    else:
        raise HTTPException(status_code=500, detail=result.error)

@app.get("/vector-db/list")
async def list_vector_db_documents(agent_id: Optional[str] = None):
    """List documents in the vector database"""
    vector_tool = available_tools["vector_db"]
    result = await vector_tool.execute("", context={"operation": "list", "agent_id": agent_id})
    
    if result.success:
        return result.data
    else:
        raise HTTPException(status_code=500, detail=result.error)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
