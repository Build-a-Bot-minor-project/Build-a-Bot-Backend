"""
Vector Database Tool using Qdrant DB
"""

import os
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from .base_tool import BaseTool, ToolConfig, ToolResult

class VectorDBTool(BaseTool):
    """Tool for vector database operations using Qdrant"""
    
    def __init__(self, config: ToolConfig):
        super().__init__(config)
        
        # Qdrant configuration
        self.qdrant_url = config.parameters.get("qdrant_url", os.getenv("QDRANT_URL", "http://localhost:6333"))
        self.qdrant_api_key = config.parameters.get("qdrant_api_key", os.getenv("QDRANT_API_KEY"))
        self.collection_name = config.parameters.get("collection_name", "knowledge_base")
        self.vector_size = config.parameters.get("vector_size", 1536)  # OpenAI embedding size
        self.max_results = config.parameters.get("max_results", 5)
        
        # Initialize Qdrant client
        try:
            if self.qdrant_api_key:
                # Use API key for Qdrant Cloud
                self.client = QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key
                )
            else:
                # Use local Qdrant (no API key)
                self.client = QdrantClient(url=self.qdrant_url)
            
            self._ensure_collection_exists()
        except Exception as e:
            print(f"Failed to connect to Qdrant: {e}")
            self.client = None
    
    def validate_input(self, input_data: str) -> bool:
        """Validate input data - allow empty input for list operations"""
        return True  # Vector DB operations can work with empty input (list operation)
    
    def _ensure_collection_exists(self):
        """Ensure the collection exists in Qdrant"""
        if not self.client:
            return
        
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created Qdrant collection: {self.collection_name}")
            
            # Ensure index exists for agent_id filtering (for both new and existing collections)
            try:
                print(f"📊 Ensuring index exists for agent_id filtering in collection {self.collection_name}")
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="metadata.agent_id",
                    field_schema=PayloadSchemaType.KEYWORD
                )
            except Exception as index_error:
                # Index might already exist, which is fine
                if "already exists" not in str(index_error).lower():
                    print(f"⚠️ Could not create index (may already exist): {index_error}")
        except Exception as e:
            print(f"Error ensuring collection exists: {e}")
    
    def _get_openai_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI API using latest model"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.embeddings.create(
                model="text-embedding-3-large",  # Latest and most powerful OpenAI embedding model
                input=text,
                dimensions=1536  # Explicitly set dimensions for consistency
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting OpenAI embedding: {e}")
            # Fallback to simple embedding
            return self._simple_embedding(text)
    
    def _simple_embedding(self, text: str) -> List[float]:
        """Simple fallback embedding function"""
        import numpy as np
        
        words = text.lower().split()
        embedding = [0.0] * self.vector_size
        
        for i, word in enumerate(words[:self.vector_size]):
            hash_val = hash(word) % self.vector_size
            embedding[hash_val] += 1.0 / (i + 1)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for better retrieval"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + chunk_size - 100, start)
                sentence_end = text.rfind('.', search_start, end)
                if sentence_end > search_start:
                    end = sentence_end + 1
                else:
                    # Look for word boundary
                    word_end = text.rfind(' ', search_start, end)
                    if word_end > search_start:
                        end = word_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    async def execute(self, input_data: str, context: Dict[str, Any] = None) -> ToolResult:
        """Execute vector database operation"""
        if not self.validate_input(input_data):
            return ToolResult(
                success=False,
                data=None,
                error="Invalid input data"
            )
        
        if not self.client:
            return ToolResult(
                success=False,
                data=None,
                error="Qdrant client not available"
            )
        
        try:
            operation = context.get("operation", "search") if context else "search"
            
            if operation == "search":
                return await self._search(input_data, context)
            elif operation == "add":
                content = context.get("content", "") if context else ""
                title = context.get("title", input_data) if context else input_data
                agent_id = context.get("agent_id") if context else None
                return await self._add_document(title, content, agent_id)
            elif operation == "list":
                agent_id = context.get("agent_id") if context else None
                return await self._list_documents(agent_id)
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown operation: {operation}"
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Vector DB operation failed: {str(e)}"
            )
    
    async def _search(self, query: str, context: Dict[str, Any] = None) -> ToolResult:
        """Search the knowledge base using semantic similarity"""
        try:
            print(f"🔍 VectorDB Search - Query: '{query}', Context: {context}")
            
            # Get query embedding
            query_embedding = self._get_openai_embedding(query)
            
            # Get limit from context or use default
            limit = context.get("limit", self.max_results) if context else self.max_results
            agent_filter = None
            agent_id = context.get("agent_id") if context else None
            if agent_id:
                print(f"🎯 Filtering by agent_id: {agent_id}")
                # Build Qdrant payload filter on metadata.agent_id
                agent_filter = models.Filter(must=[
                    models.FieldCondition(
                        key="metadata.agent_id",
                        match=models.MatchValue(value=agent_id)
                    )
                ])
            else:
                print("⚠️ No agent_id in context - searching all documents")
            
            # Search in Qdrant with error handling
            try:
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=limit,
                    with_payload=True,
                    query_filter=agent_filter
                )
            except Exception as search_error:
                print(f"Search failed: {search_error}")
                # Return empty result instead of failing
                return ToolResult(
                    success=True,
                    data={
                        "query": query,
                        "documents": [],
                        "total_found": 0
                    },
                    metadata={
                        "operation": "search",
                        "results_returned": 0,
                        "collection": self.collection_name,
                        "warning": "Search failed due to API compatibility issue"
                    }
                )
            
            # Process results
            results = []
            sources = []
            
            print(f"📊 Found {len(search_results)} search results")
            
            for result in search_results:
                payload = result.payload or {}
                results.append({
                    "id": str(result.id),
                    "content": payload.get("content", ""),
                    "title": payload.get("title", ""),
                    "metadata": payload.get("metadata", {}),
                    "score": result.score  # Changed from "similarity" to "score"
                })
                
                # Add to sources for citation
                sources.append({
                    "title": payload.get("title", ""),
                    "url": f"qdrant://{self.collection_name}/{result.id}",
                    "score": result.score,
                    "content": payload.get("content", "")[:200] + "..." if len(payload.get("content", "")) > 200 else payload.get("content", "")
                })
                
                print(f"📄 Result: {payload.get('title', '')} (score: {result.score:.3f})")
            
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "documents": results,  # Changed from "results" to "documents"
                    "total_found": len(results)
                },
                metadata={
                    "operation": "search",
                    "results_returned": len(results),
                    "collection": self.collection_name,
                    "sources": sources  # Include sources for chat display
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Search failed: {str(e)}"
            )
    
    async def _add_document(self, title: str, content: str, agent_id: Optional[str]) -> ToolResult:
        """Add a document to the knowledge base with automatic chunking"""
        try:
            # Chunk the content
            chunks = self._chunk_text(content)
            
            # Add each chunk as a separate document
            chunk_ids = []
            for i, chunk in enumerate(chunks):
                # Generate unique ID for this chunk
                chunk_id = str(uuid.uuid4())
                chunk_ids.append(chunk_id)
                
                # Create full text for embedding
                full_text = f"{title} {chunk}"
                
                # Get embedding
                embedding = self._get_openai_embedding(full_text)
                
                # Prepare payload for this chunk
                payload = {
                    "title": title,
                    "content": chunk,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "original_title": title,
                    "metadata": {
                        "created_at": datetime.now().isoformat(),
                        "source": "user_input",
                        "text_length": len(full_text),
                        "chunk_size": len(chunk),
                        "agent_id": agent_id
                    }
                }
                
                # Add to Qdrant
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        PointStruct(
                            id=chunk_id,
                            vector=embedding,
                            payload=payload
                        )
                    ]
                )
            
            return ToolResult(
                success=True,
                data={
                    "document_id": chunk_ids[0],  # Return first chunk ID as main document ID
                    "chunk_ids": chunk_ids,
                    "title": title,
                    "chunks_created": len(chunks),
                    "content_preview": content[:100] + "..." if len(content) > 100 else content,
                    "embedding_size": len(embedding)
                },
                metadata={
                    "operation": "add",
                    "collection": self.collection_name,
                    "chunked": True
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Add document failed: {str(e)}"
            )
    
    async def _list_documents(self, agent_id: Optional[str]) -> ToolResult:
        """List all documents in the knowledge base, grouping chunks by original title"""
        try:
            # Get collection info with error handling
            try:
                collection_info = self.client.get_collection(self.collection_name)
            except Exception as collection_error:
                print(f"Collection info failed: {collection_error}")
                # Return empty result instead of failing
                return ToolResult(
                    success=True,
                    data={
                        "documents": [],
                        "total_documents": 0,
                        "total_chunks": 0
                    },
                    metadata={
                        "operation": "list",
                        "collection": self.collection_name,
                        "warning": "Could not retrieve collection info due to API compatibility issue"
                    }
                )
            
            # Get all documents with error handling
            try:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,  # Get more documents to group properly
                    with_payload=True
                )
            except Exception as scroll_error:
                print(f"Scroll failed: {scroll_error}")
                # Return empty result instead of failing
                return ToolResult(
                    success=True,
                    data={
                        "documents": [],
                        "total_documents": 0,
                        "total_chunks": 0
                    },
                    metadata={
                        "operation": "list",
                        "collection": self.collection_name,
                        "warning": "Could not retrieve documents due to API compatibility issue"
                    }
                )
            
            # Group chunks by original title
            document_groups = {}
            for point in scroll_result[0]:
                payload = point.payload or {}
                original_title = payload.get("original_title", payload.get("title", ""))
                meta = payload.get("metadata", {})
                if agent_id and meta.get("agent_id") != agent_id:
                    continue
                
                if original_title not in document_groups:
                    document_groups[original_title] = {
                        "id": str(point.id),
                        "title": original_title,
                        "chunks": [],
                        "total_chunks": payload.get("total_chunks", 1),
                        "created_at": meta.get("created_at", ""),
                        "agent_id": meta.get("agent_id")
                    }
                
                document_groups[original_title]["chunks"].append({
                    "chunk_id": str(point.id),
                    "chunk_index": payload.get("chunk_index", 0),
                    "content": payload.get("content", ""),
                    "content_preview": payload.get("content", "")[:100] + "..." if len(payload.get("content", "")) > 100 else payload.get("content", "")
                })
            
            # Convert to list and sort chunks
            documents = []
            for title, doc_data in document_groups.items():
                doc_data["chunks"].sort(key=lambda x: x["chunk_index"])
                # Combine all chunks for preview
                full_content = " ".join([chunk["content"] for chunk in doc_data["chunks"]])
                doc_data["content_preview"] = full_content[:200] + "..." if len(full_content) > 200 else full_content
                documents.append(doc_data)
            
            return ToolResult(
                success=True,
                data={
                    "documents": documents,
                    "total_documents": len(documents),
                    "total_chunks": collection_info.points_count,
                    "collection_name": self.collection_name
                },
                metadata={
                    "operation": "list",
                    "collection": self.collection_name
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"List documents failed: {str(e)}"
            )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM function calling"""
        return {
            "name": "vector_db_search",
            "description": "Search through a knowledge base using semantic similarity with Qdrant",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant information"
                    }
                },
                "required": ["query"]
            }
        }