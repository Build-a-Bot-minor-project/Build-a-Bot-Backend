"""
Web Search Tool using Tavily API
"""

import httpx
import os
from typing import Dict, Any
from .base_tool import BaseTool, ToolConfig, ToolResult

class WebSearchTool(BaseTool):
    """Tool for searching the web using Tavily API"""
    
    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.api_url = "https://api.tavily.com/search"
        
    async def execute(self, input_data: str, context: Dict[str, Any] = None) -> ToolResult:
        """Execute web search"""
        if not self.validate_input(input_data):
            return ToolResult(
                success=False,
                data=None,
                error="Invalid input data"
            )
        
        if not self.api_key:
            return ToolResult(
                success=False,
                data=None,
                error="Tavily API key not configured"
            )
        
        try:
            # Prepare search parameters
            search_params = {
                "api_key": self.api_key,
                "query": input_data,
                "search_depth": self.config.parameters.get("search_depth", "basic"),
                "include_answer": self.config.parameters.get("include_answer", True),
                "include_images": self.config.parameters.get("include_images", False),
                "include_raw_content": self.config.parameters.get("include_raw_content", False),
                "max_results": self.config.parameters.get("max_results", 5)
            }
            
            # Make API request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    json=search_params,
                    timeout=30.0
                )
                response.raise_for_status()
                search_results = response.json()
            
            # Process results
            processed_results = self._process_search_results(search_results)
            
            return ToolResult(
                success=True,
                data=processed_results,
                metadata={
                    "query": input_data,
                    "results_count": len(processed_results.get("results", [])),
                    "sources_count": len(processed_results.get("sources", [])),
                    "search_depth": search_params["search_depth"],
                    "sources": processed_results.get("sources", [])
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Web search failed: {str(e)}"
            )
    
    def _process_search_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw search results into a clean format"""
        processed = {
            "query": raw_results.get("query", ""),
            "answer": raw_results.get("answer", ""),
            "results": [],
            "sources": []  # List of URLs used
        }
        
        for result in raw_results.get("results", []):
            processed["results"].append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0)
            })
            
            # Add URL to sources if it exists
            if result.get("url"):
                processed["sources"].append({
                    "url": result.get("url"),
                    "title": result.get("title", ""),
                    "score": result.get("score", 0)
                })
        
        # Sort sources by score (highest first)
        processed["sources"].sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return processed
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM function calling"""
        return {
            "name": "web_search",
            "description": "Search the web for current information on any topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web"
                    }
                },
                "required": ["query"]
            }
        }
