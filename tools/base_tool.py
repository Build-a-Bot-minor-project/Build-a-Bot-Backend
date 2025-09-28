"""
Base Tool Class for BuildABot Tools
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class ToolConfig(BaseModel):
    """Configuration for a tool"""
    name: str
    description: str
    parameters: Dict[str, Any] = {}
    enabled: bool = True

class ToolResult(BaseModel):
    """Result from tool execution"""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}

class BaseTool(ABC):
    """Base class for all tools"""
    
    def __init__(self, config: ToolConfig):
        self.config = config
        self.name = config.name
        self.description = config.description
        self.enabled = config.enabled
    
    @abstractmethod
    async def execute(self, input_data: str, context: Dict[str, Any] = None) -> ToolResult:
        """Execute the tool with given input"""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's schema for LLM function calling"""
        pass
    
    def validate_input(self, input_data: str) -> bool:
        """Validate input data - can be overridden by subclasses"""
        return bool(input_data and input_data.strip())
    
    def __str__(self):
        return f"{self.name}: {self.description}"
