"""
Agent Pipeline System for BuildABot
Allows users to create custom tool execution flows for agents
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
from tools.base_tool import BaseTool, ToolResult

class PipelineStep(BaseModel):
    """A step in the agent pipeline"""
    step_id: str
    tool_name: str
    tool_config: Dict[str, Any]
    input_mapping: Dict[str, str]  # Maps input to tool parameters
    output_mapping: Dict[str, str]  # Maps tool output to next step input
    condition: Optional[str] = None  # Optional condition for step execution
    order: int

class AgentPipeline(BaseModel):
    """Agent pipeline configuration"""
    id: str
    name: str
    description: str
    steps: List[PipelineStep]
    created_at: str
    updated_at: str

class PipelineExecutor:
    """Executes agent pipelines"""
    
    def __init__(self, available_tools: Dict[str, BaseTool]):
        self.available_tools = available_tools
    
    async def execute_pipeline(self, pipeline: AgentPipeline, initial_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a complete pipeline"""
        if not pipeline.steps:
            return {
                "success": False,
                "error": "Pipeline has no steps",
                "results": []
            }
        
        # Sort steps by order
        sorted_steps = sorted(pipeline.steps, key=lambda x: x.order)
        
        execution_results = []
        current_context = context or {}
        current_input = initial_input
        
        for step in sorted_steps:
            try:
                # Check condition if exists
                if step.condition and not self._evaluate_condition(step.condition, current_context):
                    execution_results.append({
                        "step_id": step.step_id,
                        "skipped": True,
                        "reason": f"Condition not met: {step.condition}"
                    })
                    continue
                
                # Get tool
                tool = self.available_tools.get(step.tool_name)
                if not tool:
                    execution_results.append({
                        "step_id": step.step_id,
                        "success": False,
                        "error": f"Tool '{step.tool_name}' not found"
                    })
                    continue
                
                # Prepare tool input
                tool_input = self._prepare_tool_input(step.input_mapping, current_input, current_context)
                
                # Execute tool
                result = await tool.execute(tool_input, current_context)
                
                # Store result
                execution_results.append({
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "success": result.success,
                    "data": result.data,
                    "error": result.error,
                    "metadata": result.metadata
                })
                
                # Update context for next step
                if result.success:
                    current_context = self._update_context(step.output_mapping, result.data, current_context)
                    # Use tool output as next input if no specific mapping
                    if not step.output_mapping:
                        current_input = str(result.data)
                
            except Exception as e:
                execution_results.append({
                    "step_id": step.step_id,
                    "success": False,
                    "error": f"Step execution failed: {str(e)}"
                })
        
        # Determine overall success
        successful_steps = [r for r in execution_results if r.get("success", False)]
        overall_success = len(successful_steps) > 0
        
        return {
            "success": overall_success,
            "pipeline_id": pipeline.id,
            "pipeline_name": pipeline.name,
            "total_steps": len(sorted_steps),
            "successful_steps": len(successful_steps),
            "results": execution_results,
            "final_context": current_context
        }
    
    def _prepare_tool_input(self, input_mapping: Dict[str, str], current_input: str, context: Dict[str, Any]) -> str:
        """Prepare input for tool based on mapping"""
        if not input_mapping:
            return current_input
        
        # Simple mapping - in production, implement more sophisticated mapping
        mapped_input = current_input
        for key, value in input_mapping.items():
            if key == "query":
                mapped_input = value.replace("{input}", current_input)
            elif key in context:
                mapped_input = mapped_input.replace(f"{{{key}}}", str(context[key]))
        
        return mapped_input
    
    def _update_context(self, output_mapping: Dict[str, str], tool_data: Any, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Update context with tool output"""
        if not output_mapping:
            return current_context
        
        updated_context = current_context.copy()
        
        # Simple mapping - in production, implement more sophisticated mapping
        for key, value in output_mapping.items():
            if isinstance(tool_data, dict) and value in tool_data:
                updated_context[key] = tool_data[value]
            else:
                updated_context[key] = str(tool_data)
        
        return updated_context
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate pipeline step condition"""
        # Simple condition evaluation - in production, implement more sophisticated logic
        try:
            # Replace context variables in condition
            evaluated_condition = condition
            for key, value in context.items():
                evaluated_condition = evaluated_condition.replace(f"{{{key}}}", str(value))
            
            # Simple boolean evaluation
            return eval(evaluated_condition.lower()) if evaluated_condition.lower() in ["true", "false"] else True
        except:
            return True  # Default to True if condition evaluation fails
