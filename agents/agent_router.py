"""
Intelligent Agent Router using OpenAI Function Calling
Automatically selects the best agent based on query analysis
"""

from openai import OpenAI
import os
from typing import Dict, List, Tuple
from .preset_agents import PRESET_AGENTS
import json

class AgentRouter:
    def __init__(self, openai_client: OpenAI):
        self.openai_client = openai_client
        self.preset_agents = PRESET_AGENTS
        
    def analyze_query_intent(self, query: str, available_agents: List[Dict]) -> Dict:
        """
        Analyze the query using OpenAI function calling to determine the best agent
        """
        if not available_agents:
            return {
                "primary_intent": "general",
                "confidence": 0.0,
                "recommended_agent_name": None,
                "reasoning": "No agents available",
                "query_type": "conversation",
                "complexity": "simple",
                "use_raw_llm": True
            }
        
        # Define the function schema for agent routing
        routing_function = {
            "name": "route_query_to_agent",
            "description": "Route a user query to the most appropriate agent or raw LLM",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_analysis": {
                        "type": "object",
                        "properties": {
                            "primary_intent": {
                                "type": "string",
                                "description": "The main intent category of the query"
                            },
                            "query_type": {
                                "type": "string",
                                "enum": ["greeting", "question", "request", "task", "conversation"],
                                "description": "Type of query"
                            },
                            "complexity": {
                                "type": "string",
                                "enum": ["simple", "medium", "complex"],
                                "description": "Complexity level of the query"
                            },
                            "domain": {
                                "type": "string",
                                "description": "Domain/category the query belongs to (e.g., cooking, programming, business)"
                            },
                            "requires_specialized_knowledge": {
                                "type": "boolean",
                                "description": "Whether the query requires specialized domain knowledge"
                            }
                        },
                        "required": ["primary_intent", "query_type", "complexity", "domain", "requires_specialized_knowledge"]
                    },
                    "routing_decision": {
                        "type": "object",
                        "properties": {
                            "use_raw_llm": {
                                "type": "boolean",
                                "description": "Whether to use raw LLM instead of specialized agent"
                            },
                            "recommended_agent_name": {
                                "type": "string",
                                "description": "Name of the recommended agent (null if use_raw_llm is true)"
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Confidence in the routing decision"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Explanation for the routing decision"
                            }
                        },
                        "required": ["use_raw_llm", "recommended_agent_name", "confidence", "reasoning"]
                    }
                },
                "required": ["query_analysis", "routing_decision"]
            }
        }
        
        # Create agent descriptions for the prompt
        agent_descriptions = []
        for agent in available_agents:
            agent_descriptions.append(f"- {agent['name']}: {agent['description']}")
        
        agent_list = "\n".join(agent_descriptions)
        
        # System prompt for intelligent routing
        system_prompt = f"""You are an intelligent agent router that analyzes user queries and routes them to the most appropriate agent or raw LLM.

Available agents:
{agent_list}

ROUTING GUIDELINES:
1. Use RAW LLM for:
   - Simple greetings: "hi", "hello", "hey", "good morning"
   - Basic conversation: "how are you", "thanks", "bye"
   - General questions that don't require specialized knowledge
   - Very short queries (1-2 words)

2. Use SPECIALIZED AGENTS for:
   - Domain-specific queries that match an agent's expertise
   - Tasks requiring specialized knowledge or skills
   - Complex questions in the agent's domain
   - Requests for specific outputs (recipes, code, analysis, etc.)

3. Consider query complexity and domain alignment when making routing decisions.

Analyze the user query and make an intelligent routing decision."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",  # Use GPT-4 for better function calling
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Route this query: \"{query}\""}
                ],
                functions=[routing_function],
                function_call={"name": "route_query_to_agent"},
                temperature=0.1,  # Low temperature for consistent routing
                max_tokens=1000
            )
            
            # Extract function call result
            function_call = response.choices[0].message.function_call
            if function_call and function_call.name == "route_query_to_agent":
                result = json.loads(function_call.arguments)
                
                # Combine query analysis and routing decision
                analysis = {
                    **result["query_analysis"],
                    **result["routing_decision"]
                }
                
                return analysis
            else:
                # Fallback if function calling fails
                return self._fallback_analysis(query)
                
        except Exception as e:
            print(f"Error in function calling analysis: {e}")
            return self._fallback_analysis(query)
    
    
    def _fallback_analysis(self, query: str) -> Dict:
        """
        Fallback analysis using keyword matching
        """
        query_lower = query.lower()
        
        # Keyword-based matching
        if any(word in query_lower for word in ['code', 'programming', 'python', 'javascript', 'function', 'debug', 'error', 'syntax']):
            return {
                "primary_intent": "programming",
                "confidence": 0.8,
                "recommended_agent": "code_assistant",
                "reasoning": "Query contains programming-related keywords",
                "alternative_agents": ["technical_expert"],
                "query_type": "question",
                "complexity": "medium"
            }
        elif any(word in query_lower for word in ['write', 'story', 'creative', 'poem', 'article', 'content', 'blog']):
            return {
                "primary_intent": "creative_writing",
                "confidence": 0.8,
                "recommended_agent": "creative_writer",
                "reasoning": "Query contains creative writing keywords",
                "alternative_agents": ["learning_tutor"],
                "query_type": "request",
                "complexity": "medium"
            }
        elif any(word in query_lower for word in ['business', 'strategy', 'marketing', 'sales', 'management', 'startup', 'company']):
            return {
                "primary_intent": "business",
                "confidence": 0.8,
                "recommended_agent": "business_advisor",
                "reasoning": "Query contains business-related keywords",
                "alternative_agents": ["personal_coach"],
                "query_type": "question",
                "complexity": "medium"
            }
        elif any(word in query_lower for word in ['learn', 'teach', 'explain', 'understand', 'study', 'education', 'tutorial']):
            return {
                "primary_intent": "learning",
                "confidence": 0.8,
                "recommended_agent": "learning_tutor",
                "reasoning": "Query contains learning-related keywords",
                "alternative_agents": ["technical_expert"],
                "query_type": "question",
                "complexity": "simple"
            }
        elif any(word in query_lower for word in ['motivate', 'goal', 'personal', 'development', 'life', 'coach', 'advice']):
            return {
                "primary_intent": "personal_development",
                "confidence": 0.8,
                "recommended_agent": "personal_coach",
                "reasoning": "Query contains personal development keywords",
                "alternative_agents": ["business_advisor"],
                "query_type": "conversation",
                "complexity": "medium"
            }
        else:
            return {
                "primary_intent": "general",
                "confidence": 0.5,
                "recommended_agent": "technical_expert",
                "reasoning": "General query, using technical expert as default",
                "alternative_agents": ["learning_tutor", "business_advisor"],
                "query_type": "conversation",
                "complexity": "simple"
            }
    
    def get_agent_recommendations(self, query: str) -> List[Dict]:
        """
        Get multiple agent recommendations with confidence scores
        """
        analysis = self.analyze_query_intent(query)
        
        recommendations = []
        
        # Primary recommendation
        primary_agent = analysis["recommended_agent"]
        recommendations.append({
            "agent_id": primary_agent,
            "agent_name": self.agents[primary_agent]["name"],
            "confidence": analysis["confidence"],
            "reasoning": analysis["reasoning"],
            "is_primary": True
        })
        
        # Alternative recommendations
        for alt_agent in analysis.get("alternative_agents", []):
            if alt_agent in self.agents:
                recommendations.append({
                    "agent_id": alt_agent,
                    "agent_name": self.agents[alt_agent]["name"],
                    "confidence": analysis["confidence"] * 0.7,  # Lower confidence for alternatives
                    "reasoning": f"Alternative option for {analysis['primary_intent']}",
                    "is_primary": False
                })
        
        return recommendations
    
    def get_best_agent(self, query: str, available_agents: List[Dict]) -> Tuple[str, Dict]:
        """

        Get the best agent for the query from available agents using function calling
        Returns (agent_id, analysis) - agent_id can be None for raw LLM
        """
        analysis = self.analyze_query_intent(query, available_agents)
        
        # If analysis recommends raw LLM, return None
        if analysis.get("use_raw_llm") or not analysis.get("recommended_agent_name"):
            return None, analysis
        
        # Find the agent by name
        recommended_name = analysis["recommended_agent_name"]
        for agent in available_agents:
            if agent["name"] == recommended_name:
                return agent["id"], analysis
        
        # If exact match not found, return None (use raw LLM) instead of forcing first agent
        return None, analysis
