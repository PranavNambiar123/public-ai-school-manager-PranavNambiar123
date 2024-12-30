from typing import Dict, List, Optional, Set, Union, Any
import os
import logging
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class QueryType(Enum):
    """Enum for different types of queries"""
    CODE_STRUCTURE = "code_structure"
    CODE_ANALYSIS = "code_analysis"
    DOCUMENTATION = "documentation"
    REPOSITORY = "repository"
    SEARCH = "search"
    UNKNOWN = "unknown"

@dataclass
class QueryContext:
    """Data class to store query context"""
    original_query: str
    query_type: QueryType
    relevant_files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

@dataclass
class AgentResponse:
    """Data class to store agent response"""
    agent_name: str
    response: Any
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class QueryRouterAgent:
    """Agent responsible for routing queries to appropriate specialized agents"""
    
    def __init__(self, base_path: str):
        """
        Initialize the Query Router Agent
        
        Args:
            base_path: Root path of the repository
        """
        self.base_path = os.path.abspath(base_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Query type patterns
        self.query_patterns = {
            QueryType.CODE_STRUCTURE: [
                "structure", "architecture", "design", "layout",
                "organization", "hierarchy", "dependency"
            ],
            QueryType.CODE_ANALYSIS: [
                "analyze", "complexity", "quality", "metric",
                "performance", "issue", "bug", "smell"
            ],
            QueryType.DOCUMENTATION: [
                "document", "doc", "comment", "explain",
                "readme", "guide", "tutorial", "example"
            ],
            QueryType.REPOSITORY: [
                "repository", "repo", "file", "folder",
                "directory", "size", "content"
            ],
            QueryType.SEARCH: [
                "search", "find", "locate", "where",
                "similar", "related", "matching"
            ]
        }
        
    @tool
    async def route_query(self, query: str) -> List[AgentResponse]:
        """
        Route a query to appropriate agents and aggregate responses
        
        Args:
            query: User query string
            
        Returns:
            List of agent responses
        """
        try:
            # Analyze query
            context = self._analyze_query(query)
            
            # Get relevant agents
            agents = self._get_relevant_agents(context)
            
            # Process query with selected agents
            tasks = [
                self._process_with_agent(agent, context)
                for agent in agents
            ]
            
            # Gather responses
            responses = await asyncio.gather(*tasks)
            
            # Filter and sort responses by confidence
            valid_responses = [r for r in responses if r and r.confidence > 0.3]
            sorted_responses = sorted(
                valid_responses,
                key=lambda x: x.confidence,
                reverse=True
            )
            
            return sorted_responses
            
        except Exception as e:
            logging.error(f"Error routing query: {str(e)}")
            return []
    
    @tool
    def analyze_query_type(self, query: str) -> QueryContext:
        """
        Analyze query to determine its type and context
        
        Args:
            query: User query string
            
        Returns:
            QueryContext object with analysis results
        """
        return self._analyze_query(query)
    
    def _analyze_query(self, query: str) -> QueryContext:
        """Analyze query to determine type and extract context"""
        # Preprocess query
        tokens = word_tokenize(query.lower())
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and token.isalnum()
        ]
        
        # Calculate query type scores
        type_scores = {}
        query_embedding = self.model.encode([query])[0]
        
        for query_type, patterns in self.query_patterns.items():
            patterns_embedding = self.model.encode(patterns)
            similarity_scores = [
                self._cosine_similarity(query_embedding, pattern_embedding)
                for pattern_embedding in patterns_embedding
            ]
            type_scores[query_type] = max(similarity_scores)
        
        # Determine query type
        best_type = max(type_scores.items(), key=lambda x: x[1])
        
        return QueryContext(
            original_query=query,
            query_type=best_type[0],
            confidence=best_type[1]
        )
    
    def _get_relevant_agents(self, context: QueryContext) -> List[str]:
        """Determine which agents should handle the query"""
        agents = []
        
        # Map query types to agents
        type_to_agents = {
            QueryType.CODE_STRUCTURE: ["code_analyzer"],
            QueryType.CODE_ANALYSIS: ["code_analyzer", "vector_store"],
            QueryType.DOCUMENTATION: ["documentation", "vector_store"],
            QueryType.REPOSITORY: ["repository_scanner", "vector_store"],
            QueryType.SEARCH: ["vector_store", "documentation"],
            QueryType.UNKNOWN: ["vector_store"]  # Default to search
        }
        
        agents = type_to_agents.get(context.query_type, ["vector_store"])
        
        # Add additional agents based on context
        if context.confidence < 0.5:
            agents.append("vector_store")  # Add search capability for low confidence
            
        return list(set(agents))  # Remove duplicates
    
    async def _process_with_agent(self, agent_name: str, context: QueryContext) -> Optional[AgentResponse]:
        """Process query with a specific agent"""
        try:
            start_time = datetime.now()
            
            # Implement agent-specific processing logic here
            # This would integrate with your actual agent implementations
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                agent_name=agent_name,
                response=None,  # Replace with actual agent response
                confidence=context.confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            logging.error(f"Error processing with agent {agent_name}: {str(e)}")
            return None
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_v1 = sum(a * a for a in v1) ** 0.5
        norm_v2 = sum(b * b for b in v2) ** 0.5
        return dot_product / (norm_v1 * norm_v2) if norm_v1 * norm_v2 != 0 else 0.0
    
    @tool
    async def batch_process_queries(self, queries: List[str]) -> Dict[str, List[AgentResponse]]:
        """
        Process multiple queries in batch
        
        Args:
            queries: List of query strings
            
        Returns:
            Dictionary mapping queries to their responses
        """
        results = {}
        for query in queries:
            results[query] = await self.route_query(query)
        return results
