from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import functools
import operator
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langgraph.graph import StateGraph, END

from .vector_store_agent import VectorStoreAgent
from .repository_scanner_agent import RepositoryScannerAgent
from .code_analyzer_agent import CodeAnalyzerAgent
from .documentation_agent import DocumentationAgent
from .query_router_agent import QueryRouterAgent, QueryContext, AgentResponse

@dataclass
class AgentState:
    """Data class to store the current state of the agent ecosystem"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    final_output: Optional[str]
    repository_state: Dict[str, Any] = field(default_factory=dict)
    analysis_state: Dict[str, Any] = field(default_factory=dict)
    documentation_state: Dict[str, Any] = field(default_factory=dict)
    vector_store_state: Dict[str, Any] = field(default_factory=dict)
    query_history: List[QueryContext] = field(default_factory=list)

class AgentSupervisor:
    """Supervisor to coordinate multiple specialized agents"""
    
    def __init__(self, base_path: str, llm: Optional[ChatOpenAI] = None):
        """
        Initialize the Agent Supervisor
        
        Args:
            base_path: Root path of the repository
            llm: Language model for agent communication
        """
        self.base_path = base_path
        self.llm = llm or ChatOpenAI(temperature=0)
        
        # Initialize specialized agents
        self.repo_scanner = RepositoryScannerAgent(base_path)
        self.code_analyzer = CodeAnalyzerAgent("https://github.com/PranavNambiar123/public-ai-school-manager-PranavNambiar123")
        self.documentation = DocumentationAgent(base_path)
        self.vector_store = VectorStoreAgent(f"{base_path}/.vector_store")
        self.query_router = QueryRouterAgent(base_path)
        
        # Initialize state
        self.state = AgentState(
            messages=[],
            next="ROUTE",
            final_output=None
        )
        
        # Setup workflow graph
        self.workflow = self._create_workflow_graph()
        
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query through the agent ecosystem
        
        Args:
            query: User query string
            
        Returns:
            Dictionary containing processed results
        """
        try:
            # Route query to appropriate agents
            responses = await self.query_router.route_query(query)
            
            # Process with selected agents
            results = await self._process_with_agents(query, responses)
            
            # Update state
            self._update_state(query, responses, results)
            
            return self._format_response(results)
            
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return {"error": str(e)}
    
    async def initialize_repository(self) -> None:
        """Initialize the repository analysis"""
        try:
            # Scan repository
            repo_structure = self.repo_scanner.scan_repository()
            self.state.repository_state["structure"] = repo_structure
            
            # Initialize vector store
            self.vector_store.initialize_store()
            
            # Process important files
            await self._process_important_files(repo_structure.important_files)
            
        except Exception as e:
            logging.error(f"Error initializing repository: {str(e)}")
            raise
    
    async def _process_important_files(self, files: List[str]) -> None:
        """Process important repository files"""
        for file_path in files:
            try:
                # Get file content
                content = self.repo_scanner.get_file_content(file_path)
                if content is None:
                    continue
                    
                # Analyze code if it's a Python file
                if file_path.endswith('.py'):
                    analysis = self.code_analyzer.analyze_file(file_path)
                    if analysis:
                        self.state.analysis_state[file_path] = analysis
                
                # Analyze documentation
                doc_analysis = self.documentation.analyze_documentation(file_path)
                if doc_analysis:
                    self.state.documentation_state[file_path] = doc_analysis
                
                # Add to vector store
                self.vector_store.add_file(file_path, content)
                
            except Exception as e:
                logging.warning(f"Error processing file {file_path}: {str(e)}")
    
    async def _process_with_agents(self, query: str, 
                                 agent_responses: List[AgentResponse]) -> Dict[str, Any]:
        """Process query with selected agents"""
        results = {}
        
        for response in agent_responses:
            try:
                if response.agent_name == "repository_scanner":
                    results["repository"] = self.repo_scanner.scan_repository()
                    
                elif response.agent_name == "code_analyzer":
                    if "repository" in results:
                        for file in results["repository"].important_files:
                            if file.endswith('.py'):
                                analysis = self.code_analyzer.analyze_file(file)
                                if analysis:
                                    results.setdefault("code_analysis", {})[file] = analysis
                                    
                elif response.agent_name == "documentation":
                    if "repository" in results:
                        for file in results["repository"].important_files:
                            doc_analysis = self.documentation.analyze_documentation(file)
                            if doc_analysis:
                                results.setdefault("documentation", {})[file] = doc_analysis
                                
                elif response.agent_name == "vector_store":
                    search_results = self.vector_store.similarity_search(query)
                    results["search"] = search_results
                    
            except Exception as e:
                logging.error(f"Error with agent {response.agent_name}: {str(e)}")
                
        return results
    
    def _update_state(self, query: str, responses: List[AgentResponse], 
                     results: Dict[str, Any]) -> None:
        """Update supervisor state with query results"""
        # Update query history
        query_context = self.query_router.analyze_query_type(query)
        self.state.query_history.append(query_context)
        
        # Update agent states
        if "repository" in results:
            self.state.repository_state.update(results["repository"].__dict__)
        if "code_analysis" in results:
            self.state.analysis_state.update(results["code_analysis"])
        if "documentation" in results:
            self.state.documentation_state.update(results["documentation"])
        if "search" in results:
            self.state.vector_store_state["last_search"] = results["search"]
    
    def _format_response(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format the final response"""
        response = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "metadata": {
                "query_count": len(self.state.query_history),
                "repository_files": len(self.state.repository_state.get("important_files", [])),
                "analyzed_files": len(self.state.analysis_state)
            }
        }
        return response
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the agent workflow graph"""
        workflow = StateGraph()
        
        # Add nodes for each agent type
        members = ["ROUTE", "SCAN", "ANALYZE", "DOCUMENT", "SEARCH"]
        for member in members:
            workflow.add_node(member, getattr(self, member.lower()).__call__)
        
        # Define edges
        for member in members:
            workflow.add_edge(member, "supervisor")
        
        # Define conditional edges
        conditional_map = {k: k for k in members}
        conditional_map["FINISH"] = END
        workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
        
        # Set the entry point
        workflow.set_entry_point("supervisor")
        
        # Compile the workflow into a graph
        graph = workflow.compile()
        return graph