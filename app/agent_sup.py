# Step 1: Import necessary modules and classes
# Fill in any additional imports you might need
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import functools
import operator
import requests
import asyncio
import aiohttp

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from langgraph.graph import StateGraph, END
from .vector_store_agent import vector_store, text_splitter, add_texts, add_file, similarity_search, similarity_search_with_score, initialize_store
from .repository_scanner_agent import scan_repository, get_file_content, get_directory_structure 
from .code_analyzer_agent import CodeAnalyzerAgent
from .documentation_agent import analyze_documentation, generate_documentation
from .query_router_agent import answer_query

# Step 2: Define tools
# Here, define any tools the agents might use. Example given:
# Set env variables for tracking
# LANGCHAIN_API_KEY
# LANGCHAIN_PROJECT
# LANGCHAIN_TRACING_V2
# OPENAI_API_KEY

# This tool executes code locally, which can be unsafe. Use with caution:
python_repl_tool = PythonREPLTool()

@tool
def make_post_request(url: str, data: dict) -> str:
    """Make a POST request to the specified URL with the given data."""
    try:
        response = requests.post(url, json=data)
        return f"Status: {response.status_code}, Response: {response.text}"
    except Exception as e:
        return f"Error making request: {str(e)}"

@tool
async def stress_test_endpoint(url: str, iterations: int = 10) -> str:
    """Make multiple concurrent POST requests to test endpoint stability."""
    async def single_request(session, url):
        try:
            async with session.post(url) as response:
                return response.status == 200
        except:
            return False

    async with aiohttp.ClientSession() as session:
        tasks = [single_request(session, url) for _ in range(iterations)]
        results = await asyncio.gather(*tasks)
        success_rate = sum(results) / len(results)
        return f"Success rate: {success_rate * 100}%, {sum(results)}/{len(results)} requests succeeded"

@tool
def analyze_code_style(code: str) -> str:
    """Analyze code for PEP 8 compliance and common Python style issues."""
    try:
        # Simulate style checking
        issues = []
        if "    " in code:  # Check for spaces instead of tabs
            issues.append("- Use spaces for indentation")
        if len(max(code.split('\n'), key=len)) > 79:  # Check line length
            issues.append("- Line too long (>79 characters)")
        return "\n".join(issues) if issues else "Code follows PEP 8 style guide"
    except Exception as e:
        return f"Error analyzing code: {str(e)}"

# Step 3: Define the system prompt for the supervisor agent
# Customize the members list as needed.
members = [
    "CodeAnalyzer", "VectorStore", "Documentation", "QueryRouter"
]

system_prompt = f"""You are the supervisor agent coordinating a team of specialized agents:
- CodeAnalyzer: Reads the repository and responds with the file contents in it
- VectorStore: Stores the code embeddings and generates a vector store
- Documentation: Handles documentation tasks
- QueryRouter: Uses the vector store to answer queries. Vector store must be initialized before invoking this agent

Route tasks to the most suitable agent. Only use available agents."""

# Step 4: Define options for the supervisor to choose from
options = members + ["FINISH"]

# Step 5: Define the function for OpenAI function calling
# Define what the function should do and its parameters.
function_def = {
    "name": "route",
    "description": "Route the task to the appropriate worker",
    "parameters": {
        "title": "Route the task to the appropriate worker",
        "type": "object",
        "properties": {"next": {"title": "Next", "anyOf": [{"enum": options}]}},
        "required": ["next"],
    },
}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", f"Given the conversation above who should act next?\n"
                  f"Or should we FINISH? Select one of {', '.join(options)}")
    ]
).partial(options=str(options), members=', '.join(members))

# Step 6: Define the prompt for the supervisor agent
# Customize the prompt if needed.

# Step 7: Initialize the language model
# Choose the model you need, e.g., "gpt-4o"
llm = ChatOpenAI(model="gpt-4o")

# Step 8: Create the supervisor chain
# Define how the supervisor chain will process messages.
supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

# Step 9: Define a typed dictionary for agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    final_output: Optional[str] = None

# Step 10: Function to create an agent
# Fill in the system prompt and tools for each agent you need to create.
def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

# Step 11: Function to create an agent node
# This function processes the state through the agent and returns the result.
def agent_node(state, agent, name):
    """Process a state with an agent and return the updated state"""
    # Extract the latest message
    if state.get("messages"):
        latest_message = state["messages"][-1]["content"]
    else:
        latest_message = "No previous messages"
    
    # Create input for the agent
    agent_input = {
        "input": latest_message,
        "messages": state.get("messages", []),  # Add messages directly
        "chat_history": state.get("messages", []),
    }
    
    # Add callbacks if present
    if "callbacks" in state:
        agent_input["callbacks"] = state["callbacks"]
    
    # Run the agent
    try:
        result = agent.invoke(agent_input)
        output = result.get("output", "No response from agent")
    except Exception as e:
        output = f"Error in {name}: {str(e)}"
    
    # Return updated state with new message
    return {
        "messages": [
            *state.get("messages", []),
            {"role": "assistant", "name": name, "content": output}
        ]
    }

# Step 12: Create agents and their corresponding nodes
# Define the specific role and tools for each agent.
code_analyzer = CodeAnalyzerAgent(repo_url="https://github.com/PranavNambiar123/public-ai-school-manager-PranavNambiar123")
code_analyzer_agent = create_agent(
    llm,
    [code_analyzer.analyze_file, code_analyzer.get_complexity_report],
    """You are an expert code analyzer specializing in Python code analysis.
    Your responsibilities:
    - Analyze code structure and patterns using static analysis
    - Calculate and report code complexity metrics
    - Identify maintainability issues and code smells
    - Generate detailed code quality reports
    - Provide actionable recommendations for code improvements
    
    You have access to these tools:
    - analyze_file: Performs deep analysis of Python files including metrics, structure, and quality issues
    - get_complexity_report: Generates detailed complexity metrics and identifies hotspots
    
    Focus on providing data-driven insights and specific improvement suggestions."""
)
code_analyzer_node = functools.partial(agent_node, agent=code_analyzer_agent, name="CodeAnalyzer")

vector_store_agent = create_agent(
    llm,
    [
        add_texts,
        add_file,
        similarity_search,
        similarity_search_with_score
    ],
    """You are a vector store agent responsible for storing and retrieving code embeddings from a central vector store."""
)
vector_store_node = functools.partial(agent_node, agent=vector_store_agent, name="VectorStore")

documentation_agent = create_agent(
    llm,
    [
        analyze_documentation,
        generate_documentation,
    ],
    """Documentation specialist generating clear technical documentation.
    Tasks: Create READMEs, docstrings, API docs."""
)
documentation_node = functools.partial(agent_node, agent=documentation_agent, name="Documentation")

repository_agent = create_agent(
    llm,
    [
        scan_repository,
        get_file_content,
        get_directory_structure
    ],
    """You can read repositories, get directory structure and read file content and respond with the same. Do not respond with queries regarding the content, only respond with the content."""
)
repository_node = functools.partial(agent_node, agent=repository_agent, name="Repository")

query_router_agent = create_agent(
    llm,
    [
        answer_query
    ],
    """You are a query response agent which given a query uses the vector store generated by the vector store agent to answer the queries.
    Tasks: Process queries, retrieve context, generate responses."""
)
query_router_node = functools.partial(agent_node, agent=query_router_agent, name="QueryRouter")

# Step 13: Define the workflow using StateGraph
# Add nodes and their corresponding functions to the workflow.
workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_chain)
workflow.add_node("CodeAnalyzer", repository_node)
workflow.add_node("VectorStore", vector_store_node)
workflow.add_node("Documentation", documentation_node)
# workflow.add_node("Repository", repository_node)
workflow.add_node("QueryRouter", query_router_node)

# Step 14: Add edges to the workflow
# Ensure that all workers report back to the supervisor.
for member in members:
    workflow.add_edge(member, "supervisor")

# Step 15: Define conditional edges
# The supervisor determines the next step or finishes the process.
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

# Step 16: Set the entry point
workflow.set_entry_point("supervisor")

# Step 17: Compile the workflow into a graph
# This creates the executable workflow.
graph = workflow.compile()

async def process_query(query: str):
    """
    Process a query through the agent workflow
    
    Args:
        query: The query string to process
        
    Returns:
        The final state after processing
    """
    # Create initial state
    state = {
        "messages": [{"role": "user", "content": query}],
        "current_agent": "supervisor"
    }
    
    # Run the graph with the initial state
    result = await graph.ainvoke(state)
    return result
