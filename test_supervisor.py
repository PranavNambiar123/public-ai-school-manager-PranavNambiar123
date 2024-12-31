import asyncio
import os
from app.agent_sup import graph
from langchain_openai import ChatOpenAI
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.callbacks.manager import CallbackManager

# Set up environment variables for tracking
# Set environment variables for tracking
# LANGCHAIN_API_KEY
# LANGCHAIN_PROJECT
# LANGCHAIN_TRACING_V2
# OPENAI_API_KEY

async def main():    
    # Set up LangChain tracing
    tracer = LangChainTracer(
        project_name=os.environ["LANGCHAIN_PROJECT"]
    )
    callback_manager = CallbackManager([tracer])
    
    # Create initial state with tracing
    state = {
        "messages": [{
            "role": "user",
            "content": "Find out and read only server.py file from the repository and store it in the vector store.",
            "name": "user"
        }],
        "current_agent": "supervisor",
        "callbacks": callback_manager,
        "input": "Find out and read only server.py file from the repository and store it in the vector store."
    }
    
    try:
        # Run the graph with the initial state
        result = await graph.ainvoke(state)
        print("\nFinal Messages:")
        for msg in result.get("messages", []):
            print(f"\n{msg.get('name', 'unknown')}: {msg.get('content', 'no content')}")
    except Exception as e:
        print(f"Error running graph: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
