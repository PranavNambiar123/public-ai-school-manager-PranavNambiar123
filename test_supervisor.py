import asyncio
import os
from app.agent_sup import graph
from langchain_openai import ChatOpenAI
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.callbacks.manager import CallbackManager

# Set up environment variables for tracking

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
            "content": "Explain what is the difference between agent_sup and agent?",
            "name": "user"
        }],
        "current_agent": "supervisor",
        "callbacks": callback_manager,
        "input": "What are the main components of the codebase?"
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
