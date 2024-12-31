from fastapi.responses import RedirectResponse
from langserve import add_routes
from app.agent_sup import graph
from fastapi import FastAPI
from langchain_core.tools import tool

app = FastAPI()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# Add routes for the graph
add_routes(app, graph, enable_feedback_endpoint=True)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)