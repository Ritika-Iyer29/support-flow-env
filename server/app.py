import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

# Import your custom logic and models
from .core import SupportFlowEnv
from .models import Action, Observation, ActionRequest, State

app = FastAPI(title="SupportFlow OpenEnv Service")

# Global environment instance (in a production RL env, you'd manage multiple sessions)
# For the hackathon, a single active session per container is standard.
env = SupportFlowEnv()

# --- Request/Response Wrappers ---

class StepRequest(BaseModel):
    action: Action

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]

# --- API Endpoints ---

@app.get("/")
async def root():
    """Health check endpoint for Hugging Face."""
    return {"status": "live", "env": "SupportFlow-v1", "spec": "OpenEnv 1.0"}

@app.post("/reset", response_model=Observation)
async def reset():
    """
    Resets the environment to the initial state.
    Mandatory for 'openenv validate' and the validator script.
    """
    try:
        observation = await env.reset()
        return observation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step", response_model=StepResponse)
async def step(request: ActionRequest): # Use ActionRequest here
    """
    Executes one action in the environment.
    """
    try:
        # Crucial: Ensure you are passing request.action to env.step
        obs, reward, done, info = await env.step(request.action)
        return StepResponse(
            observation=obs,
            reward=float(reward), # Force cast to float to prevent Pydantic errors
            done=done,
            info=info
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid step: {str(e)}")

@app.get("/state")
async def get_state():
    """
    Returns the full internal state (for graders).
    """
    return env.get_state()

# Add these endpoints to app.py
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/metadata")
async def metadata():
    return {
        "name": "support-flow-crm",
        "description": "Enterprise CRM triage environment for logistics resolution.",
        "version": "0.1.0"
    }

@app.get("/schema")
async def schema():
    return {
        "action": ActionRequest.schema(),
        "observation": Observation.schema(),
        "state": State.schema()
    }

@app.post("/mcp")
async def mcp():
    return {"jsonrpc": "2.0", "result": "MCP enabled"}

if __name__ == "__main__":
    import uvicorn
    # Port 7860 is required for Hugging Face Spaces
    uvicorn.run(app, host="0.0.0.0", port=7860)

