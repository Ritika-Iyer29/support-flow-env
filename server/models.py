from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union

class Action(BaseModel):
    """
    The Action model defines what the AI Agent can do.
    In a real-world CRM, an agent searches, queries, and executes logic.
    """
    command: str = Field(
        ..., 
        description="The tool to call: 'search_user', 'get_order_details', 'track_package', 'issue_refund', or 'respond'"
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value pairs for the command (e.g., {'email': 'user@example.com'})"
    )

class Observation(BaseModel):
    """
    The Observation is what the AI Agent 'sees' after every step.
    It mimics a terminal output or a CRM dashboard screen.
    """
    display_output: str = Field(..., description="The text shown to the agent after an action.")
    last_action_error: Optional[str] = Field(None, description="Error message if the last action failed.")
    is_terminal: bool = Field(False, description="Whether the episode has ended.")

class Reward(BaseModel):
    """
    The Reward model tracks progress. 
    OpenEnv expects a float between 0.0 and 1.0.
    """
    value: float = Field(0.0, ge=0.0, le=1.0, description="The reward score for the current step.")
    explanation: Optional[str] = Field(None, description="Why this reward was given (useful for debugging).")

class State(BaseModel):
    """
    The State is the 'Ground Truth'. 
    The Agent NEVER sees this. Only the Environment and the Grader use it.
    It represents the backend database.
    """
    users: List[Dict[str, Any]]
    orders: List[Dict[str, Any]]
    shipments: List[Dict[str, Any]]
    refund_processed: bool = False
    history: List[str] = []
    step_count: int = 0

# Add this to the bottom of models.py
class ActionRequest(BaseModel):
    """Wrapper required for the OpenEnv /step endpoint"""
    action: Action
