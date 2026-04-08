from typing import Dict, Any, List
from models import State

class SupportFlowGrader:
    """
    Programmatic grader to ensure the agent didn't just 'talk' 
    but actually performed the correct backend actions.
    """

    @staticmethod
    def grade_easy(state: State, last_message: str) -> float:
        """
        Task: Retrieve Tracking Number for customer@example.com
        Criteria: Must mention 'TRK_7721' in the final response.
        """
        # Did the agent provide the correct tracking ID in their response?
        if "TRK_7721" in last_message.upper():
            return 1.0
        return 0.0

    @staticmethod
    def grade_medium(state: State, last_message: str) -> float:
        """
        Task: Refund Eligibility Audit
        Criteria: Agent must confirm status is LOST.
        Score: 0.5 for checking tracking, 1.0 for correct verbal confirmation.
        """
        score = 0.0
        # Check if they at least called the tracking tool (partial progress)
        # We can track this by looking at step history or if they reached a certain state
        if state.step_count > 1:
            score += 0.4
        
        # Did they give the correct answer? (Yes, it is lost/eligible)
        if any(word in last_message.upper() for word in ["ELIGIBLE", "YES", "LOST"]):
            score += 0.6
            
        return min(score, 1.0)

    @staticmethod
    def grade_hard(state: State, last_message: str) -> float:
        """
        Task: Lost Package Resolution
        Criteria: MUST call issue_refund tool AND notify customer.
        """
        # If they didn't trigger the refund tool in core.py, they fail.
        if not state.refund_processed:
            return 0.0
        
        # If they refunded AND sent a closing message
        if state.refund_processed and len(last_message) > 5:
            return 1.0
        
        return 0.5 # Refunded but forgot to tell the customer

class TaskManager:
    """
    Handles the initialization of different scenarios.
    """
    @staticmethod
    def get_task_setup(task_id: str) -> Dict[str, Any]:
        if task_id == "easy_tracking":
            return {"instruction": "Find the tracking number for customer@example.com"}
        
        elif task_id == "medium_policy_check":
            return {"instruction": "Audit Order ord_101. Is it eligible for a refund per policy?"}
        
        elif task_id == "hard_full_resolution":
            return {"instruction": "Order ord_101 is missing. Resolve the issue fully for the customer."}
        
        return {"instruction": "Assist the customer with their inquiry."}