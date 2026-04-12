from typing import Dict, Any, List
from .models import State

class SupportFlowGrader:
    @staticmethod
    def grade_easy(state: State, last_message: str) -> float:
        # Instead of 1.0, return 0.9. Instead of 0.0, return 0.1
        if "TRK_7721" in last_message.upper():
            return 0.9
        return 0.1

    @staticmethod
    def grade_medium(state: State, last_message: str) -> float:
        score = 0.1 # Minimum non-zero score
        if state.step_count > 1:
            score += 0.3
        if any(word in last_message.upper() for word in ["ELIGIBLE", "YES", "LOST"]):
            score += 0.5
        return score # Results in 0.1, 0.4, or 0.9

    @staticmethod
    def grade_hard(state: State, last_message: str) -> float:
        if not state.refund_processed:
            return 0.1
        if state.refund_processed and len(last_message) > 5:
            return 0.95 # Maximum non-one score
        return 0.5

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
