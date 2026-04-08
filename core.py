import asyncio
from typing import Tuple, Dict, Any
from models import Action, Observation, Reward, State

class SupportFlowEnv:
    def __init__(self):
        # Initialize with mock data (The "Database")
        self.state = self._get_initial_state()
        self.max_steps = 10

    def _get_initial_state(self) -> State:
        """Resets the mock database to its starting point."""
        return State(
            users=[
                {"id": "u_882", "email": "customer@example.com", "name": "Jane Doe", "address": "123 Maple St"}
            ],
            orders=[
                {"id": "ord_101", "user_id": "u_882", "item": "Premium Headphones", "amount": 150.0, "status": "processed"}
            ],
            shipments=[
                {"order_id": "ord_101", "tracking_id": "TRK_7721", "status": "LOST", "location": "Unknown"}
            ],
            refund_processed=False,
            history=[],
            step_count=0
        )

    async def reset(self) -> Observation:
        """OpenEnv standard: Resets environment and returns initial observation."""
        self.state = self._get_initial_state()
        return Observation(
            display_output="CONNECTED TO CRM API v2.1. System ready. Please enter command."
        )

    async def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        OpenEnv standard: Processes action and returns (obs, reward, done, info).
        """
        self.state.step_count += 1
        cmd = action.command.lower()
        params = action.params
        
        reward = 0.0
        done = False
        display = ""
        error = None

        # 1. TOOL: Search User
        if cmd == "search_user":
            email = params.get("email")
            user = next((u for u in self.state.users if u["email"] == email), None)
            if user:
                display = f"USER_FOUND: {user['name']} | ID: {user['id']} | Address: {user['address']}"
                reward = 0.2  # Partial reward for finding the customer
            else:
                display = "ERROR: No user found with that email."
                error = "user_not_found"

        # 2. TOOL: Get Order Details
        elif cmd == "get_order_details":
            order_id = params.get("order_id")
            order = next((o for o in self.state.orders if o["id"] == order_id), None)
            if order:
                display = f"ORDER_INFO: Item: {order['item']} | Amount: ${order['amount']} | Status: {order['status']}"
                reward = 0.2
            else:
                display = "ERROR: Invalid Order ID."
                error = "invalid_order_id"

        # 3. TOOL: Track Package
        elif cmd == "track_package":
            order_id = params.get("order_id")
            shipment = next((s for s in self.state.shipments if s["order_id"] == order_id), None)
            if shipment:
                display = f"TRACKING_REPORT: ID: {shipment['tracking_id']} | STATUS: {shipment['status']} | Last Loc: {shipment['location']}"
                reward = 0.3 # Higher reward for investigating the root cause (the loss)
            else:
                display = "ERROR: No shipping data for this order."
                error = "no_tracking"

        # 4. TOOL: Issue Refund
        elif cmd == "issue_refund":
            order_id = params.get("order_id")
            order = next((o for o in self.state.orders if o["id"] == order_id), None)
            shipment = next((s for s in self.state.shipments if s["order_id"] == order_id), None)
            
            # Logic Gate: Only refund if the item is actually LOST
            if shipment and shipment["status"] == "LOST":
                self.state.refund_processed = True
                display = f"SUCCESS: Refund of ${order['amount']} processed for Order {order_id}."
                reward = 0.5 
            else:
                display = "REFUND_DENIED: Policy requires 'LOST' status for automated refunds."
                reward = -0.2 # Penalty for trying to refund without justification
                error = "policy_violation"

        # 5. TOOL: Respond (Final Action)
        elif cmd == "respond":
            message = params.get("message", "")
            display = f"SENT TO CUSTOMER: {message}"
            done = True
            # Final logic check: Did they actually solve the problem?
            if self.state.refund_processed:
                reward = 1.0 # Max reward
            else:
                reward = 0.0 # Failed task
        
        else:
            display = f"UNKNOWN_COMMAND: {cmd}"
            error = "unrecognized_action"

        # Check for timeout (Max steps)
        if self.state.step_count >= self.max_steps:
            done = True

        obs = Observation(display_output=display, last_action_error=error, is_terminal=done)
        
        # OpenEnv requires returning (Observation, float, bool, info_dict)
        return obs, float(reward), done, {"step": self.state.step_count}

    def get_state(self) -> Dict[str, Any]:
        """Returns current full state for the grader."""
        return self.state.dict()