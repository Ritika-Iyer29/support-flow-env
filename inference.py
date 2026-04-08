import asyncio
import os
import json
import textwrap
from typing import List, Optional
from openai import OpenAI

# Environment Variables
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# Your Space URL - Ensure this is the .hf.space URL
ENV_URL = os.getenv("ENV_URL", "https://ritikaiyer29-support-flow-env.hf.space") 

# Task Settings
TASK_NAME = os.getenv("TASK_NAME", "hard_full_resolution")
BENCHMARK = "support-flow-crm"
MAX_STEPS = 10

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a Senior Support Specialist. You MUST solve the customer's problem using the provided tools.
    
    AVAILABLE COMMANDS:
    - 'search_user': params {'email': '...'}
    - 'get_order_details': params {'order_id': '...'}
    - 'track_package': params {'order_id': '...'}
    - 'issue_refund': params {'order_id': '...'}
    - 'respond': params {'message': '...'}

    STRICT JSON FORMAT:
    {
      "command": "command_name",
      "params": {"key": "value"}
    }

    CRITICAL RULES:
    1. Output ONLY the JSON object. No preamble or explanation.
    2. Do NOT use 'get_user_id'. Use 'search_user' to find customer info.
    3. Follow the sequence: Search -> Order Details -> Track -> Refund -> Respond.
    """
).strip()
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # Corrected: Removed 'score=' for strict hackathon compliance
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

async def call_env(endpoint: str, payload: dict = None):
    import httpx
    async with httpx.AsyncClient() as client:
        url = f"{ENV_URL}/{endpoint}"
        if payload:
            resp = await client.post(url, json=payload, timeout=30.0)
        else:
            resp = await client.post(url, timeout=30.0)
        
        # Safety check: if server returns 4xx or 5xx, print raw text for debugging
        if resp.status_code != 200:
            print(f"[DEBUG] API Error {resp.status_code}: {resp.text}")
            return {}
        return resp.json()

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Track history to prevent looping
    action_history = []
    rewards = []
    steps_taken = 0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # 1. Reset Environment
        obs_data = await call_env("reset")
        current_obs = obs_data.get("display_output", "System Ready")

        for step in range(1, MAX_STEPS + 1):
            # 2. Get Model Action with Context
            history_str = "\n".join(action_history[-3:]) # Last 3 actions
            user_prompt = (
                f"TASK: customer@example.com says order ord_101 is missing. Resolve fully.\n"
                f"History:\n{history_str}\n"
                f"Current Observation: {current_obs}\n"
                f"Next Action (JSON):"
            )
            
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            action_json = json.loads(completion.choices[0].message.content)
            
            # 3. Step Environment
            step_result = await call_env("step", {"action": action_json})
            
            # 4. Process Results with safety .get()
            reward = step_result.get("reward", 0.0)
            done = step_result.get("done", False)
            obs_payload = step_result.get("observation", {})
            
            current_obs = obs_payload.get("display_output", "No response from CRM")
            error = obs_payload.get("last_action_error")
            
            # Record history to break loops
            action_history.append(f"Step {step}: {action_json.get('command')} -> {current_obs}")
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=json.dumps(action_json), reward=reward, done=done, error=error)

            if done:
                break

        # 5. Final Grading
        total_score = sum(rewards)
        final_score = min(max(total_score, 0.0), 1.0)
        success = final_score >= 0.7 and any(r > 0.4 for r in rewards) # Must have reached refund

    except Exception as e:
        print(f"[DEBUG] Error during inference: {e}")
    finally:
        log_end(success=success, steps=steps_taken, score=final_score if 'final_score' in locals() else 0.0, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())