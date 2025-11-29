# tools.py

from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List, Dict, Callable
from schemas import (
    ActivitySummary, 
    FundraisingSummary, 
    PostRequest, 
    PostCandidate, 
    JudgeFeedback, 
    Channel, 
    Tone
)
import json
import os
import uuid

# --- Initialize Gemini Client ---
# NOTE: Ensure your GEMINI_API_KEY is set in your environment
try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}. Ensure GEMINI_API_KEY is set.")
    client = None

# --- Helper Function for JSON-like structured output ---
def call_model_with_structured_output(
    system_instruction: str, 
    prompt: str, 
    response_schema: BaseModel, 
    model: str = 'gemini-2.5-flash'
):
    """A reusable helper to get structured Pydantic output from the model."""
    if not client:
        raise ConnectionError("Gemini client is not initialized.")
        
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=response_schema,
    )
    
    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=config,
    )
    
    # The response text will be a JSON string conforming to the schema
    # We parse it and return a Pydantic object
    return response_schema.model_validate_json(response.text)


# --- TOOL DEFINITIONS: Retrieving Information (R) ---

def get_activity_summary() -> ActivitySummary:
    """
    (R) RETRIEVING INFORMATION. Simulates fetching the latest team activity data
    from a fitness tracker (e.g., Strava or similar corporate wellness system).
    
    Returns:
        ActivitySummary: The latest running data.
    """
    # Stub Data representing a successful 10k team run
    data = {
        "event_name": "RMHC November Corporate 10K",
        "date": "2025-11-29",
        "distance_km": 10.0,
        "duration_minutes": 65,
        "runners_participated": 12,
    }
    return ActivitySummary(**data)


def get_fundraising_summary() -> FundraisingSummary:
    """
    (R) RETRIEVING INFORMATION. Simulates fetching current fundraising totals
    from a platform like JustGiving or corporate charity dashboard.
    
    Returns:
        FundraisingSummary: The latest financial data.
    """
    # Stub Data representing a significant milestone reached (75%)
    total_raised = 7500.00
    campaign_target = 10000.00
    percent = (total_raised / campaign_target) * 100.0
    is_milestone = percent >= 75 and percent < 100
    
    data = {
        "total_raised_usd": total_raised,
        "campaign_target_usd": campaign_target,
        "percent_to_goal": round(percent, 2),
        "latest_donor_name": "Sarah K.",
        "is_milestone": is_milestone,
    }
    return FundraisingSummary(**data)


# --- TOOL DEFINITIONS: Executing Actions (A) ---

def generate_post_candidates(post_request: PostRequest) -> List[PostCandidate]:
    """
    (A) EXECUTING ACTION. Calls the LLM to generate multiple, high-quality, 
    channel-specific post candidates based on the agent's plan.
    
    Args:
        post_request: A structured request guiding the LLM's generation task.
        
    Returns:
        List[PostCandidate]: A list of 2-3 post options for final review.
    """
    print(f"\n--- Calling LLM to Generate Candidates for {post_request.channel.value} ---")
    
    # The system instruction for the generation model
    generation_instruction = f"""
    You are a professional corporate communications specialist for Ronald McDonald House Charities (RMHC) Corporate Sponsor. 
    Your task is to generate 3 distinct social media post candidates for the target channel and objective.
    
    CRITICAL RULES:
    1. Tone must be exactly {post_request.tone.value}.
    2. The post must be optimized for the {post_request.channel.value} audience.
    3. Be highly mindful of RMHC's brand: use encouraging language, avoid hyperbole, and always clearly state the 'why' (supporting families).
    4. Include relevant hashtags.
    5. The final output MUST be a JSON list of objects conforming to the PostCandidate schema.
    """
    
    prompt = f"""
    The target channel is: {post_request.channel.value}.
    The specific objective is: {post_request.objective}.
    The available data is: {json.dumps(post_request.context_data, indent=2)}.
    
    Generate 3 distinct PostCandidate objects now. Ensure each one uses the data provided.
    """
    
    # We define a custom list type for the response schema
    class PostCandidateList(BaseModel):
        candidates: List[PostCandidate]

    try:
        response_list = call_model_with_structured_output(
            system_instruction=generation_instruction,
            prompt=prompt,
            response_schema=PostCandidateList,
            model='gemini-2.5-flash'
        )
        # We need to assign a temporary unique ID for the Judge to reference
        for i, candidate in enumerate(response_list.candidates):
            candidate.candidate_id = f"CANDIDATE_{i+1}_{uuid.uuid4().hex[:4]}"
        
        return response_list.candidates
    except Exception as e:
        print(f"Error during post generation: {e}")
        # Return a fallback list in case of error
        return [PostCandidate(
            channel=post_request.channel,
            text=f"ERROR: Could not generate post. Objective: {post_request.objective}",
            rationale="Fallback post due to model error.",
            risk_flags=["MODEL_FAILURE"]
        )]


def judge_post_quality(candidates: List[PostCandidate], data_context: dict) -> JudgeFeedback:
    """
    (A) EXECUTING ACTION. The core LLM-as-a-Judge function (Day 4).
    The model evaluates the post candidates against strict RMHC brand guidelines.
    
    Args:
        candidates: The list of posts generated by generate_post_candidates.
        data_context: The raw data used, for context-aware judging (e.g., is the data too low?).
        
    Returns:
        JudgeFeedback: The final verdict (APPROVED, REJECTED, or NEEDS_EDIT) on the best candidate.
    """
    print(f"\n--- Calling LLM-as-a-Judge for {len(candidates)} Candidates ---")
    
    # The system instruction for the Judge model
    judge_instruction = """
    You are the 'RMHC Quality Gate', an essential component of Agent Evaluation. 
    Your role is to critically assess a set of PostCandidates for brand safety, tone, and effectiveness.
    
    CRITICAL JUDGMENT CRITERIA (Score out of 10):
    1. Brand Alignment (3 pts): Is the tone professional/appropriate for RMHC? Does it clearly state the charity's mission?
    2. Data Accuracy & Value (3 pts): Does the post correctly use the provided data (e.g., 75% goal)? Is the data worth celebrating?
    3. Call-to-Action (2 pts): Is the call-to-action clear and appropriate for the channel?
    4. Sensitivity (2 pts): Does the post avoid any language that could be seen as pressuring donors or boastful?
    
    Your final task is to select the single BEST post, provide a score and reasoning, and deliver the final verdict using the JudgeFeedback schema.
    """
    
    # Format candidates for the judge prompt
    candidate_list_str = "\n---\n".join([
        f"ID: {c.candidate_id}\nChannel: {c.channel.value}\nText: {c.text}\nRationale: {c.rationale}"
        for c in candidates
    ])
    
    prompt = f"""
    Evaluate the following candidates against the context data and the critical criteria.

    CONTEXT DATA USED: {json.dumps(data_context, indent=2)}
    
    CANDIDATES FOR EVALUATION:
    {candidate_list_str}
    
    Select the single best PostCandidate. If its score is 8 or higher, set approval_status to APPROVED. If it's a 7 and needs minor changes, use NEEDS_EDIT and provide the corrected text. Otherwise, set to REJECTED.
    """
    
    try:
        feedback = call_model_with_structured_output(
            system_instruction=judge_instruction,
            prompt=prompt,
            response_schema=JudgeFeedback,
            model='gemini-2.5-flash'
        )
        return feedback
    except Exception as e:
        print(f"Error during LLM-as-a-Judge call: {e}")
        # Return a safe, rejected response on error
        return JudgeFeedback(
            candidate_id="ERROR",
            score_out_of_10=0,
            approval_status=ApprovalStatus.REJECTED,
            reasoning="Model failure during Quality Gate check. Cannot approve.",
        )


def simulate_publish_post(final_post: PostCandidate) -> str:
    """
    (A) EXECUTING ACTION. Simulates the final publication step.
    This replaces a real API call (e.g., LinkedIn API, Slack webhook).
    
    Args:
        final_post: The post that was APPROVED by the judge.
        
    Returns:
        str: A log message confirming the simulated action.
    """
    # This function would contain the real API logic for the target channel.
    log_message = f"""
    [ACTION SUCCESS]: POSTING TO {final_post.channel.value}
    
    --- POST CONTENT ---
    {final_post.text}
    --- END POST ---
    
    Simulated POST request completed successfully.
    """
    return log_message

# --- List of all available tools for the Agent ---
AGENT_TOOLS: List[Callable] = [
    get_activity_summary,
    get_fundraising_summary,
    generate_post_candidates,
    judge_post_quality,
    simulate_publish_post
]
