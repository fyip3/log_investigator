import art
import json
import openai
import textwrap
import traceback
import os
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, List, Optional

from art import Trajectory
from langchain_core.utils.function_calling import convert_to_openai_tool
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openpipe import AsyncOpenPipe
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential


from project_types import ProjectPolicyConfig
from log_search_tools import search_logs, read_trace, LogEntry

load_dotenv()

DEBUG_MODE = False

def debug_print(*args):
    if DEBUG_MODE:
        print("[DEBUG]", *args, flush=True)


try:
    op_client: Optional[AsyncOpenPipe] = AsyncOpenPipe()
except Exception:
    op_client = None


class IncidentScenario(BaseModel):
    """Synthetic incident description for log investigation scenarios."""

    id: str
    description: str
    time_hint: Optional[str] = None
    expected_root_cause: Optional[str] = None


@dataclass
class FinalRubric:
    answer_correct: bool = False
    num_turns: int = 0
    attempted_answer: bool = False
    cant_parse_tool_call: bool = False
    bad_tool_call_name: bool = False
    bad_tool_call_args: bool = False
    ran_out_of_turns: bool = False
    returned_i_dont_know: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_metrics(self) -> dict[str, float | int]:
        return {
            "answer_correct": int(self.answer_correct),
            "num_turns": self.num_turns,
            "attempted_answer": int(self.attempted_answer),
            "cant_parse_tool_call": int(self.cant_parse_tool_call),
            "bad_tool_call_name": int(self.bad_tool_call_name),
            "bad_tool_call_args": int(self.bad_tool_call_args),
            "ran_out_of_turns": int(self.ran_out_of_turns),
            "returned_i_dont_know": int(self.returned_i_dont_know),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


def calculate_reward(
    policy_config: ProjectPolicyConfig, rubric: FinalRubric, traj: Trajectory
) -> float:
    """Reward function for log investigations."""
    if policy_config.stupid_simple_reward_fn:
        return float(rubric.answer_correct)

    # Hard failures
    if rubric.cant_parse_tool_call:
        return -2.0
    if rubric.bad_tool_call_name:
        return -1.9
    if rubric.bad_tool_call_args:
        return -1.8

    # Reached turn limit or said "I don't know"
    if rubric.ran_out_of_turns or rubric.returned_i_dont_know:
        return -0.5

    # Attempted an answer
    if rubric.attempted_answer:
        if rubric.answer_correct:
            # Reward faster answers slightly more
            speed_bonus = 0.5 * (1 - rubric.num_turns / max(policy_config.max_turns, 1))
            return 1.5 + speed_bonus
        else:
            return -1.5

    # Never really answered
    return 0.0


def tool_response(response: Any, tool_call_id: str | None) -> ChatCompletionMessageParam:
    """Generate a tool message for a tool call."""
    if tool_call_id is not None:
        # Native function calling mode
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(response),
        }
    else:
        # String-based mode - return as user message
        return {
            "role": "user",
            "content": json.dumps(response),
        }


tools: List[ChatCompletionToolParam] = [
    convert_to_openai_tool(search_logs),
    convert_to_openai_tool(read_trace),
    convert_to_openai_tool(lambda answer, metadata=None: answer),
]

# The agent should NOT see or specify this parameter
del tools[0]["function"]["parameters"]["properties"]["scenario_id"]
tools[0]["function"]["parameters"]["required"].remove("scenario_id")

del tools[1]["function"]["parameters"]["properties"]["scenario_id"]
tools[1]["function"]["parameters"]["required"].remove("scenario_id")

tools[-1]["function"]["name"] = "return_final_answer"
tools[-1]["function"]["description"] = (
    "Use this tool to return your final JSON investigation result to the user. "
    "The `answer` field must be a JSON string with keys: "
    "root_cause (str), suspected_service (str or null), "
    "primary_trace_ids (list[str]), key_log_ids (list[int]), "
    "timeline (list[str]), next_actions (list[str]). "
    "You may optionally include arbitrary metadata in the `metadata` object."
)
tools[-1]["function"]["parameters"] = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "JSON string with the final investigation result.",
        },
        "metadata": {
            "type": ["object", "null"],
            "description": "Optional extra structured data about the investigation.",
        },
    },
    "required": ["answer"],
}


def _extract_function_from_tool_call(tool_call: Any) -> tuple[Optional[str], Optional[str]]:
    """Handle both dict-style and OpenAI-object-style tool_calls."""
    func = None
    if hasattr(tool_call, "function"):
        func = tool_call.function
        name = getattr(func, "name", None)
        arguments = getattr(func, "arguments", None)
    elif isinstance(tool_call, dict):
        func = tool_call.get("function")
        if not isinstance(func, dict):
            return None, None
        name = func.get("name")
        arguments = func.get("arguments")
    else:
        return None, None

    if arguments is not None and not isinstance(arguments, str):
        arguments = json.dumps(arguments)

    return name, arguments


def _get_tool_call_id(tool_call: Any) -> str:
    if hasattr(tool_call, "id"):
        return getattr(tool_call, "id") or "missing_tool_call_id"
    if isinstance(tool_call, dict):
        return str(tool_call.get("id", "missing_tool_call_id"))
    return "missing_tool_call_id"

@retry(stop=stop_after_attempt(3))
async def determine_if_root_cause_is_correct(
    ai_root_cause: str,
    scenario: IncidentScenario,
    client: Any,
    judge_model: str,
    debug: bool = False,
) -> bool:
    """
    Judge whether the AI's root_cause matches the canonical label.
    Returns True if they clearly describe the same underlying issue.
    """

    system_prompt = (
        "You will be given an incident description, a canonical root cause label, "
        "and an AI's free-text root cause explanation. "
        "Return True only if the AI's explanation clearly corresponds to the same "
        "underlying technical root cause as the canonical label (even if the wording differs). "
        "Return False if the explanation is vague, generic, inconsistent with the description, "
        "or if the AI says the root cause is unknown or unclear. "
        "Return ONLY the word True or False."
    )

    user_content = (
        f"Incident description: {scenario.description}\n"
        f"Canonical root cause label: {scenario.expected_root_cause}\n"
        f"AI root cause explanation: {ai_root_cause}"
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    if debug:
        print(f"[Judge] Scenario ID: {scenario.id}")
        print(f"[Judge] Canonical: {scenario.expected_root_cause}")
        print(f"[Judge] AI Explanation: {ai_root_cause}\n")
        print("[Judge] Sending messages:")
        print(system_prompt)
        print(user_content)

    completion = await client.chat.completions.create(
        model=judge_model,
        messages=messages,
        max_completion_tokens=4,
        temperature=0,
    )

    if not completion.choices:
        if debug:
            print("No choices returned from judge model")
        return False

    text = (completion.choices[0].message.content or "").strip().lower()

    if debug:
        print(f"Raw judge output: {repr(text)}")

    return text.startswith("t")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
async def call_policy(client, model_name, messages, tools, tool_choice, max_tokens):
    return await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice,
    )

async def rollout(
    model: art.Model,
    scenario: IncidentScenario,
) -> Trajectory:
    """Multi-turn log-investigation rollout, ART-compatible."""
    rollout_start_time = datetime.now()
    rubric = FinalRubric()
    traj = Trajectory(
        messages_and_choices=[],
        reward=0,
        metadata={"scenario_id": scenario.id},
    )
    assert isinstance(model.config, ProjectPolicyConfig)

    system_prompt = textwrap.dedent(f"""\
    You are a production incident investigation agent. You are given a vague incident report and a set of tools for searching structured application logs. Use the tools to discover the technical root cause. You may take up to {model.config.max_turns} turns, so if one search doesn’t reveal useful information, you can try different keywords, services, or time ranges. The root cause is GUARANTEED to exist in the logs.

    IMPORTANT:
    - The root cause is the UNDERLYING issue, not superficial symptoms.
    - If you find generic errors, keep digging for WHY they happened.
    - ALL parameters to search_logs are optional, so you can use any combination of parameters or none at all. For example, you can search with ONLY a time window and min_level to see all errors in that period.
    - The time_hint is often an APPROXIMATE estimate of the incident. The root cause evidence may appear BEFORE/AFTER the hint. MUST start by searching wider time windows, such as the full 30 minutes before and after the time hint. Once you are more confident, you can narrow the search if needed.

    The available logs come from multiple services in a distributed system. 
    Your goal is to identify the root cause and return a final JSON incident 
    report using return_final_answer.

    For reference, the system contains the following known services. 
    These are the canonical service names you may see in log entries or 
    use when filtering with the 'service' parameter in search_logs:

    Known services:
    - auth-service
    - billing-service
    - cache-redis
    - cache-service
    - checkout-web
    - db-service
    - deploy-controller
    - gateway
    - inventory-service
    - kube-events
    - notifications-service
    - orders-worker
    - payments-api
    - queue-worker
    - search-service
    - user-service

    You are NOT being told which of these services are involved in the 
    current incident. You must still rely on the logs and your searches to 
    identify the true root cause.

    """)

    if not model.config.use_tools:
        system_prompt += textwrap.dedent(f"""\
            
            Here are the tools you can use:
            {tools}

            Respond with a valid JSON object with exactly these fields:
            - "tool_name": (str) the tool to call
            - "tool_args": (dict) the arguments (all parameters are optional)

            Example with all parameters:
            {{
                "tool_name": "search_logs",
                "tool_args": {{
                    "keywords": ["error", "timeout"],
                    "start_time": "2025-01-20T16:30:00Z",
                    "end_time": "2025-01-20T16:45:00Z",
                    "service": "payments-api"
                }}
            }}

            Example with minimal parameters (time and level only):
            {{
                "tool_name": "search_logs",
                "tool_args": {{
                    "start_time": "2025-01-20T16:00:00Z",
                    "end_time": "2025-01-20T17:00:00Z",
                    "min_level": "ERROR"
                }}
            }}
        """)



    user_message = scenario.description
    if scenario.time_hint:
        user_message = f"{scenario.description}\n\nTime hint: {scenario.time_hint}"

    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    client = model.openai_client()
    completion = None
    final_answer_json: Optional[dict[str, Any]] = None

    while True:
        rubric.num_turns += 1

        if rubric.num_turns > model.config.max_turns:
            rubric.ran_out_of_turns = True
            break

        try:
            completion = await call_policy(
                client,
                model.get_inference_name(),
                traj.messages(),
                tools if model.config.use_tools else None,
                "auto" if model.config.use_tools and not model.trainable else None,
                model.config.max_tokens,
            )
        except openai.InternalServerError as e:
            debug_print("POLICY CALL FAILED:", e)
            rubric.cant_parse_tool_call = True
            break

        # Track token usage if available
        if completion.usage is not None:
            rubric.prompt_tokens += completion.usage.prompt_tokens or 0
            rubric.completion_tokens += completion.usage.completion_tokens or 0

        if not completion.choices:
            rubric.cant_parse_tool_call = True
            break

        choice = completion.choices[0]
        message = choice.message

        debug_print("Turn", rubric.num_turns)
        debug_print("Assistant message:", message)

        if getattr(model, "trainable", False):
            traj.messages_and_choices.append(choice)
        else:
            try:
                assistant_msg_dict = message.model_dump(
                    exclude={"refusal", "annotations", "reasoning_content"}
                )
            except Exception:
                assistant_msg_dict = {
                    "role": message.role,
                    "content": message.content,
                    "tool_calls": getattr(message, "tool_calls", None),
                }
            traj.messages_and_choices.append(assistant_msg_dict)

        if model.config.use_tools:
            # Native function calling mode
            tool_calls = getattr(message, "tool_calls", None)
            debug_print("Tool calls:", tool_calls)
            if not tool_calls:
                print(f"\n!!! BAD TOOL CALL: No tool calls in message")
                print(f"Message content: {message.content}")
                rubric.bad_tool_call_args = True
                break

            # Only handle one tool call at a time
            if len(tool_calls) > 1:
                tool_calls = tool_calls[:1]

            tool_call = tool_calls[0]
            tool_call_id = _get_tool_call_id(tool_call)
            tool_name, tool_args_json = _extract_function_from_tool_call(tool_call)

            if not tool_name or tool_args_json is None:
                print(f"\n!!! BAD TOOL CALL: Missing tool name or args")
                print(f"Tool name: {tool_name}")
                print(f"Tool args JSON: {tool_args_json}")
                print(f"Tool call: {tool_call}")
                rubric.bad_tool_call_args = True
                break

            try:
                tool_args = json.loads(tool_args_json)
                assert isinstance(tool_args, dict)
            except Exception as e:
                print(f"\n!!! BAD TOOL CALL: Failed to parse tool args JSON")
                print(f"Tool name: {tool_name}")
                print(f"Tool args JSON: {tool_args_json}")
                print(f"Error: {e}")
                rubric.bad_tool_call_args = True
                break
        else:
            # String-based tool calling mode
            raw_content = message.content
            if raw_content is None or not isinstance(raw_content, str):
                print(f"\n!!! BAD TOOL CALL: Message content is None or not a string")
                rubric.cant_parse_tool_call = True
                break

            # Extract JSON from content
            start_index = raw_content.find("{")
            end_index = raw_content.rfind("}")
            if not (start_index != -1 and end_index != -1 and start_index < end_index):
                print(f"\n!!! BAD TOOL CALL: No JSON object found in content")
                rubric.cant_parse_tool_call = True
                break
            json_str = raw_content[start_index : end_index + 1]

            try:
                tool_call_json = json.loads(json_str)
            except Exception as e:
                print(f"\n!!! BAD TOOL CALL: Failed to parse JSON - {e}")
                rubric.cant_parse_tool_call = True
                break

            if "tool_name" not in tool_call_json or "tool_args" not in tool_call_json:
                rubric.bad_tool_call_args = True
                print(f"\n!!! BAD TOOL CALL: Missing tool_name or tool_args")
                break

            tool_name = tool_call_json.get("tool_name")
            tool_args = tool_call_json.get("tool_args")
            tool_call_id = None  # No tool_call_id in string-based mode

            print(f"\nAI tool call: {tool_name}")
            print(f"   Args: {json.dumps(tool_args, indent=6)}")

        debug_print("Tool:", tool_name)
        debug_print("Args:", tool_args)

        match tool_name:
            case "search_logs":
                # Inject scenario_id
                tool_args["scenario_id"] = scenario.id
                result = search_logs(**tool_args)
                payload = [asdict(r) for r in result]
                print(f"📋 Tool output: Found {len(result)} log entries")
                traj.messages_and_choices.append(
                    tool_response(payload, tool_call_id)
                )
            case "read_trace":
                # Inject scenario_id
                tool_args["scenario_id"] = scenario.id
                result = read_trace(**tool_args)
                payload = [asdict(r) for r in result]
                print(f"📋 Tool output: Found {len(result)} trace events")
                traj.messages_and_choices.append(
                    tool_response(payload, tool_call_id)
                )
            case "return_final_answer":
                raw_answer = tool_args.get("answer")

                # Normalise to a string
                if isinstance(raw_answer, str):
                    answer_str = raw_answer
                else:
                    answer_str = json.dumps(raw_answer)

                # Try to parse once for convenience
                parsed_answer = None
                try:
                    parsed_answer = json.loads(answer_str)
                    if not isinstance(parsed_answer, dict):
                        parsed_answer = None
                except Exception:
                    parsed_answer = None

                rubric.attempted_answer = True

                if parsed_answer is not None:
                    final_answer_json = parsed_answer
                else:
                    final_answer_json = {"raw_answer": answer_str}

                # Store only scalar in metadata
                if parsed_answer is not None:
                    root_cause_text = str(parsed_answer.get("root_cause", ""))
                else:
                    root_cause_text = answer_str

                traj.metadata["final_answer"] = root_cause_text  # <- string only

                print(f"\n📋 {scenario.id}")
                if parsed_answer is not None:
                    print(f"\n AI final answer:")
                    print(f"   Root cause: {root_cause_text[:150]}...")
                    print(f"   Suspected service: {parsed_answer.get('suspected_service', 'N/A')}")
                else:
                    print(f"\n AI final answer: {answer_str[:150]}...")

                # If we have a canonical label, use LLM-as-judge
                try:
                    if scenario.expected_root_cause:
                        ai_root_cause_only = root_cause_text

                        print(f"\n True answer: {scenario.expected_root_cause}")

                        rubric.answer_correct = await determine_if_root_cause_is_correct(
                            ai_root_cause=ai_root_cause_only,
                            scenario=scenario,
                            client=client,
                            judge_model=os.getenv("JUDGE_MODEL"), 
                            debug=False,
                        )

                        verdict = "✅ CORRECT" if rubric.answer_correct else "❌ INCORRECT"
                        print(f"\n LLM judge verdict: {verdict}")
                except Exception as e:
                    debug_print(f"JUDGE FAILED WITH EXCEPTION: {type(e).__name__}: {e}")
                    debug_print(f"JUDGE TRACEBACK:\n{traceback.format_exc()}")
                    rubric.answer_correct = False


                # Done with this trajectory
                break


            case _:
                rubric.bad_tool_call_name = True
                break

        # If we just processed return_final_answer, stop
        if final_answer_json is not None:
            break

    reward = calculate_reward(model.config, rubric, traj)
    traj.reward = reward
    traj.metrics = rubric.to_metrics()
    rollout_end_time = datetime.now()
    duration_seconds = (rollout_end_time - rollout_start_time).total_seconds()

    traj.metrics["duration"] = duration_seconds

    debug_print("Rubric:", rubric.to_metrics())
    debug_print("Reward:", reward)
    debug_print("Trajectory length:", len(traj.messages_and_choices))

    return traj
