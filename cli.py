#!/usr/bin/env python3
"""CLI runner for Sir Alex FPL Agent.

Runs the agent headlessly for research, evals, and autoresearch loops.
Outputs structured JSON to stdout for machine consumption.

Usage:
    # Single message (default: ChatAnthropic with ANTHROPIC_API_KEY)
    python cli.py "Who should I captain this week?"

    # Specify model
    python cli.py --model claude-sonnet-4-20250514 "Compare Salah and Palmer"

    # With FPL team context
    python cli.py --team-id 7704194 "Rate my team"

    # Multi-turn conversation from JSONL file
    python cli.py --conversation conversation.jsonl

    # Pipe-friendly: read message from stdin
    echo "Top 3 differentials under 5% ownership?" | python cli.py

    # Disable Sigil/OTel (for offline testing without Sigil running)
    SIGIL_DISABLED=1 python cli.py "Who should I captain?"

Environment variables:
    ANTHROPIC_API_KEY    - Required. Anthropic API key.
    AGENT_MODEL          - Default model (overridden by --model).
    FPL_TEAM_ID          - Default team ID (overridden by --team-id).
    SIGIL_DISABLED       - Set to 1 to disable Sigil/OTel instrumentation.
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv()


def _disable_sigil_if_requested():
    """Disable OTel/Sigil by setting env vars before agent module import."""
    if os.getenv("SIGIL_DISABLED", "").strip() in ("1", "true", "yes"):
        os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:1")
        os.environ.setdefault("SIGIL_GENERATION_ENDPOINT", "http://localhost:1/noop")


def build_llm(model_name: str):
    """Build a LangChain chat model from the model name."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model_name)


def run_single(
    message: str,
    model_name: str,
    thread_id: str,
    fpl_team_id: str | None,
) -> dict:
    """Run a single message through the agent and return structured output."""
    from agent import run_agent

    llm = build_llm(model_name)

    start = time.time()
    response = run_agent(
        user_message=message,
        model_name=model_name,
        thread_id=thread_id,
        fpl_team_id=fpl_team_id,
        llm=llm,
    )
    elapsed = time.time() - start

    return {
        "response": response.content,
        "tool_calls": [
            {"name": tc.name, "args": tc.args, "result": tc.result}
            for tc in response.tool_calls
        ],
        "metrics": {
            "latency_seconds": round(elapsed, 3),
            "tool_call_count": len(response.tool_calls),
            "response_length": len(response.content),
        },
        "config": {
            "model": model_name,
            "thread_id": thread_id,
            "fpl_team_id": fpl_team_id,
        },
    }


def run_conversation(
    conversation_path: str,
    model_name: str,
    fpl_team_id: str | None,
) -> list[dict]:
    """Run a multi-turn conversation from a JSONL file.

    Each line should be a JSON object with a "message" field.
    Returns a list of results, one per turn.
    """
    thread_id = str(uuid.uuid4())
    results = []

    with open(conversation_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            turn = json.loads(line)
            message = turn["message"]
            result = run_single(message, model_name, thread_id, fpl_team_id)
            results.append(result)

    return results


def flush_sigil() -> None:
    """Flush Sigil telemetry before the CLI process exits."""
    try:
        from agent.agent import sigil_client
    except Exception:
        return

    if sigil_client is None:
        return

    for method_name in ("flush", "force_flush", "shutdown"):
        method = getattr(sigil_client, method_name, None)
        if callable(method):
            method()
            return


def main():
    parser = argparse.ArgumentParser(
        description="Sir Alex FPL Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "message",
        nargs="?",
        help="Message to send to the agent (reads from stdin if omitted)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("AGENT_MODEL", "claude-haiku-4-5-20251001"),
        help="Model name (default: AGENT_MODEL env or claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--team-id",
        default=os.getenv("FPL_TEAM_ID"),
        help="FPL team ID for personalized context",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Thread ID for conversation continuity (auto-generated if omitted)",
    )
    parser.add_argument(
        "--conversation",
        help="Path to JSONL file with multi-turn conversation",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )

    args = parser.parse_args()

    # Disable Sigil if requested (must happen before importing agent)
    _disable_sigil_if_requested()

    thread_id = args.thread_id or str(uuid.uuid4())

    if args.conversation:
        results = run_conversation(args.conversation, args.model, args.team_id)
        output = {"conversation": results, "turns": len(results)}
    else:
        message = args.message
        if message is None:
            if sys.stdin.isatty():
                parser.error("No message provided. Pass as argument or pipe via stdin.")
            message = sys.stdin.read().strip()
            if not message:
                parser.error("Empty message from stdin.")

        output = run_single(message, args.model, thread_id, args.team_id)

    indent = 2 if args.pretty else None
    json.dump(output, sys.stdout, indent=indent, ensure_ascii=False)
    sys.stdout.write("\n")
    flush_sigil()


if __name__ == "__main__":
    main()
