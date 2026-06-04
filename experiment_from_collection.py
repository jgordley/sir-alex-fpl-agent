"""Run a Sigil experiment whose dataset is pulled from a Sigil collection.

Unlike ``experiment.py`` (which hard-codes a dataset and uses the LangGraph
adapter), this script demonstrates the *core* ``sigil-sdk`` feature:

    1. Pull a dataset straight from a Sigil collection of saved conversations
       (``dataset_from_collection``). Each item's ``input`` is the *initial user
       prompt* of that conversation.
    2. Re-run the agent from scratch on that prompt (the "user-prompt kickoff").
    3. Grade the fresh answer with a pure numeric LLM-judge score and publish.

The experiment run is linked back to the collection: its ``collection_id`` is
set and a ``collectionId:<id>`` tag is added, so you can find the run from the
collection in the Sigil UI / ``gcx``.

Run it:

    COLLECTION_ID=<id> PYTHONPATH=. python experiment_from_collection.py

Env (same Sigil config as experiment.py — see .env):
    COLLECTION_ID       required; the collection to build the dataset from
    ANTHROPIC_API_KEY   required; drives the agent and the LLM judge
    AGENT_MODEL         agent + judge model (default from .env)
    RUN_ID              stable run id (defaults to a timestamp)
    DATASET_LIMIT       optional cap on number of conversations pulled
"""

from __future__ import annotations

import json
import os
import re
import time

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from sigil_sdk import (
    ApiConfig,
    Client,
    ClientConfig,
    DatasetItem,
    ExperimentRun,
    ExperimentRunner,
    Generation,
    GenerationStart,
    ModelRef,
    ScoreOutput,
    ScoreValue,
    TargetResult,
    assistant_text_message,
    dataset_from_collection,
    user_text_message,
)

from agent.agent import create_agent
from agent.utils import build_system_prompt

MODEL = os.getenv("AGENT_MODEL", "claude-haiku-4-5-20251001")
JUDGE_VERSION = "2026-06-03"


def _final_answer(result: dict) -> str:
    """Extract the agent's final textual answer from the graph result."""
    for message in reversed(result.get("messages", [])):
        content = getattr(message, "content", "")
        tool_calls = getattr(message, "tool_calls", None)
        if content and not tool_calls:
            return str(content)
    return ""


def make_target(agent):
    """Target: re-run the agent from the dataset item's initial user prompt.

    The agent call is recorded as a Sigil generation via ``run.start_generation``
    so it carries the experiment run_id and its id is captured for scoring — this
    is the framework-free (core SDK) integration point.
    """

    def target(item: DatasetItem, run: ExperimentRun) -> TargetResult:
        with run.start_generation(
            GenerationStart(
                model=ModelRef(provider="anthropic", name=MODEL),
                agent_name=os.getenv("SIGIL_AGENT_NAME", "sir-alex-fpl"),
                operation_name="generateText",
                conversation_title=str(item.input)[:120],
            )
        ) as rec:
            result = agent.invoke(
                {"messages": [("system", build_system_prompt()), ("user", str(item.input))]}
            )
            answer = _final_answer(result)
            rec.set_result(
                Generation(
                    model=ModelRef(provider="anthropic", name=MODEL),
                    input=[user_text_message(str(item.input))],
                    output=[assistant_text_message(answer)],
                )
            )
        return TargetResult(output=answer)

    return target


def make_quality_judge(client: Client):
    """A pure numeric quality score (0..1) from an LLM judge, recorded in Sigil."""

    judge_llm = ChatAnthropic(model=MODEL)
    prompt_template = (
        "You are grading a Fantasy Premier League assistant's answer for overall quality.\n\n"
        "User question:\n{input}\n\n"
        "Assistant answer:\n{output}\n\n"
        "Score 1.0 for an answer that directly addresses the question, is specific, and is useful. "
        "Score 0.0 for an answer that misses the request or gives unusable advice.\n\n"
        'Reply with ONLY a JSON object: {{"score": <float 0-1>, "pass": <true|false>, "reason": "<one sentence>"}}.'
    )

    def judge(item: DatasetItem, result: TargetResult) -> list[ScoreOutput]:
        prompt = prompt_template.format(input=item.input, output=result.output)
        with client.start_generation(
            GenerationStart(
                model=ModelRef(provider="anthropic", name=MODEL),
                agent_name="fpl-llm-judge",
                operation_name="llm-judge-quality",
            )
        ) as rec:
            raw = str(judge_llm.invoke(prompt).content)
            rec.set_result(
                Generation(
                    model=ModelRef(provider="anthropic", name=MODEL),
                    input=[user_text_message(prompt)],
                    output=[assistant_text_message(raw)],
                )
            )
        score, passed, reason = _parse_judgement(raw)
        return [
            ScoreOutput(
                evaluator_id="fpl_llm_judge.quality",
                evaluator_version=JUDGE_VERSION,
                score_key="quality",
                value=ScoreValue(number=score),
                passed=passed,
                explanation=reason,
            )
        ]

    return judge


def _parse_judgement(raw: str) -> tuple[float, bool, str]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            score = max(0.0, min(1.0, float(data.get("score", 0.0))))
            passed = bool(data.get("pass")) if isinstance(data.get("pass"), bool) else score >= 0.5
            reason = str(data.get("reason", "")).strip() or "(no reason given)"
            return score, passed, reason
        except (ValueError, TypeError):
            pass
    return 0.0, False, f"could not parse judge output: {raw[:160]}"


def build_client() -> Client:
    endpoint = os.getenv("SIGIL_ENDPOINT", "http://localhost:8080")
    return Client(ClientConfig(api=ApiConfig(endpoint=endpoint)))


def main() -> None:
    load_dotenv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True
    )
    global MODEL
    MODEL = os.getenv("AGENT_MODEL", MODEL)

    collection_id = os.getenv("COLLECTION_ID", "").strip()
    if not collection_id:
        raise SystemExit("COLLECTION_ID is required (the Sigil collection to build the dataset from).")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is required to run the agent and judge.")

    client = build_client()

    limit_env = os.getenv("DATASET_LIMIT", "").strip()
    limit = int(limit_env) if limit_env else None
    print(f"[sigil] pulling dataset from collection {collection_id} ...")
    dataset = dataset_from_collection(client, collection_id, limit=limit)
    if not dataset:
        raise SystemExit(f"collection {collection_id} produced no dataset items.")
    print(f"[sigil] dataset has {len(dataset)} item(s):")
    for item in dataset:
        print(f"  - {item.id}: {str(item.input)[:80]!r}")

    agent = create_agent(MODEL, llm=ChatAnthropic(model=MODEL))
    run_id = os.getenv("RUN_ID", f"sir-alex-fpl-collection-{int(time.time())}")

    runner = ExperimentRunner(
        client=client,
        run_id=run_id,
        name=f"Sir Alex FPL — collection replay ({MODEL})",
        description=f"User-prompt kickoff replay of collection {collection_id}",
        dataset={"id": f"collection:{collection_id}", "version": JUDGE_VERSION},
        candidate={"model": MODEL},
        tags=["sir-alex-fpl", "collection-replay", MODEL],
        collection_id=collection_id,
        agent_name=os.getenv("SIGIL_AGENT_NAME", "sir-alex-fpl"),
        agent_version=os.getenv("SIGIL_AGENT_VERSION", "1.0.0"),
    )

    result = runner.run(dataset, make_target(agent), [make_quality_judge(client)])

    print(f"\n[sigil] run '{run_id}' done — {result.accepted_scores} score(s)")
    print(f"[sigil] view: {result.url}")
    if result.report and result.report.summary:
        s = result.report.summary
        print(f"[sigil] mean quality={s.mean_score:.3f}  pass_rate={s.pass_rate:.3f}  generations={s.n_generations}")

    client.shutdown()


if __name__ == "__main__":
    main()
