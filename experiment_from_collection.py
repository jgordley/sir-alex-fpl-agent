"""Run a Sigil experiment from a saved-conversation collection.

This uses the current ``sigil_sdk.experiments`` API. The collection is read with
the core Sigil conversation APIs, converted into a ``TestSuite``, and replayed as
typed experiment trials.

Run it:

    COLLECTION_ID=<id> PYTHONPATH=. python experiment_from_collection.py
"""

from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from sigil_sdk import experiments as sigil

import experiment as fixed_suite
from agent.agent import create_agent

MODEL = os.getenv("AGENT_MODEL", "claude-haiku-4-5-20251001")
SUITE_VERSION = "2026-06-29"


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is required.")
    return value


def _build_client() -> sigil.Client:
    return sigil.Client(
        _required_env("SIGIL_ENDPOINT"),
        tenant_id=os.getenv("SIGIL_AUTH_TENANT_ID", "").strip(),
        ingest_token=_required_env("SIGIL_AUTH_TOKEN"),
        grafana_url=os.getenv("SIGIL_GRAFANA_URL", "").strip(),
        actor=os.getenv("SIGIL_INGEST_ACTOR", "").strip(),
    )


def _role_is_user(role: Any) -> bool:
    normalized = str(role or "").strip().lower()
    return normalized in ("user", "message_role_user") or normalized.endswith(
        "_role_user"
    )


def _part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return ""
    value = part.get("text")
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("text") or value.get("content") or "")
    return str(part.get("content") or "")


def _message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    parts = message.get("parts")
    if isinstance(parts, list):
        text = "\n".join(
            chunk for chunk in (_part_text(part) for part in parts) if chunk
        )
        if text:
            return text
    content = message.get("content")
    if isinstance(content, str):
        return content
    return ""


def _initial_user_prompt(conversation: dict[str, Any]) -> str:
    for generation in conversation.get("generations") or []:
        if not isinstance(generation, dict):
            continue
        for message in generation.get("input") or []:
            if isinstance(message, dict) and _role_is_user(message.get("role")):
                text = _message_text(message).strip()
                if text:
                    return text
    for message in conversation.get("messages") or []:
        if isinstance(message, dict) and _role_is_user(
            message.get("role") or message.get("type")
        ):
            text = _message_text(message).strip()
            if text:
                return text
    return ""


def suite_from_collection(
    client: sigil.Client,
    collection_id: str,
    *,
    limit: int | None = None,
) -> sigil.TestSuite:
    members = client.core.list_collection_members(collection_id)
    if limit is not None:
        members = members[:limit]

    cases: list[sigil.TestCase] = []
    for index, member in enumerate(members, start=1):
        conversation_id = str(member.get("conversation_id") or "").strip()
        if not conversation_id:
            continue
        conversation = client.core.get_conversation(conversation_id)
        prompt = _initial_user_prompt(conversation)
        if not prompt:
            print(f"[sigil] skipping {conversation_id}: no initial user prompt found")
            continue
        saved_id = str(member.get("saved_id") or member.get("id") or conversation_id)
        case_id = sigil.stable_id("case", collection_id, saved_id, index)
        cases.append(
            sigil.TestCase(
                test_case_id=case_id,
                name=str(member.get("name") or f"Collection item {index}"),
                input=prompt,
                expected="Answer the user's Fantasy Premier League prompt directly and helpfully.",
                category="collection-replay",
                metadata={
                    "collection_id": collection_id,
                    "source_conversation_id": conversation_id,
                    "saved_id": saved_id,
                },
            )
        )

    if not cases:
        raise SystemExit(
            f"collection {collection_id} produced no replayable test cases."
        )
    return sigil.TestSuite(
        suite_id=f"collection:{collection_id}",
        name=f"Collection replay {collection_id}",
        version=SUITE_VERSION,
        test_cases=cases,
    )


def main() -> None:
    load_dotenv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True
    )
    global MODEL
    MODEL = os.getenv("AGENT_MODEL", MODEL)
    fixed_suite.MODEL = MODEL

    collection_id = os.getenv("COLLECTION_ID", "").strip()
    if not collection_id:
        raise SystemExit("COLLECTION_ID is required.")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is required to run the agent and judge.")

    client = _build_client()
    limit_env = os.getenv("DATASET_LIMIT", "").strip()
    limit = int(limit_env) if limit_env else None
    print(f"[sigil] pulling dataset from collection {collection_id} ...")
    suite = suite_from_collection(client, collection_id, limit=limit)
    print(f"[sigil] dataset has {len(suite.cases)} item(s):")
    for case in suite.cases:
        print(f"  - {case.test_case_id}: {str(case.input)[:80]!r}")

    experiment_id = (
        os.getenv("SIGIL_EXPERIMENT_ID", "").strip()
        or os.getenv("RUN_ID", "").strip()
        or f"sir-alex-fpl-collection-{int(time.time())}"
    )
    os.environ["SIGIL_EXPERIMENT_ID"] = experiment_id

    agent = create_agent(MODEL, llm=ChatAnthropic(model=MODEL))
    judge_llm = ChatAnthropic(model=MODEL)
    quality = sigil.Evaluator(
        "fpl_llm_judge.quality", version=fixed_suite.JUDGE_VERSION, kind="llm_judge"
    )
    skip_scores = os.getenv("SKIP_SCORES", "").strip().lower() in ("1", "true", "yes")
    report = None

    try:
        with sigil.Experiment(
            client,
            experiment_id=experiment_id,
            name=f"Sir Alex FPL collection replay ({MODEL})",
            description=f"User-prompt replay of collection {collection_id}",
            suite=suite,
            candidate={
                "agent_name": os.getenv("SIGIL_AGENT_NAME", "sir-alex-fpl"),
                "agent_version": os.getenv("SIGIL_AGENT_VERSION", "1.0.0"),
                "model_provider": "anthropic",
                "model_name": MODEL,
            },
            tags=[
                "sir-alex-fpl",
                "collection-replay",
                MODEL,
                f"collectionId:{collection_id}",
            ],
            metadata={"collection_id": collection_id},
        ) as experiment:
            for case in suite.cases:
                with experiment.trial(case, metadata=case.metadata) as trial:
                    conversation_id = sigil.stable_id(
                        "conv", experiment.experiment_id, case.test_case_id
                    )
                    trial.bind_conversation(conversation_id)
                    answer = fixed_suite.run_agent_case(
                        agent, experiment.client, case, conversation_id
                    )
                    trial.record_io(
                        input=case.input,
                        output=answer,
                        model_provider="anthropic",
                        model_name=MODEL,
                        agent_name=os.getenv("SIGIL_AGENT_NAME", "sir-alex-fpl"),
                    )
                    if skip_scores:
                        continue
                    judgement = fixed_suite.run_judge(
                        client=experiment.client,
                        judge_llm=judge_llm,
                        experiment_id=experiment.experiment_id,
                        case=case,
                        answer=answer,
                        evaluator_id=quality.evaluator_id,
                        operation_name="llm-judge-quality",
                        prompt_template=fixed_suite.QUALITY_PROMPT,
                    )
                    trial.score(
                        "final",
                        judgement.score,
                        passed=judgement.passed,
                        explanation=judgement.reason,
                        evaluator=quality,
                        grader_conversation_id=judgement.conversation_id,
                        grader_generation_id=judgement.generation_id,
                    )
            if not skip_scores:
                report = experiment.report()
    finally:
        client.shutdown()

    print(
        f"\n[sigil] run '{experiment_id}' done - {experiment.accepted_scores} score(s)"
    )
    print(f"[sigil] view: {experiment.url}")
    if report is not None:
        print(
            f"[sigil] mean quality={report.summary.final_score_avg:.3f}  "
            f"pass_rate={report.summary.pass_rate:.3f}  "
            f"generations={report.summary.n_generations}"
        )


if __name__ == "__main__":
    main()
