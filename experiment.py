"""Run the Sir Alex FPL agent as a Sigil offline-evaluation experiment.

This uses the current ``sigil_sdk.experiments`` API:

    PYTHONPATH=. python experiment.py

Required env:
    ANTHROPIC_API_KEY   drives the agent and LLM judges
    SIGIL_ENDPOINT      Grafana AI Observability Sigil ingest URL
    SIGIL_AUTH_TOKEN    Grafana Cloud access policy token with sigil:write

Optional env:
    SIGIL_AUTH_TENANT_ID    Grafana Cloud stack/tenant id
    SIGIL_GRAFANA_URL       Grafana stack URL for deep links
    SIGIL_EXPERIMENT_ID     stable experiment id; RUN_ID is accepted as a legacy alias
    SKIP_SCORES             set to 1 to run trials without LLM judges
    MIN_PASS_RATE           fail the process if the report pass rate is lower
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from sigil_sdk import experiments as sigil

from agent.agent import create_agent
from agent.utils import build_system_prompt

MODEL = os.getenv("AGENT_MODEL", "claude-haiku-4-5-20251001")
SUITE_VERSION = "2026-06-29"
JUDGE_VERSION = "2026-06-29"


DATASET: list[sigil.TestCase] = [
    sigil.TestCase(
        test_case_id="captain-advice",
        input="Who should I consider captaining this gameweek?",
        expected="Recommends at least one specific player to captain and gives brief reasoning.",
        category="advice",
        metadata={"task_id": "captain_advice", "task_category": "advice"},
    ),
    sigil.TestCase(
        test_case_id="player-stats-haaland",
        input="How is Erling Haaland performing this season?",
        expected="Reports Haaland's FPL stats (for example goals, points, or form) and a short assessment.",
        category="stats",
        metadata={"task_id": "player_stats", "task_category": "stats"},
    ),
    sigil.TestCase(
        test_case_id="compare-salah-palmer",
        input="Compare Mohamed Salah and Cole Palmer for FPL.",
        expected="Compares both players with relevant stats and ends with a recommendation.",
        category="compare",
        metadata={"task_id": "compare_players", "task_category": "compare"},
    ),
    sigil.TestCase(
        test_case_id="leaders-goals",
        input="Who are the top goal scorers in the Premier League right now?",
        expected="Lists several specific players as the leading goal scorers.",
        category="leaders",
        metadata={"task_id": "league_leaders_goals", "task_category": "leaders"},
    ),
    sigil.TestCase(
        test_case_id="differentials",
        input="Suggest a couple of differential picks for my FPL team.",
        expected="Names specific players framed as differentials or low-ownership picks.",
        category="advice",
        metadata={"task_id": "differentials", "task_category": "advice"},
    ),
    sigil.TestCase(
        test_case_id="budget-defender",
        input="Recommend a budget defender under 5.0 million.",
        expected="Recommends at least one specific defender and acknowledges the budget constraint.",
        category="advice",
        metadata={"task_id": "budget_pick", "task_category": "advice"},
    ),
    sigil.TestCase(
        test_case_id="transfer-hit",
        input="Should I take a -4 point hit to bring in a player this week?",
        expected="Gives reasoned advice about taking a points hit and mentions the trade-offs.",
        category="advice",
        metadata={"task_id": "transfer_decision", "task_category": "advice"},
    ),
    sigil.TestCase(
        test_case_id="player-stats-saka",
        input="How has Bukayo Saka done this season?",
        expected="Reports Saka's FPL stats or form and a short assessment.",
        category="stats",
        metadata={"task_id": "player_stats", "task_category": "stats"},
    ),
    sigil.TestCase(
        test_case_id="leaders-assists",
        input="Who leads the Premier League in assists?",
        expected="Names one or more players leading the league in assists.",
        category="leaders",
        metadata={"task_id": "league_leaders_assists", "task_category": "leaders"},
    ),
    sigil.TestCase(
        test_case_id="persona-team-talk",
        input="Give me a motivational team talk for my FPL decisions this week.",
        expected="Responds in the Sir Alex Ferguson motivational persona and stays on FPL/football.",
        category="persona",
        metadata={"task_id": "persona", "task_category": "persona"},
    ),
]


@dataclass(frozen=True)
class Judgement:
    score: float
    passed: bool
    reason: str
    generation_id: str
    conversation_id: str


def _final_answer(result: dict[str, Any]) -> str:
    for message in reversed(result.get("messages", [])):
        content = getattr(message, "content", "")
        tool_calls = getattr(message, "tool_calls", None)
        if content and not tool_calls:
            return str(content)
    return ""


def _agent_config(
    *,
    client: sigil.Client,
    case: sigil.TestCase,
    conversation_id: str,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "metadata": {
            "conversation_id": conversation_id,
            "experiment_run_id": os.getenv("SIGIL_EXPERIMENT_ID")
            or os.getenv("RUN_ID", ""),
            "task_id": case.test_case_id,
            **case.metadata,
        }
    }
    if os.getenv("SIGIL_DISABLED", "").strip().lower() in ("1", "true", "yes"):
        return config

    try:
        from sigil_sdk_langgraph import SigilLangGraphHandler
    except Exception as exc:
        print(
            f"[sigil] LangGraph handler unavailable; trial will use record_io only: {exc}"
        )
        return config

    title = str(case.input)[:120] or case.test_case_id
    config["callbacks"] = [
        SigilLangGraphHandler(
            client=client.core,
            provider_resolver="auto",
            agent_name=os.getenv("SIGIL_AGENT_NAME", "sir-alex-fpl"),
            agent_version=os.getenv("SIGIL_AGENT_VERSION", "1.0.0"),
            conversation_title=title,
            capture_workflow_steps=True,
            extra_tags={
                "experiment.run_id": os.getenv("SIGIL_EXPERIMENT_ID")
                or os.getenv("RUN_ID", "")
            },
            extra_metadata={"task_id": case.test_case_id, **case.metadata},
        )
    ]
    return config


def run_agent_case(
    agent: Any, client: sigil.Client, case: sigil.TestCase, conversation_id: str
) -> str:
    result = agent.invoke(
        {"messages": [("system", build_system_prompt()), ("user", str(case.input))]},
        config=_agent_config(client=client, case=case, conversation_id=conversation_id),
    )
    try:
        client.core.flush()
    except Exception:
        pass
    return _final_answer(result)


def _judge_prompt(template: str, case: sigil.TestCase, answer: str) -> str:
    return (
        template.format(input=case.input, expected=case.expected, output=answer)
        + "\n\nReply with ONLY a JSON object: "
        '{"score": <float 0-1>, "pass": <true|false>, "reason": "<one sentence>"}.'
    )


def run_judge(
    *,
    client: sigil.Client,
    judge_llm: ChatAnthropic,
    experiment_id: str,
    case: sigil.TestCase,
    answer: str,
    evaluator_id: str,
    operation_name: str,
    prompt_template: str,
) -> Judgement:
    prompt = _judge_prompt(prompt_template, case, answer)
    raw = str(judge_llm.invoke(prompt).content)
    score, passed, reason = _parse_judgement(raw)
    generation_id = sigil.stable_id(
        "gen", experiment_id, case.test_case_id, evaluator_id
    )
    conversation_id = sigil.stable_id(
        "conv", experiment_id, case.test_case_id, evaluator_id
    )
    client.record_generation(
        generation_id,
        conversation_id=conversation_id,
        input_text=prompt,
        output_text=raw,
        model_provider="anthropic",
        model_name=MODEL,
        agent_name="fpl-llm-judge",
        operation_name=operation_name,
        tags={
            "experiment.run_id": experiment_id,
            "task_id": case.test_case_id,
            "judge": evaluator_id,
        },
        metadata={"experiment_run_id": experiment_id, "task_id": case.test_case_id},
    )
    return Judgement(score, passed, reason, generation_id, conversation_id)


def _parse_judgement(raw: str) -> tuple[float, bool, str]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            score = max(0.0, min(1.0, float(data.get("score", 0.0))))
            passed = _parse_pass(data.get("pass"), score)
            reason = str(data.get("reason", "")).strip()
            return score, passed, reason or "(no reason given)"
        except (ValueError, TypeError):
            pass
    return 0.0, False, f"could not parse judge output: {raw[:160]}"


def _parse_pass(value: object, score: float) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "yes", "1", "pass", "passed"):
            return True
        if normalized in ("false", "no", "0", "fail", "failed"):
            return False
    return score >= 0.5


def _experiment_id() -> str:
    return (
        os.getenv("SIGIL_EXPERIMENT_ID", "").strip()
        or os.getenv("RUN_ID", "").strip()
        or f"sir-alex-fpl-{int(time.time())}"
    )


def _print_run_targets(experiment_id: str) -> None:
    def tok(name: str) -> str:
        return "set" if os.getenv(name, "").strip() else "unset"

    print("-- Sigil experiment targets " + "-" * 38)
    print(f"  experiment_id : {experiment_id}")
    print(f"  model         : {MODEL}")
    print(f"  endpoint      : {os.getenv('SIGIL_ENDPOINT', '(unset)')}")
    print(f"  grafana_url   : {os.getenv('SIGIL_GRAFANA_URL', '(unset)')}")
    print(f"  tenant        : {os.getenv('SIGIL_AUTH_TENANT_ID', '(unset)')}")
    print(f"  token         : {tok('SIGIL_AUTH_TOKEN')}")
    print("-" * 68)


QUALITY_PROMPT = (
    "You are grading a Fantasy Premier League assistant's answer for overall quality.\n\n"
    "User question:\n{input}\n\n"
    "What a good answer should contain:\n{expected}\n\n"
    "Assistant answer:\n{output}\n\n"
    "Score 1.0 for an answer that satisfies the rubric, is clear, and is useful. "
    "Score 0.0 for an answer that misses the request or gives unusable advice."
)

HELPFULNESS_PROMPT = (
    "You are grading a Fantasy Premier League assistant's answer for helpfulness.\n\n"
    "User question:\n{input}\n\n"
    "Expected task intent:\n{expected}\n\n"
    "Assistant answer:\n{output}\n\n"
    "Score 1.0 for an answer that is actionable, specific, well-structured, and directly "
    "helps the user make an FPL decision. Score 0.0 for a vague, generic, or unhelpful answer."
)

HALLUCINATION_PROMPT = (
    "You are grading a Fantasy Premier League assistant's answer for hallucination risk.\n\n"
    "User question:\n{input}\n\n"
    "Expected task intent:\n{expected}\n\n"
    "Assistant answer:\n{output}\n\n"
    "Score 1.0 if the answer appears factual, appropriately cautious, and does not invent "
    "specific stats, injuries, fixtures, prices, ownership, or source-backed claims. "
    "Score 0.0 if it contains clear unsupported or fabricated factual claims."
)


def main() -> None:
    loaded = load_dotenv(override=True)
    print(f"[sigil] loaded .env: {loaded}")
    global MODEL
    MODEL = os.getenv("AGENT_MODEL", MODEL)
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is required to run the agent and judge.")

    experiment_id = _experiment_id()
    os.environ["SIGIL_EXPERIMENT_ID"] = experiment_id
    _print_run_targets(experiment_id)

    suite = sigil.TestSuite(
        suite_id="sir-alex-fpl-smoke",
        name="Sir Alex FPL smoke suite",
        version=SUITE_VERSION,
        test_cases=DATASET,
    )
    quality = sigil.Evaluator(
        "fpl_llm_judge.quality", version=JUDGE_VERSION, kind="llm_judge"
    )
    helpfulness = sigil.Evaluator(
        "fpl_llm_judge.helpfulness", version=JUDGE_VERSION, kind="llm_judge"
    )
    hallucination = sigil.Evaluator(
        "fpl_llm_judge.hallucination_detection",
        version=JUDGE_VERSION,
        kind="llm_judge",
    )

    agent = create_agent(MODEL, llm=ChatAnthropic(model=MODEL))
    judge_llm = ChatAnthropic(model=MODEL)
    skip_scores = os.getenv("SKIP_SCORES", "").strip().lower() in ("1", "true", "yes")
    min_pass_rate_raw = os.getenv("MIN_PASS_RATE", "").strip()
    min_pass_rate = float(min_pass_rate_raw) if min_pass_rate_raw else None
    report = None

    with sigil.experiment(
        name=f"Sir Alex FPL eval ({MODEL})",
        experiment_id=experiment_id,
        suite=suite,
        candidate={
            "agent_name": os.getenv("SIGIL_AGENT_NAME", "sir-alex-fpl"),
            "agent_version": os.getenv("SIGIL_AGENT_VERSION", "1.0.0"),
            "model_provider": "anthropic",
            "model_name": MODEL,
        },
        tags=["sir-alex-fpl", "smoke", MODEL],
    ) as experiment:
        print(
            f"Running experiment '{experiment_id}' over {len(suite.cases)} scenarios"
            f"{' without scoring' if skip_scores else ' with 3 LLM judges'}...\n"
        )
        for case in suite.cases:
            with experiment.trial(case, metadata=case.metadata) as trial:
                conversation_id = sigil.stable_id(
                    "conv", experiment.experiment_id, case.test_case_id
                )
                trial.bind_conversation(conversation_id)
                answer = run_agent_case(agent, experiment.client, case, conversation_id)
                trial.record_io(
                    input=case.input,
                    output=answer,
                    model_provider="anthropic",
                    model_name=MODEL,
                    agent_name=os.getenv("SIGIL_AGENT_NAME", "sir-alex-fpl"),
                )
                if skip_scores:
                    continue

                quality_result = run_judge(
                    client=experiment.client,
                    judge_llm=judge_llm,
                    experiment_id=experiment.experiment_id,
                    case=case,
                    answer=answer,
                    evaluator_id=quality.evaluator_id,
                    operation_name="llm-judge-quality",
                    prompt_template=QUALITY_PROMPT,
                )
                trial.score(
                    "final",
                    quality_result.score,
                    passed=quality_result.passed,
                    explanation=quality_result.reason,
                    evaluator=quality,
                    grader_conversation_id=quality_result.conversation_id,
                    grader_generation_id=quality_result.generation_id,
                )

                helpfulness_result = run_judge(
                    client=experiment.client,
                    judge_llm=judge_llm,
                    experiment_id=experiment.experiment_id,
                    case=case,
                    answer=answer,
                    evaluator_id=helpfulness.evaluator_id,
                    operation_name="llm-judge-helpfulness",
                    prompt_template=HELPFULNESS_PROMPT,
                )
                trial.rubric_score(
                    "helpfulness",
                    helpfulness_result.score,
                    passed=helpfulness_result.passed,
                    explanation=helpfulness_result.reason,
                    evaluator=helpfulness,
                    grader_conversation_id=helpfulness_result.conversation_id,
                    grader_generation_id=helpfulness_result.generation_id,
                )

                hallucination_result = run_judge(
                    client=experiment.client,
                    judge_llm=judge_llm,
                    experiment_id=experiment.experiment_id,
                    case=case,
                    answer=answer,
                    evaluator_id=hallucination.evaluator_id,
                    operation_name="llm-judge-hallucination",
                    prompt_template=HALLUCINATION_PROMPT,
                )
                trial.rubric_score(
                    "hallucination_detection",
                    hallucination_result.score,
                    passed=hallucination_result.passed,
                    explanation=hallucination_result.reason,
                    evaluator=hallucination,
                    grader_conversation_id=hallucination_result.conversation_id,
                    grader_generation_id=hallucination_result.generation_id,
                )

        if not skip_scores:
            report = experiment.report()

    expected_scores = 0 if skip_scores else len(suite.cases) * 3
    print(f"\nDone: {experiment.accepted_scores}/{expected_scores} scores accepted.")
    pass_rate = None
    if report is not None:
        pass_rate = report.summary.pass_rate
        print(
            f"pass_rate={report.summary.pass_rate:.2f}  "
            f"mean_score={report.summary.final_score_avg:.2f}  "
            f"trials={report.summary.completed_count}/{report.summary.trial_count}"
        )
    print(f"\nView in Sigil: {experiment.url}")

    if min_pass_rate is not None:
        if pass_rate is None:
            raise SystemExit("MIN_PASS_RATE was set but no scored report is available.")
        if pass_rate < min_pass_rate:
            raise SystemExit(
                f"pass_rate {pass_rate:.2f} is below the required minimum {min_pass_rate:.2f}."
            )
        print(
            f"pass_rate {pass_rate:.2f} meets the required minimum {min_pass_rate:.2f}."
        )


if __name__ == "__main__":
    main()
