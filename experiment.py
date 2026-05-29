"""Run the Sir Alex FPL agent as a Sigil offline-evaluation experiment.

This is the *only* file added to instrument experiments — the agent code is
untouched. It reuses the existing compiled LangGraph agent (`create_agent`) and
system prompt (`build_system_prompt`), runs a small dataset through it, grades
each answer with a basic LLM judge, and publishes everything to Sigil under one
run_id so you can browse and compare runs in the UI.

Run it (see .env.experiment.sample for the variables):

    PYTHONPATH=. python experiment.py

Useful env:
    ANTHROPIC_API_KEY   required; drives the agent and the LLM judge
    AGENT_MODEL         agent + judge model (Anthropic id; default claude-haiku-4-5-20251001)
    RUN_ID              stable run id (defaults to a timestamp); reuse to retry idempotently
    SIGIL_*             Sigil config (Grafana Cloud: basic auth over HTTP)
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
    Generation,
    GenerationStart,
    ModelRef,
    ScoreValue,
    assistant_text_message,
    user_text_message,
)
from sigil_sdk_langgraph import (
    DatasetItem,
    ExperimentRun,
    ExperimentRunner,
    ScoreOutput,
    TargetResult,
)

from agent.agent import create_agent
from agent.utils import build_system_prompt

# Anthropic model id (same convention as cli.py / AGENT_MODEL).
MODEL = os.getenv("AGENT_MODEL", "claude-haiku-4-5-20251001")
JUDGE_VERSION = "2026-05-28"


# --------------------------------------------------------------------------- #
# 10 test scenarios. `expected` is the grading rubric the LLM judge checks
# against — these are behavioral criteria, not exact facts, since live FPL data
# changes week to week.
# --------------------------------------------------------------------------- #
DATASET: list[DatasetItem] = [
    DatasetItem(
        id="captain-advice",
        input="Who should I consider captaining this gameweek?",
        expected="Recommends at least one specific player to captain and gives brief reasoning.",
        metadata={"task_id": "captain_advice", "task_category": "advice"},
    ),
    DatasetItem(
        id="player-stats-haaland",
        input="How is Erling Haaland performing this season?",
        expected="Reports Haaland's FPL stats (e.g. goals, points, or form) and a short assessment.",
        metadata={"task_id": "player_stats", "task_category": "stats"},
    ),
    DatasetItem(
        id="compare-salah-palmer",
        input="Compare Mohamed Salah and Cole Palmer for FPL.",
        expected="Compares both players with relevant stats and ends with a recommendation.",
        metadata={"task_id": "compare_players", "task_category": "compare"},
    ),
    DatasetItem(
        id="leaders-goals",
        input="Who are the top goal scorers in the Premier League right now?",
        expected="Lists several specific players as the leading goal scorers.",
        metadata={"task_id": "league_leaders_goals", "task_category": "leaders"},
    ),
    DatasetItem(
        id="differentials",
        input="Suggest a couple of differential picks for my FPL team.",
        expected="Names specific players framed as differentials or low-ownership picks.",
        metadata={"task_id": "differentials", "task_category": "advice"},
    ),
    DatasetItem(
        id="budget-defender",
        input="Recommend a budget defender under 5.0 million.",
        expected="Recommends at least one specific defender and acknowledges the budget constraint.",
        metadata={"task_id": "budget_pick", "task_category": "advice"},
    ),
    DatasetItem(
        id="transfer-hit",
        input="Should I take a -4 point hit to bring in a player this week?",
        expected="Gives reasoned advice about taking a points hit and mentions the trade-offs.",
        metadata={"task_id": "transfer_decision", "task_category": "advice"},
    ),
    DatasetItem(
        id="player-stats-saka",
        input="How has Bukayo Saka done this season?",
        expected="Reports Saka's FPL stats or form and a short assessment.",
        metadata={"task_id": "player_stats", "task_category": "stats"},
    ),
    DatasetItem(
        id="leaders-assists",
        input="Who leads the Premier League in assists?",
        expected="Names one or more players leading the league in assists.",
        metadata={"task_id": "league_leaders_assists", "task_category": "leaders"},
    ),
    DatasetItem(
        id="persona-team-talk",
        input="Give me a motivational team talk for my FPL decisions this week.",
        expected="Responds in the Sir Alex Ferguson motivational persona and stays on FPL/football.",
        metadata={"task_id": "persona", "task_category": "persona"},
    ),
]


def _final_answer(result: dict) -> str:
    """Extract the agent's final textual answer from the graph result."""
    for message in reversed(result.get("messages", [])):
        content = getattr(message, "content", "")
        tool_calls = getattr(message, "tool_calls", None)
        if content and not tool_calls:
            return str(content)
    return ""


def make_target(agent):
    """Build the experiment target: invoke the compiled agent for one item."""

    def target(item: DatasetItem, run: ExperimentRun) -> TargetResult:
        messages = [
            ("system", build_system_prompt()),
            ("user", str(item.input)),
        ]
        # Passing run.langgraph_config() is the whole integration: it tags every
        # generation with the experiment run_id and captures their ids so scores
        # attach automatically.
        result = agent.invoke(
            {"messages": messages},
            config=run.langgraph_config(conversation_title=str(item.input)[:120]),
        )
        # A turn can emit several generations (agent -> tools -> agent). Attach the
        # quality score to the final one (the answer the user sees).
        produced = run.produced_generation_ids
        return TargetResult(output=_final_answer(result), generation_ids=produced[-1:])

    return target


def make_judge(client: Client):
    """Build a basic LLM-as-judge scorer that records its own Sigil generation."""

    judge_llm = ChatAnthropic(model=MODEL, temperature=0)

    def judge(item: DatasetItem, result: TargetResult) -> list[ScoreOutput]:
        prompt = (
            "You are grading a Fantasy Premier League assistant's answer.\n\n"
            f"User question:\n{item.input}\n\n"
            f"What a good answer should contain:\n{item.expected}\n\n"
            f"Assistant answer:\n{result.output}\n\n"
            "Reply with ONLY a JSON object: "
            '{"score": <float 0-1>, "pass": <true|false>, "reason": "<one sentence>"}.'
        )

        # Record the judge call as a Sigil generation so the grade is auditable.
        raw = ""
        with client.start_generation(
            GenerationStart(
                model=ModelRef(provider="anthropic", name=MODEL),
                agent_name="fpl-llm-judge",
                operation_name="llm-judge",
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
            score = float(data.get("score", 0.0))
            passed = bool(data.get("pass", score >= 0.5))
            reason = str(data.get("reason", "")).strip()
            return max(0.0, min(1.0, score)), passed, reason or "(no reason given)"
        except (ValueError, TypeError):
            pass
    return 0.0, False, f"could not parse judge output: {raw[:160]}"


def build_client() -> Client:
    """Sigil client driven by SIGIL_* env (Grafana Cloud: basic auth over HTTP).

    The agent's own generation export reads SIGIL_* automatically; the only extra
    bit experiments need is the HTTP control-plane base URL (create experiment /
    export scores), which is not an env var by default — so point ApiConfig at the
    same SIGIL_ENDPOINT.
    """
    endpoint = os.getenv("SIGIL_ENDPOINT", "http://localhost:8080")
    return Client(ClientConfig(api=ApiConfig(endpoint=endpoint)))


def main() -> None:
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is required to run the agent and judge.")

    client = build_client()
    # Reuse the existing compiled agent, but drive it with an Anthropic model.
    agent = create_agent(MODEL, llm=ChatAnthropic(model=MODEL, temperature=0))
    run_id = os.getenv("RUN_ID", f"sir-alex-fpl-{int(time.time())}")

    runner = ExperimentRunner(
        client=client,
        run_id=run_id,
        name=f"Sir Alex FPL eval ({MODEL})",
        dataset={"id": "sir-alex-fpl-smoke", "version": "2026-05-28"},
        candidate={"model": MODEL},
        tags=["sir-alex-fpl", "smoke", MODEL],
        agent_name=os.getenv("SIGIL_AGENT_NAME", "sir-alex-fpl"),
        agent_version=os.getenv("SIGIL_AGENT_VERSION", "1.0.0"),
        provider_resolver="auto",
    )

    # SKIP_SCORES=1 runs the agent and publishes generations + the experiment
    # lifecycle without grading/publishing scores. Useful on Grafana Cloud, where
    # the score-export endpoint is not currently exposed externally (the LLM judge
    # is also skipped, saving its model calls).
    skip_scores = os.getenv("SKIP_SCORES", "").strip().lower() in ("1", "true", "yes")
    scorers = [] if skip_scores else [make_judge(client)]

    mode = "no scoring (SKIP_SCORES)" if skip_scores else "with LLM judge"
    print(
        f"Running experiment '{run_id}' over {len(DATASET)} scenarios with model {MODEL} ({mode})...\n"
    )
    try:
        result = runner.run(DATASET, make_target(agent), scorers)
        print(f"\nDone: {result.accepted_scores}/{len(DATASET)} scores accepted.")
        if result.report is not None:
            s = result.report.summary
            print(
                f"pass_rate={s.pass_rate:.2f}  mean_score={s.mean_score:.2f}  n_scores={s.n_scores}"
            )
        print(f"\nView in Sigil: {result.url}")
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
