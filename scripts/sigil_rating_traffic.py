#!/usr/bin/env python3
"""Send synthetic Sir Alex conversations with random ratings to local Sigil.

The script uses only Python's standard library. It writes one generation per
conversation through Sigil's HTTP generation ingest endpoint, then optionally
adds good/bad ratings through the conversation ratings endpoint.

Examples:
    python scripts/sigil_rating_traffic.py
    python scripts/sigil_rating_traffic.py --count 100 --seed 7 --verify
    python scripts/sigil_rating_traffic.py --base-url http://localhost:8080 --tenant fake
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

GOOD = "CONVERSATION_RATING_VALUE_GOOD"
BAD = "CONVERSATION_RATING_VALUE_BAD"

PROMPTS = [
    "Who should I captain this gameweek?",
    "Should I start my Arsenal defender or bench him?",
    "Compare Salah and Haaland for captaincy.",
    "Find a budget midfielder with strong form.",
    "Rate my wildcard draft.",
    "Which transfer has the best upside this week?",
    "Should I use my bench boost now?",
    "Who are the best differentials under 5 percent ownership?",
    "Is it worth taking a hit for a premium defender?",
    "Which goalkeeper rotation looks safest?",
]

RESPONSES = [
    "I'd lean toward the player with penalties, home fixture, and stronger expected minutes.",
    "Start the defender if the clean-sheet odds are favorable and bench risk is low.",
    "Salah has the safer floor, while Haaland gives you the higher ceiling in a strong fixture.",
    "Look for a nailed midfielder with set pieces, good xGI, and a short-term fixture swing.",
    "The draft is balanced, but I would not overcommit budget to the bench this early.",
    "The best transfer is the one that improves your next three fixtures without blocking future moves.",
    "Save bench boost unless all four bench players have strong expected minutes.",
    "A good differential should still have minutes security. Low ownership alone is not enough.",
    "Only take the hit if the move fixes a longer-term structural issue or unlocks captaincy upside.",
    "Prioritize fixture rotation, save points, and guaranteed starts over chasing one clean sheet.",
]

TOPICS = [
    "captaincy",
    "lineup",
    "premium-comparison",
    "budget-pick",
    "wildcard",
    "transfer",
    "chip-strategy",
    "differential",
    "points-hit",
    "goalkeeper-rotation",
]

MANAGER_TIERS = ["casual", "engaged", "elite"]
LEAGUES = ["work-mini-league", "friends-league", "overall-rank-chase", "head-to-head"]


@dataclass(frozen=True)
class ConversationPlan:
    conversation_id: str
    generation_id: str
    title: str
    user_id: str
    manager_id: str
    manager_tier: str
    league: str
    gameweek: int
    topic: str
    user_prompt: str
    assistant_response: str
    started_at: datetime
    completed_at: datetime
    ratings: tuple[str, ...]


def post_json(
    url: str, payload: dict[str, Any], tenant: str, timeout: float
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Scope-OrgID": tenant,
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"POST {url} failed: {exc.reason}") from exc

    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def post_otlp_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"POST {url} failed: {exc.reason}") from exc

    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def rfc3339(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def unix_nanos(value: datetime) -> str:
    return str(int(value.timestamp() * 1_000_000_000))


def filter_string(value: str) -> str:
    return json.dumps(value)


def stable_hex(value: str, length: int) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def default_otlp_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "localhost"
    if host.startswith("sigil.") and host.endswith(".orb.local"):
        suffix = host.removeprefix("sigil.")
        return f"{scheme}://alloy.{suffix}:4318/v1/traces"
    return "http://localhost:4318/v1/traces"


def choose_ratings(rng: random.Random) -> tuple[str, ...]:
    bucket = rng.choices(
        ["good", "bad", "mixed", "unrated"],
        weights=[45, 35, 10, 10],
        k=1,
    )[0]
    if bucket == "good":
        return (GOOD,)
    if bucket == "bad":
        return (BAD,)
    if bucket == "mixed":
        return (GOOD, BAD)
    return ()


def build_plans(count: int, seed: int | None, run_id: str) -> list[ConversationPlan]:
    rng = random.Random(seed)
    now = datetime.now(UTC)
    plans: list[ConversationPlan] = []
    for idx in range(count):
        prompt_index = rng.randrange(len(PROMPTS))
        prompt = PROMPTS[prompt_index]
        response = rng.choice(RESPONSES)
        conversation_id = f"sir-alex-rating-{run_id}-{idx + 1:03d}"
        generation_id = f"gen-{conversation_id}"
        user_id = f"traffic-user-{(idx % 7) + 1}"
        started_at = now - timedelta(seconds=(count - idx) * 3)
        completed_at = started_at + timedelta(milliseconds=rng.randint(350, 2400))
        plans.append(
            ConversationPlan(
                conversation_id=conversation_id,
                generation_id=generation_id,
                title=prompt[:120],
                user_id=user_id,
                manager_id=f"manager-{(idx % 17) + 1:02d}",
                manager_tier=rng.choice(MANAGER_TIERS),
                league=rng.choice(LEAGUES),
                gameweek=rng.randint(1, 38),
                topic=TOPICS[prompt_index],
                user_prompt=prompt,
                assistant_response=response,
                started_at=started_at,
                completed_at=completed_at,
                ratings=choose_ratings(rng),
            )
        )
    return plans


def generation_payload(
    plan: ConversationPlan, model_provider: str, model_name: str, run_id: str
) -> dict[str, Any]:
    input_tokens = max(8, len(plan.user_prompt.split()) + 10)
    output_tokens = max(12, len(plan.assistant_response.split()) + 14)
    return {
        "id": plan.generation_id,
        "conversation_id": plan.conversation_id,
        "operation_name": "chat",
        "mode": "GENERATION_MODE_SYNC",
        "model": {"provider": model_provider, "name": model_name},
        "response_id": f"resp-{plan.generation_id}",
        "response_model": model_name,
        "system_prompt": "You are Sir Alex, a Fantasy Premier League assistant.",
        "input": [
            {
                "role": "MESSAGE_ROLE_USER",
                "parts": [{"text": plan.user_prompt}],
            }
        ],
        "output": [
            {
                "role": "MESSAGE_ROLE_ASSISTANT",
                "parts": [{"text": plan.assistant_response}],
            }
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "stop_reason": "end_turn",
        "started_at": rfc3339(plan.started_at),
        "completed_at": rfc3339(plan.completed_at),
        "tags": {
            "sigil.conversation.title": plan.title,
            "traffic.source": "sir-alex-rating-traffic",
            "traffic.run_id": run_id,
            "user.id": plan.user_id,
            "sir_alex.manager.id": plan.manager_id,
            "sir_alex.manager.tier": plan.manager_tier,
            "sir_alex.league": plan.league,
            "sir_alex.gameweek": str(plan.gameweek),
            "sir_alex.request.topic": plan.topic,
        },
        "metadata": {
            "sigil.conversation.title": plan.title,
            "sigil.user.id": plan.user_id,
            "user.id": plan.user_id,
            "sir_alex.traffic_run_id": run_id,
            "sir_alex.manager.id": plan.manager_id,
            "sir_alex.manager.tier": plan.manager_tier,
            "sir_alex.league": plan.league,
            "sir_alex.gameweek": plan.gameweek,
            "sir_alex.request.topic": plan.topic,
            "attributes": {
                "user.id": plan.user_id,
                "sir_alex.manager.id": plan.manager_id,
                "sir_alex.manager.tier": plan.manager_tier,
                "sir_alex.league": plan.league,
                "sir_alex.gameweek": str(plan.gameweek),
                "sir_alex.request.topic": plan.topic,
                "traffic.run_id": run_id,
            },
        },
        "agent_name": "sir-alex-fpl-traffic",
        "agent_version": f"traffic-{run_id}",
    }


def otlp_string(value: str) -> dict[str, str]:
    return {"stringValue": value}


def otlp_int(value: int) -> dict[str, str]:
    return {"intValue": str(value)}


def otlp_attribute(key: str, value: dict[str, Any]) -> dict[str, Any]:
    return {"key": key, "value": value}


def otlp_trace_payload(
    plan: ConversationPlan, model_provider: str, model_name: str, run_id: str
) -> dict[str, Any]:
    trace_id = stable_hex(plan.conversation_id, 32)
    span_id = stable_hex(plan.generation_id, 16)
    attrs = [
        otlp_attribute("sigil.sdk.name", otlp_string("sir-alex-fpl-agent-traffic")),
        otlp_attribute("sigil.generation.id", otlp_string(plan.generation_id)),
        otlp_attribute("gen_ai.conversation.id", otlp_string(plan.conversation_id)),
        otlp_attribute("gen_ai.operation.name", otlp_string("chat")),
        otlp_attribute("gen_ai.provider.name", otlp_string(model_provider)),
        otlp_attribute("gen_ai.request.model", otlp_string(model_name)),
        otlp_attribute("gen_ai.agent.name", otlp_string("sir-alex-fpl-traffic")),
        otlp_attribute("gen_ai.agent.version", otlp_string(f"traffic-{run_id}")),
        otlp_attribute("sigil.conversation.title", otlp_string(plan.title)),
        otlp_attribute("user.id", otlp_string(plan.user_id)),
        otlp_attribute("traffic.source", otlp_string("sir-alex-rating-traffic")),
        otlp_attribute("traffic.run_id", otlp_string(run_id)),
        otlp_attribute("sir_alex.manager.id", otlp_string(plan.manager_id)),
        otlp_attribute("sir_alex.manager.tier", otlp_string(plan.manager_tier)),
        otlp_attribute("sir_alex.league", otlp_string(plan.league)),
        otlp_attribute("sir_alex.gameweek", otlp_int(plan.gameweek)),
        otlp_attribute("sir_alex.request.topic", otlp_string(plan.topic)),
    ]
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        otlp_attribute(
                            "service.name", otlp_string("sir-alex-fpl-agent")
                        ),
                        otlp_attribute(
                            "deployment.environment", otlp_string("local-traffic")
                        ),
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {
                            "name": "sir-alex-rating-traffic",
                            "version": "1.0.0",
                        },
                        "spans": [
                            {
                                "traceId": trace_id,
                                "spanId": span_id,
                                "name": "sir-alex.chat",
                                "kind": 3,
                                "startTimeUnixNano": unix_nanos(plan.started_at),
                                "endTimeUnixNano": unix_nanos(plan.completed_at),
                                "attributes": attrs,
                            }
                        ],
                    }
                ],
            }
        ]
    }


def send_generation(
    base_url: str,
    tenant: str,
    timeout: float,
    plan: ConversationPlan,
    model_provider: str,
    model_name: str,
    run_id: str,
) -> None:
    payload = {
        "generations": [generation_payload(plan, model_provider, model_name, run_id)],
    }
    response = post_json(
        f"{base_url}/api/v1/generations:export", payload, tenant, timeout
    )
    results = response.get("results", [])
    if not results or not results[0].get("accepted", False):
        raise RuntimeError(
            f"generation rejected for {plan.conversation_id}: {response}"
        )


def send_otlp_span(
    otlp_url: str,
    timeout: float,
    plan: ConversationPlan,
    model_provider: str,
    model_name: str,
    run_id: str,
) -> None:
    post_otlp_json(
        otlp_url, otlp_trace_payload(plan, model_provider, model_name, run_id), timeout
    )


def send_ratings(
    base_url: str, tenant: str, timeout: float, plan: ConversationPlan
) -> None:
    for idx, rating in enumerate(plan.ratings, start=1):
        label = "good" if rating == GOOD else "bad"
        payload = {
            "rating_id": f"rating-{plan.conversation_id}-{idx}",
            "rating": rating,
            "generation_id": plan.generation_id,
            "comment": f"synthetic {label} rating for {plan.title}",
            "rater_id": "sir-alex-rating-traffic",
            "source": "traffic-script",
            "metadata": {
                "traffic": True,
                "rating_label": label,
            },
        }
        post_json(
            f"{base_url}/api/v1/conversations/{plan.conversation_id}/ratings",
            payload,
            tenant,
            timeout,
        )


def count_matching_conversations(
    base_url: str,
    tenant: str,
    timeout: float,
    filters: str,
    from_time: datetime,
    to_time: datetime,
) -> int:
    cursor = ""
    total = 0
    while True:
        payload: dict[str, Any] = {
            "filters": filters,
            "select": [],
            "time_range": {
                "from": rfc3339(from_time),
                "to": rfc3339(to_time),
            },
            "page_size": 50,
        }
        if cursor:
            payload["cursor"] = cursor
        response = post_json(
            f"{base_url}/api/v1/conversations/search", payload, tenant, timeout
        )
        total += len(response.get("conversations", []))
        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor", "")
        if not cursor:
            raise RuntimeError(
                "search verification returned has_more without next_cursor"
            )
    return total


def wait_for_search_index(
    base_url: str,
    tenant: str,
    timeout: float,
    plans: list[ConversationPlan],
    base_filter: str,
    from_time: datetime,
    to_time: datetime,
    verify_timeout: float,
    verify_interval: float,
) -> None:
    deadline = time.monotonic() + verify_timeout
    while True:
        observed = count_matching_conversations(
            base_url, tenant, timeout, base_filter, from_time, to_time
        )
        if observed >= len(plans):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"search verification timed out waiting for indexed conversations "
                f"(observed {observed}/{len(plans)})"
            )
        time.sleep(verify_interval)


def verify_counts(
    base_url: str,
    tenant: str,
    timeout: float,
    plans: list[ConversationPlan],
    run_id: str,
    verify_timeout: float,
    verify_interval: float,
) -> dict[str, int]:
    from_time = min(plan.started_at for plan in plans) - timedelta(minutes=1)
    to_time = max(plan.completed_at for plan in plans) + timedelta(minutes=1)
    base_filter = f'agent = "sir-alex-fpl-traffic" agent.version = {filter_string(f"traffic-{run_id}")}'
    wait_for_search_index(
        base_url,
        tenant,
        timeout,
        plans,
        base_filter,
        from_time,
        to_time,
        verify_timeout,
        verify_interval,
    )
    filters = {
        "bad": "rating=bad",
        "good": "rating=good",
        "rated": "rating=rated",
        "unrated": "rating=unrated",
        "has_bad": "rating.has_bad=true",
        "bad_count": "rating.bad_count > 0",
        "first_user": f"span.user.id = {filter_string(plans[0].user_id)}",
        "manager_tier": f"span.sir_alex.manager.tier = {filter_string(plans[0].manager_tier)}",
    }
    counts: dict[str, int] = {}
    for key, expression in filters.items():
        counts[key] = count_matching_conversations(
            base_url,
            tenant,
            timeout,
            f"{base_filter} {expression}",
            from_time,
            to_time,
        )
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", default="http://localhost:8080", help="Sigil HTTP base URL"
    )
    parser.add_argument("--tenant", default="fake", help="X-Scope-OrgID tenant header")
    parser.add_argument(
        "--count", type=int, default=100, help="Number of conversations to create"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for repeatable ratings"
    )
    parser.add_argument(
        "--run-id", default=None, help="Run identifier embedded in conversation IDs"
    )
    parser.add_argument(
        "--model-provider", default="openrouter", help="Synthetic model provider"
    )
    parser.add_argument(
        "--model-name",
        default="anthropic/claude-haiku-4.5",
        help="Synthetic model name",
    )
    parser.add_argument(
        "--otlp-url",
        default=None,
        help="OTLP HTTP traces URL for filterable span attributes",
    )
    parser.add_argument(
        "--skip-otlp", action="store_true", help="Do not emit matching OTLP spans"
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="HTTP timeout in seconds"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between conversations",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Run rating-filter searches after ingest"
    )
    parser.add_argument(
        "--verify-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for Tempo search indexing",
    )
    parser.add_argument(
        "--verify-interval",
        type=float,
        default=2.0,
        help="Seconds between verification search attempts",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print planned summary without sending"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")

    base_url = args.base_url.rstrip("/")
    otlp_url = (
        ""
        if args.skip_otlp
        else (args.otlp_url or default_otlp_url(base_url)).rstrip("/")
    )
    run_id = args.run_id or uuid.uuid4().hex[:8]
    plans = build_plans(args.count, args.seed, run_id)

    expected = {
        "good": sum(GOOD in plan.ratings for plan in plans),
        "bad": sum(BAD in plan.ratings for plan in plans),
        "rated": sum(bool(plan.ratings) for plan in plans),
        "unrated": sum(not plan.ratings for plan in plans),
        "mixed": sum(len(plan.ratings) == 2 for plan in plans),
    }
    print(
        json.dumps(
            {
                "run_id": run_id,
                "count": args.count,
                "base_url": base_url,
                "otlp_url": otlp_url or None,
                "tenant": args.tenant,
                "expected": expected,
                "example_span_filters": {
                    "user": f'span.user.id = "{plans[0].user_id}"',
                    "manager_tier": f'span.sir_alex.manager.tier = "{plans[0].manager_tier}"',
                    "gameweek": f"span.sir_alex.gameweek = {plans[0].gameweek}",
                },
            },
            indent=2,
        )
    )

    if args.dry_run:
        return 0

    for idx, plan in enumerate(plans, start=1):
        send_generation(
            base_url,
            args.tenant,
            args.timeout,
            plan,
            args.model_provider,
            args.model_name,
            run_id,
        )
        if otlp_url:
            send_otlp_span(
                otlp_url,
                args.timeout,
                plan,
                args.model_provider,
                args.model_name,
                run_id,
            )
        send_ratings(base_url, args.tenant, args.timeout, plan)
        if idx % 10 == 0 or idx == len(plans):
            print(f"sent {idx}/{len(plans)} conversations")
        if args.sleep > 0:
            time.sleep(args.sleep)

    if args.verify:
        observed = verify_counts(
            base_url,
            args.tenant,
            args.timeout,
            plans,
            run_id,
            args.verify_timeout,
            args.verify_interval,
        )
        print(json.dumps({"observed_search_counts": observed}, indent=2))

    print(f"done: run_id={run_id}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
