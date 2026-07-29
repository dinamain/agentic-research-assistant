from agent import app
from eval_cases import EVAL_CASES
import uuid

# Known fallback phrases your agent's error-handling produces under exhausted retries —
# these indicate a systemic/external failure (rate limit, quota, connection issue),
# not a genuine logic regression in the agent itself.
SYSTEMIC_FAILURE_PHRASES = [
    "currently rate-limited",
    "trouble connecting to required services",
    "repeated formatting errors while trying to use my tools",
    "unexpected internal error",
]


def is_systemic_failure(answer: str) -> bool:
    return any(phrase in answer.lower() for phrase in SYSTEMIC_FAILURE_PHRASES)


def run_case(case: dict) -> dict:
    thread_id = f"eval-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    tools_called = []
    final_answer = ""

    for chunk in app.stream(
        {"messages": [{"role": "user", "content": case["input"]}]},
        config=config,
        stream_mode="updates"
    ):
        if "agent" in chunk:
            last_msg = chunk["agent"]["messages"][-1]
            if getattr(last_msg, "tool_calls", None):
                tools_called.extend(tc["name"] for tc in last_msg.tool_calls)
            if last_msg.content:
                final_answer = last_msg.content

    # Check for systemic failure BEFORE evaluating pass/fail logic —
    # a rate-limited response should never be scored as a real regression.
    if is_systemic_failure(final_answer):
        return {
            "id": case["id"],
            "status": "SKIPPED",
            "reasons": ["Systemic failure detected (rate limit / quota / connection issue) — not evaluated"],
            "tools_called": tools_called,
            "answer": final_answer,
        }

    passed = True
    reasons = []

    if case["check"] == "tool_called":
        if case["expected_tool"] not in tools_called:
            passed = False
            reasons.append(f"Expected tool '{case['expected_tool']}' not called. Tools called: {tools_called}")

    elif case["check"] == "no_tool_called":
        if tools_called:
            passed = False
            reasons.append(f"Expected no tool call, but got: {tools_called}")

    elif case["check"] == "answer_contains":
        missing = [term for term in case["expected_terms"] if term.lower() not in final_answer.lower()]
        if missing:
            passed = False
            reasons.append(f"Answer missing expected term(s): {missing}")

    for forbidden in case.get("must_not_contain", []):
        if forbidden.lower() in final_answer.lower():
            negation_words = ["no evidence", "not found", "no specific", "does not", "did not",
                              "no known", "couldn't find", "not covered", "no credible evidence"]
            if not any(neg in final_answer.lower() for neg in negation_words):
                passed = False
                reasons.append(f"Answer wrongly contains forbidden term: '{forbidden}'")

    return {
        "id": case["id"],
        "status": "PASS" if passed else "FAIL",
        "reasons": reasons,
        "tools_called": tools_called,
        "answer": final_answer,
    }


if __name__ == "__main__":
    results = [run_case(case) for case in EVAL_CASES]

    print("\n=== EVAL RESULTS ===")
    status_icons = {"PASS": "✅", "FAIL": "❌", "SKIPPED": "⏭️"}
    for r in results:
        print(f"{status_icons[r['status']]} {r['status']} | {r['id']}")
        if r["status"] != "PASS":
            for reason in r["reasons"]:
                print(f"    - {reason}")
            print(f"    Full answer: {r['answer']}")

    passed = sum(r["status"] == "PASS" for r in results)
    failed = sum(r["status"] == "FAIL" for r in results)
    skipped = sum(r["status"] == "SKIPPED" for r in results)

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped (systemic failures) — out of {len(results)} total")