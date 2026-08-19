"""Shared, lenient JSON parsing for LLM responses that are supposed to be JSON but often aren't quite.

Two model-produced malformations show up often enough in this project's own
real usage to justify one shared, tested repair step rather than
reimplementing it per agent (this exact repair started life inline in
clinical_agent.py on Day 5, then proved itself live again during Day 9's PHI
batching work — worth sharing rather than a third copy): markdown code
fences wrapped around otherwise-valid JSON, and a trailing comma before a
closing ]/} bracket (the single most common malformation smaller/weaker
models produce — confirmed live multiple times, see PROGRESS.md Days 5-9).
Neither repair changes the meaning of well-formed JSON; both only fix syntax
a model got wrong.
"""

import json
import re

# Strips ```json ... ``` or ``` ... ``` fences some models wrap JSON in
# despite being told not to — cheaper to tolerate than to keep tuning the
# prompt against every model's formatting habit.
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# Matches a comma immediately followed (across whitespace/newlines) by a
# closing ] or } — see module docstring.
_TRAILING_COMMA_RE = re.compile(r",(\s*[\]}])")


def parse_llm_json(raw_text: str):
    """Parse `raw_text` as JSON, tolerating code fences and a trailing comma before a closing bracket.

    Args:
        raw_text: The LLM's raw response text, expected to be (mostly) JSON.

    Returns:
        The parsed JSON value — a dict or list, depending on what the
        prompt asked the model to return.

    Raises:
        ValueError: still not valid JSON even after both repairs — an edge
            condition per rule 8. Deliberately not swallowed into a default
            value here: callers decide how to degrade (raise louder, fall
            back to a default, skip one item vs. the whole document) based
            on their own context, which this shared helper doesn't have.

    Use when: parsing any LLM response a prompt asked to be JSON —
    clinical_agent.py's finding extraction, phi_agent.py's batch PHI
    verification (Day 9).
    """
    cleaned = _CODE_FENCE_RE.sub("", raw_text.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        repaired = _TRAILING_COMMA_RE.sub(r"\1", cleaned)
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Not valid JSON even after trailing-comma repair: {exc}. Raw response: {raw_text!r}") from exc
        return parsed


if __name__ == "__main__":
    # Self-test: no external dependency.
    assert parse_llm_json('{"a": 1}') == {"a": 1}
    assert parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_llm_json('{"a": [1, 2,],}') == {"a": [1, 2]}  # two trailing commas, both repaired
    assert parse_llm_json("[1, 2, 3]") == [1, 2, 3]

    try:
        parse_llm_json("not json at all")
        raise AssertionError("expected ValueError for genuinely non-JSON input")
    except ValueError:
        pass

    print("llm_json.py self-test passed: code-fence stripping and trailing-comma repair verified.")
