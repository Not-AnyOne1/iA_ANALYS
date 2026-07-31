"""Unit tests for claude_client.py's pure helpers and CLI invocation.

Mostly the module-level, side-effect-free helpers: the retry-classification
heuristic and the JSON-schema compaction pipeline, which decision_engine.py
also reuses for its own schema.

Plus one group that asserts the CLI *argument list* and error reporting.
Those exist because a real production bug slipped through: `--disallowedTools
"*"` also denied the CLI-internal `StructuredOutput` tool, so every analysis
failed with `subtype=error_max_turns` — and the old error handler reported it
as a bare "unknown error". No test caught either, because they all mock
_run_cli. No subprocess is spawned here; the boundary is patched.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from claude_client import (
    _ARGV_SAFETY_BUDGET_BYTES,
    AnalysisError,
    ClaudeAnalyzer,
    _build_schema,
    _classify_failure,
    _exclude_field,
    _inline_refs,
    _strip_prose,
    _tighten_objects,
)
from config import Settings
from market_data import Timeframe


# --------------------------------------------------------------------------- _classify_failure

@pytest.mark.parametrize("text", [
    "Error: not a valid json schema",
    "authentication_failed",
    "You are not logged in",
    "Please run claude auth login",
    "claude auth login",
    "oauth_org_not_allowed",
    "invalid_request",
    "model_not_found",
    "permission denied",
])
def test_non_retryable_markers_are_classified_as_permanent(text):
    assert _classify_failure(text) is False


@pytest.mark.parametrize("text", [
    "rate_limit exceeded",
    "rate limit exceeded",
    "the service is overloaded",
    "HTTP 529",
    "HTTP 503",
    "ECONNRESET",
    "ETIMEDOUT",
    "temporarily unavailable",
    "server_error",
    "internal server error",
    "network unreachable",
])
def test_retryable_markers_are_classified_as_transient(text):
    assert _classify_failure(text) is True


def test_classification_is_case_insensitive():
    assert _classify_failure("AUTHENTICATION_FAILED") is False
    assert _classify_failure("Rate_Limit") is True


def test_unknown_failure_defaults_to_retryable():
    # Documented behaviour: one wasted retry is cheaper than giving up on a
    # transient hiccup, and retries are bounded by max_retries anyway.
    assert _classify_failure("something nobody has seen before") is True
    assert _classify_failure("") is True


def test_non_retryable_marker_wins_over_a_retryable_one():
    # Both markers present: the permanent one must win, otherwise an
    # auth failure mentioning "network" would be retried pointlessly.
    assert _classify_failure("network error: authentication_failed") is False


# --------------------------------------------------------------------------- _inline_refs

def test_inline_refs_resolves_a_single_ref():
    defs = {"Child": {"type": "object", "properties": {"x": {"type": "integer"}}}}
    node = {"type": "object", "properties": {"child": {"$ref": "#/$defs/Child"}}}

    result = _inline_refs(node, defs)

    assert result["properties"]["child"] == defs["Child"]
    assert "$ref" not in json.dumps(result)


def test_inline_refs_resolves_nested_refs():
    defs = {
        "Inner": {"type": "string"},
        "Outer": {"type": "object", "properties": {"inner": {"$ref": "#/$defs/Inner"}}},
    }
    node = {"$ref": "#/$defs/Outer"}

    result = _inline_refs(node, defs)

    assert result["properties"]["inner"] == {"type": "string"}


def test_inline_refs_walks_into_lists():
    defs = {"Item": {"type": "number"}}
    node = {"anyOf": [{"$ref": "#/$defs/Item"}, {"type": "null"}]}

    result = _inline_refs(node, defs)

    assert result["anyOf"][0] == {"type": "number"}
    assert result["anyOf"][1] == {"type": "null"}


def test_inline_refs_leaves_a_ref_free_schema_unchanged():
    node = {"type": "object", "properties": {"x": {"type": "integer"}}}
    assert _inline_refs(node, {}) == node


def test_inline_refs_passes_scalars_through():
    assert _inline_refs("plain", {}) == "plain"
    assert _inline_refs(42, {}) == 42
    assert _inline_refs(None, {}) is None


# --------------------------------------------------------------------------- _tighten_objects

def test_tighten_objects_adds_additional_properties_false():
    node = {"type": "object", "properties": {}}
    _tighten_objects(node)
    assert node["additionalProperties"] is False


def test_tighten_objects_recurses_into_nested_objects():
    node = {
        "type": "object",
        "properties": {"child": {"type": "object", "properties": {}}},
    }
    _tighten_objects(node)

    assert node["additionalProperties"] is False
    assert node["properties"]["child"]["additionalProperties"] is False


def test_tighten_objects_recurses_into_lists():
    node = {"anyOf": [{"type": "object", "properties": {}}]}
    _tighten_objects(node)
    assert node["anyOf"][0]["additionalProperties"] is False


def test_tighten_objects_respects_an_existing_explicit_value():
    node = {"type": "object", "properties": {}, "additionalProperties": True}
    _tighten_objects(node)
    assert node["additionalProperties"] is True  # not overwritten


def test_tighten_objects_ignores_non_object_schemas():
    node = {"type": "string"}
    _tighten_objects(node)
    assert "additionalProperties" not in node


# --------------------------------------------------------------------------- _strip_prose

def test_strip_prose_removes_description_and_title():
    node = {"type": "string", "description": "some prose", "title": "Some Title"}
    _strip_prose(node)
    assert node == {"type": "string"}


def test_strip_prose_recurses_into_nested_dicts_and_lists():
    node = {
        "type": "object",
        "title": "Root",
        "properties": {"x": {"type": "integer", "description": "an int"}},
        "anyOf": [{"type": "null", "description": "nothing"}],
    }
    _strip_prose(node)

    assert "title" not in node
    assert "description" not in node["properties"]["x"]
    assert "description" not in node["anyOf"][0]


def test_strip_prose_is_a_noop_without_prose_keys():
    node = {"type": "integer"}
    _strip_prose(node)
    assert node == {"type": "integer"}


def test_strip_prose_measurably_shrinks_the_payload():
    node = {"type": "string", "description": "x" * 500, "title": "y" * 100}
    before = len(json.dumps(node))
    _strip_prose(node)
    assert len(json.dumps(node)) < before


# --------------------------------------------------------------------------- _exclude_field

def test_exclude_field_removes_the_property():
    schema = {"properties": {"keep": {"type": "string"}, "drop": {"type": "string"}}}
    _exclude_field(schema, "drop")
    assert "drop" not in schema["properties"]
    assert "keep" in schema["properties"]


def test_exclude_field_also_removes_it_from_required():
    schema = {
        "properties": {"keep": {"type": "string"}, "drop": {"type": "string"}},
        "required": ["keep", "drop"],
    }
    _exclude_field(schema, "drop")
    assert schema["required"] == ["keep"]


def test_exclude_field_is_a_noop_for_an_absent_field():
    schema = {"properties": {"keep": {"type": "string"}}, "required": ["keep"]}
    _exclude_field(schema, "not-there")  # must not raise
    assert schema["properties"] == {"keep": {"type": "string"}}
    assert schema["required"] == ["keep"]


def test_exclude_field_handles_a_schema_without_required():
    schema = {"properties": {"drop": {"type": "string"}}}
    _exclude_field(schema, "drop")  # must not raise
    assert schema["properties"] == {}


# --------------------------------------------------------------------------- _build_schema

def test_build_schema_produces_a_flat_object_schema():
    schema = _build_schema()

    assert schema["type"] == "object"
    assert "$defs" not in schema
    assert "$ref" not in json.dumps(schema)  # every ref inlined


def test_build_schema_never_exposes_the_source_field():
    # source is local metadata stamped by signal_parser.py/claude_client.py —
    # Claude must never see, decide, or spend output tokens on it.
    schema = _build_schema()
    assert "source" not in schema["properties"]
    assert "source" not in schema.get("required", [])


def test_build_schema_keeps_the_fields_claude_must_produce():
    schema = _build_schema()
    for field in ("is_signal", "category", "setup", "summary", "confidence"):
        assert field in schema["properties"]


def test_build_schema_tightens_every_object_and_strips_prose():
    schema = _build_schema()
    serialized = json.dumps(schema)

    assert schema["additionalProperties"] is False
    assert schema["properties"]["setup"]["additionalProperties"] is False
    assert '"description"' not in serialized
    assert '"title"' not in serialized


def test_build_schema_stays_within_the_argv_safety_budget():
    # Regression guard for the Windows .cmd-shim command-line cap that made
    # every analysis fail silently when the schema grew too large.
    payload = json.dumps(_build_schema(), separators=(",", ":"))
    assert len(payload.encode("utf-8")) <= _ARGV_SAFETY_BUDGET_BYTES


def test_build_schema_is_deterministic():
    assert _build_schema() == _build_schema()


def test_build_schema_does_not_mutate_the_pydantic_model_schema():
    # _build_schema pops/mutates its working copy; calling it must not
    # corrupt SignalAnalysis.model_json_schema() for any other caller.
    from models import SignalAnalysis

    _build_schema()
    fresh = SignalAnalysis.model_json_schema()

    assert "source" in fresh["properties"]  # untouched by the call above


# ------------------------------------------------------- CLI arguments (regression)

def _settings(**overrides) -> Settings:
    base = dict(
        api_id=1, api_hash="x", session_name="t", phone="", target_chat="t",
        claude_cli_path="claude", model="", claude_max_turns=3,
        signal_parser_mode="off", worker_count=1, queue_maxsize=10,
        max_message_chars=8000, analyse_edits=False, max_retries=0,
        request_timeout=5.0, log_level="INFO", log_file=None,
        jsonl_output=None, color=False, show_json=False,
        storage_db_path=Path("data/signals.db"), twelve_data_api_key="",
        market_data_cache_ttl=30.0, market_data_timeframe=Timeframe.H1,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def captured_args(monkeypatch):
    """Build a ClaudeAnalyzer and capture the argv it would spawn."""
    monkeypatch.setattr(shutil, "which", lambda cmd: "/fake/claude")
    seen: list[list[str]] = []

    def _make(**settings_overrides):
        analyzer = ClaudeAnalyzer(_settings(**settings_overrides))

        async def fake_spawn(args):
            seen.append(list(args))
            proc = MagicMock()
            proc.returncode = 0
            proc.communicate = AsyncMock(return_value=(b'{"structured_output":null}', b""))
            return proc

        monkeypatch.setattr(analyzer, "_spawn", fake_spawn)
        asyncio.run(analyzer._run_cli("instruction", "payload"))
        return seen[-1]

    return _make


def test_structured_output_tool_is_allowed(captured_args):
    """The tool that delivers --json-schema results must be permitted.

    Denying it makes the model's (correct) answer get refused, retried, and
    the run fail with error_max_turns — every analysis failing in production.
    """
    args = captured_args()
    assert "--allowedTools" in args
    assert args[args.index("--allowedTools") + 1] == "StructuredOutput"


def test_blanket_tool_denial_is_never_used(captured_args):
    args = captured_args()
    assert "--disallowedTools" not in args, (
        "a blanket deny-list also denies StructuredOutput and breaks every call"
    )


def test_json_schema_and_core_flags_are_passed(captured_args):
    args = captured_args()
    for flag in ("-p", "--output-format", "--json-schema",
                 "--system-prompt-file", "--permission-mode",
                 "--setting-sources", "--max-turns"):
        assert flag in args, f"{flag} missing"
    assert args[args.index("--output-format") + 1] == "json"
    assert args[args.index("--max-turns") + 1] == "3"


def test_model_flag_only_present_when_configured(captured_args):
    assert "--model" not in captured_args(model="")
    args = captured_args(model="sonnet")
    assert args[args.index("--model") + 1] == "sonnet"


# ------------------------------------------------------- error reporting (regression)

def _analyzer(monkeypatch) -> ClaudeAnalyzer:
    monkeypatch.setattr(shutil, "which", lambda cmd: "/fake/claude")
    return ClaudeAnalyzer(_settings())


def test_error_envelope_without_result_is_not_reported_as_unknown(monkeypatch):
    """A failed run carries no `result`; the reason lives in subtype/errors.

    Reading only `result` collapsed every such failure into "unknown error",
    which is what made this class of bug undiagnosable from the logs.
    """
    analyzer = _analyzer(monkeypatch)
    envelope = json.dumps({
        "is_error": True,
        "subtype": "error_max_turns",
        "terminal_reason": "max_turns",
        "errors": ["Reached maximum number of turns (3)"],
        "permission_denials": [{"tool_name": "StructuredOutput", "tool_input": {}}],
    })

    with pytest.raises(AnalysisError) as exc_info:
        analyzer._parse_output(1, envelope, "")

    detail = str(exc_info.value)
    assert "unknown error" not in detail
    assert "error_max_turns" in detail
    assert "StructuredOutput" in detail          # names the denied tool
    assert "maximum number of turns" in detail   # surfaces the errors array


def test_error_envelope_still_includes_result_when_present(monkeypatch):
    analyzer = _analyzer(monkeypatch)
    envelope = json.dumps({"is_error": True, "result": "authentication_failed"})

    with pytest.raises(AnalysisError) as exc_info:
        analyzer._parse_output(1, envelope, "")

    assert "authentication_failed" in str(exc_info.value)
    assert exc_info.value.retryable is False  # auth failure must not retry


def test_error_detail_falls_back_to_exit_code_when_envelope_is_bare(monkeypatch):
    analyzer = _analyzer(monkeypatch)

    with pytest.raises(AnalysisError) as exc_info:
        analyzer._parse_output(7, json.dumps({"is_error": True}), "")

    assert "unknown error" not in str(exc_info.value)
    assert "exit=7" in str(exc_info.value)
