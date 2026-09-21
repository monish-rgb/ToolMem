"""Phase 1 token-telemetry tests (plan section 6)."""

from __future__ import annotations

from toolatlas.token_usage import (
    RequestLedger,
    TokenUsage,
    as_token_usage,
    check_resume_compatible,
    cross_check_usage,
    normalize_gemini_usage,
    normalize_openai_usage,
    prompt_anatomy,
    tokenizer_estimate_usage,
)


def test_gemini_exact_normalization():
    usage = normalize_gemini_usage(
        {
            "promptTokenCount": 120,
            "cachedContentTokenCount": 20,
            "candidatesTokenCount": 40,
            "thoughtsTokenCount": 10,
            "totalTokenCount": 160,
        },
        "r1",
    )
    assert usage.input_tokens == 120
    assert usage.cached_input_tokens == 20
    assert usage.uncached_input_tokens == 100
    assert usage.output_tokens == 40
    assert usage.reasoning_tokens == 10
    # No double counting: total is the provider total, not output+reasoning.
    assert usage.total_tokens == 160
    assert usage.usage_source == "provider"
    assert usage.has_authoritative_input is True
    assert usage.raw["promptTokenCount"] == 120


def test_openai_exact_normalization():
    usage = normalize_openai_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "total_tokens": 130,
            "prompt_tokens_details": {"cached_tokens": 25},
            "completion_tokens_details": {"reasoning_tokens": 12},
        },
        "r2",
    )
    assert usage.input_tokens == 100
    assert usage.cached_input_tokens == 25
    assert usage.uncached_input_tokens == 75
    assert usage.output_tokens == 30
    assert usage.reasoning_tokens == 12
    assert usage.total_tokens == 130
    assert usage.has_authoritative_input is True


def test_cached_absent_stays_none():
    usage = normalize_openai_usage({"prompt_tokens": 50, "completion_tokens": 5})
    assert usage.cached_input_tokens is None
    assert usage.uncached_input_tokens is None
    assert usage.input_tokens == 50


def test_missing_usage_never_zero():
    for usage in (
        normalize_gemini_usage({}),
        normalize_gemini_usage(None),
        normalize_openai_usage(None),
        normalize_openai_usage({}),
        normalize_openai_usage({"completion_tokens": 5}),
    ):
        assert usage.input_tokens is None
        assert usage.usage_source == "missing"
        assert usage.has_authoritative_input is False
        assert usage.is_complete is False


def test_reasoning_not_double_counted_openai():
    usage = normalize_openai_usage(
        {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30,
         "completion_tokens_details": {"reasoning_tokens": 15}}
    )
    assert usage.total_tokens == 30  # provider total, not 10+20+15
    assert usage.output_tokens == 20
    assert usage.reasoning_tokens == 15


def test_retries_are_separate_rows_and_totals_sum():
    ledger = RequestLedger()
    first = normalize_openai_usage({"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}, "a")
    retry = normalize_openai_usage({"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}, "b")
    ledger.log(experiment_id="e", task_id="t", arm="baseline", attempt=1,
               model="m", provider="openai_compatible", usage=first)
    ledger.log(experiment_id="e", task_id="t", arm="baseline", attempt=1,
               model="m", provider="openai_compatible", usage=retry, is_retry=True)
    rows = ledger.attempt_rows("t", "baseline", 1)
    assert len(rows) == 2
    assert rows[0].request_id != rows[1].request_id
    totals = ledger.attempt_totals("t", "baseline", 1)
    assert totals["model_requests"] == 2
    assert totals["input_tokens"] == 200
    assert totals["total_tokens"] == 220
    assert totals["has_complete_input_usage"] is True


def test_incomplete_requests_excluded_from_primary():
    ledger = RequestLedger()
    ledger.log(experiment_id="e", task_id="t", arm="a", attempt=1, model="m",
               provider="gemini", usage=normalize_gemini_usage({"promptTokenCount": 50}, "ok"))
    ledger.log(experiment_id="e", task_id="t", arm="a", attempt=1, model="m",
               provider="gemini", usage=TokenUsage.missing("bad"))
    totals = ledger.attempt_totals("t", "a", 1)
    assert totals["model_requests"] == 2
    assert totals["incomplete_requests"] == 1
    assert totals["has_complete_input_usage"] is False
    # Only the authoritative row contributes; missing row is not zero-filled.
    assert totals["input_tokens"] == 50


def test_as_token_usage_accepts_normalized_form():
    # Regression: GeminiRestClient.generate returns both normalized "usage"
    # and raw "raw_usage". Passing the normalized form back must not mark
    # the request missing (double-normalization bug: avg_tokens 0).
    normalized = normalize_openai_usage(
        {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}, "r1")
    revived = as_token_usage(normalized.to_dict(), "openai_compatible")
    assert revived.is_complete is True
    assert revived.input_tokens == 100
    assert revived.usage_source == "provider"
    # Raw provider payloads still normalize through the provider path.
    assert as_token_usage({"promptTokenCount": 50}, "gemini").input_tokens == 50
    assert as_token_usage(None, "gemini").is_complete is False


def test_resume_rejects_mismatch():
    prior = {"model": "gemini-2.5-flash", "provider": "gemini", "temperature": 0.0}
    same = dict(prior)
    assert check_resume_compatible(prior, same) == []
    changed = dict(prior, model="other")
    assert "model" in check_resume_compatible(prior, changed)


def test_prompt_anatomy_sections():
    anatomy = prompt_anatomy(system="sys", tool_schemas=[{"name": "a"}],
                             guidance="guide", history="h", task_text="task")
    assert anatomy["estimated_tokens"]["total_estimate"] == sum(
        v for k, v in anatomy["estimated_tokens"].items() if k != "total_estimate"
    )
    assert "toolatlas_guidance" in anatomy["sections"]


def test_cross_check_flags_anomaly_only_with_authoritative():
    usage = normalize_openai_usage({"prompt_tokens": 100, "completion_tokens": 10})
    ok = cross_check_usage(usage, prompt_text="x" * 400)  # ~100 estimate tokens
    assert ok["comparable"] is True and ok["anomaly"] is False
    assert cross_check_usage(TokenUsage.missing())["comparable"] is False
    assert tokenizer_estimate_usage("hello").usage_source == "tokenizer_estimate"
