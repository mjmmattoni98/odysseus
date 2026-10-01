"""Agent metrics keep backend-reported speed and load time (native Ollama)."""
from src.agent_loop import (
    _accumulate_backend_timings,
    _backend_speed_metrics,
    _compute_final_metrics,
)


def _metrics(**kwargs):
    base = dict(
        messages=[{"role": "user", "content": "hi"}],
        full_response="hello there",
        total_duration=30.0,
        time_to_first_token=20.0,
        context_length=32768,
        real_input_tokens=900,
        real_output_tokens=120,
        has_real_usage=True,
        tool_events=[],
        round_texts=[],
    )
    base.update(kwargs)
    return _compute_final_metrics(**base)


def test_backend_generation_speed_replaces_the_wall_clock_estimate():
    metrics = _metrics(
        backend_gen_tps=40.0,
        backend_prefill_tps=600.0,
        backend_timings={"load_ms": 18200.04, "prefill_ms": 1500.0, "gen_ms": 3000.0},
    )

    assert metrics["tokens_per_second"] == 40.0
    assert metrics["tps_source"] == "backend"
    assert metrics["prefill_tps"] == 600.0
    assert metrics["load_ms"] == 18200.0
    assert metrics["prefill_ms"] == 1500.0
    assert metrics["gen_ms"] == 3000.0
    assert "finish_reason" not in metrics


def test_estimate_is_kept_when_the_backend_reports_no_speed():
    metrics = _metrics()

    assert metrics["tokens_per_second"] == 4.0
    assert metrics["tps_source"] == "computed"
    assert "load_ms" not in metrics


def test_length_cut_is_recorded_in_metrics():
    assert _metrics(finish_reason="length")["finish_reason"] == "length"
    assert "finish_reason" not in _metrics(finish_reason="stop")


def test_round_timings_are_summed_and_malformed_values_ignored():
    totals = {}
    _accumulate_backend_timings(totals, {"load_ms": 18000.0, "prefill_ms": 900.0, "gen_ms": 100.0})
    _accumulate_backend_timings(totals, {"load_ms": 40.0, "prefill_ms": "fast", "gen_ms": float("nan")})
    _accumulate_backend_timings(totals, {"load_ms": True, "gen_ms": -5})

    assert totals == {"load_ms": 18040.0, "prefill_ms": 900.0, "gen_ms": 100.0}


def test_direct_reply_metrics_use_backend_speed_from_usage():
    metrics = _backend_speed_metrics({
        "input_tokens": 10,
        "output_tokens": 20,
        "gen_tps": 55.123,
        "prefill_tps": 800.0,
        "load_ms": 700.0,
        "finish_reason": "length",
    })

    assert metrics == {
        "tokens_per_second": 55.12,
        "tps_source": "backend",
        "prefill_tps": 800.0,
        "load_ms": 700.0,
        "finish_reason": "length",
    }
    assert _backend_speed_metrics({"input_tokens": 1, "output_tokens": 1}) == {}
