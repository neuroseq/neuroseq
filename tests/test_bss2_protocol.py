from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.device_control import bss2_protocol as bp
from neuroseq.device_control.bss2_protocol import Bss2Settings


def _settings(**kwargs) -> Bss2Settings:
    settings = Bss2Settings()
    for key, value in kwargs.items():
        setattr(settings, key, value)
    return settings


# ---- arrays and time base ---------------------------------------------------


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        (np.array([1.5, -2.25], dtype=np.float32), "<f4"),
        (np.arange(6, dtype=np.uint32).reshape(2, 3), "<u4"),
        (np.array([[True, False]]), "|u1"),
        (np.array([1022, 0], dtype=np.uint16), "<u2"),
    ],
)
def test_array_roundtrip(values, dtype) -> None:
    payload = bp.encode_array(values, dtype)
    assert isinstance(payload["data"], str)
    decoded = bp.decode_array(payload)
    assert decoded.dtype == np.dtype(dtype)
    assert np.array_equal(decoded, values.astype(dtype))


def test_array_decode_validates() -> None:
    payload = bp.encode_array(np.zeros(3), "<f4")
    with pytest.raises(ValueError, match="size"):
        bp.decode_array({**payload, "shape": [4]})
    with pytest.raises(ValueError, match="dtype"):
        bp.decode_array({**payload, "dtype": "<f8"})
    with pytest.raises(ValueError, match="Negative"):
        bp.decode_array({**payload, "shape": [-3]})
    with pytest.raises(ValueError, match="dtype"):
        bp.encode_array(np.zeros(1), "object")


def test_time_base_conversion() -> None:
    assert bp.bio_s_to_hw_ms(1.0, 1000) == pytest.approx(1.0)
    assert bp.hw_ms_to_bio_s(0.5, 1000) == pytest.approx(0.5)
    assert bp.bio_s_to_hw_ms(2.0, 100) == pytest.approx(20.0)
    with pytest.raises(ValueError, match="speed-up"):
        bp.bio_s_to_hw_ms(1.0, 0)


# ---- settings ---------------------------------------------------------------


def test_settings_roundtrip_and_sanitising() -> None:
    settings = _settings(container_image="/img.sif", madc_neuron=3, speedup=500.0)
    restored = Bss2Settings.from_dict(settings.to_dict())
    assert restored == settings
    assert "token" not in " ".join(settings.to_dict()).lower()

    odd = Bss2Settings.from_dict(
        {
            "input_mode": "bogus",
            "spike_source": "bogus",
            "i_offset_max": 5000,
            "spike_weight": -4,
            "speedup": -1,
            "max_neurons": 0,
            "madc_neuron": "not a number",
            "unknown": 1,
            "token": "secret",
        }
    )
    assert odd.input_mode == bp.INPUT_CONSTANT_CURRENT
    assert odd.spike_source == bp.SPIKE_SOURCE_STIM
    assert odd.i_offset_max == bp.I_OFFSET_LIMIT
    assert odd.spike_weight == 0
    assert odd.speedup == bp.DEFAULT_SPEEDUP
    assert odd.max_neurons == 1
    assert odd.madc_neuron == 0
    assert not hasattr(odd, "token")
    assert Bss2Settings.from_dict(None) == Bss2Settings()


# ---- segmentation -----------------------------------------------------------


def test_change_segments_merges_equal_runs() -> None:
    values = np.array([[0, 0], [0, 0], [1, 0], [1, 0], [1, 2], [0, 0]], np.float32)
    starts, lengths = bp.change_segments(values)
    assert starts.tolist() == [0, 2, 4, 5]
    assert lengths.tolist() == [2, 2, 1, 1]
    with pytest.raises(ValueError):
        bp.change_segments(np.zeros((0, 2)))


def test_constant_current_plan_maps_values() -> None:
    values = np.array([[0, 0], [2, 0], [2, 1], [4, 1], [0, 0]], np.float32)
    plan = bp.plan_constant_current(values, 0.1, _settings(i_offset_max=1000))
    assert plan.n_neurons == 2
    assert plan.n_segments == 5
    assert plan.duration_s == pytest.approx(0.5)
    assert plan.value_at_max == 4
    assert plan.enable.tolist() == [
        [False, False],
        [True, False],
        [True, True],
        [True, True],
        [False, False],
    ]
    assert plan.offset.tolist() == [[0, 0], [500, 0], [500, 250], [1000, 250], [0, 0]]
    assert plan.warnings == []


def test_constant_current_plan_value_at_max_clips_and_requantised_runs_merge() -> None:
    # 1.0 and 1.001 quantise to the same offset; 9 clips to i_offset_max.
    values = np.array([[1.0], [1.001], [9.0]], np.float32)
    plan = bp.plan_constant_current(
        values, 0.01, _settings(i_offset_max=10, value_at_max=2.0)
    )
    assert plan.n_segments == 2
    assert plan.durations_s.tolist() == pytest.approx([0.02, 0.01])
    assert plan.offset[:, 0].tolist() == [5, 10]


def test_constant_current_plan_all_zero_and_single_sample() -> None:
    plan = bp.plan_constant_current(np.zeros((50, 3)), 0.001, _settings())
    assert plan.n_segments == 1
    assert not plan.enable.any() and not plan.offset.any()
    assert plan.duration_s == pytest.approx(0.05)

    single = bp.plan_constant_current(np.array([[3.0, 0.0]]), 0.2, _settings())
    assert single.n_segments == 1
    assert single.enable.tolist() == [[True, False]]
    assert single.offset.tolist() == [[500, 0]]

    one_d = bp.plan_constant_current(np.array([0.0, 1.0]), 0.1, _settings())
    assert one_d.n_neurons == 1 and one_d.n_segments == 2


def test_constant_current_plan_change_on_one_neuron_only() -> None:
    values = np.zeros((6, 3), np.float32)
    values[2:4, 1] = 1.0
    plan = bp.plan_constant_current(values, 0.1, _settings())
    assert plan.n_segments == 3
    assert plan.enable[:, 0].tolist() == [False] * 3
    assert plan.enable[:, 1].tolist() == [False, True, False]
    assert plan.durations_s.tolist() == pytest.approx([0.2, 0.2, 0.2])


def test_neuron_count_truncates_and_pads() -> None:
    values = np.ones((2, 6), np.float32)
    truncated = bp.plan_constant_current(values, 0.1, _settings(max_neurons=4))
    assert truncated.n_neurons == 4
    assert any("first 4" in w for w in truncated.warnings)

    padded = bp.plan_constant_current(values[:, :2], 0.1, _settings(n_neurons=5))
    assert padded.n_neurons == 5
    assert padded.enable[0].tolist() == [True, True, False, False, False]
    assert any("receive no input" in w for w in padded.warnings)

    capped = bp.plan_constant_current(
        values, 0.1, _settings(n_neurons=10, max_neurons=8)
    )
    assert capped.n_neurons == 8
    assert any("at most 8" in w for w in capped.warnings)


def test_constant_current_plan_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        bp.plan_constant_current(np.zeros((0, 2)), 0.1, _settings())
    with pytest.raises(ValueError, match="dt"):
        bp.plan_constant_current(np.zeros((2, 2)), 0.0, _settings())
    with pytest.raises(ValueError, match="no channels"):
        bp.resolve_neuron_count(0, _settings())


def test_many_segments_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bp, "SEGMENT_WARNING_LIMIT", 3)
    plan = bp.plan_constant_current(np.arange(10.0)[:, None], 0.1, _settings())
    assert any("10 segments" in w for w in plan.warnings)


# ---- spike input ------------------------------------------------------------


def test_channels_to_neurons_and_stim_to_spike_times() -> None:
    sampled = np.array([[1.0, 2.0]])
    assert np.array_equal(bp.channels_to_neurons(sampled[:, :1], 2), [[1.0, 1.0]])
    trains = bp.stim_to_spike_times(
        [(np.array([0.0, 1.0, 0.0, 2.0]), 0.1)], 0.4, 2, 0.5, 100.0
    )
    assert [t.tolist() for t in trains] == [[150.0, 350.0], [150.0, 350.0]]
    with pytest.raises(ValueError, match="interval"):
        bp.stim_to_spike_times([], 1.0, 1, 0.0, 0.0)


def test_spike_plan_from_stim() -> None:
    values = np.zeros((10, 2), np.float32)
    values[2:5, 1] = 1.0
    plan = bp.plan_spikes_from_stim(
        values, 0.1, _settings(spike_event_ms=100.0, spike_weight=40)
    )
    assert plan.n_neurons == 2
    assert plan.duration_s == pytest.approx(1.0)
    assert plan.indices.tolist() == [1, 1, 1]
    assert plan.times_s.tolist() == pytest.approx([0.25, 0.35, 0.45])
    assert plan.weight == 40

    silent = bp.plan_spikes_from_stim(np.zeros(5), 0.1, _settings())
    assert silent.n_spikes == 0
    assert any("No input spikes" in w for w in silent.warnings)
    with pytest.raises(ValueError, match="empty"):
        bp.plan_spikes_from_stim(np.zeros((0, 1)), 0.1, _settings())
    with pytest.raises(ValueError, match="dt"):
        bp.plan_spikes_from_stim(np.zeros((1, 1)), -1, _settings())


def test_spike_plan_from_data_sorts_and_filters() -> None:
    spikes = SpikeData(
        "s",
        spike_indices=np.array([2, 0, 5, 0]),
        spike_times=np.array([0.3, 0.2, 0.1, 0.05]),
    )
    plan = bp.plan_spikes_from_data(spikes, 1.0, _settings(max_neurons=4))
    assert plan.n_neurons == 4
    assert plan.indices.tolist() == [0, 0, 2]
    assert plan.times_s.tolist() == pytest.approx([0.05, 0.2, 0.3])

    no_duration = bp.plan_spikes_from_data(spikes, 0.0, _settings())
    assert no_duration.duration_s == pytest.approx(0.3)
    with pytest.raises(ValueError, match="no duration"):
        bp.plan_spikes_from_data(
            SpikeData("e", np.zeros(0), np.zeros(0)), 0.0, _settings()
        )


# ---- request / readback -----------------------------------------------------


def test_build_run_arguments_constant_current() -> None:
    values = np.array([[0.0, 1.0], [2.0, 1.0]], np.float32)
    settings = _settings(madc_neuron=1, speedup=1000.0)
    plan = bp.plan_constant_current(values, 0.5, settings)
    arguments = bp.build_run_arguments(plan, settings)
    assert arguments["mode"] == bp.INPUT_CONSTANT_CURRENT
    assert arguments["n_neurons"] == 2
    assert arguments["duration_ms"] == pytest.approx(1.0)
    assert arguments["record_v_neuron"] == 1
    # Empty: the helper uses HXNeuron with the chip's nightly calibration.
    assert arguments["neuron_parameters"] == {}
    segments = arguments["segments"]
    assert bp.decode_array(segments["durations_ms"]).tolist() == pytest.approx(
        [0.5, 0.5]
    )
    assert bp.decode_array(segments["enable"]).tolist() == [[0, 1], [1, 1]]
    assert bp.decode_array(segments["offset"]).tolist() == [[0, 250], [500, 250]]

    with pytest.raises(ValueError, match="recorded neuron"):
        bp.build_run_arguments(plan, _settings(madc_neuron=2))


def test_build_run_arguments_spikes() -> None:
    settings = _settings(speedup=100.0, spike_weight=12)
    plan = bp.plan_spikes_from_data(
        SpikeData("s", np.array([0, 1]), np.array([0.1, 0.2])), 1.0, settings
    )
    arguments = bp.build_run_arguments(plan, settings)
    assert arguments["mode"] == bp.INPUT_SPIKES
    assert arguments["duration_ms"] == pytest.approx(10.0)
    spikes = arguments["spikes"]
    assert spikes["weight"] == 12
    assert bp.decode_array(spikes["indices"]).tolist() == [0, 1]
    assert bp.decode_array(spikes["times_ms"]).tolist() == pytest.approx([1.0, 2.0])


def _result(v_times_ms, v_values, *, neuron=1) -> dict:
    return {
        "duration_ms": 2.0,
        "spikes": {
            "indices": bp.encode_array(np.array([1, 0]), "<u4"),
            "times_ms": bp.encode_array(np.array([0.5, 1.25]), "<f4"),
        },
        "v": {
            "neuron": neuron,
            "times_ms": bp.encode_array(np.asarray(v_times_ms), "<f4"),
            "values": bp.encode_array(np.asarray(v_values), "<f4"),
        },
    }


def test_decode_run_result_converts_to_biological_time() -> None:
    readback = bp.decode_run_result(
        _result([0.002, 0.001, 0.003], [3.0, 1.0, 5.0]), 1000.0, label="BSS-2 x"
    )
    assert readback.duration_s == pytest.approx(2.0)
    spikes = readback.spikes
    assert spikes.spike_indices.dtype == np.uint32
    assert spikes.spike_indices.tolist() == [1, 0]
    assert spikes.spike_times.tolist() == pytest.approx([0.5, 1.25])
    trace = readback.trace
    assert isinstance(trace, TraceData)
    assert trace.units == "a.u."
    assert trace.indices.tolist() == [1]
    assert "neuron 1" in trace.name
    assert trace.dt == pytest.approx(0.001)
    # Leading sample pads the trace back to t=0; samples were sorted by time.
    assert trace.traces.shape == (1, 4)
    assert trace.traces[0].tolist() == pytest.approx([1.0, 1.0, 3.0, 5.0])


def test_decode_run_result_without_or_with_degenerate_v() -> None:
    result = _result([0.1], [1.0])
    assert bp.decode_run_result(result, 1000.0).trace is None
    result["v"] = None
    assert bp.decode_run_result(result, 1000.0).trace is None
    with pytest.raises(ValueError, match="spacing"):
        bp.decode_run_result(_result([0.1, 0.1], [1.0, 2.0]), 1000.0)
    with pytest.raises(ValueError, match="differ"):
        bp.decode_run_result(_result([0.1, 0.2], [1.0]), 1000.0)
    bad = _result([0.1, 0.2], [1.0, 2.0])
    bad["spikes"]["indices"] = bp.encode_array(np.array([1]), "<u4")
    with pytest.raises(ValueError, match="differ"):
        bp.decode_run_result(bad, 1000.0)


def test_readback_experiment_name_and_data() -> None:
    readback = bp.decode_run_result(_result([0.0, 0.001], [1, 2]), 1000.0)
    experiment = bp.readback_experiment(
        readback, "opsin", datetime(2026, 1, 2, 3, 4, 5)
    )
    assert experiment.name == "BSS-2 opsin 2026-01-02 03-04-05"
    assert experiment.duration == pytest.approx(2.0)
    assert [type(d) for d in experiment.data] == [SpikeData, TraceData]
    readback.trace = None
    assert len(bp.readback_experiment(readback, "opsin").data) == 1


def test_plan_summary_mentions_key_numbers() -> None:
    settings = _settings(madc_neuron=0)
    plan = bp.plan_constant_current(np.ones((10, 2)), 0.1, settings)
    text = bp.plan_summary(plan, settings)
    assert "Neurons: 2" in text and "1 segment" in text and "1 ms on hardware" in text
    spike_plan = bp.plan_spikes_from_stim(np.ones((10, 2)), 0.1, settings)
    assert "spike(s)" in bp.plan_summary(spike_plan, settings)


# ---- quiggeldy setups -------------------------------------------------------


def test_parse_setups_csv() -> None:
    text = (
        'Host,Port,Identifier\n"quiggeldy.bioai.eu",11717,"hxcube7fpga3chip61_1"\n'
        "other.host, 1234 , chip2\nbad,port,chip\n,1,x\n"
    )
    setups = bp.parse_setups_csv(text)
    assert setups == [
        bp.QuiggeldySetup("quiggeldy.bioai.eu", 11717, "hxcube7fpga3chip61_1"),
        bp.QuiggeldySetup("other.host", 1234, "chip2"),
    ]
    assert "quiggeldy.bioai.eu:11717" in setups[0].label
    assert bp.parse_setups_csv("") == []
    with pytest.raises(ValueError, match="identifier"):
        bp.parse_setups_csv("Host,Port\na,1\n")


def test_default_setup_index_matches_collaborator_formula() -> None:
    expected = int(hashlib.sha256(b"alice").hexdigest(), 16) % 3
    assert bp.default_setup_index("alice", 3) == expected
    assert bp.default_setup_index("alice", 1) == 0
    with pytest.raises(ValueError):
        bp.default_setup_index("alice", 0)


def test_current_username_prefers_user_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER", "bob")
    assert bp.current_username() == "bob"
    monkeypatch.delenv("USER")
    assert bp.current_username()


# ---- launch -----------------------------------------------------------------


def test_launch_command_keeps_token_out_of_argv(tmp_path: Path) -> None:
    token = "eyJhbGciOi.secret-payload.signature"
    config = bp.LaunchConfig(
        container_image="/c/image.sif",
        container_app="dls",
        host="quiggeldy.bioai.eu",
        port=11717,
        user="alice",
        helper_path=tmp_path / "dir with space" / "helper.py",
    )
    command = bp.helper_launch_command(config, token)
    assert command.program == "singularity"
    argv = command.arguments
    assert argv[:3] == ["exec", "--bind", str(tmp_path / "dir with space")]
    assert argv[3:5] == ["--app", "dls"]
    assert argv[5:8] == ["/c/image.sif", "bash", "-c"]
    script = argv[8]
    assert "spack env activate esd" in script and "1>&2" in script
    assert "exec python -u" in script and "'" in script  # quoted helper path
    assert all(token not in part for part in argv)
    env = command.environment
    assert env["SINGULARITYENV_QUIGGELDY_TOKEN"] == token
    assert env["APPTAINERENV_QUIGGELDY_TOKEN"] == token
    assert env["QUIGGELDY_TOKEN"] == token
    assert env["SINGULARITYENV_QUIGGELDY_IP"] == "quiggeldy.bioai.eu"
    assert env["QUIGGELDY_PORT"] == "11717"
    assert env["SINGULARITYENV_QUIGGELDY_USER_NO_MUNGE"] == "alice"
    assert env["QUIGGELDY_ENABLED"] == "1"

    no_app = bp.helper_launch_command(
        bp.LaunchConfig("/img", "", "h", 1, "u"), token=""
    )
    assert "--app" not in no_app.arguments
    assert "QUIGGELDY_TOKEN" not in no_app.environment
    with pytest.raises(ValueError, match="image"):
        bp.helper_launch_command(bp.LaunchConfig(" ", "", "h", 1, "u"), "")


def test_scrub_secrets() -> None:
    text = "token=abcd1234 eyJa.bcd.efg QUIGGELDY_TOKEN=xyz other"
    scrubbed = bp.scrub_secrets(text, ["abcd1234", ""])
    assert "abcd1234" not in scrubbed
    assert "eyJa.bcd.efg" not in scrubbed
    assert "xyz" not in scrubbed
    assert scrubbed.endswith("other")


def test_summarize_message_replaces_arrays_and_truncates() -> None:
    message = {
        "id": 3,
        "command": "run",
        "arguments": {"segments": {"offset": bp.encode_array(np.ones((2, 3)), "<u2")}},
    }
    text = bp.summarize_message(message)
    assert "<array <u2 [2, 3]>" in text and "data" not in text
    long = bp.summarize_message({"x": "y" * 50}, limit=20)
    assert long.startswith('{"x": "yyyyyyyyyyyy') and "more characters" in long


def test_describe_launch_hides_the_token_even_after_scrubbing() -> None:
    config = bp.LaunchConfig(
        container_image="/img.sif", container_app="", host="h", port=1, user="u"
    )
    token = "secret-token-value"
    lines = bp.describe_launch(bp.helper_launch_command(config, token))
    text = "\n".join(lines)
    assert token not in text
    assert lines[0].startswith("Launch: singularity exec --bind ")
    assert f"QUIGGELDY_TOKEN (value hidden): {len(token)} characters" in text
    # Debug lines are scrubbed before display; the length note must survive.
    assert "value hidden" in bp.scrub_secrets(text, [token])
    assert (
        "QUIGGELDY_IP=h" in text
        and "SINGULARITYENV_ and APPTAINERENV_ prefixes" in text
    )
    no_token = "\n".join(bp.describe_launch(bp.helper_launch_command(config, "")))
    assert "QUIGGELDY_TOKEN not set by NeuroSeq" in no_token


def test_debug_report_header_lists_context_but_not_the_token() -> None:
    lines = bp.debug_report_header(
        Bss2Settings(container_image="/img.sif"),
        token_given=True,
        neuroseq_version="9.9",
        singularity_version="singularity 4.0",
        connection_state="connected",
        identifier="chip_x",
    )
    text = "\n".join(lines)
    assert "neuroseq:    9.9" in text and "singularity 4.0" in text
    assert "connected (identifier chip_x)" in text and "token:       given" in text
    assert "container_image = /img.sif" in text
    assert lines[-1] == "=== log ==="


def test_launch_caches_the_spack_activation(tmp_path) -> None:
    image = tmp_path / "image.sif"
    image.write_bytes(b"")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    config = bp.LaunchConfig(
        container_image=str(image),
        container_app="",
        host="h",
        port=1,
        user="u",
        cache_dir=cache_dir,
    )
    command = bp.helper_launch_command(config, "")
    bind = command.arguments[command.arguments.index("--bind") + 1]
    assert bind.split(",")[1] == str(cache_dir.resolve())
    script = command.arguments[-1]
    cache = bp.activation_cache_file(cache_dir, str(image))
    assert cache.parent == cache_dir and cache.name.startswith("esd-activate-")
    assert f"if [ -s {cache} ]; then source {cache};" in script
    assert f"spack env activate --sh esd > {cache}.partial" in script
    assert script.endswith(f"exec python -u {bp.HELPER_PATH}")
    # Updating the image changes the cache key.
    import os

    os.utime(image, ns=(1, 1))
    assert bp.activation_cache_file(cache_dir, str(image)) != cache
    assert bp.activation_cache_file(cache_dir, str(tmp_path / "missing")).name


def test_launch_without_cache_dir_activates_spack_each_time() -> None:
    command = bp.helper_launch_command(bp.LaunchConfig("/img", "", "h", 1, "u"), "")
    assert "spack env activate esd" in command.arguments[-1]
    assert "--sh" not in command.arguments[-1]
