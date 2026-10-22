from neuroseq.stimulation.dynamic_input.rheobase_dialog import RheobaseBackend
from neuroseq.stimulation.dynamic_input.rheobase_dialog import RheobaseRunConfig


class Host:
    def __init__(self):
        self.requests = []
        self.starts = 0

    def request_simulation(self, seconds):
        self.requests.append(seconds)

    def append_block(self, block):
        start = self.starts
        self.starts += len(block)
        return start


def test_rheobase_backend_bisects_without_qt():
    host = Host()
    backend = RheobaseBackend(host)
    finished = []
    backend.on_finished = finished.append
    backend.on_summarize = lambda: "summary"
    config = RheobaseRunConfig(
        "Fixed duration, varying amplitude",
        0.0,
        10.0,
        5.0,
        5.0,
        5.0,
        1.0,
        1.0,
        1.0,
        0,
        -20.0,
        0.1,
        20,
        0.001,
        1,
    )
    backend.begin_search(config)
    while backend.running:
        backend.record_probe_result(int(backend.pending_amplitude >= 5.0))
    assert backend.rheobase == 5.0
    assert backend.samples
    assert backend.thresholds == [(5.0, 5.0)]
    assert finished == ["summary"]


def _config(strategy, min_amp, max_amp, pulse_ms, max_pulse_ms, repeats=1):
    return RheobaseRunConfig(
        strategy=strategy,
        min_amplitude=min_amp,
        max_amplitude=max_amp,
        fixed_amplitude=min_amp,
        pulse_ms=pulse_ms,
        max_pulse_ms=max_pulse_ms,
        duration_tolerance_ms=1.0,
        recovery_ms=1.0,
        response_ms=1.0,
        neuron_id=0,
        threshold_mv=-20.0,
        tolerance=0.01,
        max_iterations=30,
        stim_dt=0.001,
        n_neurons=1,
        repeats=repeats,
    )


def test_repeats_decide_each_probe_by_majority():
    """Noisy hardware: one trial in three disagrees, the majority wins."""
    host = Host()
    backend = RheobaseBackend(host)
    logs = []
    backend.on_log = logs.append
    backend.begin_search(
        _config("Fixed duration, varying amplitude", 0.0, 10.0, 5.0, 5.0, repeats=3)
    )
    trial = 0
    while backend.running:
        above = backend.pending_amplitude >= 5.0
        # Above threshold one trial in three fails; below, one in three fires.
        flipped = trial % 3 == 1
        backend.record_probe_result(int(above != flipped) * 2)
        trial += 1
    assert abs(backend.rheobase - 5.0) <= 0.01
    assert len(host.requests) == 3 * len(backend.samples)
    assert any("2/3 trials spiked" in line for line in logs)
    assert all(count in (0, 2) for _d, _a, count in backend.samples)


def _strength_duration(max_amp):
    """thr(d) = 0.4 (1 + 20 / d): rheobase 0.4, chronaxie 20 ms."""
    host = Host()
    backend = RheobaseBackend(host)
    finished = []
    backend.on_finished = finished.append
    backend.on_summarize = lambda: "summary"
    backend.begin_search(_config("Amplitude and duration", 0.05, max_amp, 10.0, 50.0))
    while backend.running:
        threshold = 0.4 * (1 + 20.0 / backend.current_duration_ms)
        backend.record_probe_result(int(backend.pending_amplitude >= threshold))
    return backend, finished


def test_unreached_short_duration_keeps_the_measured_rheobase():
    """At 10 ms even the maximum (1.0) does not fire, and 1.0 < 2 x rheobase, so
    the chronaxie cannot be located; the rheobase found at 50 ms is kept."""
    backend, finished = _strength_duration(1.0)
    assert backend.unreached_durations == [10.0]
    assert backend.thresholds and abs(backend.thresholds[0][1] - 0.56) < 0.02
    assert finished[0].startswith("summary")
    assert "raise the maximum amplitude to at least" in finished[0]


def test_unreached_short_duration_still_brackets_the_chronaxie():
    """With a maximum of at least twice the rheobase, an unreachable duration
    means its threshold is above 2 x rheobase: the search goes on."""
    backend, finished = _strength_duration(1.15)
    assert 10.0 in backend.unreached_durations
    assert finished == ["summary"]
    assert backend.chronaxie_ms is not None
    # Twice the rheobase measured at 50 ms (0.56) is reached at 11.1 ms.
    assert 10.0 < backend.chronaxie_ms < 12.5


def test_unreached_longest_duration_still_finishes():
    backend, finished = _strength_duration(0.3)
    assert backend.unreached_durations == [50.0]
    assert "Increase the maximum amplitude." in finished[0]
