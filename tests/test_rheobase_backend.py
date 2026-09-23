from neuroseq.rheobase_dialog import RheobaseBackend
from neuroseq.rheobase_dialog import RheobaseRunConfig


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
