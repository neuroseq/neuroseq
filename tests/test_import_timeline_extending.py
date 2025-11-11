from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import pytest

from neuroseq.import_dialog import Suite2pImportResult
from neuroseq.importer import prepare_suite2p_spikes
from neuroseq.importer import prepare_suite2p_traces

TRACE_KEY = "suite2p_contrasting_traces"
SPIKE_KEY = "suite2p_spikes"
SAMPLING_RATE = 30.0
EXPECTED_DT = 1.0 / SAMPLING_RATE


class MockExternalDataHandler:
    """Mock handler that simulates the GUI's external data handling without graphics."""

    def __init__(self) -> None:
        self.external_simulation_data: dict[str, dict[str, Any]] = {}
        self.max_timepoints = 0

    def handle_general_import(self, imported_data: Suite2pImportResult) -> None:
        """Simplified version of GUI's handle_general_import method."""
        replace_mode = bool(imported_data.get("_replace_existing"))
        cleared_widget_types: set[str] = set()

        for key, value in imported_data.items():
            if key.startswith("_"):
                continue

            if not isinstance(value, dict):
                continue

            widget_type = value.get("widget", "")

            if (
                replace_mode
                and widget_type in {"trace", "spike"}
                and widget_type not in cleared_widget_types
            ):
                self._clear_external_widget_data(widget_type)
                cleared_widget_types.add(widget_type)

            if widget_type == "trace":
                dataset_dt = value.get("dt", 1.0 / 30.0)
                self._concatenate_trace_data(key, value, dataset_dt)
            elif widget_type == "spike":
                dataset_dt = value.get("dt", 1.0 / 30.0)
                self._concatenate_spike_data(key, value, dataset_dt)
            else:
                self.external_simulation_data[key] = value

        self.max_timepoints = self._calculate_max_timepoints()

    def _clear_external_widget_data(self, widget_type: str) -> None:
        """Clear existing data for a widget type."""
        keys_to_remove = []
        for key, data in self.external_simulation_data.items():
            if isinstance(data, dict) and data.get("widget") == widget_type:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.external_simulation_data[key]

    def _concatenate_trace_data(
        self, key: str, payload: dict[str, Any], dt: float
    ) -> None:
        """Concatenate trace data to existing timeline."""
        new_timestamps = payload["timestamps"]
        new_traces = payload["traces"]

        if key not in self.external_simulation_data:
            self.external_simulation_data[key] = payload.copy()
            return

        existing_data = self.external_simulation_data[key]
        existing_timestamps = existing_data["timestamps"]
        existing_traces = existing_data["traces"]

        # Calculate offset for new timestamps
        if len(existing_timestamps) > 0:
            last_timestamp = existing_timestamps[-1]
            time_offset = last_timestamp + dt
            adjusted_new_timestamps = new_timestamps + time_offset
        else:
            adjusted_new_timestamps = new_timestamps

        # Concatenate data
        concatenated_timestamps = np.concatenate(
            [existing_timestamps, adjusted_new_timestamps]
        )
        concatenated_traces = np.concatenate([existing_traces, new_traces], axis=1)

        # Update stored data
        existing_data["timestamps"] = concatenated_timestamps
        existing_data["traces"] = concatenated_traces

    def _concatenate_spike_data(
        self, key: str, payload: dict[str, Any], dt: float
    ) -> None:
        """Concatenate spike data to existing timeline."""
        new_timestamps = payload["timestamps"]
        new_spikes = payload["spikes"]

        if key not in self.external_simulation_data:
            self.external_simulation_data[key] = payload.copy()
            return

        existing_data = self.external_simulation_data[key]
        existing_timestamps = existing_data["timestamps"]
        existing_spikes = existing_data["spikes"]

        # Calculate offset for new timestamps
        if len(existing_timestamps) > 0:
            last_timestamp = existing_timestamps[-1]
            time_offset = last_timestamp + dt
            adjusted_new_timestamps = new_timestamps + time_offset
        else:
            adjusted_new_timestamps = new_timestamps

        # Concatenate data
        concatenated_timestamps = np.concatenate(
            [existing_timestamps, adjusted_new_timestamps]
        )
        concatenated_spikes = np.concatenate([existing_spikes, new_spikes], axis=1)

        # Update stored data
        existing_data["timestamps"] = concatenated_timestamps
        existing_data["spikes"] = concatenated_spikes

    def _calculate_max_timepoints(self) -> int:
        """Calculate maximum timepoints across all trace datasets."""
        max_timepoints = 0
        for data in self.external_simulation_data.values():
            if isinstance(data, dict) and data.get("widget") == "trace":
                timestamps = data.get("timestamps")
                if isinstance(timestamps, np.ndarray):
                    max_timepoints = max(max_timepoints, len(timestamps))
        return max_timepoints

    def get_trace_dataset(self, key: str) -> dict[str, Any] | None:
        """Get trace dataset by key."""
        return self.external_simulation_data.get(key)

    def get_max_timepoints(self) -> int:
        """Get maximum timepoints."""
        return self.max_timepoints


def _write_temp_f_file(tmp_path, n_timepoints: int, n_neurons: int = 2) -> str:
    """Create temporary F.npy file with test data."""
    file_path = tmp_path / f"F_{n_neurons}n_{n_timepoints}t_{uuid.uuid4().hex}.npy"
    data = np.linspace(0.0, 1.0, n_neurons * n_timepoints, dtype=np.float64).reshape(
        n_neurons, n_timepoints
    )
    np.save(file_path, data)
    return str(file_path)


def _create_import_data(
    tmp_path, n_timepoints: int, *, replace_existing: bool
) -> Suite2pImportResult:
    """Create import data payload from temporary files."""
    f_path = _write_temp_f_file(tmp_path, n_timepoints)
    f_data = np.load(f_path)

    trace_payload = prepare_suite2p_traces(
        f_data=f_data,
        fneu_data=None,
        max_neurons=f_data.shape[0],
        sampling_rate=SAMPLING_RATE,
    )

    return {TRACE_KEY: trace_payload, "_replace_existing": replace_existing}


def _timeline_metrics(trace_dataset: dict[str, Any]) -> tuple[int, float]:
    """Calculate timeline metrics from trace dataset."""
    timestamps = trace_dataset["timestamps"]
    assert isinstance(timestamps, np.ndarray)
    count = len(timestamps)
    dt = float(timestamps[1] - timestamps[0]) if count > 1 else EXPECTED_DT
    duration_ms = count * dt * 1000.0
    return count, duration_ms


@pytest.fixture
def handler() -> MockExternalDataHandler:
    """Provide a mock external data handler."""
    return MockExternalDataHandler()


def test_import_timeline_single_insert(
    handler: MockExternalDataHandler, tmp_path
) -> None:
    """Test single import creates correct timeline."""
    imported = _create_import_data(tmp_path, n_timepoints=500, replace_existing=True)
    handler.handle_general_import(imported)

    trace_dataset = handler.get_trace_dataset(TRACE_KEY)
    assert trace_dataset is not None

    count, duration_ms = _timeline_metrics(trace_dataset)

    assert count == 500
    assert duration_ms == pytest.approx(500 * (1000.0 / SAMPLING_RATE), rel=1e-3)
    assert handler.get_max_timepoints() == 500


def test_import_timeline_append_mode(
    handler: MockExternalDataHandler, tmp_path
) -> None:
    """Test append mode correctly extends existing timeline."""
    # First import
    first_payload = _create_import_data(
        tmp_path, n_timepoints=500, replace_existing=True
    )
    handler.handle_general_import(first_payload)

    # Second import in append mode
    append_payload = _create_import_data(
        tmp_path, n_timepoints=500, replace_existing=False
    )
    handler.handle_general_import(append_payload)

    trace_dataset = handler.get_trace_dataset(TRACE_KEY)
    assert trace_dataset is not None

    count, duration_ms = _timeline_metrics(trace_dataset)

    assert count == 1000
    assert duration_ms == pytest.approx(1000 * (1000.0 / SAMPLING_RATE), rel=1e-3)
    assert handler.get_max_timepoints() == 1000


def test_import_timeline_replace_mode(
    handler: MockExternalDataHandler, tmp_path
) -> None:
    """Test replace mode overwrites existing timeline."""
    # First import
    first_payload = _create_import_data(
        tmp_path, n_timepoints=500, replace_existing=True
    )
    handler.handle_general_import(first_payload)

    # Second import in replace mode
    replace_payload = _create_import_data(
        tmp_path, n_timepoints=300, replace_existing=True
    )
    handler.handle_general_import(replace_payload)

    trace_dataset = handler.get_trace_dataset(TRACE_KEY)
    assert trace_dataset is not None

    count, duration_ms = _timeline_metrics(trace_dataset)

    assert count == 300
    assert duration_ms == pytest.approx(300 * (1000.0 / SAMPLING_RATE), rel=1e-3)
    assert handler.get_max_timepoints() == 300


def test_timeline_concatenation_preserves_offset(
    handler: MockExternalDataHandler, tmp_path
) -> None:
    """Test that timeline concatenation maintains proper time offsets."""
    # First import - 200 timepoints
    first_payload = _create_import_data(
        tmp_path, n_timepoints=200, replace_existing=True
    )
    handler.handle_general_import(first_payload)

    # Second import - 300 timepoints in append mode
    append_payload = _create_import_data(
        tmp_path, n_timepoints=300, replace_existing=False
    )
    handler.handle_general_import(append_payload)

    trace_dataset = handler.get_trace_dataset(TRACE_KEY)
    assert trace_dataset is not None

    timestamps = trace_dataset["timestamps"]
    traces = trace_dataset["traces"]

    # Check total length
    assert len(timestamps) == 500
    assert traces.shape[1] == 500

    # Check that timestamps are continuous and increasing
    assert np.all(np.diff(timestamps) > 0)

    # Check that the second segment starts after the first
    expected_first_end_time = (200 - 1) * EXPECTED_DT
    expected_second_start_time = expected_first_end_time + EXPECTED_DT

    # The 200th timestamp (index 200) should be approximately the start of second segment
    actual_second_start_time = timestamps[200]
    assert actual_second_start_time == pytest.approx(
        expected_second_start_time, rel=1e-3
    )


def test_import_with_spikes_and_traces(
    handler: MockExternalDataHandler, tmp_path
) -> None:
    """Test importing both spikes and traces together."""
    # Create trace data
    f_path = _write_temp_f_file(tmp_path, n_timepoints=400, n_neurons=3)
    f_data = np.load(f_path)

    # Create spike data with some high values to ensure spikes are detected
    spike_data = np.zeros((3, 400))
    spike_data[0, [10, 50, 100]] = 0.1  # Spike events for neuron 0
    spike_data[1, [25, 75]] = 0.28  # Spike events for neuron 1
    spike_data[2, [200, 300]] = 0.12  # Spike events for neuron 2

    trace_payload = prepare_suite2p_traces(
        f_data=f_data,
        fneu_data=None,
        max_neurons=f_data.shape[0],
        sampling_rate=SAMPLING_RATE,
    )

    spike_payload = prepare_suite2p_spikes(
        spike_data=spike_data,
        max_neurons=spike_data.shape[0],
        dt=EXPECTED_DT,
        threshold=0.05,
    )

    import_data: Suite2pImportResult = {
        TRACE_KEY: trace_payload,
        SPIKE_KEY: spike_payload,
        "_replace_existing": True,
    }

    handler.handle_general_import(import_data)

    # Check both datasets were imported
    trace_dataset = handler.get_trace_dataset(TRACE_KEY)
    spike_dataset = handler.get_trace_dataset(SPIKE_KEY)

    assert trace_dataset is not None
    assert spike_dataset is not None

    # Check trace timeline length
    trace_count, _ = _timeline_metrics(trace_dataset)
    assert trace_count == 400

    # Check spike data structure (spikes are individual events, not timeline)
    spike_timestamps = spike_dataset["timestamps"]
    spike_neurons = spike_dataset["spikes"]
    assert isinstance(spike_timestamps, np.ndarray)
    assert isinstance(spike_neurons, np.ndarray)
    assert len(spike_timestamps) == len(spike_neurons)
    assert len(spike_timestamps) == 7  # Total spike events we created

    # Max timepoints should be based on traces (which have timeline structure)
    assert handler.get_max_timepoints() == 400
