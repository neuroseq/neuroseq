from __future__ import annotations

import numpy as np
import pytest

from neuroseq.importers.importer_suite2p import _resolve_suite2p_indices
from neuroseq.importers.importer_suite2p import compute_normalized_suite2p_spikes
from neuroseq.importers.importer_suite2p import compute_normalized_suite2p_traces
from neuroseq.importers.importer_suite2p import prepare_suite2p_spikes
from neuroseq.importers.importer_suite2p import prepare_suite2p_traces
from neuroseq.importers.importer_suite2p import remap_raw_cellpose_ids


def test_prepare_suite2p_traces_produces_widget_payload(suite2p_import_data_dir):
    f_data = np.load(suite2p_import_data_dir / "F.npy")
    fneu_data = np.load(suite2p_import_data_dir / "Fneu.npy")

    data = prepare_suite2p_traces(
        f_data,
        fneu_data,
        max_neurons=3,
        sampling_rate=30.0,
    )

    assert "trace" in data.name.lower()
    assert data.traces.shape == (3, 900)
    assert data.dt == pytest.approx(1 / 30.0)
    expected_traces, _, _ = compute_normalized_suite2p_traces(f_data, fneu_data)
    np.testing.assert_allclose(data.traces, expected_traces[:3, :])


def test_prepare_suite2p_traces_works_without_fneu(suite2p_import_data_dir):
    f_data = np.load(suite2p_import_data_dir / "F.npy")
    data = prepare_suite2p_traces(
        f_data,
        None,
        max_neurons=2,
        sampling_rate=15.0,
    )

    assert "trace" in data.name.lower()
    assert data.traces.shape[0] == 2
    assert data.dt == pytest.approx(1 / 15.0)
    expected_traces, _, _ = compute_normalized_suite2p_traces(f_data, None)
    np.testing.assert_allclose(data.traces, expected_traces[:2, :])


def test_prepare_suite2p_traces_remaps_indices_by_cellpose_order(
    suite2p_import_data_dir,
):
    f_data = np.load(suite2p_import_data_dir / "F.npy")
    data = prepare_suite2p_traces(
        f_data,
        None,
        max_neurons=2,
        stimulus_to_cellpose_mapping={4: 0, 7: 1},
    )

    np.testing.assert_array_equal(data.indices, np.array([4, 7], dtype=np.uint32))
    np.testing.assert_array_equal(
        data.raw_cellpose_ids, np.array([0, 1], dtype=np.uint32)
    )


def test_prepare_suite2p_spikes_respects_threshold_and_limits_neurons(
    suite2p_import_data_dir,
):
    """

    :param suite2p_import_data_dir:
    :return:
    """
    spike_data = np.load(suite2p_import_data_dir / "spks.npy")
    dt = 1 / 30.0

    threshold = 0.2
    data = prepare_suite2p_spikes(
        spike_data,
        dt=dt,
        threshold=threshold,
        max_neurons=4,
    )

    assert "spike" in data.name.lower()
    assert len(data.spike_indices) == len(data.spike_times)

    time_indices = np.rint(data.spike_times / dt).astype(int)
    normalized_spikes, _, _ = compute_normalized_suite2p_spikes(spike_data)
    trimmed_data = normalized_spikes[:4, :]
    values = trimmed_data[data.spike_indices, time_indices]
    assert np.all(
        values >= threshold
    )  # detected a regression here. we now see resutls like:
    """
    >       assert np.all(values >= threshold)
    E       assert np.False_
    E        +  where np.False_ = <function all at 0x7cdef276f170>(array([0.5150124 , 0.9885168 , 0.10664748, 1.        , 0.6523537 ,\n       0.29718414, 0.1466293 , 0.26053137, 0.27846375, 0.25474724,\n       0.61752146, 0.30598238, 0.8945163 , 0.6416693 , 0.7615432 ,\n       0.89310557, 0.63276815, 0.9714045 , 0.46945933, 0.9162624 ,\n       0.550443  , 0.37802076, 0.17140116, 0.86899847, 0.33517653,\n       0.8196693 , 0.9928306 , 0.11396623, 0.8070358 , 0.11437435,\n       0.22229856, 0.85346013, 0.5908078 , 0.96274316, 0.89447355,\n       0.15064798, 0.45507854, 0.5350444 , 0.37590244, 0.8332498 ,\n       0.8441347 , 0.27324748, 0.08113655, 0.12833838, 0.4851915 ,\n       0.5017272 , 0.41835028, 0.7761316 , 0.46966183, 0.73085123,\n       0.42288566, 0.44247127, 0.10338625, 0.75927824, 0.85167545,\n       0.3469438 , 0.16358255, 0.9367483 , 0.41809708, 0.9016271 ,\n       0.9760032 , 0.11841686, 0.8308901 , 0.78086966, 0.596526  ,\n       1.        , 0.4547433 , 0.79808   , 0.7351832 , 0.39820093,\n       0.64166254, 0.78691816, 0.23727685, 0.8046875 , 0.67822224,\n       0.2657009 , 0.49587592, 0.968925  , 0.9777021 , 0.20871072,\n       0.27458408, 0.8248748 , 0.46612424, 0.14335115, 0.8712204 ,\n       0.14624012, 1.        , 0.26103976, 0.6518295 , 0.39677134,\n       0.13394874, 0.42202446, 0.19508348, 0.77223694, 0.9870696 ,\n       0.7319426 , 0.7622101 , 0.8651438 , 0.99697423, 0.18572904,\n       0.07031433, 0.21974424, 0.98191375, 0.20109744, 0.19966218,\n       0.7262079 , 0.10114086, 0.28988844, 0.34453458, 0.8660409 ,\n       0.9977638 , 0.5780134 , 0.43784592, 0.46107855, 0.27956823,\n       0.27754351, 0.47518986, 0.20581415, 0.17566703, 0.19999686,\n       0.7853043 , 0.5963357 , 0.9012837 , 0.98581576, 0.1916846 ,\n       0.6310945 , 1.        , 0.9486884 , 0.8517052 , 0.90641105,\n       0.17229426, 0.8523556 , 0.22446194, 0.36487725, 0.6750837 ,\n       0.20915009, 0.5802005 , 0.05367818, 0.9929934 , 0.98066473,\n       0.4278413 , 0.3707346 , 0.10823226, 0.48071438, 0.7209617 ,\n       0.8351283 , 0.74016964], dtype=float32) >= 0.2)

    """
    # We need to disentangle the different thresholding functions in the codebase

    high_threshold_data = prepare_suite2p_spikes(
        spike_data,
        dt=dt,
        threshold=1.1,
        max_neurons=4,
    )
    assert len(high_threshold_data.spike_indices) == 0


def test_resolve_suite2p_indices_uses_identity_for_missing_ids() -> None:
    indices = _resolve_suite2p_indices(
        4,
        {
            7: 1,
            4: 0,
        },
    )
    np.testing.assert_array_equal(indices, np.array([4, 7, 2, 3], dtype=np.uint32))


def test_resolve_suite2p_indices_gives_unmapped_ids_non_colliding_stim_ids() -> None:
    """Cellpose 0 -> stim 1; unmapped Cellpose 1 cannot keep stim 1, so it takes
    the stim ID freed by the mapping (0) instead of raising."""
    indices = _resolve_suite2p_indices(2, {1: 0})
    np.testing.assert_array_equal(indices, np.array([1, 0], dtype=np.uint32))


def test_resolve_suite2p_indices_accepts_sparse_custom_mapping() -> None:
    indices = _resolve_suite2p_indices(20, {1: 12, 2: 3})
    assert len(set(indices.tolist())) == 20
    assert indices[12] == 1
    assert indices[3] == 2
    # Unmapped cells never claim a stimulated neuron's ID.
    unmapped = [i for i in range(20) if i not in (12, 3)]
    assert not {1, 2} & set(indices[unmapped].tolist())


def test_remap_raw_cellpose_ids_handles_cellpose_id_gaps() -> None:
    """Linear mapping over Cellpose IDs [0, 1, 3] (ID 2 absent)."""
    raw_ids = np.array([0, 1, 3], dtype=np.uint32)
    result = remap_raw_cellpose_ids(raw_ids, {0: 0, 1: 1, 2: 3})
    np.testing.assert_array_equal(result, np.array([0, 1, 2], dtype=np.uint32))


def test_remap_raw_cellpose_ids_is_consistent_for_subsets() -> None:
    mapping = {1: 12, 2: 3}
    full = remap_raw_cellpose_ids(np.arange(20, dtype=np.uint32), mapping)
    subset = np.array([1, 2, 5], dtype=np.uint32)
    np.testing.assert_array_equal(remap_raw_cellpose_ids(subset, mapping), full[subset])


def test_resolve_suite2p_indices_rejects_duplicate_cellpose_ids() -> None:
    with pytest.raises(ValueError, match="duplicate Cellpose IDs"):
        _resolve_suite2p_indices(2, {0: 0, 1: 0})


def test_prepare_suite2p_spikes_remaps_indices_by_cellpose_order() -> None:
    spike_data = np.zeros((2, 4), dtype=np.float32)
    spike_data[0, 1] = 1.0
    spike_data[1, 2] = 1.0

    result = prepare_suite2p_spikes(
        spike_data,
        dt=1.0,
        threshold=0.5,
        max_neurons=2,
        stimulus_to_cellpose_mapping={4: 0, 7: 1},
    )

    np.testing.assert_array_equal(
        result.spike_indices, np.array([4, 7], dtype=np.uint32)
    )
    np.testing.assert_array_equal(
        result.raw_cellpose_ids, np.array([0, 1], dtype=np.uint32)
    )


def test_remap_raw_cellpose_ids_is_identity_without_mapping() -> None:
    raw_ids = np.array([0, 2, 1], dtype=np.uint32)
    np.testing.assert_array_equal(remap_raw_cellpose_ids(raw_ids, None), raw_ids)
    np.testing.assert_array_equal(remap_raw_cellpose_ids(raw_ids, {}), raw_ids)


def test_remap_raw_cellpose_ids_handles_empty_array() -> None:
    raw_ids = np.array([], dtype=np.uint32)
    result = remap_raw_cellpose_ids(raw_ids, {4: 0})
    assert result.size == 0
    assert result.dtype == np.uint32


def test_remap_raw_cellpose_ids_applies_mapping() -> None:
    raw_ids = np.array([0, 1, 0, 1], dtype=np.uint32)
    result = remap_raw_cellpose_ids(raw_ids, {4: 0, 7: 1})
    np.testing.assert_array_equal(result, np.array([4, 7, 4, 7], dtype=np.uint32))


def test_remap_raw_cellpose_ids_avoids_colliding_fallback() -> None:
    raw_ids = np.array([0, 1], dtype=np.uint32)
    result = remap_raw_cellpose_ids(raw_ids, {1: 0})
    np.testing.assert_array_equal(result, np.array([1, 0], dtype=np.uint32))


def test_remap_raw_cellpose_ids_matches_a_fresh_import_under_new_mapping(
    suite2p_import_data_dir,
) -> None:
    """Reproduces the real use case: rather than re-importing from disk under a
    new mapping, remap_raw_cellpose_ids on the already-imported raw ids should
    produce exactly what a fresh import under that new mapping would."""
    f_data = np.load(suite2p_import_data_dir / "F.npy")
    imported_under_mapping_a = prepare_suite2p_traces(
        f_data,
        None,
        max_neurons=3,
        stimulus_to_cellpose_mapping={4: 0, 7: 1},
    )

    new_mapping = {9: 0, 2: 1, 5: 2}
    assert imported_under_mapping_a.raw_cellpose_ids is not None
    remapped = remap_raw_cellpose_ids(
        imported_under_mapping_a.raw_cellpose_ids, new_mapping
    )

    fresh_import_under_mapping_b = prepare_suite2p_traces(
        f_data,
        None,
        max_neurons=3,
        stimulus_to_cellpose_mapping=new_mapping,
    )
    np.testing.assert_array_equal(remapped, fresh_import_under_mapping_b.indices)


def test_compute_normalized_suite2p_traces_uses_neuropil_coefficient() -> None:
    f_data = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    fneu_data = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    _, mins, maxs = compute_normalized_suite2p_traces(f_data, fneu_data, 0.5)
    # f - 0.5 * fneu = [1.0, 1.5, 4.0]
    assert mins[0] == pytest.approx(1.0)
    assert maxs[0] == pytest.approx(4.0)
    _, mins_default, _ = compute_normalized_suite2p_traces(f_data, fneu_data)
    assert mins_default[0] == pytest.approx(1.0)  # 2.0 - 0.7 = 1.3 > 1.0


def test_prepare_suite2p_rows_follow_given_raw_cellpose_ids() -> None:
    f_data = np.random.default_rng(0).random((3, 5)).astype(np.float32)
    traces = prepare_suite2p_traces(
        f_data,
        None,
        max_neurons=3,
        raw_cellpose_ids=np.array([0, 2, 5]),
        stimulus_to_cellpose_mapping={1: 2},
    )
    np.testing.assert_array_equal(traces.raw_cellpose_ids, [0, 2, 5])
    assert traces.indices[1] == 1  # Cellpose 2 -> stimulus 1

    spike_data = np.zeros((3, 4), dtype=np.float32)
    spike_data[2, 1] = 1.0
    spikes = prepare_suite2p_spikes(
        spike_data,
        dt=1.0,
        threshold=0.5,
        max_neurons=3,
        raw_cellpose_ids=np.array([0, 2, 5]),
    )
    np.testing.assert_array_equal(spikes.raw_cellpose_ids, [5])
    np.testing.assert_array_equal(spikes.spike_indices, [5])
    with pytest.raises(ValueError, match="raw_cellpose_ids"):
        prepare_suite2p_traces(
            f_data, None, max_neurons=3, raw_cellpose_ids=np.array([0])
        )
