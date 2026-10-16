"""Tests for the ELISE kinetic-units archive migration.

Covers:
- Loading a pre-kinetic archive (no schema_version, stims in [0,1])
  converts the opsin stims to photon-flux density.
- A kinetic archive (schema_version=2) roundtrips cleanly: units
  string and values preserved, no double migration.
- A pre-kinetic-style archive whose values already look like flux
  (max > 2.0) is left alone defensively.
"""

from __future__ import annotations

import copy
import json
import pathlib
import zipfile

import numpy as np

from neuroseq.example_models.elise import DEFAULT_PARAMS
from neuroseq.example_projects import elise_project
from neuroseq.project import Project
from neuroseq.save_load_options import SaveLoadOptions
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType


def _save_options() -> SaveLoadOptions:
    return SaveLoadOptions(
        include_stim_arrays=True,
        include_model=True,
        include_simulation_data=False,
        include_experimental_data=False,
    )


def _build_legacy_elise_project(tfinal_ms: int = 100) -> Project:
    """A Project shaped like a pre-kinetic ELISE archive: opsin stims
    with values in [0, 1] and the legacy ``units="1"``."""
    project = Project()
    project.external_simulate_path = "neuroseq:elise"
    project.external_simulate_params = copy.deepcopy(DEFAULT_PARAMS)
    project.load_NN()
    n_pitches = 13
    exc = np.full((tfinal_ms, n_pitches), 1.0 / 15.0, dtype=np.float32)
    inh = np.full((tfinal_ms, n_pitches), 14.0 / 15.0, dtype=np.float32)
    exc[10:20, 0] = 1.0 / 3.0
    inh[10:20, 0] = 2.0 / 3.0
    project.stim_arrays["opsin_exc"] = StimArray(
        name="opsin_exc",
        stim_type=StimType.Excitation,
        values=exc,
        units="1",
        dt=0.001,
    )
    project.stim_arrays["opsin_inh"] = StimArray(
        name="opsin_inh",
        stim_type=StimType.Inhibition,
        values=inh,
        units="1",
        dt=0.001,
    )
    return project


def _save_then_drop_schema_version(project: Project, path: pathlib.Path) -> None:
    """Save the project, then strip ``schema_version`` from its
    ``project_metadata.json`` so reload triggers the legacy migration."""
    project.save(path, _save_options())
    with zipfile.ZipFile(path, "r") as zf:
        contents = {name: zf.read(name) for name in zf.namelist()}
    meta = json.loads(contents["project_metadata.json"])
    meta.pop("schema_version", None)
    contents["project_metadata.json"] = json.dumps(meta).encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in contents.items():
            zf.writestr(name, data)


def test_legacy_pre_kinetic_archive_migrates_to_photon_flux(
    tmp_path: pathlib.Path,
) -> None:
    project = _build_legacy_elise_project()
    archive = tmp_path / "legacy.nseq"
    _save_then_drop_schema_version(project, archive)

    loaded = Project(filepath=archive)
    exc = loaded.stim_arrays["opsin_exc"]
    inh = loaded.stim_arrays["opsin_inh"]

    # Photon-flux scale (≥ 1e13 mm^-2 s^-1, well above the legacy [0,1]).
    assert float(np.max(exc.values)) > 1e13
    assert float(np.max(inh.values)) > 1e13
    assert exc.units == "mm**-2 / second"
    assert inh.units == "mm**-2 / second"


def test_kinetic_archive_roundtrip_preserves_schema_and_units(
    tmp_path: pathlib.Path,
) -> None:
    project = elise_project()
    archive = tmp_path / "kinetic.nseq"
    project.save(archive, _save_options())

    with zipfile.ZipFile(archive) as zf:
        meta = json.loads(zf.read("project_metadata.json"))
    assert meta["schema_version"] == 2

    loaded = Project(filepath=archive)
    assert loaded._loaded_schema_version == 2
    original_max = float(np.max(project.stim_arrays["opsin_exc"].values))
    loaded_max = float(np.max(loaded.stim_arrays["opsin_exc"].values))
    assert abs(loaded_max - original_max) / original_max < 1e-3
    assert loaded.stim_arrays["opsin_exc"].units == "mm**-2 / second"
    assert loaded.stim_arrays["opsin_inh"].units == "mm**-2 / second"


def test_archive_without_schema_version_but_flux_values_is_left_alone(
    tmp_path: pathlib.Path,
) -> None:
    """Defensive: if the archive lacks schema_version but the opsin
    values already look like photon flux (max > 2.0), the migration
    declines to convert them."""
    project = _build_legacy_elise_project()
    project.stim_arrays["opsin_exc"].values[:] = 5e17
    project.stim_arrays["opsin_inh"].values[:] = 5e19
    archive = tmp_path / "flux_no_version.nseq"
    _save_then_drop_schema_version(project, archive)

    loaded = Project(filepath=archive)
    assert np.isclose(
        float(np.max(loaded.stim_arrays["opsin_exc"].values)), 5e17, rtol=1e-6
    )
    assert np.isclose(
        float(np.max(loaded.stim_arrays["opsin_inh"].values)), 5e19, rtol=1e-6
    )
