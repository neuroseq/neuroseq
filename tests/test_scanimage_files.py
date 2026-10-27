import datetime

from neuroseq.device_control.scanimage_files import next_recording_number
from neuroseq.device_control.scanimage_files import recording_file_name
from neuroseq.device_control.scanimage_files import recording_stem


def test_recordings_are_numbered_per_day_in_the_folder(tmp_path):
    stem = recording_stem(datetime.date(2026, 10, 22))
    assert stem == "2026-10-22"
    assert next_recording_number(tmp_path / "missing", stem) == 1
    assert next_recording_number(tmp_path, stem) == 1
    for name in (
        "2026-10-22_00002.tif",
        "2026-10-22_00005.json",
        "2026-10-21_00009.tif",
    ):
        (tmp_path / name).write_text("")
    assert next_recording_number(tmp_path, stem) == 6
    assert recording_file_name(stem, 6) == "2026-10-22_00006.tif"
