import pytest
import pathlib
import shutil
import joblib
import numpy as np
import astropy.units as u
import astropy.table
import aiapy.calibrate.utils
import named_arrays as na
import sdo

_time_start = "2021-09-23T06:00:00"
_time_stop = "2021-09-23T06:00:12"

_wavelength = 304 * u.AA

_urls = sdo.aia.urls(
    time_start=_time_start,
    time_stop=_time_stop,
    wavelength=_wavelength,
)

_files = sdo.aia.download(_urls)


@pytest.mark.parametrize(
    argnames="time_start",
    argvalues=[
        _time_start,
    ],
)
@pytest.mark.parametrize(
    argnames="time_stop",
    argvalues=[
        _time_stop,
    ],
)
@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        _wavelength,
    ],
)
@pytest.mark.parametrize(
    argnames="cache",
    argvalues=[
        sdo.directory_default,
    ],
)
def test_urls(
    time_start: str,
    time_stop: str,
    wavelength: u.Quantity | na.ScalarArray,
    cache: None | str | joblib.Memory,
):
    result = sdo.aia.urls(
        time_start=time_start,
        time_stop=time_stop,
        wavelength=wavelength,
        cache=cache,
    )

    for i in result.ndindex():
        assert isinstance(result[i].ndarray, str)


@pytest.mark.parametrize(
    argnames="urls",
    argvalues=[
        _urls,
    ],
)
@pytest.mark.parametrize(
    argnames="cache",
    argvalues=[
        sdo.directory_default,
    ],
)
def test_download(
    urls: na.ScalarArray,
    cache: None | str | joblib.Memory,
):
    result = sdo.aia.download(urls, cache=cache)

    for i in result.ndindex():
        path = pathlib.Path(result[i].ndarray)
        assert path.is_file()


@pytest.mark.parametrize(
    argnames="files",
    argvalues=[
        _files,
    ],
)
@pytest.mark.parametrize(
    argnames="register",
    argvalues=[
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    argnames="cache",
    argvalues=[
        sdo.directory_default,
    ],
)
def test_prep(
    files: na.ScalarArray,
    register: bool,
    cache: None | str | joblib.Memory,
):
    result = sdo.aia.prep(
        files=files,
        register=register,
        cache=cache,
    )

    for i in result.ndindex():
        path = pathlib.Path(result[i].ndarray)
        assert path.is_file()


def test_prep_pointing_table(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The pointing table, which aiapy downloads again on every call, is fetched
    once for all the images to prepare, and not at all once they are prepared.
    """
    original = aiapy.calibrate.utils.get_pointing_table
    calls = []

    def get_pointing_table(**kwargs) -> astropy.table.QTable:
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(aiapy.calibrate.utils, "get_pointing_table", get_pointing_table)
    sdo.aia._data._pointing_table.cache_clear()

    file = pathlib.Path(_files.ndarray.item(0))
    copies = []
    for name in ["a", "b"]:
        (tmp_path / name).mkdir()
        copies.append(shutil.copy(file, tmp_path / name / file.name))
    files = na.ScalarArray(np.array(copies, dtype=object), axes="time")

    try:
        sdo.aia._data._prep(files)
        assert len(calls) == 1
        sdo.aia._data._pointing_table.cache_clear()
        sdo.aia._data._prep(files)
        assert len(calls) == 1
    finally:
        sdo.aia._data._pointing_table.cache_clear()
