"""Explicit AER MT_CKD water acquisition and local classic-NetCDF reading."""

import shutil
from pathlib import Path
from typing import Literal

import numpy as np
from orsia_radiative import WaterContinuumReference
from scipy.io import netcdf_file

from ..acquisition import download_file


def download_mt_ckd_water(
    *,
    edition: Literal["4.3"],
    destination: Path,
    timeout_s: float = 30.0,
    max_bytes: int = 128 * 1024 * 1024,
    expected_sha256: str | None = None,
) -> Path:
    """Download the native AER water ABSCO reference, without parsing it.

    Only explicit edition ``"4.3"`` is supported, from the official
    ``AER-RC/MT_CKD_H2O`` release tag, never a mutable 'current' branch.
    Return the caller's destination; no temperature envelope is needed to obtain
    the file. Bounds, optional digest, atomic no-clobber publication and failure
    cleanup follow ``download_file``. Network/HTTP errors propagate. Download
    success is not format or scientific qualification. Retain AER's embedded
    copyright/licence; no archive extraction, implicit cache or conversion.

    Args:
        edition: Supported AER water-reference release, currently "4.3".
        destination: New path for the native classic-NetCDF reference file.
        timeout_s: Network-operation timeout in seconds.
        max_bytes: Maximum native file size in bytes.
        expected_sha256: Optional expected lowercase hexadecimal file digest.

    Returns:
        Destination after bounded atomic transfer and optional digest verification.

    Raises:
        ValueError: Edition or transfer bounds are invalid.
        FileExistsError: Destination already exists.
    """
    if edition != "4.3":
        raise ValueError("MT_CKD water acquisition supports only explicit edition 4.3")
    return download_file(
        f"https://raw.githubusercontent.com/AER-RC/MT_CKD_H2O/{edition}/data/absco-ref_wv-mt-ckd.nc",
        destination,
        timeout_s=timeout_s,
        max_bytes=max_bytes,
        expected_sha256=expected_sha256,
    )


def fetch_mt_ckd_water(
    *,
    edition: Literal["4.3"],
    directory: Path,
    temperature_range_k: tuple[float, float],
    timeout_s: float = 30.0,
    max_bytes: int = 128 * 1024 * 1024,
    expected_sha256: str | None = None,
) -> WaterContinuumReference:
    """Download and read AER's H2O self/foreign ABSCO, retaining native bytes.

    Create a new exclusively owned directory with ``absco-ref_wv-mt-ckd.nc``.
    Existing directories fail before network access; remove the new directory on
    transfer or decode failure (parent directories may remain). Do not modify it
    concurrently. Use ``read_mt_ckd_water`` for offline replay. No resampling or
    absorption evaluation occurs. ``temperature_range_k`` [K] declares the caller's
    evaluation envelope, not a model-accuracy guarantee. Transport parameters and
    supported editions follow ``download_mt_ckd_water``; reader errors propagate.

    Args:
        edition: Supported AER water-reference release, currently "4.3".
        directory: New directory for the native reference file.
        temperature_range_k: Inclusive caller-declared temperature support in K.
        timeout_s: Network-operation timeout in seconds.
        max_bytes: Maximum native file size in bytes.
        expected_sha256: Optional expected hexadecimal file digest.

    Returns:
        Parsed self/foreign continuum reference with fresh coefficient arrays.

    Raises:
        FileExistsError: The output directory already exists.
        ValueError: Edition, temperature envelope or file contents are invalid.
    """
    if edition != "4.3":
        raise ValueError("MT_CKD water acquisition supports only explicit edition 4.3")
    directory.mkdir(parents=True, exist_ok=False)
    try:
        path = download_mt_ckd_water(
            edition=edition,
            destination=directory / "absco-ref_wv-mt-ckd.nc",
            timeout_s=timeout_s,
            max_bytes=max_bytes,
            expected_sha256=expected_sha256,
        )
        return read_mt_ckd_water(path, temperature_range_k=temperature_range_k)
    except BaseException:
        shutil.rmtree(directory)
        raise


def read_mt_ckd_water(
    path: Path, *, temperature_range_k: tuple[float, float]
) -> WaterContinuumReference:
    """Read AER MT_CKD ABSCO classic NetCDF (CDF-1/CDF-2), with mbar -> Pa.

    Required variables: wavenumbers,self_absco_ref,for_absco_ref,self_texp,
    ref_press,ref_temp. Optional HDF5/NetCDF4 backends are NOT imported; convert
    such files explicitly outside this reader and identify the transformed
    file separately. This reader does not load other MT_CKD continua.

    Args:
        path: Local classic-NetCDF reference file to read.
        temperature_range_k: Inclusive caller-declared valid temperatures in K.

    Returns:
        Water self/foreign ABSCO coefficients and reference conditions with
        independent arrays; reference pressure is converted to pascals.

    Raises:
        ValueError: Input is not classic NetCDF or required variables are invalid.
    """
    with path.open("rb") as stream:
        if stream.read(4) not in (b"CDF\x01", b"CDF\x02"):
            raise ValueError("MT_CKD reader supports classic NetCDF only, not HDF5/NetCDF4")
    with netcdf_file(path, "r", mmap=False) as dataset:
        try:
            arrays = [
                np.array(dataset.variables[name].data, dtype=np.float64, copy=True)
                for name in ("wavenumbers", "self_absco_ref", "for_absco_ref", "self_texp")
            ]
            pressure = float(np.asarray(dataset.variables["ref_press"].data).item()) * 100
            temperature = float(np.asarray(dataset.variables["ref_temp"].data).item())
        except (KeyError, ValueError) as error:
            raise ValueError("invalid MT_CKD reference variables") from error
    return WaterContinuumReference(
        *arrays,
        pressure,
        temperature,
        temperature_range_k,
    )
