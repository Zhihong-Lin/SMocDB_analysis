from __future__ import annotations

import io
from importlib import resources
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import quote
from urllib.request import urlopen

import anndata as ad
import pandas as pd


BASE_URL = "https://smart0414.oss-cn-shanghai.aliyuncs.com"
DEFAULT_SUBDIR = "4.Pathological image"
DEFAULT_FILENAME = "Download.h5ad"
DEFAULT_TIMEOUT = 120
DEFAULT_METADATA_RESOURCE = "data/1.infro_df.xlsx"


def build_download_url(dataset_id: str) -> str:
    dataset_id = str(dataset_id).strip()
    if not dataset_id:
        raise ValueError("dataset_id must not be empty")
    return (
        f"{BASE_URL}/{quote(dataset_id)}/{quote(DEFAULT_SUBDIR)}/{quote(DEFAULT_FILENAME)}"
    )


def download_dataset(
    dataset_id: str,
    output_dir: str | Path = ".",
    overwrite: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
) -> Path:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / f"{dataset_id}.h5ad"

    if target_file.exists() and not overwrite:
        return target_file

    url = build_download_url(dataset_id)
    with urlopen(url, timeout=timeout) as response:
        content = response.read()
    target_file.write_bytes(content)
    return target_file


def load_dataset(
    dataset_id: str,
    output_dir: str | Path = ".",
    overwrite: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
):
    file_path = download_dataset(
        dataset_id=dataset_id,
        output_dir=output_dir,
        overwrite=overwrite,
        timeout=timeout,
    )
    return ad.read_h5ad(file_path)


def _read_packaged_metadata() -> pd.DataFrame:
    package_resource = resources.files("pysmocdb").joinpath(DEFAULT_METADATA_RESOURCE)
    if package_resource.exists():
        return pd.read_excel(io.BytesIO(package_resource.read_bytes()))

    source_fallback = Path(__file__).resolve().parent / "data" / "1.infro_df.xlsx"
    if source_fallback.exists():
        return pd.read_excel(source_fallback)

    raise FileNotFoundError(
        "Default metadata file is not bundled in this installation. "
        "Please provide metadata_path explicitly."
    )


def _read_metadata(metadata_path: str | Path | None = None) -> pd.DataFrame:
    if metadata_path is None:
        return _read_packaged_metadata()

    xlsx_path = Path(metadata_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {xlsx_path}")
    return pd.read_excel(xlsx_path)


def _resolve_id_column(df: pd.DataFrame, id_column: str | None = None) -> str:
    if id_column:
        if id_column not in df.columns:
            raise ValueError(f"Specified id_column does not exist: {id_column}")
        return id_column

    candidates = [
        "Data ID",
        "Dataset ID",
        "ID",
        "dataset_id",
        "data_id",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    return df.columns[0]


def list_dataset_ids(
    metadata_path: str | Path | None = None,
    id_column: str | None = None,
) -> list[str]:
    df = _read_metadata(metadata_path)
    id_col = _resolve_id_column(df, id_column)

    ids = (
        df[id_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    ids = ids[ids != ""]
    return ids.drop_duplicates().tolist()


def _normalize_filter_values(value: str | Sequence[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(v) for v in value]
    raise TypeError("Filter values must be a string, a sequence of strings, or None")


def list_available_metadata_values(
    metadata_path: str | Path | None = None,
) -> dict[str, list[str]]:
    df = _read_metadata(metadata_path)

    if "Cancer Type" not in df.columns:
        raise ValueError("Missing required metadata column: Cancer Type")
    if "Modality Type" not in df.columns:
        raise ValueError("Missing required metadata column: Modality Type")

    cancer_types = (
        df["Cancer Type"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    modality_types = (
        df["Modality Type"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    cancer_types = sorted(cancer_types[cancer_types != ""].drop_duplicates().tolist())
    modality_types = sorted(modality_types[modality_types != ""].drop_duplicates().tolist())

    return {
        "cancer_types": cancer_types,
        "modality_types": modality_types,
    }


def filter_dataset_ids(
    cancer_type: str | Sequence[str] | None = None,
    modality_type: str | Sequence[str] | None = None,
    metadata_path: str | Path | None = None,
    id_column: str | None = None,
) -> list[str]:
    df = _read_metadata(metadata_path)

    if "Cancer Type" not in df.columns:
        raise ValueError("Missing required metadata column: Cancer Type")
    if "Modality Type" not in df.columns:
        raise ValueError("Missing required metadata column: Modality Type")

    ct_values = _normalize_filter_values(cancer_type)
    mt_values = _normalize_filter_values(modality_type)

    filtered = df.copy()
    if ct_values:
        allowed = {v.lower() for v in ct_values}
        filtered = filtered[
            filtered["Cancer Type"].astype(str).str.lower().isin(allowed)
        ]
    if mt_values:
        allowed = {v.lower() for v in mt_values}
        filtered = filtered[
            filtered["Modality Type"].astype(str).str.lower().isin(allowed)
        ]

    id_col = _resolve_id_column(filtered, id_column)
    ids = (
        filtered[id_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    ids = ids[ids != ""]
    return ids.drop_duplicates().tolist()