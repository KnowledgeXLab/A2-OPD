"""Shared robust Parquet reader for G-OPD-style nested columns.

PyArrow on some parquet files raises ``Repetition level histogram size mismatch``.
fastparquet can read those files but often decodes LIST<STRUCT> ``prompt`` as null.
Prefer Polars when available—it usually preserves nested ``prompt``.
"""

from __future__ import annotations

import pandas as pd


def read_parquet_flexible(path: str, engine: str = "auto") -> pd.DataFrame:
    """Read parquet; ``engine`` is ``auto`` | ``pyarrow`` | ``polars`` | ``fastparquet``."""

    def _pandas(eng: str) -> pd.DataFrame:
        return pd.read_parquet(path, engine=eng)

    if engine == "polars":
        try:
            import polars as pl
        except ImportError:
            return read_parquet_flexible(path, "auto")
        return pl.read_parquet(path).to_pandas()

    if engine != "auto":
        return _pandas(engine)

    errors: list[tuple[str, BaseException]] = []
    try:
        return _pandas("pyarrow")
    except Exception as e:  # noqa: BLE001
        errors.append(("pyarrow", e))

    try:
        import polars as pl

        return pl.read_parquet(path).to_pandas()
    except ImportError:
        errors.append(("polars", ImportError('install with: pip install polars')))
    except Exception as e:  # noqa: BLE001
        errors.append(("polars", e))

    try:
        return _pandas("fastparquet")
    except ImportError:
        errors.append(("fastparquet", ImportError('install with: pip install fastparquet')))
    except Exception as e:  # noqa: BLE001
        errors.append(("fastparquet", e))

    err_lines = "; ".join(f"{n}: {ex!r}" for n, ex in errors)
    raise OSError(
        "读取 Parquet 失败: {!r}. 尝试过: {}\n"
        "若需 G-OPD math_and_code 且 PyArrow 报 histogram/mismatch：请 ``pip install polars`` "
        "后重跑 split（先 polars 再写出），或直接 ``--engine polars`` 读.".format(path, err_lines),
    ) from errors[0][1]
