from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STATUS_VALUES = ("accepted", "review", "rejected")
STATUS_COLUMNS = ("final_status", "status", "filter_status")
CLIP_COLUMNS = ("clip_score", "image_text_similarity", "similarity_score")
SAMPLE_ID_COLUMNS = ("sample_id", "image_id")


def _connect():
    try:
        import duckdb
    except Exception as exc:  # pragma: no cover - exercised only when dependency is missing
        raise RuntimeError("DuckDB is required. Install with `pip install duckdb>=0.10.0`.") from exc
    return duckdb.connect(database=":memory:")


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _schema_columns(connection, parquet_path: Path) -> set[str]:
    result = connection.execute("SELECT * FROM read_parquet(?) LIMIT 0", [str(parquet_path)])
    return {column[0] for column in result.description}


def _first_existing(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _round_or_none(value: object, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _scalar(connection, sql: str, params: list[Any]) -> Any:
    return connection.execute(sql, params).fetchone()[0]


def _status_counts(connection, parquet_path: Path, status_column: str | None, total: int) -> tuple[dict[str, int], dict[str, float]]:
    counts = {status: 0 for status in STATUS_VALUES}
    if status_column is not None:
        rows = connection.execute(
            f"""
            SELECT CAST({_quote(status_column)} AS VARCHAR) AS status, COUNT(*) AS samples
            FROM read_parquet(?)
            WHERE {_quote(status_column)} IS NOT NULL
            GROUP BY 1
            """,
            [str(parquet_path)],
        ).fetchall()
        for status, samples in rows:
            if status in counts:
                counts[status] = int(samples)
    ratios = {status: round(counts[status] / total, 4) if total else 0.0 for status in STATUS_VALUES}
    return counts, ratios


def _filter_reason_topn(connection, parquet_path: Path, columns: set[str], limit: int) -> list[dict[str, Any]]:
    if "filter_reason" not in columns:
        return []
    rows = connection.execute(
        """
        SELECT reason AS filter_reason, COUNT(*) AS samples
        FROM (
            SELECT trim(unnest(string_split(COALESCE(CAST(filter_reason AS VARCHAR), ''), ';'))) AS reason
            FROM read_parquet(?)
        )
        WHERE reason <> ''
        GROUP BY 1
        ORDER BY samples DESC, filter_reason ASC
        LIMIT ?
        """,
        [str(parquet_path), limit],
    ).fetchall()
    return [{"filter_reason": reason, "samples": int(samples)} for reason, samples in rows]


def _distribution(connection, parquet_path: Path, column: str | None) -> dict[str, int]:
    if column is None:
        return {}
    rows = connection.execute(
        f"""
        SELECT CAST({_quote(column)} AS VARCHAR) AS value, COUNT(*) AS samples
        FROM read_parquet(?)
        WHERE {_quote(column)} IS NOT NULL AND CAST({_quote(column)} AS VARCHAR) <> ''
        GROUP BY 1
        ORDER BY samples DESC, value ASC
        """,
        [str(parquet_path)],
    ).fetchall()
    return {str(value): int(samples) for value, samples in rows}


def _clip_buckets(connection, parquet_path: Path, clip_column: str | None) -> list[dict[str, Any]]:
    if clip_column is None:
        return []
    rows = connection.execute(
        f"""
        SELECT
            CASE
                WHEN {_quote(clip_column)} < 0.2 THEN '[0.0,0.2)'
                WHEN {_quote(clip_column)} < 0.4 THEN '[0.2,0.4)'
                WHEN {_quote(clip_column)} < 0.6 THEN '[0.4,0.6)'
                WHEN {_quote(clip_column)} < 0.8 THEN '[0.6,0.8)'
                ELSE '[0.8,1.0]'
            END AS bucket,
            COUNT(*) AS samples
        FROM read_parquet(?)
        WHERE {_quote(clip_column)} IS NOT NULL
        GROUP BY 1
        ORDER BY bucket
        """,
        [str(parquet_path)],
    ).fetchall()
    return [{"bucket": bucket, "samples": int(samples)} for bucket, samples in rows]


def _duplicate_count(connection, parquet_path: Path, columns: set[str]) -> int:
    if "is_duplicate_image" in columns:
        return int(
            _scalar(
                connection,
                "SELECT COUNT(*) FROM read_parquet(?) WHERE COALESCE(CAST(is_duplicate_image AS BOOLEAN), false)",
                [str(parquet_path)],
            )
        )
    if "duplicate_group_size" in columns:
        return int(
            _scalar(
                connection,
                "SELECT COUNT(*) FROM read_parquet(?) WHERE COALESCE(CAST(duplicate_group_size AS DOUBLE), 1) > 1",
                [str(parquet_path)],
            )
        )
    return 0


def _avg_caption_length(connection, parquet_path: Path, columns: set[str]) -> float | None:
    if "caption_word_count" in columns:
        return _round_or_none(
            _scalar(
                connection,
                "SELECT AVG(CAST(caption_word_count AS DOUBLE)) FROM read_parquet(?)",
                [str(parquet_path)],
            )
        )
    if "caption" not in columns:
        return None
    return _round_or_none(
        _scalar(
            connection,
            """
            SELECT AVG(
                CASE
                    WHEN caption IS NULL OR trim(CAST(caption AS VARCHAR)) = '' THEN 0
                    ELSE array_length(regexp_split_to_array(trim(CAST(caption AS VARCHAR)), '\\s+'))
                END
            )
            FROM read_parquet(?)
            """,
            [str(parquet_path)],
        )
    )


def aggregate_quality_metadata(parquet_path: str | Path, filter_topn: int = 10) -> dict[str, Any]:
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Input Parquet not found: {path}")

    with _connect() as connection:
        columns = _schema_columns(connection, path)
        status_column = _first_existing(columns, STATUS_COLUMNS)
        clip_column = _first_existing(columns, CLIP_COLUMNS)
        sample_id_column = _first_existing(columns, SAMPLE_ID_COLUMNS)
        total = int(_scalar(connection, "SELECT COUNT(*) FROM read_parquet(?)", [str(path)]))
        status_counts, status_ratios = _status_counts(connection, path, status_column, total)
        duplicate_count = _duplicate_count(connection, path, columns)

        quality_expr = "final_quality_score" if "final_quality_score" in columns else None
        summary = {
            "input_path": str(path),
            "total_samples": total,
            "sample_id_column": sample_id_column,
            "status_column": status_column,
            "clip_score_column": clip_column,
            "status_counts": status_counts,
            "status_ratios": status_ratios,
            "avg_quality_score": _round_or_none(
                _scalar(connection, f"SELECT AVG({_quote(quality_expr)}) FROM read_parquet(?)", [str(path)])
                if quality_expr
                else None
            ),
            "min_quality_score": _round_or_none(
                _scalar(connection, f"SELECT MIN({_quote(quality_expr)}) FROM read_parquet(?)", [str(path)])
                if quality_expr
                else None
            ),
            "max_quality_score": _round_or_none(
                _scalar(connection, f"SELECT MAX({_quote(quality_expr)}) FROM read_parquet(?)", [str(path)])
                if quality_expr
                else None
            ),
            "avg_clip_score": _round_or_none(
                _scalar(connection, f"SELECT AVG({_quote(clip_column)}) FROM read_parquet(?)", [str(path)])
                if clip_column
                else None
            ),
            "clip_score_buckets": _clip_buckets(connection, path, clip_column),
            "duplicate_count": duplicate_count,
            "duplicate_ratio": round(duplicate_count / total, 4) if total else 0.0,
            "source_distribution": _distribution(connection, path, "source" if "source" in columns else None),
            "split_distribution": _distribution(connection, path, "split" if "split" in columns else None),
            "avg_caption_length": _avg_caption_length(connection, path, columns),
            "filter_reason_topn": _filter_reason_topn(connection, path, columns, filter_topn),
        }
    return summary


def write_quality_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        import pandas as pd

        pd.DataFrame([summary]).to_parquet(path, index=False)
    else:
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate multimodal quality metadata with DuckDB.")
    parser.add_argument("--input", required=True, help="Input quality metadata Parquet path.")
    parser.add_argument("--output", required=True, help="Output JSON or Parquet summary path.")
    parser.add_argument("--filter-topn", type=int, default=10, help="Number of filter_reason rows to keep.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = aggregate_quality_metadata(args.input, filter_topn=args.filter_topn)
    output = write_quality_summary(summary, args.output)
    print(f"Wrote quality summary: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
