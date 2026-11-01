#!/usr/bin/env python3
"""Download QASPER v0.3 and BEIR-SciFact into ``data/``.

QASPER lands in ``data/qasper/`` as ``qasper-{train,dev,test}-v0.3.json``.
SciFact lands in ``data/scifact/`` as ``corpus.jsonl``, ``queries.jsonl``,
and ``qrels/test.tsv`` (300 test queries over the full abstract corpus).

Examples::

    python scripts/download_data.py
    python scripts/download_data.py --dataset scifact
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

QASPER_TRAIN_DEV = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
QASPER_TEST = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"
SCIFACT_ZIP = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

_QASPER_JSON = {
    "qasper-train-v0.3.json",
    "qasper-dev-v0.3.json",
    "qasper-test-v0.3.json",
}
_SCIFACT_FILES = {
    "corpus.jsonl",
    "queries.jsonl",
    "qrels/train.tsv",
    "qrels/test.tsv",
}


def main() -> None:
    args = _parse_args()
    if args.dataset in ("qasper", "both"):
        out = DATA / "qasper"
        out.mkdir(parents=True, exist_ok=True)
        _extract_qasper(_download(QASPER_TRAIN_DEV, DATA / "_download" / "qasper-train-dev-v0.3.tgz"), out)
        _extract_qasper(_download(QASPER_TEST, DATA / "_download" / "qasper-test-v0.3.tgz"), out)
        print(f"QASPER v0.3 -> {out}")
    if args.dataset in ("scifact", "both"):
        out = DATA / "scifact"
        out.mkdir(parents=True, exist_ok=True)
        _extract_scifact(_download(SCIFACT_ZIP, DATA / "_download" / "scifact.zip"), out)
        print(f"BEIR-SciFact -> {out}")


def _download(url: str, dest: Path) -> Path:
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"Already downloaded {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "rag-hybrid/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            handle.write(block)
    partial.replace(dest)
    return dest


def _extract_qasper(archive: Path, out: Path) -> None:
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if name not in _QASPER_JSON or not member.isfile():
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            target = out / name
            target.write_bytes(extracted.read())
            print(f"  {target.relative_to(ROOT)}")


def _extract_scifact(archive: Path, out: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            relative = Path(info.filename)
            parts = relative.parts
            if not parts or parts[0] != "scifact":
                continue
            key = "/".join(parts[1:])
            if key not in _SCIFACT_FILES:
                continue
            target = out.joinpath(*parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(info))
            print(f"  {target.relative_to(ROOT)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("qasper", "scifact", "both"), default="both")
    return parser.parse_args()


if __name__ == "__main__":
    main()
