# -*- coding: utf-8 -*-
"""检查主源码中是否残留已移除的旧路径标记。"""
from __future__ import annotations

from base64 import b64decode
from pathlib import Path


ROOT = Path(__file__).resolve().parent


_TOKEN_B64 = [
    "S19TVEFSX1JFUFJFU0VOVEFUSU9OX01PREU=",
    "a19zdGFyX3JlcHJlc2VudGF0aW9uX21vZGU=",
    "YXJtX3Jlc29sdmVk",
    "Y29tcGxleF9wcmltaXRpdmU=",
    "c3VwZXJjZWxsX2lmX2NvbW1lbnN1cmF0ZQ==",
    "YWxsb3dfaW5jb21tZW5zdXJhdGVfY29tcGxleF9tb2Rlcw==",
    "aXJyZXBfbGFiZWw=",
    "SXJyZXBMYWJlbA==",
    "cmVzb2x2ZV9pcnJlcF9sYWJlbA==",
    "bWFrZV91bmlxdWVfaXJyZXBfa2V5",
    "aXJyZXBfbmFtaW5n",
    "c3RhbmRhcmRfaXJyZXBfZGF0YWJhc2U=",
    "bnVtZXJpY19jaGFyYWN0ZXJfZmFsbGJhY2s=",
    "TEdOVU1JUg==",
    "U1RBUl9OVU1JUg==",
    "TlVNSVI=",
    "X2dldF9rcGF0aF9weW1hdGdlbg==",
    "X2dldF9rcGF0aF9hc2U=",
    "aW5jbHVkZV9rX3N0YXI=",
    "aGlnaF9zeW1tZXRyeV9tb2RlX291dHB1dA==",
    "Y29tcGxleF9rcG9pbnRfYmFzaXM=",
    "ayBAIFIuVA==",
    "cm93OiBrIEAgUi5U",
    "X2RpYWdvbmFsX3RyYW5zbGF0aW9ucw==",
    "T25seSBkaWFnb25hbA==",
    "aXJyZXBfbW9kZXM=",
    "bl9pcnJlcHM=",
    "QUxMT1dfUkFUSU9OQUxfQVBQUk9YSU1BVElPTg==",
    "RVhQTElDSVRfS1BPSU5UX1NUUklDVA==",
    "QUxMT1dfRU1QVFlfQ09NQk9fU0NBTg==",
    "QUxMT1dfRU1QVFlfSElHSF9LX01PREVT",
    "QUxMT1dfU0tJUF9GQUlMRURfQ0FORElEQVRFUw==",
    "YWxsb3dfcmF0aW9uYWxfYXBwcm94aW1hdGlvbg==",
    "ZXhwbGljaXRfa3BvaW50X3N0cmljdA==",
    "YWxsb3dfZW1wdHlfY29tYm9fc2Nhbg==",
    "YWxsb3dfZW1wdHlfaGlnaF9rX21vZGVz",
    "YWxsb3dfc2tpcF9mYWlsZWRfY2FuZGlkYXRlcw==",
    "cGF0aF9saW5lcw==",
    "aGlnaF9zeW1tZXRyeV9rcGF0aF9jb252ZW50aW9u",
    "SElHSF9TWU1NRVRSWV9LUEFUSF9DT05WRU5USU9O",
    "aGlnaF9zeW1tZXRyeV9zdXBlcmNlbGxfcG9saWN5",
    "SElHSF9TWU1NRVRSWV9TVVBFUkNFTExfUE9MSUNZ",
    "UkVRVUlSRURfU1VQRVJDRUxMX1BPTElDWQ==",
    "aGlnaF9zeW1tZXRyeV9zdXBlcmNlbGxfbWF0cml4",
    "SElHSF9TWU1NRVRSWV9TVVBFUkNFTExfTUFUUklY",
    "cG9saWN5X25vcm0=",
    "dXNlcl9tYXRyaXg=",
    "RU5BQkxFX0hJR0hfU1lNTUVUUllfS1BPSU5UX01PREVT",
    "ZW5hYmxlX2hpZ2hfc3ltbWV0cnlfa3BvaW50X21vZGVz",
    "LS1lbmFibGUtaGlnaC1zeW1tZXRyeS1rcG9pbnQtbW9kZXM=",
    "SU5DTFVERV9BTExfSElHSF9TWU1NRVRSWV9QT0lOVF9DT09SRFM=",
    "aW5jbHVkZV9hbGxfaGlnaF9zeW1tZXRyeV9wb2ludF9jb29yZHM=",
    "QVhJQUxfUE9JTlRfR1JPVVBTXzI3",
]

_KPATH_TOKEN_B64 = [
    "cHltYXRnZW4=",
    "YXNl",
    "YXR0ZW1wdHMgPSA=",
    "YmFja2VuZD0iYXV0byI=",
]


def _tokens() -> list[str]:
    return [b64decode(item).decode("utf-8") for item in _TOKEN_B64]


def main() -> None:
    failures: list[str] = []
    for path in sorted(ROOT.glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8")
        for token in _tokens():
            if token in text:
                failures.append(f"{path.name}: {token}")

    kpath_text = (ROOT / "kpath_backend.py").read_text(encoding="utf-8")
    for token in [b64decode(item).decode("utf-8") for item in _KPATH_TOKEN_B64]:
        if token in kpath_text:
            failures.append(f"kpath_backend.py: {token}")

    if failures:
        raise SystemExit("LEGACY_PATH_CHECK_FAIL\n" + "\n".join(failures))
    print("LEGACY_PATH_CHECK_PASS")


if __name__ == "__main__":
    main()
