from __future__ import annotations

import urllib.request
from pathlib import Path


PAPERS = {
    "attention_is_all_you_need_v7.pdf": "https://arxiv.org/pdf/1706.03762v7",
    "moshi_v1.pdf": "https://arxiv.org/pdf/2410.00037v1",
    "dynamic_chunking_v2.pdf": "https://arxiv.org/pdf/2507.07955v2",
}

ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = ROOT / "papers"


def main() -> None:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in PAPERS.items():
        dest = PAPERS_DIR / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"Exists, skipping: {dest}")
            continue
        print(f"Downloading {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "TheRelayAssignment/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response, dest.open("wb") as out:
            out.write(response.read())
        print(f"Saved {dest} ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
