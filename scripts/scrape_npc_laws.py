from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests
from tqdm import tqdm

BASE_URL = "https://flk.npc.gov.cn"
DATA_SOURCE = "国家法律法规数据库"
DEFAULT_CATEGORIES = {
    120: "民法商法",
    150: "社会法",
}

STATUS_LABELS = {
    1: "已废止",
    2: "已修改",
    3: "有效",
    4: "尚未生效",
}


class NPCClient:
    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: int = 30,
        trust_env_proxy: bool = False,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.session = requests.Session()
        self.session.trust_env = trust_env_proxy
        self.session.headers.update(
            {
                "Accept": "application/json, text/plain, */*",
                "Content-Type": "application/json;charset=utf-8",
                "Origin": self.base_url,
                "Referer": f"{self.base_url}/search",
                "User-Agent": "LegalEagle-AI/0.1 (+local research; contact project owner)",
            }
        )

    def request(self, method: str, path_or_url: str, **kwargs: Any) -> Any:
        url = path_or_url if path_or_url.startswith("http") else f"{self.base_url}{path_or_url}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(method, url, timeout=self.timeout, **kwargs)
                response.raise_for_status()
                try:
                    return response.json()
                except ValueError:
                    if is_waf_response(response):
                        raise RuntimeError(
                            "国家法律法规数据库返回了访问验证页面。请降低抓取频率，稍后使用 "
                            "同一命令断点续跑；已下载文件不会重复下载。"
                        )
                    return response
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(self.retry_backoff * (attempt + 1))
        raise RuntimeError(f"Request failed after retries: {method} {url}") from last_error

    def search_laws(self, category_code: int, page_num: int, page_size: int) -> Dict[str, Any]:
        payload = {
            "searchRange": 1,
            "sxrq": [],
            "gbrq": [],
            "searchType": 2,
            "sxx": [],
            "gbrqYear": [],
            "flfgCodeId": [category_code],
            "zdjgCodeId": [],
            "searchContent": "",
            "pageNum": page_num,
            "pageSize": page_size,
        }
        return self.request("POST", "/law-search/search/list", json=payload)

    def law_detail(self, bbbs: str) -> Dict[str, Any]:
        return self.request("GET", "/law-search/search/flfgDetails", params={"bbbs": bbbs})

    def download_url(self, bbbs: str, fmt: str = "docx", file_id: str = "") -> str:
        result = self.request(
            "GET",
            "/law-search/download/pc",
            params={"format": fmt, "bbbs": bbbs, "fileId": file_id},
        )
        if not isinstance(result, dict) or result.get("code") != 200:
            raise RuntimeError(f"Download URL failed for {bbbs}: {response_preview(result)}")
        return result["data"]["url"]

    def download_file(self, url: str, output_path: Path) -> None:
        response = self.session.get(url, stream=True, timeout=self.timeout)
        response.raise_for_status()
        if is_waf_response(response):
            raise RuntimeError(
                "下载正文时触发访问验证。请稍后使用同一命令断点续跑，并适当增大 --sleep。"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    file.write(chunk)


def fetch_laws(client: NPCClient, category_code: int, page_size: int) -> List[Dict[str, Any]]:
    first = client.search_laws(category_code, 1, page_size)
    total = int(first.get("total", 0))
    rows = list(first.get("rows", []))

    pages = (total + page_size - 1) // page_size
    for page_num in range(2, pages + 1):
        result = client.search_laws(category_code, page_num, page_size)
        rows.extend(result.get("rows", []))

    if len(rows) != total:
        raise RuntimeError(f"Category {category_code} expected {total} rows, got {len(rows)}")
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def response_preview(result: Any) -> str:
    if isinstance(result, requests.Response):
        return result.text[:500]
    return str(result)[:500]


def is_waf_response(response: requests.Response) -> bool:
    content_type = response.headers.get("content-type", "").lower()
    if "text/html" not in content_type:
        return False
    preview = response.text[:2000]
    return "WZWS" in preview or "wzwschallenge" in preview or "WZWSREL" in preview


def manifest_item_from_record(record_path: Path, files_dir: Path) -> Dict[str, Any]:
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    detail = payload.get("detail", {})
    list_record = payload.get("list_record", {})
    bbbs = detail.get("bbbs") or list_record.get("bbbs") or record_path.stem
    category_code = payload.get("category_code")
    category_name = payload.get("category_name")
    return {
        "bbbs": bbbs,
        "title": detail.get("title") or list_record.get("title"),
        "category_code": category_code,
        "category_name": category_name,
        "status": STATUS_LABELS.get(
            detail.get("sxx") or list_record.get("sxx"),
            detail.get("sxx") or list_record.get("sxx"),
        ),
        "record_path": str(record_path),
        "docx_path": str(files_dir / f"{bbbs}.docx"),
    }


def load_or_init_manifest(output_dir: Path, records_dir: Path, files_dir: Path) -> Dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    return {
        "source": DATA_SOURCE,
        "source_url": BASE_URL,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "categories": [],
        "records": [],
    }


def upsert_manifest_record(manifest: Dict[str, Any], item: Dict[str, Any]) -> None:
    records = manifest.setdefault("records", [])
    for index, existing in enumerate(records):
        if existing.get("bbbs") == item.get("bbbs"):
            records[index] = item
            break
    else:
        records.append(item)


def set_manifest_categories(manifest: Dict[str, Any], source_totals: Dict[int, int]) -> None:
    records = manifest.get("records", [])
    categories: List[Dict[str, Any]] = []
    for code, name in DEFAULT_CATEGORIES.items():
        count = sum(1 for record in records if int(record.get("category_code", 0)) == code)
        categories.append(
            {
                "code": code,
                "name": name,
                "count": count,
                "source_total": source_totals.get(code),
            }
        )
    manifest["categories"] = categories


def validate_complete_manifest(
    manifest: Dict[str, Any],
    records_dir: Path,
    files_dir: Path,
    expected_bbbss: Dict[int, set[str]],
) -> None:
    records = manifest.get("records", [])
    by_bbbs = {record.get("bbbs"): record for record in records}
    if len(by_bbbs) != len(records):
        raise RuntimeError("Manifest contains duplicate records.")

    for category_code, bbbss in expected_bbbss.items():
        missing = sorted(bbbs for bbbs in bbbss if bbbs not in by_bbbs)
        if missing:
            raise RuntimeError(f"Category {category_code} missing manifest records: {missing}")

    for record in records:
        bbbs = record.get("bbbs")
        if not bbbs:
            raise RuntimeError(f"Manifest record missing bbbs: {record}")
        record_path = records_dir / f"{bbbs}.json"
        docx_path = files_dir / f"{bbbs}.docx"
        if not record_path.exists():
            raise FileNotFoundError(f"Missing record JSON: {record_path}")
        if not docx_path.exists():
            raise FileNotFoundError(f"Missing DOCX: {docx_path}")

    for category in manifest.get("categories", []):
        source_total = category.get("source_total")
        if source_total is not None and category.get("count") != source_total:
            raise RuntimeError(
                f"Category {category.get('code')} incomplete: "
                f"{category.get('count')}/{source_total}"
            )


def scrape(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    records_dir = output_dir / "records"
    files_dir = output_dir / "files"
    client = NPCClient(
        timeout=args.timeout,
        trust_env_proxy=args.trust_env_proxy,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
    )

    manifest = load_or_init_manifest(output_dir, records_dir, files_dir)
    manifest["scraped_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)

    source_totals: Dict[int, int] = {}
    expected_bbbss: Dict[int, set[str]] = {}
    skip_existing = True
    for category_code, category_name in DEFAULT_CATEGORIES.items():
        rows = fetch_laws(client, category_code, args.page_size)
        source_totals[category_code] = len(rows)
        expected_bbbss[category_code] = {row["bbbs"] for row in rows}
        set_manifest_categories(manifest, source_totals)
        write_json(manifest_path, manifest)

        for row in tqdm(rows, desc=f"{category_name}({category_code})"):
            bbbs = row["bbbs"]
            record_path = records_dir / f"{bbbs}.json"
            docx_path = files_dir / f"{bbbs}.docx"

            if not (record_path.exists() and skip_existing):
                detail_result = client.law_detail(bbbs)
                if not isinstance(detail_result, dict) or detail_result.get("code") != 200:
                    raise RuntimeError(f"Detail failed for {bbbs}: {response_preview(detail_result)}")
                detail_payload = {
                    "source": DATA_SOURCE,
                    "source_url": BASE_URL,
                    "detail_url": f"{BASE_URL}/detail?id={bbbs}",
                    "category_code": category_code,
                    "category_name": category_name,
                    "list_record": row,
                    "detail": detail_result["data"],
                }
                write_json(record_path, detail_payload)

            if not docx_path.exists():
                url = client.download_url(bbbs, "docx")
                client.download_file(url, docx_path)
                time.sleep(args.sleep)

            item = manifest_item_from_record(record_path, files_dir)
            upsert_manifest_record(manifest, item)
            set_manifest_categories(manifest, source_totals)
            write_json(manifest_path, manifest)
            time.sleep(args.sleep)

    validate_complete_manifest(manifest, records_dir, files_dir, expected_bbbss)
    write_json(manifest_path, manifest)
    print(f"Wrote manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape NPC civil/commercial and social law DOCX files.")
    parser.add_argument("--output-dir", default="data/raw/npc")
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.4)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--trust-env-proxy", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    scrape(parse_args())
