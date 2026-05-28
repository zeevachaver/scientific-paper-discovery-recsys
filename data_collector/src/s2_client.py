from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from .io_utils import ensure_dir


class S2Client:
    """Semantic Scholar Graph API client with caching, retry, and 1-request/sec pacing."""

    def __init__(self, config: dict[str, Any]):
        load_dotenv()
        self.base_url = config["api"].get("base_url", "https://api.semanticscholar.org/graph/v1").rstrip("/")
        self.cache_dir = ensure_dir(config["project"].get("cache_dir", "raw_cache"))
        self.rate_limit_seconds = float(config["api"].get("rate_limit_seconds", 1.1))
        self.max_retries = int(config["api"].get("max_retries", 6))
        self.timeout_seconds = int(config["api"].get("timeout_seconds", 60))
        self.last_request_time = 0.0
        self.session = requests.Session()
        api_key = os.getenv("S2_API_KEY", "").strip()
        if api_key:
            self.session.headers.update({"x-api-key": api_key})
        self.session.headers.update({"User-Agent": "ecs172-relation-aware-recommender/1.0"})

    def _cache_path(self, method: str, path: str, params: dict[str, Any] | None, json_body: Any | None) -> Path:
        payload = json.dumps({"method": method, "path": path, "params": params, "json": json_body}, sort_keys=True, ensure_ascii=False)
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        safe_name = path.strip("/").replace("/", "_").replace(":", "_")[:140]
        return self.cache_dir / f"{safe_name}_{h}.json"

    def request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: Any | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any] | list[Any]:
        method = method.upper()
        params = {k: v for k, v in (params or {}).items() if v is not None and v != ""}
        cache_path = self._cache_path(method, path, params, json_body)
        if use_cache and cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries):
            self._pace()
            response = self.session.request(
                method,
                url,
                params=params,
                json=json_body,
                timeout=self.timeout_seconds,
            )
            if response.status_code == 200:
                data = response.json()
                if use_cache:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False)
                return data

            if response.status_code in {429, 500, 502, 503, 504}:
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_s = float(retry_after)
                else:
                    sleep_s = min(60.0, (2 ** attempt) + random.random())
                print(f"[WARN] {response.status_code} {path}; retrying in {sleep_s:.1f}s")
                time.sleep(sleep_s)
                continue

            # 404 can happen for title match or stale paper IDs. Return an empty marker.
            if response.status_code == 404:
                data = {"error": "404", "message": response.text}
                if use_cache:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False)
                return data

            raise RuntimeError(f"Semantic Scholar API error {response.status_code}: {response.text[:1000]}")

        raise RuntimeError(f"Semantic Scholar API failed after {self.max_retries} retries: {method} {path}")

    def _pace(self) -> None:
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - elapsed)
        self.last_request_time = time.time()

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | list[Any]:
        return self.request("GET", path, params=params)

    def post(self, path: str, params: dict[str, Any] | None = None, json_body: Any | None = None) -> dict[str, Any] | list[Any]:
        return self.request("POST", path, params=params, json_body=json_body)
