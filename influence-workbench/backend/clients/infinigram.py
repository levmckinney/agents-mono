"""Infini-gram API client using the search_docs endpoint."""

from __future__ import annotations

import httpx

API_PAGE_SIZE = 10  # Hard server-side limit for search_docs maxnum
DEFAULT_MAX_ATTEMPTS = 10  # Default cap on API calls to avoid infinite loops


class InfinigramClient:
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        api_url: str,
        index: str,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ):
        self._http = http_client
        self._api_url = api_url.rstrip("/")
        self._index = index
        self._max_attempts = max_attempts

    async def _post(self, payload: dict) -> dict:
        resp = await self._http.post(self._api_url, json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()

    def _parse_document(self, doc: dict) -> dict:
        """Parse a raw API document into our internal format."""
        raw_spans = doc.get("spans", [])
        doc_spans = []
        for span in raw_spans:
            if isinstance(span, list) and len(span) >= 2:
                doc_spans.append(
                    {
                        "text": span[0],
                        "is_match": span[1] is not None,
                    }
                )

        full_text = "".join(s["text"] for s in doc_spans)
        return {
            "doc_ix": doc.get("doc_ix", 0),
            "doc_len": doc.get("doc_len", 0),
            "disp_len": doc.get("disp_len", 0),
            "spans": doc_spans,
            "full_text": full_text,
        }

    async def search(self, query: str, max_docs: int = 10) -> dict:
        """Search for *query* and return up to *max_docs* unique documents.

        The infini-gram ``search_docs`` API returns at most 10 results per
        call.  When more are requested, this method makes repeated calls
        and deduplicates by ``doc_ix``.

        Returns a dict with keys: documents, query, count.
        """
        seen_doc_ixs: set[int] = set()
        collected: list[dict] = []
        count: int | None = None

        page_size = min(max_docs, API_PAGE_SIZE)

        for _ in range(self._max_attempts):
            if len(collected) >= max_docs:
                break

            result = await self._post(
                {
                    "index": self._index,
                    "query_type": "search_docs",
                    "query": query,
                    "maxnum": page_size,
                }
            )

            batch_count = result.get("cnt", 0)
            if count is None:
                count = batch_count
            if batch_count == 0:
                break

            batch_docs = result.get("documents", [])
            if not batch_docs:
                break

            new_in_batch = 0
            for doc in batch_docs:
                doc_ix = doc.get("doc_ix", 0)
                if doc_ix in seen_doc_ixs:
                    continue
                seen_doc_ixs.add(doc_ix)
                new_in_batch += 1
                collected.append(self._parse_document(doc))

                if len(collected) >= max_docs:
                    break

            # No new unique docs — result space likely exhausted
            if new_in_batch == 0:
                break

        return {"documents": collected, "query": query, "count": count or 0}
