"""Infini-gram API client using the search_docs endpoint."""

from __future__ import annotations

import httpx


class InfinigramClient:
    def __init__(self, http_client: httpx.AsyncClient, api_url: str, index: str):
        self._http = http_client
        self._api_url = api_url.rstrip("/")
        self._index = index

    async def _post(self, payload: dict) -> dict:
        resp = await self._http.post(self._api_url, json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()

    async def search(self, query: str, max_docs: int = 10) -> dict:
        """Search for *query* and return matching documents.

        Uses the ``search_docs`` query type which returns documents with
        properly highlighted match spans.

        Returns a dict with keys: documents, query, count.
        """
        result = await self._post(
            {
                "index": self._index,
                "query_type": "search_docs",
                "query": query,
                "maxnum": max_docs,
            }
        )

        count = result.get("cnt", 0)
        if count == 0:
            return {"documents": [], "query": query, "count": 0}

        documents = []
        for doc in result.get("documents", []):
            # Spans from search_docs are 2-element arrays: [text, highlight]
            # highlight is None for non-match, an integer (e.g. 0) for match
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
            documents.append(
                {
                    "doc_ix": doc.get("doc_ix", 0),
                    "doc_len": doc.get("doc_len", 0),
                    "disp_len": doc.get("disp_len", 0),
                    "spans": doc_spans,
                    "full_text": full_text,
                }
            )

        return {"documents": documents, "query": query, "count": count}
