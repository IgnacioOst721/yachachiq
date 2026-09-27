"""Downloaders for the offline catalog: The Met, Art Institute of Chicago, Cleveland Museum of Art.

Only open-access material is stored:
  * The Met Collection API: objects with isPublicDomain=true (Open Access, CC0).
  * Art Institute of Chicago API: is_public_domain=true artworks, image via IIIF (CC0 data).
  * Cleveland Museum of Art Open Access API: cc0 filter.
Polite by design: one rate limiter per host, a descriptive User-Agent, retries with backoff,
and everything is resumable (state/done.tsv remembers every object already handled).
Run it through tools/box_analysis_build_catalog.py (needs internet; the robot never does).
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from .catalog import CatalogStore, save_thumb
from .taxonomy import excluded_classification, normalize_record

USER_AGENT = "Yachachiq/2.0 (WRO 2026 student robot, offline museum reference catalog; https://ignacioost721.github.io/yachachiq/)"

# Department priority: the Andes and the rest of the Americas first (the robot is shown in Peru).
MET_DEPARTMENTS = [5, 10, 3, 13, 14, 6, 17, 7, 12, 18, 4]
# The Met CSV department names -> maximum objects taken (priority order; Andes/Americas first).
MET_CSV_CAPS = {"Arts of Africa, Oceania, and the Americas": 20000, "Egyptian Art": 13000, "Ancient Near Eastern Art": 7000,
                "Greek and Roman Art": 15000, "Islamic Art": 8000, "Asian Art": 12000, "Medieval Art": 5000,
                "The Cloisters": 2500, "Musical Instruments": 2500, "European Sculpture and Decorative Arts": 6000,
                "Arms and Armor": 1500}
AIC_DEPARTMENTS = ["Arts of the Americas", "Arts of Africa", "Arts of Greece, Rome, and Byzantium", "Arts of Asia",
                   "Textiles", "Applied Arts of Europe"]
AIC_EXCLUDED_TYPES = ["Print", "Drawing and Watercolor", "Photograph", "Painting", "Book", "Miniature Painting",
                      "Architectural Drawing", "Furniture", "Miniature room", "non-art", "Installation", "Mixed Media"]
CMA_DEPARTMENTS = ["Art of the Americas", "African Art", "Egyptian and Ancient Near Eastern Art", "Greek and Roman Art",
                   "Islamic Art", "Indian and Southeast Asian Art", "Chinese Art", "Korean Art", "Japanese Art", "Oceania",
                   "Medieval Art", "Textiles", "Decorative Art and Design"]


class RateLimiter:
    def __init__(self, per_second: float):
        self.interval = 1.0 / per_second
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            t = max(now, self._next)
            self._next = t + self.interval
        if t > now:
            time.sleep(t - now)


class Http:
    def __init__(self, log: Callable[[str], None] = print):
        import requests
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT, "AIC-User-Agent": USER_AGENT})
        self.log = log
        self._last_note = {}

    def _note(self, msg: str) -> None:
        """Log throttling/errors at most once a minute per message kind."""
        key = msg.split(" (")[0]
        now = time.time()
        if now - self._last_note.get(key, 0) > 60:
            self._last_note[key] = now
            self.log("[http] " + msg)

    def get(self, url: str, limiter: RateLimiter, timeout: float = 60.0, retries: int = 4, **kw):
        delay = 5.0
        last = None
        for _ in range(retries):
            limiter.wait()
            try:
                r = self.s.get(url, timeout=timeout, **kw)
                if r.status_code == 200:
                    return r
                if r.status_code == 404:
                    return None
                last = "HTTP %d" % r.status_code
                self._note("HTTP %d from %s (retrying in %.0f s)" % (r.status_code, url.split("/")[2], delay))
                if r.status_code in (403, 429, 500, 502, 503, 504):
                    time.sleep(delay * (4 if r.status_code in (403, 429) else 1))
                    delay *= 2
                    continue
                return None
            except Exception as e:  # network hiccup
                last = str(e)
                self._note("%s on %s" % (type(e).__name__, url.split("/")[2]))
                time.sleep(delay)
                delay *= 2
        raise IOError("GET %s failed: %s" % (url, last))

    def post_json(self, url: str, payload: dict, limiter: RateLimiter, timeout: float = 60.0, retries: int = 4):
        delay = 5.0
        last = None
        for _ in range(retries):
            limiter.wait()
            try:
                r = self.s.post(url, json=payload, timeout=timeout)
                if r.status_code == 200:
                    return r.json()
                last = "HTTP %d" % r.status_code
            except Exception as e:
                last = str(e)
            time.sleep(delay)
            delay *= 2
        raise IOError("POST %s failed: %s" % (url, last))


class DoneLog:
    """state/done.tsv: '<id>\\t<status>' per handled object (ok, skip:<why>, err:<why>)."""

    def __init__(self, store: CatalogStore):
        self.path = store.state / "done.tsv"
        self._lock = threading.Lock()
        self.status: dict = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    self.status[parts[0]] = parts[1]

    def done(self, item_id: str, retry_errors: bool = False) -> bool:
        st = self.status.get(item_id)
        if st is None:
            return False
        return not (retry_errors and st.startswith("err"))

    def mark(self, item_id: str, status: str) -> None:
        with self._lock:
            self.status[item_id] = status
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write("%s\t%s\n" % (item_id, status.replace("\t", " ").replace("\n", " ")[:200]))


def _cm(v) -> Optional[float]:
    try:
        v = float(v)
        return round(v, 2) if v > 0 else None
    except (TypeError, ValueError):
        return None


class Builder:
    def __init__(self, store: CatalogStore, log: Callable[[str], None] = print, retry_errors: bool = False,
                 stop: Optional[threading.Event] = None, workers: int = 4):
        self.store = store.ensure()
        self.log = log
        self.http = Http(log)
        self.done = DoneLog(store)
        self.retry_errors = retry_errors
        self.stop = stop or threading.Event()
        self.workers = workers
        self.counts: dict = {}
        self._clock = threading.Lock()

    # -- shared ---------------------------------------------------------------------------------
    def _count(self, source: str, key: str) -> None:
        with self._clock:
            c = self.counts.setdefault(source, {"ok": 0, "skip": 0, "err": 0, "t0": time.time()})
            c[key] += 1
            n = c["ok"] + c["skip"] + c["err"]
            if n % 250 == 0:
                rate = n / max(1.0, time.time() - c["t0"])
                self.log("[%s] handled %d (ok %d, skip %d, err %d) %.2f/s, catalog total %d"
                         % (source, n, c["ok"], c["skip"], c["err"], rate, self.store.count()))

    def _finish(self, rec: dict, image_url: str, limiter: RateLimiter) -> None:
        """Download the image, write the thumbnail and append the normalized record."""
        rid = rec["id"]
        try:
            r = self.http.get(image_url, limiter, timeout=90.0)
            if r is None:
                self.done.mark(rid, "skip:no image")
                self._count(rec["source"], "skip")
                return
            rel = "thumbs/%s/%s.jpg" % (rec["source"], rec["source_id"])
            w, h = save_thumb(r.content, self.store.root / rel)
            rec.update(image_url=image_url, thumb=rel, thumb_wh=[w, h], added=time.strftime("%Y-%m-%d"))
            self.store.append(normalize_record(rec))
            self.done.mark(rid, "ok")
            self._count(rec["source"], "ok")
        except Exception as e:
            self.done.mark(rid, "err:%s" % e)
            self._count(rec["source"], "err")

    def _pool(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=self.workers)

    # -- The Met --------------------------------------------------------------------------------
    MET_API = "https://collectionapi.metmuseum.org/public/collection/v1"

    def met_ids(self, dept: int) -> list:
        cache = self.store.state / ("met_dept_%d.json" % dept)
        if cache.exists():
            return json.loads(cache.read_text())
        lim = RateLimiter(2.0)
        r = self.http.get("%s/search?departmentId=%d&hasImages=true&q=*" % (self.MET_API, dept), lim, timeout=300.0)
        ids = (r.json().get("objectIDs") if r is not None else None) or []
        cache.write_text(json.dumps(ids))
        return ids

    def met_record(self, o: dict) -> Optional[dict]:
        if not o.get("isPublicDomain") or not o.get("primaryImageSmall"):
            return None
        dims = {}
        for m in o.get("measurements") or []:
            if (m.get("elementName") or "").lower() == "overall":
                em = m.get("elementMeasurements") or {}
                dims = {k: _cm(em.get(K)) for k, K in (("h", "Height"), ("w", "Width"), ("d", "Depth"), ("diam", "Diameter"))}
                dims = {k: v for k, v in dims.items() if v}
                break
        geo = ", ".join(x for x in (o.get("locale"), o.get("region"), o.get("subregion"), o.get("state"), o.get("country")) if x)
        return {"id": "met:%d" % o["objectID"], "source": "met", "source_id": str(o["objectID"]),
                "museum": "The Metropolitan Museum of Art", "title": o.get("title") or o.get("objectName") or "",
                "object_name": o.get("objectName") or "", "culture": o.get("culture") or "",
                "period": " ".join(x for x in (o.get("period"), o.get("dynasty")) if x), "date": o.get("objectDate") or "",
                "date_begin": o.get("objectBeginDate"), "date_end": o.get("objectEndDate"), "medium": o.get("medium") or "",
                "classification": o.get("classification") or "", "department": o.get("department") or "", "geography": geo,
                "dims_cm": dims, "url": o.get("objectURL") or "", "license": "CC0 (The Met Open Access)",
                "accession": o.get("accessionNumber") or ""}

    def met_one(self, oid: int, api: RateLimiter, img: RateLimiter) -> None:
        rid = "met:%d" % oid
        if self.stop.is_set() or self.done.done(rid, self.retry_errors) or self.store.has(rid):
            return
        try:
            r = self.http.get("%s/objects/%d" % (self.MET_API, oid), api)
            o = r.json() if r is not None else None
        except Exception as e:
            self.done.mark(rid, "err:%s" % e)
            self._count("met", "err")
            return
        rec = self.met_record(o) if o else None
        if rec is None or excluded_classification(rec["classification"], rec["object_name"]):
            self.done.mark(rid, "skip:not public domain / 2D")
            self._count("met", "skip")
            return
        self._finish(rec, o["primaryImageSmall"], img)

    def met_candidates(self) -> list:
        """[(department, [objectIDs])] in priority order, from The Met's CC0 open-access CSV
        (state/MetObjects.csv, github.com/metmuseum/openaccess) filtered to public-domain 3D objects.
        Without the CSV, falls back to the API search per department (slower: every object is fetched)."""
        import csv
        import random
        path = self.store.state / "MetObjects.csv"
        if not path.exists():
            return [("dept %d" % d, self.met_ids(d)) for d in MET_DEPARTMENTS]
        csv.field_size_limit(10 ** 9)
        by = {}
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("Is Public Domain") != "True" or row.get("Department") not in MET_CSV_CAPS:
                    continue
                if excluded_classification(row.get("Classification", ""), row.get("Object Name", "")):
                    continue
                try:
                    by.setdefault(row["Department"], []).append(int(row["Object ID"]))
                except ValueError:
                    continue
        out = []
        for dept, cap in MET_CSV_CAPS.items():
            ids = by.get(dept, [])
            random.Random(dept).shuffle(ids)          # a partial build still covers the whole department
            out.append((dept, ids[:cap]))
        return out

    def build_met(self, limit: Optional[int] = None, departments=None) -> None:
        api, img = RateLimiter(2.5), RateLimiter(3.0)
        n = 0
        for dept, ids in self.met_candidates():
            if self.stop.is_set():
                return
            todo = [i for i in ids if not self.done.done("met:%d" % i, self.retry_errors)]
            self.log("[met] %s: %d candidates, %d to do" % (dept, len(ids), len(todo)))
            if limit is not None:
                todo = todo[: max(0, limit - n)]
            with self._pool() as pool:
                list(pool.map(lambda i: self.met_one(i, api, img), todo))
            n += len(todo)
            if limit is not None and n >= limit:
                return

    # -- Art Institute of Chicago ------------------------------------------------------------------
    AIC_API = "https://api.artic.edu/api/v1/artworks/search"
    AIC_FIELDS = ["id", "title", "artist_display", "date_start", "date_end", "date_display", "place_of_origin",
                  "medium_display", "classification_title", "artwork_type_title", "department_title", "style_title",
                  "image_id", "dimensions_detail", "is_public_domain"]

    def aic_record(self, a: dict) -> Optional[dict]:
        if not a.get("is_public_domain") or not a.get("image_id"):
            return None
        dims = {}
        for d in a.get("dimensions_detail") or []:
            dims = {k: _cm(d.get(K)) for k, K in (("h", "height"), ("w", "width"), ("d", "depth"), ("diam", "diameter"))}
            dims = {k: v for k, v in dims.items() if v}
            break
        artist = (a.get("artist_display") or "").replace("\n", ", ")
        return {"id": "aic:%d" % a["id"], "source": "aic", "source_id": str(a["id"]), "museum": "Art Institute of Chicago",
                "title": a.get("title") or "", "object_name": a.get("artwork_type_title") or "", "culture": artist,
                "period": a.get("style_title") or "", "date": a.get("date_display") or "", "date_begin": a.get("date_start"),
                "date_end": a.get("date_end"), "medium": a.get("medium_display") or "",
                "classification": " / ".join(x for x in (a.get("classification_title"), a.get("artwork_type_title")) if x),
                "department": a.get("department_title") or "", "geography": a.get("place_of_origin") or "", "dims_cm": dims,
                "url": "https://www.artic.edu/artworks/%d" % a["id"], "license": "CC0 (Art Institute of Chicago, public domain)"}

    def build_aic(self, limit: Optional[int] = None, departments=None) -> None:
        api, img = RateLimiter(0.9), RateLimiter(1.5)
        n = 0
        for dept in departments or AIC_DEPARTMENTS:
            query = {"bool": {"filter": [{"term": {"is_public_domain": True}}, {"exists": {"field": "image_id"}},
                                         {"term": {"department_title.keyword": dept}}],
                              "must_not": [{"terms": {"artwork_type_title.keyword": AIC_EXCLUDED_TYPES}}]}}
            page, pages = 1, 1
            while page <= min(pages, 100) and not self.stop.is_set():
                d = self.http.post_json(self.AIC_API, {"query": query, "fields": self.AIC_FIELDS, "limit": 100, "page": page}, api)
                pages = int((d.get("pagination") or {}).get("total_pages") or 1)
                if page == 1:
                    self.log("[aic] %s: %s records" % (dept, (d.get("pagination") or {}).get("total")))
                todo = []
                for a in d.get("data") or []:
                    rec = self.aic_record(a)
                    if rec is None or self.done.done(rec["id"], self.retry_errors) or self.store.has(rec["id"]):
                        continue
                    if excluded_classification(rec["classification"]):
                        self.done.mark(rec["id"], "skip:2D")
                        continue
                    todo.append((rec, "https://www.artic.edu/iiif/2/%s/full/400,/0/default.jpg" % a["image_id"]))
                if limit is not None:
                    todo = todo[: max(0, limit - n)]
                with self._pool() as pool:
                    list(pool.map(lambda t: self._finish(t[0], t[1], img), todo))
                n += len(todo)
                if limit is not None and n >= limit:
                    return
                page += 1

    # -- Cleveland Museum of Art -------------------------------------------------------------------
    CMA_API = "https://openaccess-api.clevelandart.org/api/artworks/"

    def cma_record(self, a: dict) -> Optional[dict]:
        web = ((a.get("images") or {}).get("web") or {}).get("url")
        if a.get("share_license_status") != "CC0" or not web:
            return None
        ov = ((a.get("dimensions") or {}).get("overall") or {})
        dims = {k: _cm((ov.get(K) or 0) * 100.0) for k, K in (("h", "height"), ("w", "width"), ("d", "depth"), ("diam", "diameter"))}
        culture = "; ".join(a.get("culture") or [])
        return {"id": "cma:%d" % a["id"], "source": "cma", "source_id": str(a["id"]), "museum": "Cleveland Museum of Art",
                "title": a.get("title") or "", "object_name": a.get("type") or "", "culture": culture, "period": "",
                "date": a.get("creation_date") or "", "date_begin": a.get("creation_date_earliest"),
                "date_end": a.get("creation_date_latest"), "medium": a.get("technique") or "", "classification": a.get("type") or "",
                "department": a.get("department") or "", "geography": culture, "dims_cm": {k: v for k, v in dims.items() if v},
                "url": a.get("url") or "", "license": "CC0 (Cleveland Museum of Art Open Access)", "_image": web}

    def build_cma(self, limit: Optional[int] = None, departments=None) -> None:
        import urllib.parse
        api, img = RateLimiter(1.0), RateLimiter(4.0)
        fields = "id,title,culture,technique,type,department,dimensions,creation_date,creation_date_earliest,creation_date_latest,url,images,share_license_status"
        n = 0
        for dept in departments or CMA_DEPARTMENTS:
            skip, total = 0, None
            while (total is None or skip < total) and not self.stop.is_set():
                url = "%s?cc0=1&has_image=1&limit=500&skip=%d&department=%s&fields=%s" % (
                    self.CMA_API, skip, urllib.parse.quote(dept), fields)
                r = self.http.get(url, api, timeout=180.0)
                d = r.json() if r is not None else {}
                total = int((d.get("info") or {}).get("total") or 0)
                if skip == 0:
                    self.log("[cma] %s: %d records" % (dept, total))
                todo = []
                for a in d.get("data") or []:
                    rec = self.cma_record(a)
                    if rec is None or self.done.done(rec["id"], self.retry_errors) or self.store.has(rec["id"]):
                        continue
                    if excluded_classification(rec["classification"]):
                        self.done.mark(rec["id"], "skip:2D")
                        continue
                    todo.append((rec, rec.pop("_image")))
                if limit is not None:
                    todo = todo[: max(0, limit - n)]
                with self._pool() as pool:
                    list(pool.map(lambda t: self._finish(t[0], t[1], img), todo))
                n += len(todo)
                if limit is not None and n >= limit:
                    return
                skip += 500
                if not d.get("data"):
                    break

    # -- all ----------------------------------------------------------------------------------------
    def build(self, sources=("met", "aic", "cma"), limit_per_source: Optional[int] = None) -> dict:
        """Run the sources in parallel threads (each has its own hosts and rate limits)."""
        fns = {"met": self.build_met, "aic": self.build_aic, "cma": self.build_cma}
        threads = []
        for s in sources:
            def run(fn=fns[s], name=s):
                try:
                    fn(limit=limit_per_source)
                    self.log("[%s] finished" % name)
                except Exception as e:  # keep the other sources going
                    self.log("[%s] stopped with error: %s" % (name, e))
            t = threading.Thread(target=run, name="catalog-" + s, daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return {k: {kk: vv for kk, vv in v.items() if kk != "t0"} for k, v in self.counts.items()}
