"""File-based request queue for the security-hardening adapter.

Requests are atomically claimed from ``incoming/*.json`` and each produces one
schema-shaped response at ``output/<request-file-stem>.response.json``. The
response's ``request_id`` echoes the request's ID. A small SQLite ledger makes
request IDs unique across runs and supports recovery after an interrupted run.

Run one watcher process per incoming/output directory pair. Within that process,
``--workers`` allows multiple requests to be hardened concurrently.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import threading
from contextlib import contextmanager
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .integration import HardeningService, error_response

_CLAIM_RE = re.compile(r"^(?P<source>.+)\.(?P<token>[0-9a-f]{32})\.claim$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    """Write JSON by replacing a same-directory temporary file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # os.replace is atomic on the same filesystem. The worker checks that
        # destination names are unique before this function is called.
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


@dataclass(frozen=True)
class _Job:
    claim_path: Path
    source_filename: str
    recovered: bool

    @property
    def source_stem(self) -> str:
        return Path(self.source_filename).stem

    @property
    def output_filename(self) -> str:
        return f"{self.source_stem}.response.json"


@dataclass
class _JobResult:
    source_filename: str
    request_id: Optional[str]
    response_path: Optional[Path]
    status: str
    completed: bool
    message: str = ""


@dataclass
class WorkerSummary:
    discovered: int = 0
    responses_written: int = 0
    failures: int = 0
    results: List[_JobResult] = field(default_factory=list)


class DirectoryWorker:
    """Poll an inbox, harden requests in parallel, and write one paired response each."""

    def __init__(
        self,
        incoming_dir: str | Path = "incoming",
        output_dir: str | Path = "output",
        processed_dir: str | Path = "processed",
        failed_dir: str | Path = "failed",
        workers: int = 4,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self.incoming_dir = Path(incoming_dir).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.processed_dir = Path(processed_dir).expanduser().resolve()
        self.failed_dir = Path(failed_dir).expanduser().resolve()
        if len({self.incoming_dir, self.output_dir, self.processed_dir, self.failed_dir}) != 4:
            raise ValueError("incoming, output, processed and failed directories must be different")
        self.workers = workers
        self.processing_dir = self.incoming_dir / ".processing"
        self.ledger_path = self.output_dir / ".request_index.sqlite3"
        self._thread_local = threading.local()
        self._prepare_directories()
        self._init_ledger()

    def _prepare_directories(self) -> None:
        for path in (
            self.incoming_dir,
            self.output_dir,
            self.processed_dir,
            self.failed_dir,
            self.processing_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self):
        """Open, commit/rollback, and always close a per-operation SQLite connection."""
        con = sqlite3.connect(self.ledger_path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout = 30000")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_ledger(self) -> None:
        with self._connect() as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS request_ids (
                    request_id TEXT PRIMARY KEY,
                    source_filename TEXT NOT NULL,
                    output_filename TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('processing', 'completed')),
                    first_seen_at TEXT NOT NULL,
                    completed_at TEXT
                )"""
            )

    def _ledger_row(self, request_id: str) -> Optional[sqlite3.Row]:
        with self._connect() as con:
            return con.execute(
                "SELECT * FROM request_ids WHERE request_id = ?", (request_id,)
            ).fetchone()

    def _reserve_id(self, request_id: str, job: _Job) -> str:
        """Return ``reserved``, ``resume``, ``completed`` or ``duplicate`` atomically."""
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT * FROM request_ids WHERE request_id = ?", (request_id,)
            ).fetchone()
            if row is None:
                con.execute(
                    "INSERT INTO request_ids(request_id, source_filename, output_filename, state, first_seen_at) "
                    "VALUES (?, ?, ?, 'processing', ?)",
                    (request_id, job.source_filename, job.output_filename, _utc_now()),
                )
                return "reserved"
            if row["source_filename"] != job.source_filename:
                return "duplicate"
            if row["state"] == "processing":
                return "resume"
            return "completed"

    def _mark_completed(self, request_id: str) -> None:
        with self._connect() as con:
            con.execute(
                "UPDATE request_ids SET state='completed', completed_at=? WHERE request_id=?",
                (_utc_now(), request_id),
            )

    def _service(self) -> HardeningService:
        # Each thread has its own service/loaded catalogue; no mutable service
        # state is shared between concurrent requests.
        service = getattr(self._thread_local, "service", None)
        if service is None:
            service = HardeningService()
            self._thread_local.service = service
        return service

    def _claim_incoming(self, source: Path) -> Optional[_Job]:
        token = uuid.uuid4().hex
        claim_path = self.processing_dir / f"{source.name}.{token}.claim"
        try:
            # Same-filesystem rename claims a file atomically. If another worker
            # has already claimed it, this rename fails and that worker owns it.
            source.rename(claim_path)
        except FileNotFoundError:
            return None
        except OSError as exc:
            print(f"[worker] could not claim {source.name}: {type(exc).__name__}: {exc}")
            return None
        return _Job(claim_path=claim_path, source_filename=source.name, recovered=False)

    def _recover_claims(self) -> List[_Job]:
        recovered: List[_Job] = []
        for claim in sorted(self.processing_dir.glob("*.claim")):
            match = _CLAIM_RE.match(claim.name)
            if not match:
                self._record_failure_file(claim, claim.name, "INVALID_CLAIM_NAME", "cannot recover a claim with an unexpected filename")
                continue
            recovered.append(_Job(claim_path=claim, source_filename=match.group("source"), recovered=True))
        return recovered

    def process_available(self) -> WorkerSummary:
        """Process files currently in the inbox plus claims left by an interrupted run."""
        self._prepare_directories()
        jobs = self._recover_claims()
        failures: List[_JobResult] = []

        for source in sorted(self.incoming_dir.glob("*.json")):
            if not source.is_file():
                continue
            response_path = self.output_dir / f"{source.stem}.response.json"
            if response_path.exists():
                message = (
                    f"response {response_path.name} already exists; request filenames must be unique and a response is never overwritten"
                )
                self._record_failure_file(source, source.name, "DUPLICATE_INPUT_FILENAME", message)
                failures.append(_JobResult(source.name, None, None, "DUPLICATE_INPUT_FILENAME", False, message))
                continue
            job = self._claim_incoming(source)
            if job is not None:
                jobs.append(job)

        summary = WorkerSummary(discovered=len(jobs) + len(failures), failures=len(failures), results=failures)
        if not jobs:
            return summary

        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="hardening") as pool:
            futures = {pool.submit(self._process_job, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # preserve claim for retry if unexpected I/O/ledger failure
                    message = f"worker failed for {job.source_filename}: {type(exc).__name__}; claim retained for retry"
                    print(f"[worker] {message}")
                    summary.failures += 1
                    summary.results.append(_JobResult(job.source_filename, None, None, "WORKER_FAILURE", False, message))
                    continue
                summary.results.append(result)
                if result.completed:
                    summary.responses_written += 1
                    print(
                        f"[{result.status}] {job.source_filename} -> "
                        f"{result.response_path.name if result.response_path else '(recovered)'} "
                        f"request_id={result.request_id}"
                    )
                else:
                    summary.failures += 1
                    print(f"[worker] {result.source_filename}: {result.message}")
        return summary

    def _process_job(self, job: _Job) -> _JobResult:
        response_path = self.output_dir / job.output_filename
        try:
            raw = _read_json(job.claim_path)
            parse_error: Optional[str] = None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raw = None
            parse_error = f"input is not readable valid UTF-8 JSON ({type(exc).__name__})"

        request_id_value = raw.get("request_id") if isinstance(raw, dict) else None
        request_id = request_id_value if isinstance(request_id_value, str) and request_id_value.strip() else None
        response_key = request_id or job.source_stem

        if response_path.exists():
            existing = self._load_existing_response(response_path)
            # On restart, a response may have been written just before the
            # process exited. If it belongs to this claimed request, finish the
            # ledger/archive step rather than processing the same request twice.
            if job.recovered and existing and existing.get("request_id") == response_key:
                if request_id:
                    row = self._ledger_row(request_id)
                    if row and row["source_filename"] == job.source_filename:
                        self._mark_completed(request_id)
                self._archive_claim(job)
                return _JobResult(job.source_filename, response_key, response_path, existing.get("status", "UNKNOWN"), True, "recovered existing response")
            message = f"response {response_path.name} already exists; refusing to overwrite it"
            self._record_failure_file(job.claim_path, job.source_filename, "DUPLICATE_INPUT_FILENAME", message)
            return _JobResult(job.source_filename, request_id, None, "DUPLICATE_INPUT_FILENAME", False, message)

        if parse_error:
            response = error_response(response_key, "INVALID_JSON", parse_error)
            status = response["status"]
        elif not isinstance(raw, dict):
            response = error_response(response_key, "INVALID_REQUEST", "request JSON must contain one object, not an array or scalar")
            status = response["status"]
        elif request_id is None:
            response = error_response(response_key, "REQUEST_ID_MISSING", "request_id must be a non-empty string; the filename stem is used only to identify this error response")
            status = response["status"]
        else:
            reservation = self._reserve_id(request_id, job)
            if reservation == "duplicate":
                response = error_response(
                    request_id,
                    "DUPLICATE_REQUEST_ID",
                    "request_id has already been used by a different input file; generate a unique request_id for each request",
                )
                status = response["status"]
            elif reservation == "completed":
                # This is typically a re-delivery that reused a filename and ID.
                # Do not replace an older response, even when it has the same ID.
                message = f"request_id {request_id!r} was already completed; refusing to process it again"
                self._record_failure_file(job.claim_path, job.source_filename, "DUPLICATE_REQUEST", message)
                return _JobResult(job.source_filename, request_id, None, "DUPLICATE_REQUEST", False, message)
            else:
                response = self._service().handle(raw)
                status = response.get("status", "UNKNOWN")

        _atomic_write_json(response_path, response)
        if request_id:
            # For duplicate IDs we intentionally don't alter the original row.
            row = self._ledger_row(request_id)
            if row and row["source_filename"] == job.source_filename:
                self._mark_completed(request_id)
        self._archive_claim(job)
        return _JobResult(job.source_filename, request_id or response_key, response_path, status, True)

    def _load_existing_response(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            value = _read_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _archive_claim(self, job: _Job) -> None:
        destination = self.processed_dir / job.source_filename
        if destination.exists():
            destination = self.processed_dir / f"{Path(job.source_filename).stem}.{uuid.uuid4().hex}{Path(job.source_filename).suffix}"
        job.claim_path.rename(destination)

    def _record_failure_file(self, source_path: Path, source_filename: str, code: str, message: str) -> None:
        """Move an unprocessable/colliding file to failed and write an operator-readable record."""
        token = uuid.uuid4().hex
        suffix = Path(source_filename).suffix or ".dat"
        archived_name = f"{Path(source_filename).stem}.{token}.failed{suffix}"
        destination = self.failed_dir / archived_name
        try:
            source_path.rename(destination)
        except OSError:
            # Preserve the event record even if a file cannot be moved.
            pass
        event = {
            "source_filename": source_filename,
            "archived_filename": archived_name,
            "error_code": code,
            "message": message,
            "occurred_at": _utc_now(),
        }
        _atomic_write_json(self.failed_dir / f"{Path(source_filename).stem}.{token}.failure.json", event)

    def watch_forever(self, poll_interval: float = 0.5) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be greater than 0")
        print(
            f"Watching {self.incoming_dir} -> {self.output_dir} "
            f"({self.workers} worker threads; Ctrl-C to stop)"
        )
        while True:
            self.process_available()
            time.sleep(poll_interval)
