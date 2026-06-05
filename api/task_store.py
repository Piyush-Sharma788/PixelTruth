"""In-memory, thread-safe task store for asynchronous inference jobs.

Each task tracks a deepfake-detection request through its lifecycle:
PENDING → RUNNING → COMPLETED | FAILED.

The store is intentionally simple (dict + Lock) and lives in-process.
For multi-worker or persistent deployments, swap this for a Redis or
database-backed implementation.

TTL-based expiry: completed/failed tasks are automatically pruned after
TASK_TTL_SECONDS (default 300 s) to prevent unbounded memory growth.
"""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Default task retention period (seconds). Override via PIXELTRUTH_TASK_TTL
# environment variable.
# ---------------------------------------------------------------------------
import os as _os
TASK_TTL_SECONDS: int = int(_os.getenv("PIXELTRUTH_TASK_TTL", "300"))


class TaskStatus(str, Enum):
    """Lifecycle states for an inference task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskResult(BaseModel):
    """Serialisable snapshot of a task's current state."""

    task_id: str
    status: TaskStatus
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    raw_scores: Optional[list[float]] = None
    face_detected: Optional[bool] = None
    face_box: Optional[list[int]] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class TaskStore:
    """Thread-safe container for in-flight and completed tasks.

    Tasks are automatically expired after ``TASK_TTL_SECONDS`` once they
    reach a terminal state (COMPLETED or FAILED).  A lightweight background
    daemon thread runs the sweep so the main event-loop is never blocked.
    """

    _TERMINAL = {TaskStatus.COMPLETED, TaskStatus.FAILED}

    def __init__(self, ttl_seconds: int = TASK_TTL_SECONDS) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskResult] = {}
        self._ttl = ttl_seconds
        # Monotonic timestamps (seconds) for when a task became terminal
        self._terminal_ts: dict[str, float] = {}
        # Start background cleanup daemon
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="task-store-gc"
        )
        self._cleanup_thread.start()

    # -- public API ----------------------------------------------------------

    def create_task(self) -> str:
        """Register a new task in PENDING state and return its ID."""
        task_id = uuid.uuid4().hex
        task = TaskResult(
            task_id=task_id,
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )
        with self._lock:
            self._tasks[task_id] = task
        return task_id

    def get_task(self, task_id: str) -> Optional[TaskResult]:
        """Return the current snapshot for *task_id*, or ``None``."""
        with self._lock:
            return self._tasks.get(task_id)

    def mark_running(self, task_id: str) -> None:
        """Transition a task to RUNNING."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                task.status = TaskStatus.RUNNING

    def mark_completed(
        self,
        task_id: str,
        result: dict,
    ) -> None:
        """Store a successful prediction result and mark COMPLETED."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                task.status = TaskStatus.COMPLETED
                task.verdict = result["label"]
                task.confidence = result["confidence"]
                task.raw_scores = result["raw"]
                task.face_detected = result.get("face_detected", False)
                task.face_box = (
                    list(result["face_box"])
                    if result.get("face_box") is not None
                    else None
                )
                task.completed_at = datetime.now(timezone.utc)
                self._terminal_ts[task_id] = time.monotonic()

    def mark_failed(self, task_id: str, error: str) -> None:
        """Record an error message and mark FAILED."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                task.status = TaskStatus.FAILED
                task.error = error
                task.completed_at = datetime.now(timezone.utc)
                self._terminal_ts[task_id] = time.monotonic()

    # -- private helpers -----------------------------------------------------

    def _cleanup_loop(self) -> None:
        """Background daemon: sweep expired terminal tasks every 60 seconds."""
        while True:
            time.sleep(60)
            self._evict_expired()

    def _evict_expired(self) -> None:
        """Remove tasks that have been terminal for longer than *ttl_seconds*."""
        now = time.monotonic()
        with self._lock:
            expired = [
                tid
                for tid, ts in self._terminal_ts.items()
                if (now - ts) >= self._ttl
            ]
            for tid in expired:
                self._tasks.pop(tid, None)
                self._terminal_ts.pop(tid, None)
