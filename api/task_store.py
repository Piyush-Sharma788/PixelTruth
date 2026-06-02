"""In-memory, thread-safe task store for asynchronous inference jobs.

Each task tracks a deepfake-detection request through its lifecycle:
PENDING → RUNNING → COMPLETED | FAILED.

The store is intentionally simple (dict + Lock) and lives in-process.
For multi-worker or persistent deployments, swap this for a Redis or
database-backed implementation.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


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
    """Thread-safe container for in-flight and completed tasks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskResult] = {}

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
                task.face_box = list(result["face_box"]) if result.get("face_box") is not None else None
                task.completed_at = datetime.now(timezone.utc)

    def mark_failed(self, task_id: str, error: str) -> None:
        """Record an error message and mark FAILED."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                task.status = TaskStatus.FAILED
                task.error = error
                task.completed_at = datetime.now(timezone.utc)
