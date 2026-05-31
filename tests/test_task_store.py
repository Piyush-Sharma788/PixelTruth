import threading
from queue import Queue

from api.task_store import TaskResult, TaskStatus, TaskStore


def test_create_task():
    store = TaskStore()
    task_id = store.create_task()

    task = store.get_task(task_id)

    assert task is not None
    assert task.task_id == task_id
    assert task.status == TaskStatus.PENDING
    assert task.verdict is None
    assert task.error is None
    assert task.face_detected is None
    assert task.face_box is None


def test_task_lifecycle():
    store = TaskStore()
    task_id = store.create_task()

    store.mark_running(task_id)
    store.mark_completed(
        task_id,
        {
            "label": "Real",
            "confidence": 0.7,
            "raw": [0.7],
            "face_detected": True,
            "face_box": (1, 2, 3, 4),
        },
    )

    task = store.get_task(task_id)

    assert task is not None
    assert task.status == TaskStatus.COMPLETED
    assert task.verdict == "Real"
    assert task.confidence == 0.7
    assert task.raw_scores == [0.7]
    assert task.face_detected is True
    assert task.face_box == [1, 2, 3, 4]
    assert task.completed_at is not None


def test_task_failure():
    store = TaskStore()
    task_id = store.create_task()

    store.mark_running(task_id)
    store.mark_failed(task_id, "processing error")

    task = store.get_task(task_id)

    assert task is not None
    assert task.status == TaskStatus.FAILED
    assert task.error == "processing error"
    assert task.completed_at is not None
    assert task.face_detected is None
    assert task.face_box is None


def test_get_nonexistent_task():
    store = TaskStore()
    assert store.get_task("missing") is None


def test_thread_safety():
    store = TaskStore()
    queue = Queue()

    def worker():
        task_id = store.create_task()
        queue.put(task_id)
        store.mark_running(task_id)
        store.mark_completed(
            task_id,
            {
                "label": "Real",
                "confidence": 0.5,
                "raw": [0.5],
                "face_detected": False,
                "face_box": None,
            },
        )

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    ids = []
    while not queue.empty():
        ids.append(queue.get_nowait())

    assert len(ids) == 20
    for task_id in ids:
        task = store.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.COMPLETED
        assert task.verdict == "Real"
        assert task.face_detected is False
        assert task.face_box is None
