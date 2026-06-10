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


def test_get_task_returns_deep_copy():
    store = TaskStore()
    task_id = store.create_task()

    store.mark_running(task_id)
    store.mark_completed(
        task_id,
        {
            "label": "Real",
            "confidence": 0.9,
            "raw": [0.9],
            "face_detected": True,
            "face_box": [1, 2, 3, 4],
        },
    )

    task_snapshot = store.get_task(task_id)
    assert task_snapshot is not None

    another_snapshot = store.get_task(task_id)

    # Mutating mutable fields on task_snapshot should not affect another_snapshot
    task_snapshot.raw_scores.append(0.1)
    task_snapshot.face_box[0] = 99

    assert another_snapshot.raw_scores == [0.9]
    assert another_snapshot.face_box == [1, 2, 3, 4]


def test_concurrency_no_torn_reads():
    import time
    import random
    store = TaskStore()

    stop_event = threading.Event()
    active_tasks = []
    tasks_lock = threading.Lock()

    def writer():
        while not stop_event.is_set():
            task_id = store.create_task()
            with tasks_lock:
                active_tasks.append(task_id)

            store.mark_running(task_id)
            if random.random() < 0.5:
                store.mark_completed(
                    task_id,
                    {
                        "label": "Real",
                        "confidence": 0.8,
                        "raw": [0.8],
                        "face_detected": True,
                        "face_box": [10, 20, 30, 40],
                    },
                )
            else:
                store.mark_failed(task_id, "temporary error")
            time.sleep(0.001)

    errors = []

    def reader():
        while not stop_event.is_set():
            with tasks_lock:
                tids = list(active_tasks)
            if not tids:
                time.sleep(0.001)
                continue

            task_id = random.choice(tids)
            task = store.get_task(task_id)
            if task is not None:
                if task.status == TaskStatus.COMPLETED:
                    if task.verdict is None or task.confidence is None or task.raw_scores is None:
                        errors.append(
                            f"Torn read COMPLETED: verdict={task.verdict}, "
                            f"confidence={task.confidence}, raw_scores={task.raw_scores}"
                        )
                    if task.error is not None:
                        errors.append(f"Torn read COMPLETED: has error={task.error}")
                elif task.status == TaskStatus.FAILED:
                    if task.error is None:
                        errors.append("Torn read FAILED: error is None")
                    if task.verdict is not None or task.confidence is not None:
                        errors.append(f"Torn read FAILED: has verdict={task.verdict}")
                elif task.status == TaskStatus.PENDING or task.status == TaskStatus.RUNNING:
                    if task.verdict is not None or task.confidence is not None or task.error is not None:
                        errors.append(
                            f"Torn read {task.status}: verdict={task.verdict}, error={task.error}"
                        )

    writers = [threading.Thread(target=writer) for _ in range(5)]
    readers = [threading.Thread(target=reader) for _ in range(5)]

    for t in writers + readers:
        t.start()

    time.sleep(0.5)
    stop_event.set()

    for t in writers + readers:
        t.join()

    assert not errors, f"Errors detected: {errors}"


