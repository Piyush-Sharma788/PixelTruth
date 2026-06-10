# [BUG] API test collection fails when `pydantic` and `pydantic-core` drift out of sync

## Summary
The API layer imports `fastapi` and `pydantic` at module import time, but the project did not previously pin the underlying `pydantic` runtime. In a non-isolated environment, this allows `pydantic` and `pydantic-core` to drift apart and causes the API tests to fail during collection before any application code runs.

This is currently reproducible in the local environment used for validation.

---

## Evidence

### 1. `requirements.txt` previously left the Pydantic runtime implicit
The dependency file included:

```txt
fastapi>=0.100.0
```

but did not explicitly constrain `pydantic` / `pydantic-core`, even though both are imported directly by the API modules.

### 2. API test collection fails before test execution
Running:

```bash
pytest tests/test_api.py tests/test_task_store.py -q
```

currently fails with:

```txt
SystemError: The installed pydantic-core version (2.14.6) is incompatible with the current pydantic version, which requires 2.41.5.
```

### 3. The failure is independent of request-handling logic
The exception is raised during import of:

- `fastapi.testclient`
- `api.task_store`

which means the test suite never reaches any endpoint or task-store assertions.

---

## Steps To Reproduce
1. Install dependencies into an environment that already contains a mismatched `pydantic-core`.
2. Run:
   ```bash
   pytest tests/test_api.py tests/test_task_store.py -q
   ```
3. Observe import-time failure before test execution.

---

## Actual Result
API tests fail during import/collection due to a broken `pydantic` runtime pair.

## Expected Result
Installing the documented project dependencies should produce a compatible FastAPI/Pydantic runtime, and API tests should at least collect successfully.

---

## Suggested Resolution
1. Keep `pydantic` and `pydantic-core` explicitly pinned in `requirements.txt`.
2. Run `pip check` in CI after installation to catch resolver drift.
3. Prefer installing into a clean virtual environment for local validation so global site-packages cannot mask dependency conflicts.
