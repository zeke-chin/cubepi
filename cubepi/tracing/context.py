"""Per-run identity, tagging, and metadata context for cubepi.tracing.

Sets contextvar-scoped session, user, tags, and metadata values that the
:class:`~cubepi.tracing.recorder.Recorder` reads on ``AgentStartEvent``
and stamps onto the ``invoke_agent`` span. Inspired by LangSmith's
``langsmith.run_helpers.tracing_context`` (same contextvar mechanism).

Namespacing:

- Tags use a single attribute ``cubepi.tags`` (tuple of strings).
- User metadata is namespaced under ``cubepi.metadata.*`` so that
  recorder-owned schema keys (``cubepi.run_id``,
  ``cubepi.turn.index``, …) can never be overridden by
  caller-supplied values.

Usage::

    from cubepi.tracing import tracing_context

    async with tracer.attached(agent):
        with tracing_context(tags=["beta-arm"], metadata={"user_id": "u-42"}):
            await agent.prompt("hello")
        # the next run does NOT carry those tags
        await agent.prompt("goodbye")

Resulting ``invoke_agent`` span attributes:

- ``cubepi.tags`` = ``("beta-arm",)``
- ``cubepi.metadata.user_id`` = ``"u-42"``

Multiple nested ``tracing_context`` blocks merge: inner tags are
appended, inner metadata keys override outer ones (last-write-wins).
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Iterator


_run_metadata: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "cubepi.tracing.run_metadata", default={}
)

_run_tags: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "cubepi.tracing.run_tags", default=()
)

_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cubepi.tracing.session_id", default=None
)

_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cubepi.tracing.user_id", default=None
)


@contextlib.contextmanager
def tracing_context(
    *,
    tags: list[str] | tuple[str, ...] | None = None,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> Iterator[None]:
    """Scope identity, tags, and metadata onto runs started in this block.

    The recorder reads these contextvars on ``AgentStartEvent`` and
    stamps them on the ``invoke_agent`` span as:

    - ``cubepi.tags`` — tuple of strings (OTel attribute type)
    - one attribute per metadata key, namespaced under
      ``cubepi.metadata.*`` (e.g. ``metadata={"tenant": "demo"}`` →
      ``cubepi.metadata.tenant = "demo"``). The dedicated
      sub-namespace keeps recorder-owned schema keys
      (``cubepi.run_id``, ``cubepi.turn.index``, …) safe from
      caller-supplied collisions.

    The contextvar nature means this works for concurrent agents:
    each asyncio task tree gets its own value. Nested blocks merge
    additively (tags concatenate; metadata is union with inner
    keys winning).

    Args:
        tags: Tags to apply to runs started in this scope. Stored as
            a tuple on the span so it round-trips through OTel's
            attribute serializer.
        metadata: Per-run key/value pairs. Values must be types that
            OTel attributes accept (str, bool, int, float, or a
            tuple/list of those); other shapes will be silently
            dropped by the recorder.
        session_id: Backend session/conversation identity. Inner non-None
            values override outer contexts; omitted values inherit.
        user_id: Backend end-user identity, with the same nesting semantics.
    """
    new_meta = {**_run_metadata.get(), **(metadata or {})}
    new_tags = tuple(_run_tags.get()) + tuple(tags or ())
    new_session_id = session_id if session_id is not None else _session_id.get()
    new_user_id = user_id if user_id is not None else _user_id.get()
    meta_token = _run_metadata.set(new_meta)
    tag_token = _run_tags.set(new_tags)
    session_token = _session_id.set(new_session_id)
    user_token = _user_id.set(new_user_id)
    try:
        yield
    finally:
        _run_metadata.reset(meta_token)
        _run_tags.reset(tag_token)
        _session_id.reset(session_token)
        _user_id.reset(user_token)


def _current_tags() -> tuple[str, ...]:
    """Internal: return the active tag tuple for the current task.

    Called by :class:`~cubepi.tracing.recorder.Recorder` on
    ``_on_agent_start``; not part of the public API.
    """
    return _run_tags.get()


def _current_metadata() -> dict[str, Any]:
    """Internal: return the active metadata dict for the current task."""
    return _run_metadata.get()


def _current_session_id() -> str | None:
    """Internal: return the active session id for the current task."""
    return _session_id.get()


def _current_user_id() -> str | None:
    """Internal: return the active user id for the current task."""
    return _user_id.get()
