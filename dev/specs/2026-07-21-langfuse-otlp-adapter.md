# Langfuse OTLP adapter

## Goal

Make Cubepi's existing OpenTelemetry traces render correctly in Langfuse without
depending on the Langfuse SDK or provider-specific auto-instrumentors. In
particular, Langfuse must receive readable system/input/output content and trace
identity fields for sessions, users, and tags.

## Design

- Keep the existing GenAI semantic-convention attributes as the portable source
  of truth.
- Add synchronous span adapters that may derive backend-specific attributes
  while a Cubepi span is still writable. This avoids cloning `ReadableSpan`
  objects in exporters and remains compatible with `opentelemetry-sdk>=1.30`.
- Provide `LangfuseSpanAdapter`, which emits Langfuse OTLP attributes alongside
  the standard attributes. Input and output use a `{"messages": [...]}` envelope
  with string `content`, matching the shape rendered by Langfuse/OpenInference.
- Add first-class `session_id` and `user_id` fields to `tracing_context`; keep
  arbitrary metadata under `cubepi.metadata.*`.
- Normalize provider response bodies into `gen_ai.output.messages`, fixing a
  general Cubepi tracing omission rather than deriving Langfuse output from raw
  Anthropic/OpenAI payloads.

## Prior art and divergence

OpenInference's Agno instrumentation emits framework-level Agent/Generation
spans and propagates `session.id`/`user.id`. Anthropic instrumentation emits an
additional SDK-level Generation span. Cubepi already owns the framework and
provider lifecycle, so enabling either instrumentor in core would duplicate
Generation spans. Cubepi instead emits one semantic span tree and adds only the
Langfuse attribute mapping.

HTTPX instrumentation is intentionally outside Cubepi. It describes transport
latency, can expose request metadata, and adds noise below existing chat spans;
applications may enable it independently when needed.

## Privacy

Langfuse input/output attributes are emitted only when `record_content=True`.
The existing redaction hook runs before both standard and Langfuse attributes
are written.

