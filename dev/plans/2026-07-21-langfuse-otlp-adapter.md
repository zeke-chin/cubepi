# Langfuse OTLP adapter implementation plan

1. Add the span-adapter protocol and Langfuse implementation; expose adapters
   through `Tracer(span_adapters=...)` without changing exporter behavior.
2. Extend `tracing_context` with task-local session/user identity and apply it
   to normal Agent and one-shot roots.
3. Normalize Anthropic, OpenAI Chat, and OpenAI Responses bodies into standard
   output messages, then derive Langfuse message envelopes from recorded
   content.
   Snapshot each agent step's input at its first provider request so tool-loop
   turns have correctly aligned input/output boundaries.
4. Add unit/integration-style tracing tests for mapping, privacy, nesting, and
   provider response shapes; update tracing documentation.
5. Update the `alg-wos-agent` example to enable the adapter and optional
   session/user/tags configuration, then validate against QA Langfuse.
