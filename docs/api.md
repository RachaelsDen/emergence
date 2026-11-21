# Operator API and CLI

One core process serves HTTP on `127.0.0.1:8000` by default. Start it from
the checkout with either `emergence --config FILE` or
`emergence serve --config FILE` (prefix with `uv run --locked` when working
in the checkout). The CLI is an HTTP client for all other commands; it never
opens the core database. Use only the loopback or an authorized management
listener; see [operations](operations.md).

The service generates `/openapi.json`. A [schema export](openapi.json) includes
the optional metrics route; the offline suite checks it against the application.

## Authentication and routes

Set API tokens **only** in the environment variables named by
`operator_token_env` and `reader_token_env` (defaults below). Each configured
role token must be at least 32 bytes, and the two values must be distinct. Use
random printable ASCII tokens. An unset token disables that role's credential;
the operator token also authorizes reader routes. The service snapshots API
tokens at startup; restart after rotation. Pass `Authorization: Bearer <token>`
over a trusted transport; never put a token value in YAML, URLs, or CLI flags.
Missing/invalid credentials yield `401`; a valid reader on an operator route
yields `403`. Audited writes record the authenticated role (`operator`), not
the token or an asserted user identity.

| Access | Method and path | CLI / behavior |
| --- | --- | --- |
| Public | `GET /health/live` | `health --live`; process liveness |
| Public | `GET /health/ready` | `health`; local service/DB readiness, not provider or NAS health |
| Reader | `GET /api/v1/providers` | `providers`; configuration and credential-presence flags, not proof of working credentials |
| Reader | `GET /api/v1/models` | `models`; enabled aliases, upstream IDs, and declared bounds |
| Reader | `GET /api/v1/resources` | `resources`; observations, local storage and archive status even with metrics disabled |
| Reader | `GET /api/v1/budgets` | `budgets show [--provider NAME]`; current period projections |
| Reader | `GET /metrics` | Prometheus text, only when `metrics_enabled: true`; bearer protected |
| Operator | `GET /api/v1/usage` | `usage`; paginated usage, including scoped report rows |
| Operator | `GET /api/v1/history/{request_id}` | HTTP only; redacted input/output payloads (when available) and events |
| Operator | `POST /api/v1/inference` | `inference test`; bounded, non-streaming request |
| Operator | `GET /api/v1/inference/{request_id}` | `inference status`; service-issued ID |
| Operator | `POST /api/v1/inference/{request_id}/resolution` | `inference resolve`; audited outcome when unresolved |
| Operator | `POST /api/v1/providers/{provider}/exhaustion/resolution` | HTTP only; independently confirmed recovery from the current inference denial |
| Operator | `POST /api/v1/budgets/adjustments` | `budgets adjust`; audited adjustment to the current period |

Use configured provider names and model **aliases**, not the upstream IDs.
Budget reads and adjustments require an enabled provider; unknown or disabled
provider names return `404`.
Usage filters are `provider`, `model`, `start` (inclusive), `end` (exclusive),
`limit` (1–500, default 100), and `offset` (0–1,000,000, default 0).
Timestamps require an ISO 8601 offset or `Z`. CLI options use the same names
with `--`. API error bodies normally carry an `error` code; archived payloads
and CLI responses can still contain sensitive research data. Do not treat
redaction as permission to submit credentials or personal data.

## CLI examples

Put `--url` and `--token-env` **before** the subcommand. `--url` overrides
`EMERGENCE_URL` and the settings-derived address. Without `--token-env`, the
CLI uses the configured operator variable first, then the reader variable.
Name the role explicitly for least privilege.

```sh
emergence --url http://127.0.0.1:8000 --token-env EMERGENCE_READER_TOKEN resources
emergence --token-env EMERGENCE_READER_TOKEN budgets show --provider openai
emergence --token-env EMERGENCE_OPERATOR_TOKEN usage --provider openai --model chat \
  --start 2026-11-01T00:00:00Z --end 2026-12-01T00:00:00Z --limit 20
```

The next example **dispatches inference** when `local/chat` is configured,
enabled and admitted. Use a new opaque key per intended request; save it
before submitting. The prompt comes from a UTF-8 file or stdin.

```sh
emergence --token-env EMERGENCE_OPERATOR_TOKEN inference test \
  --provider local --model chat --max-output-tokens 8 \
  --request-id operator-check-unique-001 --prompt-file prompt.txt
emergence --token-env EMERGENCE_OPERATOR_TOKEN inference status "$SERVICE_REQUEST_ID"
```

HTTP equivalent: `POST /api/v1/inference` with `Content-Type: application/json`,
operator authorization, and `Idempotency-Key: operator-check-unique-001`:

```json
{
  "provider": "local",
  "model": "chat",
  "max_output_tokens": 8,
  "messages": [{"role": "user", "content": "Reply with OK."}]
}
```

The misleadingly named CLI `--request-id` sets the caller's **idempotency
key** (generated if omitted); it is not the returned `request_id`. The response
includes `request_id`, `state`, `replayed`, and, where applicable, `result` or
`error_code`. Save the returned ID for status, full history, and resolution.
The CLI does **not** retry inference automatically. A client timeout or
transport error does not prove no dispatch or no charge. Inspect status if
the service ID is known; if explicitly necessary to recover the result using
the same request, reuse the original key **and identical body**. A different
key can dispatch and charge again; a changed body with the original key is
not a valid replay. Do not blindly retry a write after an uncertain outcome.

## Audited writes

Send exact USD amounts as JSON **strings**, not JSON numbers; returned ledger
amounts are decimal strings or `null`. Append a positive grant, negative
reduction, or signed nonzero correction with an evidence-based reason:

```sh
emergence --token-env EMERGENCE_OPERATOR_TOKEN budgets adjust --provider openai \
  --kind grant --amount 1.25 --reason 'Approved virtual budget adjustment'
```

Equivalent `POST /api/v1/budgets/adjustments` body:
`{"provider":"openai","kind":"grant","amount":"1.25","reason":"Approved virtual budget adjustment"}`.

Investigate an unresolved charge before resolving it. A supplied nonnegative
`--amount` is an operator **estimate**, not a measured provider charge.
`--no-charge` is a separate audited decision to release the hold. Choose
exactly one and record the supporting reason. These commands are alternatives
for an eligible unresolved request, not a sequence:

```sh
emergence --token-env EMERGENCE_OPERATOR_TOKEN inference resolve "$SERVICE_REQUEST_ID" \
  --amount 0.004 --reason 'Estimated from confirmed usage and retained rates'
emergence --token-env EMERGENCE_OPERATOR_TOKEN inference resolve "$SERVICE_REQUEST_ID" \
  --no-charge --reason 'Provider confirmed this request incurred no charge'
```

For `POST /api/v1/inference/{request_id}/resolution`, send
`{"amount":"0.004","reason":"Estimated from confirmed usage and retained rates"}`
or `{"amount":null,"reason":"Provider confirmed this request incurred no charge"}`.

Provider exhaustion has a separate recovery operation. After independently
confirming replenishment for the configured provider account, send
`POST /api/v1/providers/{provider}/exhaustion/resolution` with operator
authorization and this body:

```json
{
  "request_id": "0123456789abcdef0123456789abcdef",
  "reason": "Confirmed replenishment for this account in the provider console"
}
```

Use the service-issued request ID of the **current** inference exhaustion
denial, not its idempotency key. IDs must be 32 lowercase hexadecimal characters.
The reason is trimmed, must be nonblank, and is limited to 2,000 characters.
Successful `200` responses contain `provider`, `request_id`,
`resolved_observation_id`, `observation_id`, `audit_event_id`, `observed_at`,
and `fresh_until`. The immutable audit records the authenticated operator role,
timestamp, original observation/request, and redacted reason. Recovery appends
an `inference` observation with `source: operator`,
`confidence: operator_confirmed_recovery`, `unit: availability`, and
`remaining: "1"`. This is a boolean availability assertion, not a monetary
balance or provider-measured credit.

Unknown or disabled providers return `404 provider_not_found`; local providers
return `400 provider_recovery_not_supported`. A wrong, absent, already resolved,
or superseded target returns `409 provider_exhaustion_target_changed`. The
service checks the target again under the database writer, so a concurrent newer
denial cannot be cleared by an older confirmation. A regressed clock that would
rank the recovery before the denial also returns `409`. Invalid bodies return
`422 invalid_request`; the usual `401`/`403` authorization rules apply.

Recovery dispatches no inference and changes no ledger entries, holds, other
resource observations, or safety limits. It does not rewrite the denial.
The assertion is fresh for the provider's configured `stale_after_seconds`;
`require_fresh` blocks once it is stale. Expiration does not reapply the old
denial. After a new independent check, an operator can submit the same original
`request_id`, a new reason, and the current recovery receipt's `observation_id`
to attest availability again. The observation ID must still be current and
linked to that request; each successful confirmation returns a new ID.
Replaying an earlier body cannot extend freshness. A later inference denial
requires a new independent confirmation targeting that denial.
`GET /api/v1/resources` retains both source histories; the newest applicable
observation for the inference resource governs admission.
