# Emergence — Project Charter & Implementation Roadmap

**Revision:** 3 — v0.1 implementation baseline  
**Date:** 2026-11-20  
**Status:** Approved architectural baseline; implementation pending  
**Project:** Emergence

## 1. Vision

> An open-ended experiment in AI autonomy, self-discovery, and self-directed evolution, starting with nothing more than a model, a terminal, and the freedom to build its own future.

Emergence is a persistent, observable experimental environment in which AI agents may explore, build and run their own software, organize their environments, and potentially cooperate. The infrastructure must support autonomy without prescribing goals, organizational roles, reproduction, or an economic strategy. Unexpected inactivity and failed projects are also results.

**v0.1 is not an agent framework.** It creates the trusted, always-on provider and accounting foundation that subsequent autonomous-agent milestones will use.

## 2. Agreed architectural decisions

| Topic | Decision |
|---|---|
| Protected core | Python 3.12+ service, always running in a dedicated **unprivileged Proxmox LXC** |
| API/UI | FastAPI + Uvicorn; API-first; Typer CLI is an API client; future web frontend; optional Prometheus metrics |
| Python tooling | `uv`, `pydantic`, `pydantic-settings`, `httpx`, SQLAlchemy 2.x, Alembic, `aiosqlite`, pytest, Ruff, Pyright |
| Provider integration | Thin Emergence contracts; vendor SDKs and/or compatible APIs for inference; HTTPX for distinct quota endpoints; PydanticAI may be evaluated later, not required in v0.1 |
| Database | SQLite on local LXC storage; SQLAlchemy/Alembic permit a later PostgreSQL migration (SQL dialect/migrations still need testing) |
| Large files | Protected NAS-backed archive accessible to core via Proxmox-managed mount; do not place active SQLite database on NFS |
| Agent environment | Separate Debian 13 minimal QEMU/KVM VM **201**, not an LXC; agent-created code stays there |
| Agent execution | **Deferred to v0.2:** SSH over a dedicated private Proxmox network with individual unprivileged Linux users and per-agent Ed25519 keys; keys remain in core |
| VM provisioning | First agent may be set up manually; later restricted, narrowly scoped guest account provisioner (no general root credential delegated to agents) |
| Experimental controls | No direct core edits or API keys in guest; operator owns quotas, network policy, snapshots, lifecycle and approvals |

### Deployment details and observed state

- **Emergence Core LXC:** Debian 13; initial allocation **2 vCPU, 2 GiB RAM, 24 GiB local disk**; LAN interface observed as `172.17.2.200` (`eth0`) and private interface `10.10.10.1/24` (`agent-bridge`). More RAM can be allocated if concurrency requires it.
- **Agent VM 201:** Debian 13 (Trixie), QEMU Guest Agent confirmed via `qm agent 201 get-osinfo`; initial allocation **4 vCPU, 8 GiB RAM, 16–24 GiB OS disk**. NAS-backed **64–128 GiB workspace virtual disk** is planned, not asserted as already attached.
- **Private control network:** guest `ens19` is `10.10.10.2/24`; core is `10.10.10.1/24`. Connectivity was verified by ping. This link stays available regardless of experimental internet privileges.
- **Guest `ens18`:** LAN/WAN interface is to be **disabled by default for experiments**. It may be enabled temporarily for maintenance (APT cache, DNS, logging, updates) and then disabled again. Previously configured maintenance destinations were DNS `172.17.0.1`, APT proxy `172.17.2.100`, and Loki `172.17.2.104` with external firewall restrictions; these should *not* be assumed reachable while the interface is disabled.
- **No default gateway for agent experiments.** Validate both IPv4 and IPv6 connectivity and verify Proxmox-side isolation, not just guest interface state. A guest administrator can re-enable an interface; therefore external bridge/firewall controls are the actual containment boundary. The Proxmox management plane, NAS administration, host services and core secrets must remain unreachable.
- **NAS:** repaired and available; dedicated core/archive and VM workspace shares or Proxmox storage definitions are **next setup tasks**. NFS is mounted/managed by Proxmox; the guest gets a virtual block disk and never sees NFS credentials or NAS addresses. Snapshot consistency and NFS storage-format support are to be tested.

These addresses are deployment facts observed during setup, not API defaults; keep them configurable.

## 3. System boundaries

```text
Operator (CLI now, Web UI later)
    | authenticated REST
    v
Emergence Core (unprivileged LXC)
    |- API and settings / credential loading
    |- provider/model registry and inference gateway
    |- provider quota monitor + normalized observations
    |- real usage + virtual budget ledger
    |- SQLite operational state and event metadata
    |- NAS archives + Prometheus endpoint (optional)
    |
    +--> Z.ai / MiniMax / OpenAI APIs
    +--> local llama.cpp inference server
    |
    +-- future AsyncSSH (private network 10.10.10.0/24) --> Agent VM 201
                                                         |- agent user homes
                                                         |- agent-created software
                                                         `- future NAS-backed workspace disk
```

The agent VM is untrusted even if individual accounts are unprivileged. Model switching and operator interventions must be logged as experiment events. Never pass cloud credentials into VM prompts, terminal environments, logs or agent-owned files. SSH host-key verification is mandatory; do not use `known_hosts=None` in production.

## 4. v0.1 provider scope

| Provider | Initial target | Inference | Resource monitoring | Certainty |
|---|---|---|---|---|
| **Z.ai** | GLM-5.3 (API model ID must be confirmed live) | Coding-plan credential; vendor/compatible endpoint | `GET https://api.z.ai/api/monitor/usage/quota/limit` | Endpoint worked in one user test; undocumented schema; experimental parser |
| **MiniMax** | M3 / M3.1 Flash (exact IDs and entitlement must be confirmed live) | Coding-plan key, not PAYG by assumption | `GET https://www.minimax.io/v1/token_plan/remains` | Endpoint identified; live output/schema not yet confirmed |
| **OpenAI** | Configurable model available to the Emergence project | Dedicated service-account inference key | Organization usage and cost endpoints with restricted Admin API key | Reporting available; exact remaining subscription-credit balance may not be exposed |
| **Local** | Configurable llama.cpp GGUF model | OpenAI-compatible llama.cpp API | Request statistics; llama.cpp metrics; optional later Prometheus/DCGM telemetry | No monetary per-token fee, but bounded compute |

Providers, models and agents are distinct identities; adding a new provider or model should require configuration when the protocol is already supported. Do not silently fallback across models: model identity is an experimental variable.

### Credential requirements

- Z.ai and MiniMax coding-plan keys are independently configured and tested against their intended subscription endpoint. Never assume a coding key is equivalent to a pay-as-you-go key.
- OpenAI is in a dedicated Emergence organization/project, with a **project service-account inference key** and a **separate restricted Admin reporting key** (usage/audit read only, 90-day expiration as configured). Validate both keys point to the intended organization; alert on expiration.
- OpenAI UI was configured with **auto-reload disabled** and a **$100 hard spending limit**. Do not treat either as instantaneous or perfect enforcement; retain Emergence's local guardrail. Confirm credit assignment/renewal against the new organization; do not infer it from older screenshots.
- Environment variables or a secret store hold keys, never YAML, git, DB event rows, exception traces or agent environments.

### Quota normalization

Each quota observation includes `provider`, `resource`, `unit`, `used`, `limit`, `remaining`, `usage_percent`, `window`, `resets_at`, `observed_at`, `source`, `confidence`, `is_stale`, and `raw_payload_ref` where available. Unknown numeric values stay `null`, never 0 or infinity. Status: `verified`, `observed`, `experimental`, `unavailable`.

Z.ai's observed response included a `TIME_LIMIT` category with `search-prime`, `web-reader`, `zread` MCP-tool usage and a `TOKENS_LIMIT` percentage entry. Do **not** infer the meaning of `unit` and `number`, an absolute token ceiling, or remaining model calls without corroboration. Preserve raw payloads with secrets scrubbed.

## 5. Financial accounting and budgets — DECIDED

### Monthly provider budgets

- **Cloud budgets:** **USD $100 per provider** for Z.ai, MiniMax, and OpenAI. A provider's budget is shared by all agents using it, **not per agent**.
- **Cycle:** calendar month, beginning on the first day and ending at the next month's start, **no rollover**. Implement calendar boundaries using a configurable accounting timezone; default **UTC** until the operator explicitly chooses another. This timezone is an implementation default, not an assertion about OpenAI's billing cycle.
- **Z.ai and MiniMax:** simulated pay-as-you-go charges using versioned price tables and actual request token figures (and cache/tool fees when applicable). Track real subscription quotas separately.
- **OpenAI:** actual reportable API costs, distinguished from local estimates; compare to an independently configured local safety limit (initial **$95** suggested). A newly started virtual month **does not** assert that actual OpenAI credits renewed; any real-credit shortage blocks dispatch regardless of virtual balance.
- **Local llama.cpp:** no dollar budget initially. Track request count, tokens, latency and future GPU energy/compute measures separately; no claim of unlimited throughput.
- **Budget adjustments:** operator-only, append-only grant/reduction/correction events with reason and timestamp. Agent requests for additional funding are a later feature and never automatically purchase services.

### Ledger and enforcement

- Use append-only `usage_transactions`, `budget_adjustments`, `budget_periods`, and `request_reservations`; every entry has provider, model, request ID and optional future agent/experiment ID.
- Reserve a configurable upper-bound cost **before** dispatch under a concurrency-safe transactional lock, reconcile actual usage on completion, and release unused reservations. Reject requests when budget or actual quota prevents safe dispatch. Treat missing/stale quota information as unknown; choose an explicit configurable policy rather than assuming unlimited capacity.
- Provider pricing is versioned by effective date/model; support input, output, cached tokens, and provider-specific extras. Use exact decimal currency values, not binary floating point.
- Record **measured provider costs**, **estimated local costs**, and **virtual charges** as separate fields. Monthly reset creates a new budget period; it never destroys transaction history.
- Quota reports and costs from a provider may lag. The local guardrail reduces risk but cannot guarantee zero overage; keep request-level maximum output token budgets and concurrency caps.
- Initial rollout may use shadow accounting first, then turn on local virtual-budget enforcement after fixture and live-token tests validate pricing.

## 6. Retention and archival — DECIDED

**Preserve full conversations and complete execution histories indefinitely**, with large data archived to NAS. This is the research default, subject to necessary deletion/redaction of secrets, personal information, legal requirements, and operator-controlled purges.

- **Local SQLite:** structured event metadata, request and pricing accounting, file references, content checksums, agent and experiment identities, timestamps; no huge blobs by default.
- **NAS archive:** complete model input/output payloads and (in later phases) terminal transcripts, streamed tool output, artifacts and environment snapshots. Use immutable/append-only conventions where practical; periodic integrity verification and backups.
- **Quota snapshots:** retain raw observations (or archive them) plus queryable aggregates. Downsampling must not silently destroy the original research record.
- **Safety:** redact API keys and other secrets before archiving; access control, encryption where appropriate, storage-capacity alerts and explicit retention exceptions. "Indefinite" means no automatic age-based deletion, **not** infinite storage.
- **Availability:** NAS outage must not corrupt transactions. Local metadata continues; archival payloads queue safely with bounded backpressure and visible failure state. No silently dropped transcripts.

## 7. v0.1 implementation boundary

### Required service capabilities

1. FastAPI app, startup/shutdown, liveness/readiness endpoints, configuration validation and structured logs.
2. Pluggable provider registry with provider-specific model configuration and error handling, including inference, usage, and limits as separate contracts (explicit `unsupported` where necessary).
3. Unified inference (including configurable token limit and optional streaming), preserving model/provider-specific features and raw usage metadata.
4. Background quota polling with bounded concurrency, jitter/backoff and configurable poll intervals; graceful partial provider outages.
5. SQLAlchemy data layer with Alembic migrations, SQLite operational DB, append-only ledger and NAS archive interface.
6. $100 monthly provider budgets, simulated billing, budget reservations, OpenAI safety cap, and operator-only adjustments.
7. REST API with authentication and role separation, explicit `/api/v1` routes, normalized health/status, models, usage, quota observations, budgets and bounded test inference.
8. Typer CLI using the REST API (not directly using the database or service internals).
9. Prometheus `/metrics` endpoint as an optional feature; do not require the existing monitoring stack for service availability.
10. Automated offline tests, opt-in live smoke tests, operator README, example config, systemd installation instructions for unprivileged LXC, and API documentation.

### Suggested routes

| Method | Route | Purpose |
|---|---|---|
| GET | `/health/live` | Process alive, no provider dependency |
| GET | `/health/ready` | Service and DB ready (partial provider outages listed separately) |
| GET | `/api/v1/providers` | Connection/capability status |
| GET | `/api/v1/models` | Configured model IDs/aliases |
| GET | `/api/v1/resources` | Normalized quota observations with freshness/provenance |
| GET | `/api/v1/usage` | Historical usage; filterable by dates/provider/model |
| GET | `/api/v1/budgets` | Period balances, reservations, real vs simulated |
| POST | `/api/v1/budgets/adjustments` | Operator-only audited change |
| POST | `/api/v1/inference` | Authenticated, rate-limited inference; not guest-accessible |
| GET | `/metrics` | Optional Prometheus exposition, protected or restricted at network layer |

Do not expose admin or inference routes on the private agent bridge. Initial API binding can be `127.0.0.1`; explicitly configure operator management access later. CORS is not a substitute for authentication.

### Proposed package structure

```text
emergence/
├── pyproject.toml
├── uv.lock
├── README.md
├── config.example.yaml
├── alembic.ini
├── migrations/
├── src/emergence/
│   ├── api/{app,dependencies,routes}.py
│   ├── cli.py
│   ├── config.py
│   ├── providers/{base,zai,minimax,openai,llamacpp,registry}.py
│   ├── resources/{schemas,quota_polling,pricing,budgets,ledger}.py
│   ├── storage/{database,models,repositories,archive}.py
│   ├── inference/{schemas,gateway}.py
│   └── telemetry/{logging,metrics}.py
└── tests/{unit,integration,fixtures}/
```

The directory organization is a suggestion, not an obligation to create empty abstraction layers.

## 8. Ordered implementation tickets and exit tests

| Ticket | Deliverable | Acceptance test |
|---|---|---|
| **E-001** | `uv` project, FastAPI service, Pydantic settings, lifecycle/health, Ruff/Pyright/pytest | Fresh clone runs via `uv sync`; liveness and readiness tests pass |
| **E-002** | SQLAlchemy/Alembic SQLite tables and repositories | Migrate empty DB, persist data across restart, migration upgrade test |
| **E-003** | Provider contracts, registry, inference adapters for all four targets | Fixture-backed successful/failure responses; opt-in minimum-cost smoke calls |
| **E-004** | Provider quota monitors, Z.ai experimental parser, MiniMax parser, OpenAI usage/cost, local metrics | No secrets in logs; unknown/stale data represented accurately; outages degrade gracefully |
| **E-005** | Versioned cost models, calendar-month budget periods, durable reservations and reconciliations | Concurrency test cannot overspend virtual budget; month boundary resets without rollover |
| **E-006** | Authenticated REST resources/inference/budget endpoints and API-first Typer CLI | CLI operates against separate running service; unauthorized write/inference denied |
| **E-007** | NAS archive backend, durable spool/backpressure, Prometheus metrics, retention protections | NAS unavailable/recovered test preserves accounting and reports archive backlog |
| **E-008** | systemd deployment docs, example settings, tests, final integration checklist | Full offline CI; explicit opt-in live four-provider smoke report and redaction check |

### Required verification before marking v0.1 complete

- Correct live **model IDs** and coding-plan API compatibility for Z.ai/MiniMax; failures documented, not disguised.
- MiniMax quota response inspected with redacted fixture; Z.ai undocumented fields remain experimental until observed repeatedly.
- OpenAI Admin API key can read intended organization usage/cost reporting; inference key independently works; credit assignment and spending policies verified in the actual account UI.
- Local llama.cpp reachable from LXC and handles a bounded inference request.
- Fixture tests cover timeouts, non-JSON responses, rate limits, unauthorized keys, schema drift, partial outage, stale quota, negative/unknown balances, crashes during budget reconciliation, concurrent requests, and calendar-month rollover.
- No secrets stored in archived full prompts, logs, or provider errors. Real calls are opt-in, budget-capped, and not part of standard test runs.

**Exit criterion:** After restart, a running Emergence background service can serve an authenticated CLI and API, test each configured provider, show honest quota freshness and monthly budget balances, archive history safely, and reject new requests beyond its local limits. **No autonomous agent is running yet.**

## 9. Later milestones (intentionally outside v0.1)

| Phase | Deliverable |
|---|---|
| **v0.2 — First agent** | Persistent minimal observe/terminal/wait loop; AsyncSSH over `ens19`; one unprivileged guest user and key; basic context/state and event history; pause/resume |
| **v0.3 — Hardened experiment world** | Automated restricted user provisioning, VM isolation tests, per-user quotas, snapshots/recovery, PTY sessions, tamper-evident logs and resource limits |
| **v0.4 — Multiple agents** | Independent guest users and home directories, optional shared workspace, agent messages, collaboration observations |
| **v0.5 — Experimental reproducibility** | Versioned initial environments, model swaps/new-provider trials, repeatable baselines and intervention timelines, comparative analysis |
| **Later — Environmental stages** | Initially isolated VM; later curated LAN services and SearXNG; later filtered web access; operator-approved capability and budget expansion; possible agent-requested creation of other agents |

Agents may inspect an exported copy of protected source code and propose patches, but cannot directly modify the running core. Later economic experiments can measure genuinely realized revenue against provider and infrastructure costs, not just self-reported claims. The core never executes purchases automatically.

## 10. Remaining questions (not blockers for v0.1)

- Exact configured model IDs, provider pricing and rate-limit details: **integration facts to confirm during E-003/E-004**.
- Precise OpenAI credit replenishment day: **verify separately from the UTC calendar-month virtual budget**.
- NAS share names, mount paths, backup policy and storage quotas: **operator deployment choices**, now that NAS is fixed.
- Budget accounting timezone: **UTC initial default**, configurable; operator can later select a different zone.
- Agent wakeups and waiting semantics; shared workspace rules; autonomous permissions; future LAN/Internet progression: **v0.2+ design questions**.
- GPU/DCGM exporter setup and Grafana dashboard: **optional later telemetry integration**.

No additional conceptual planning decisions are necessary to start E-001.

## 11. Coding agent handoff

> Implement **Emergence v0.1 only**, using this document as the source of truth. Begin with E-001 and progress through E-008 in order. Build the API-first background service in Python using `uv`, FastAPI, Pydantic, SQLAlchemy/Alembic, and SQLite. Use thin provider adapters with fixture-driven tests, separate quota reporting from inference and real costs from simulated charges. Honor monthly shared $100 cloud budgets without rollover, preserve full histories with NAS archival, and never put credentials in guest environments or persistent logs. Treat unverified model IDs and undocumented quota fields as experimental, not guaranteed. Do not build the autonomous runtime, SSH provisioning, agent creation, a web dashboard or artificial social/economic systems in v0.1. Present tests and any unresolved integration assumptions for operator review.
