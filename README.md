# emergence

> An open-ended experiment in AI autonomy, self-discovery, and self-directed evolution, starting with nothing more than a model, a terminal, and the freedom to build its own future.

The trusted v0.1 core is an operator-controlled background service: a REST
API and CLI for bounded model inference, resource observations, accounting,
and durable request history. It does not run an autonomous agent or provision
guest accounts. See the [charter](docs/Emergence_Project_Charter_and_Roadmap_v0.1_FINAL.md)
for the longer-term experiment.

## Start locally without contacting providers

From the checkout root, with Python 3.12+ and `uv` available:

```sh
uv sync --locked
umask 077
mkdir -p var
env -u EMERGENCE_DATABASE_PATH EMERGENCE_CONFIG="$PWD/config.example.yaml" \
  uv run --locked alembic upgrade head
uv run --locked emergence serve --config config.example.yaml
```

`uv run --locked emergence --config config.example.yaml` is an equivalent
launch form. In a second terminal, run `uv run --locked emergence health`
and `uv run --locked emergence health --live`. Both health checks are public
on the loopback listener; authenticated routes need a role token from a named
environment variable. The example uses checkout-relative `var/` paths, has all
four providers disabled, and ships no model entitlements, credentials, or
prices. Do not enable a provider by copying its placeholders.

- [API and CLI](docs/api.md): roles, requests, histories, and audited resolution.
- [Operations](docs/operations.md): installed paths, provider setup, budgets,
  NAS behavior, backup and restore, offline CI, and the opt-in live test contract.

Packaging and offline evidence do not establish v0.1 completion. Provider
accounts, real NAS, and target LXC operation still require live verification.
