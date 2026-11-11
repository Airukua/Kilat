# Security Policy

## Reporting a Vulnerability

Please **do not** open a public issue for security problems. Report privately through GitHub's
**Security → Report a vulnerability** (private vulnerability reporting) on this repository.
Include affected version/commit, a minimal reproduction, and impact. We aim to acknowledge
reports within 7 days.

## Supported Versions

| Version | Supported |
| :--- | :--- |
| 2.x (`main`) | ✅ |
| < 2.0 | ❌ |

## Threat Model

Kilat is a training and inference library. Its main attack surface is **loading artefacts that
someone else produced**: checkpoints, weight shards, tokenizer/configuration files, Hub
snapshots, datasets, and third-party Python packages. All of these are treated as untrusted.

| Untrusted input | Risk | Mitigation |
| :--- | :--- | :--- |
| Pickle-based weights (`*.bin`, `*.pt`) | Arbitrary code execution | `torch.load(weights_only=True)` only; safetensors preferred |
| Shard index `weight_map` | Path traversal | `kilat.utils.security.safe_join` |
| Tokenizer config inside a checkpoint | Import and call of arbitrary functions | Module allowlist (`kilat.*` + `KILAT_TRUSTED_MODULES`) |
| Hugging Face Hub repos | Upstream tampering / poisoning | `revision=<commit SHA>`; `KILAT_REQUIRE_PINNED_REVISION=1` enforces it |
| Remote model code | Code execution | `trust_remote_code=True` is banned in `src/` |
| Dependencies / CI actions | Vulnerable or compromised components | `pip-audit`, Dependabot, SHA-pinned GitHub Actions, least-privilege `GITHUB_TOKEN` |

These controls are verified automatically by **Gate 0 (`G0_SECURITY_SUPPLY_CHAIN`)** in
`kilat verify` and by `make security`.

## Operator Hardening Checklist

- Set `KILAT_REQUIRE_PINNED_REVISION=1` for production training and inference.
- Only add your own packages to `KILAT_TRUSTED_MODULES` (comma-separated module prefixes).
- Prefer `model.safetensors` over `pytorch_model.bin` when publishing checkpoints.
- Verify `checkpoint_manifest.json` SHA-256 digests before resuming from a copied checkpoint.

## Frameworks

Controls are mapped to the OWASP Top 10 for LLM Applications (2025), the OWASP Machine Learning
Security Top 10, the OWASP Top 10 (2021), and the NCSC/CISA *Guidelines for Secure AI System
Development*. See `.agents/skills/core/ml-security-assurance/SKILL.md` for the full mapping.
