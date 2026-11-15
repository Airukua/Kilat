# Security Policy

The **Kilat** project prioritizes the security, supply-chain integrity, and mathematical safety of our machine learning training and inference framework. Because ML pipelines operate on complex serialized artifacts, large datasets, and remote model hubs, we treat **all external artifacts and configurations as untrusted inputs**.

---

## Reporting a Vulnerability

Please **do not** open public GitHub issues or discussions for undisclosed security vulnerabilities.

### Secure Reporting Channels:
1. **GitHub Private Vulnerability Reporting (Preferred)**:
   Navigate to the repository: **Security → Report a vulnerability**. This allows secure, private collaboration and coordinated disclosure.
2. **Maintainer Security Email**:
   Email: **Abdul Wahid Rukua** ([rukuaabdulwahid@gmail.com](mailto:rukuaabdulwahid@gmail.com))
   Subject: `[SECURITY] Kilat Vulnerability Report - <Brief Summary>`

### Response Commitments & SLA:
* **Initial Acknowledgment**: Within **48 hours**.
* **Triage & Severity Assessment**: Within **5 business days**.
* **Remediation & Patch Release**: Coordinated with the reporter prior to public CVE disclosure.

Please include:
* Affected version, commit hash, or environment.
* Clear step-by-step reproduction instructions or proof-of-concept (PoC).
* Potential security impact (e.g., arbitrary code execution, path traversal, data/gradient tampering).

---

## Supported Versions

| Version | Status | Security Support |
| :--- | :---: | :--- |
| **2.0.x (`main`)** | Active | ✅ Supported with active security audits and patches |
| Flagship Model (`150M-Kilat-SMALL-10B`) | Active | ✅ Supported (Safetensors + pinned Hugging Face artifacts) |
| < 2.0.0 | Deprecated | ❌ Unsupported; please upgrade to Kilat v2.0 |

---

## Threat Model & Automated Mitigations

Kilat is a foundational training and inference framework. Its primary attack surfaces involve **deserializing external artifacts, tokenizers, checkpoints, remote repository downloads, and configuration files**.

The table below outlines our threat vectors, corresponding risks, and enforced architectural mitigations:

| Untrusted Input / Vector | Threat Risk | Enforced Mitigation in Kilat | Rule ID |
| :--- | :--- | :--- | :--- |
| **Pickle-based weights** (`*.bin`, `*.pt`) | Arbitrary Code Execution (RCE) via `pickle` | Strict enforcement of `torch.load(..., weights_only=True)`; mandatory `safetensors` format for production | `SEC-DESER-1`<br>`SEC-DESER-2` |
| **YAML / JSON configurations** | YAML object instantiation RCE | Strict use of `yaml.safe_load()` across all loaders; raw `yaml.load()` is strictly prohibited | `SEC-DESER-3` |
| **Checkpoint `weight_map` index** | Arbitrary file write / Path Traversal (OWASP A01) | Canonical path sanitization via `kilat.utils.security.safe_join()`; rejects absolute paths, `..`, and NUL bytes | `SEC-PATH-1` |
| **Custom Tokenizer Builders** | Arbitrary dynamic code execution (OWASP LLM03) | Strict import allowlist (`kilat.*` + `KILAT_TRUSTED_MODULES`). Arbitrary module prefixes are rejected | `SEC-TOKEN-1` |
| **Hugging Face Hub Repositories** | Upstream repository hijacking & supply-chain tampering | Support for immutable commit SHA pinning (`revision=<40-hex SHA>`); enforced via `KILAT_REQUIRE_PINNED_REVISION=1` | `SEC-HUB-1` |
| **Remote Code Execution in Core** | Supply-chain compromise via uninspected remote code | Literal `trust_remote_code=True` is prohibited within core library `src/`; remote code requires explicit opt-in | `SEC-RCE-1`<br>`SEC-RCE-2` |
| **Shell & Subprocess Invocations** | Command injection via unsanitized strings | `shell=True` is forbidden in `subprocess` calls; arguments must be tokenized lists | `SEC-SHELL-1` |
| **Cache & LoRA State Tampering** | Silent gradient corruption or state drift | Invariant validation & chaos testing (`HFC-01`..`HFC-08`) preventing dynamic cache mismatches | `INV-LORA-1` |

---

## Automated Security Verification (`make security`)

Kilat includes built-in automated static and dynamic security scanners:

### 1. Gate 0 Security & Supply-Chain Audit
Every pre-training run and pull request must pass **Gate 0 (`G0_SECURITY_SUPPLY_CHAIN`)**:
```bash
make security
```
This executes:
1. **SAST Bandit Scan (`ruff check --select S src`)**: Checks for insecure crypto, subprocess misuse, hardcoded credentials, and bad serialization idioms.
2. **Gate 0 AST Auditor (`kilat.assurance.gates.run_gate0_security`)**: An Abstract Syntax Tree scanner that verifies zero unsafe deserialization sinks, zero unverified remote code invocations, and zero path traversal vulnerabilities across all source files.

### 2. Full Invariant Test Suite
Run the dedicated security invariant tests:
```bash
pytest tests/unit/assurance/test_gate0_security.py tests/unit/utils/test_security.py -v
```

### 3. Pre-Flight Assurance Integration
Gate 0 runs as the mandatory initial step of Kilat's 6-Gate Pre-Flight verification:
```bash
kilat verify --config configs/laptop_cpu.yaml
```

---

## Operator Hardening Checklist

When deploying Kilat in production training clusters, fine-tuning environments, or public inference servers, operators should follow these hardening practices:

- [x] **Enforce Hub Revision Pinning**: Set `export KILAT_REQUIRE_PINNED_REVISION=1` in your environment to prevent silent model weight substitution attacks.
- [x] **Restrict Trusted Modules**: Never set `KILAT_TRUSTED_MODULES=*`. Only specify explicitly verified first-party package prefixes.
- [x] **Use Safetensors Exclusively**: When exporting or publishing weights, always use `use_safetensors=True` (`model.safetensors`), which prevents arbitrary code execution vulnerabilities native to Python pickle.
- [x] **Validate Checkpoint Manifests**: Verify SHA-256 hashes generated in `checkpoint_manifest.json` before loading weights on remote cluster nodes.
- [x] **Isolate Environment Secrets**: Never hardcode API tokens or credentials (`HF_TOKEN`, `WANDB_API_KEY`) in scripts or configs; always inject them via environment variables or secret managers.
- [x] **Run Chaos Pre-Flight**: Run `kilat chaos -s all` and `kilat chaos -s hf` to verify resilience against memory leaks, state corruption, and adversarial inputs.

---

## Compliance & Standards Alignment

Kilat's security policies and assurance gates are formally mapped to industry standards:

- **OWASP Top 10 for Large Language Model Applications (2025)**:
  - *LLM03: Supply Chain Vulnerabilities* — Mitigated via Hub revision pinning and strict dependency audits.
  - *LLM05: Improper Output Handling* — Mitigated via generation bounds and causal mask isolation.
  - *LLM08: Vector and Embedding Weaknesses* — Mitigated via numerical stability checks and gradient sanity gates.
- **OWASP Machine Learning Security Top 10 (ML01–ML10)**:
  - *ML06: AI Supply Chain Attacks* — Mitigated via Gate 0 AST scanner and tokenizer allowlists.
  - *ML09: Model Inversion & Data Poisoning* — Mitigated via causal loss masking (`-100`) and document boundary isolation.
- **NCSC / CISA Guidelines for Secure AI System Development**:
  - Principles of secure design, secure development, secure deployment, and continuous verification.

---

*For additional technical details regarding security specifications and invariant derivations, refer to [.agents/skills/core/ml-security-assurance/SKILL.md](.agents/skills/core/ml-security-assurance/SKILL.md).*
