---
name: ml-security-assurance
description: Senior ML Security Engineering for Kilat — OWASP Top 10 for LLM Applications (2025), OWASP Machine Learning Security Top 10, OWASP Top 10 (2021), and NCSC/CISA Guidelines for Secure AI System Development mapped onto concrete Kilat controls. Covers model/checkpoint deserialization, Hub supply-chain pinning, dynamic-import allowlists, path traversal in shard indexes, Gate 0 (G0_SECURITY_SUPPLY_CHAIN), SAST, dependency auditing, and CI hardening.
---

# ML Security Assurance Skill (OWASP + NCSC)

## 1. When to Activate This Skill
Activate whenever a change touches any **trust boundary** — anything that reads bytes Kilat did not produce in this process:
- Loading checkpoints, weights, shards, tokenizers, configs (`from_pretrained`, `load_checkpoint`, `AutoTokenizer`, `export_hf`).
- Downloading from the Hugging Face Hub or any remote source.
- Dynamic import (`importlib`, `custom_builder`, `custom_module`, plugin registries).
- Spawning processes, writing files from user-supplied paths, or adding dependencies / CI steps.
- Datasets from outside the repo (Parquet / JSONL / memmap ingestion).

> **Mandatory Cross-Reference**: `docs/04_ASSURANCE_AND_CHAOS_ENGINEERING_PLAN.md` (Gate 0), `SECURITY.md` (threat model), and the `assurance-chaos-engineering` skill (meta-assertion contract).

---

## 2. Threat Model (what an attacker controls)

| Asset an attacker can supply | Attack | Primary Kilat control |
| :--- | :--- | :--- |
| `pytorch_model.bin`, `training_state.pt`, shard `.bin` | Pickle RCE on load | `torch.load(..., weights_only=True)` everywhere; safetensors preferred |
| `*.index.json` `weight_map` | Path traversal (`../../x`) reading files outside the checkpoint | `kilat.utils.security.safe_join` |
| `tokenizer_config.json`, `byte_tokenizer_meta.json`, `config.yaml` in a checkpoint | `custom_builder: "os:system"` → import + call arbitrary function | `assert_trusted_module` allowlist (`kilat.*` + `KILAT_TRUSTED_MODULES`), re-raised by `AutoTokenizer` (no silent fallback) |
| Hub repo name / branch | Upstream swaps weights under the same tag (poisoning) | `revision=<40-char SHA>`; `enforce_revision_policy`; `KILAT_REQUIRE_PINNED_REVISION=1` |
| `*.py` next to exported weights | `trust_remote_code=True` executes it | Banned in `src/` (Gate 0 rule SEC-RCE-1); export verification reloads via `KilatTransformer.from_pretrained` |
| PyPI dependency | Known-vulnerable or typosquatted package | `pip-audit` in CI, Dependabot, actions pinned by SHA |
| Training corpus | Data poisoning / train–val contamination | Gate 4 data forensics (n-gram leak, memmap SHA-256 manifest) |

---

## 3. Framework → Control Mapping

### 3.1 OWASP Top 10 for LLM Applications (2025)
| ID | Risk | Kilat control (file) |
| :--- | :--- | :--- |
| **LLM03** | Supply Chain | `utils/security.py` (allowlist, `safe_join`, revision pinning); `models/base.py::enforce_revision_policy`; Gate 0; `pip-audit`; SHA-pinned actions |
| **LLM04** | Data & Model Poisoning | Pinned Hub revisions; Gate 4 contamination + memmap SHA-256; checkpoint manifest SHA-256 (`engine/checkpoint.py::verify_checkpoint_manifest`) |
| **LLM05** | Improper Output Handling | Out of scope for the training library; downstream apps must treat `TextGenerator` output as untrusted |
| **LLM10** | Unbounded Consumption | `GenerationConfig.max_new_tokens` / stopping criteria; never default to unbounded generation in CLI |

### 3.2 OWASP Machine Learning Security Top 10
| ID | Risk | Kilat control |
| :--- | :--- | :--- |
| **ML02** | Data Poisoning | Gate 4 forensics; dataset provenance via `.meta.json` SHA-256 |
| **ML06** | AI Supply Chain Attacks | Same as LLM03; Gate 0 rules SEC-DESER-*, SEC-RCE-1, SEC-IMPORT-1 |
| **ML10** | Model Poisoning | `weights_only=True`, safetensors, pinned revisions, checkpoint manifests |

### 3.3 OWASP Top 10 (2021) — classic web/app risks that still apply
| ID | Kilat control |
| :--- | :--- |
| **A01** Broken Access Control (path traversal) | `safe_join` on every untrusted relative path |
| **A03** Injection | No `shell=True`, no `os.system`, no `eval/exec` (Gate 0 SEC-SHELL-1, SEC-RCE-2); `subprocess` with fixed argv and absolute executable (`shutil.which`) |
| **A06** Vulnerable & Outdated Components | `pip-audit`, Dependabot |
| **A08** Software & Data Integrity Failures | `weights_only=True`, `yaml.safe_load`, no pickle/`allow_pickle=True`, SHA-256 manifests |

### 3.4 NCSC / CISA — Guidelines for Secure AI System Development (4 phases)
| Phase | Kilat practice |
| :--- | :--- |
| **1. Secure design** | Threat model in `SECURITY.md`; treat every checkpoint/config as untrusted input; secure-by-default flags (no remote code, allowlisted imports) |
| **2. Secure development** | Supply-chain controls (pinning, audit, allowlists); asset tracking (SHA-256 manifests, `config_sha256` + `git_commit` in assurance report); SAST (ruff `S`); Gate 0 in `kilat verify` |
| **3. Secure deployment** | Protect model artefacts from tampering (manifests, safetensors, pinned revisions); least-privilege CI (`permissions: contents: read`, `persist-credentials: false`) |
| **4. Secure operation & maintenance** | Dependabot + `pip-audit` on every PR; `FlightRecorderCallback` runtime telemetry; vulnerability disclosure via `SECURITY.md` |

---

## 4. Hard Rules (enforced by Gate 0, ruff `S`, and tests)

1. **Never** call `torch.load` without the literal `weights_only=True`. Never use `pickle`, `dill`, `cloudpickle`, `joblib.load`, or `numpy.load(allow_pickle=True)` on artefacts.
2. **Never** use `yaml.load` / `full_load` / `unsafe_load`; only `yaml.safe_load`.
3. **Never** pass `trust_remote_code=True` anywhere in `src/`.
4. **Never** `importlib.import_module(x)` where `x` originates from a file. Call `kilat.utils.security.assert_trusted_module(x)` first.
5. **Never** join an untrusted relative path with `/`. Use `safe_join(root, rel)`.
6. **Never** download from the Hub without threading a `revision` through; production runs set `KILAT_REQUIRE_PINNED_REVISION=1`.
7. **Never** swallow a security exception in a broad `except Exception` fallback chain. Re-raise `UntrustedModuleError` / `UnsafePathError` / `UnpinnedRevisionError` explicitly before the generic handler.
8. **Never** use `assert` for checks inside `src/kilat/assurance/` (stripped under `python -O`); raise explicitly.
9. `subprocess`: fixed argv list, absolute executable via `shutil.which`, no `shell=True`, bounded `timeout`.
10. CI: third-party actions pinned by commit SHA; top-level `permissions: contents: read`.

---

## 5. Gate 0 — `G0_SECURITY_SUPPLY_CHAIN`

Implemented in `src/kilat/assurance/gates/gate0_security.py`, runs **first** in `kilat verify`.

| Rule | Detects | Failure code |
| :--- | :--- | :--- |
| SEC-DESER-1..4 | unsafe `torch.load`, pickle family, unsafe YAML, `allow_pickle=True` | `ERR_INSECURE_DESERIALIZATION` |
| SEC-RCE-1 / 2 | `trust_remote_code=True`, builtin `eval`/`exec` | `ERR_REMOTE_CODE_EXECUTION` |
| SEC-SHELL-1 | `shell=True`, `os.system`, `os.popen` | `ERR_SHELL_INJECTION_RISK` |
| SEC-IMPORT-1 | tokenizer `custom_builder`/`custom_module` outside allowlist | `ERR_UNTRUSTED_DYNAMIC_IMPORT` |

**Meta-assertion contract**: each rule has a planted-sink test in `tests/unit/assurance/test_gate0_security.py` that must make the gate FAIL; a vacuous scan (0 files) also FAILS.

---

## 6. Verification Protocol
1. `make check` — ruff (incl. `S` SAST) + mypy: 0 errors.
2. `make security` — ruff `S` + Gate 0 PASS on the Kilat package.
3. `make test` — includes `tests/unit/utils/test_security.py`, `tests/unit/assurance/test_gate0_security.py`, and `tests/integration/architecture/test_architecture_layering.py`.
4. `kilat verify --config configs/debug_cpu.yaml` — `G0_SECURITY_SUPPLY_CHAIN` PASS before G1..G6.
5. CI `pip-audit` reports no known vulnerabilities (or each is documented with a justification).
