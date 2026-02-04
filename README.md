# Project Q-Secure: PhD Research Repository

## Hybrid Post-Quantum Cryptography for Embedded Systems

This repository contains the research materials, implementation, and documentation for a self-directed PhD research program focused on post-quantum cryptography.

---

## Repository Structure

```
phd/
├── README.md                    # This file - navigation guide
├── PhD_Research_Plan.md         # Practical roadmap & task tracking
├── PhD_Expert_Curriculum.md     # Deep academic curriculum
├── src/                         # Implementation (coming soon)
├── docs/                        # Additional documentation
├── papers/                      # Literature & notes
└── experiments/                 # Benchmarks & results
```

---

## Document Guide

### 1. [PhD_Research_Plan.md](./PhD_Research_Plan.md)
**Purpose:** Practical roadmap and progress tracking

**Use this for:**
- Daily/weekly task checklists
- Study routine and time management
- Progress tracking with checkboxes
- Quick reference of monthly goals
- Context management between sessions
- Resource links and tool lists

**Best for:** Day-to-day execution and staying on track

---

### 2. [PhD_Expert_Curriculum.md](./PhD_Expert_Curriculum.md)
**Purpose:** Deep academic and technical curriculum

**Use this for:**
- Mathematical foundations (proofs, theorems, equations)
- Formal security definitions (IND-CCA2, LWE reductions)
- Code examples and implementation patterns
- Research methodology (hypotheses, research questions)
- Publication strategy and thesis structure
- Advanced reading lists with specific papers

**Best for:** Deep learning sessions and understanding the "why"

---

## Quick Comparison

| Aspect | Research Plan | Expert Curriculum |
|--------|---------------|-------------------|
| **Focus** | What to do | How & why |
| **Depth** | Overview | Deep dive |
| **Format** | Checklists | Explanations |
| **Math** | Topics listed | Formulas included |
| **Code** | Tool names | Code snippets |
| **Length** | ~400 lines | ~1200 lines |
| **Use case** | Daily tracking | Study sessions |

---

## Research Overview

### Title
**Design and Implementation of Hybrid Post-Quantum Cryptographic Primitives with Provable Security Guarantees for Resource-Constrained Embedded Systems**

### Timeline
| Quarter | Focus | Months |
|---------|-------|--------|
| Q1 | Mathematical Foundations | 1-3 |
| Q2 | Rust & Embedded Systems | 4-6 |
| Q3 | Hybrid Protocol & Security | 7-9 |
| Q4 | Analysis & Publication | 10-12 |

### Key Technologies
- **Language:** Rust (memory-safe, no_std for embedded)
- **Cryptography:** ML-KEM (Kyber), ML-DSA (Dilithium)
- **Hardware:** ARM Cortex-M4, ESP32, RISC-V
- **Tools:** Kani (verification), ChipWhisperer (side-channel)

---

## Getting Started

### Week 1 Checklist
- [ ] Read this README
- [ ] Review [PhD_Research_Plan.md](./PhD_Research_Plan.md) for schedule
- [ ] Study [PhD_Expert_Curriculum.md](./PhD_Expert_Curriculum.md) Part II (Math)
- [ ] Set up Rust environment: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- [ ] Create study notes in `docs/` folder

### Daily Routine
```
05:30-07:00  Deep study (Expert Curriculum - theory)
08:00-09:30  Code practice (implementation)
10:00-11:00  Log progress (Research Plan checkboxes)
```

---

## Session Context Template

Copy this when starting a new AI conversation:

```
Project: Q-Secure PhD
Quarter: Q[1-4]
Month: [1-12]
Week: [1-4]
Current Topic: [from curriculum]
Last Completed: [task]
Current Focus: [task]
Blockers: [any issues]
```

---

## Progress Dashboard

### Quarterly Status
- [ ] Q1: Mathematical Foundations (Months 1-3)
- [ ] Q2: Implementation Foundations (Months 4-6)
- [ ] Q3: Novel Research (Months 7-9)
- [ ] Q4: Publication & Release (Months 10-12)

### Deliverables
- [ ] Literature Review (Month 3)
- [ ] Working Kyber Implementation (Month 6)
- [ ] Hybrid Protocol Design (Month 7)
- [ ] Security Validation (Month 9)
- [ ] Benchmarking Report (Month 10)
- [ ] Dissertation Draft (Month 11)
- [ ] Conference Paper Submission (Month 12)
- [ ] Open Source Release (Month 12)

---

## Links & Resources

### Official Standards
- [NIST FIPS 203 (ML-KEM)](https://csrc.nist.gov/pubs/fips/203/final)
- [NIST FIPS 204 (ML-DSA)](https://csrc.nist.gov/pubs/fips/204/final)

### Key Libraries
- [pqcrypto](https://crates.io/crates/pqcrypto) - PQC for Rust
- [pqm4](https://github.com/mupq/pqm4) - Reference embedded implementations

### Learning Resources
- [ArXiv Cryptography](https://arxiv.org/list/cs.CR/recent)
- [IACR ePrint](https://eprint.iacr.org/)

---

## Author

**Md Ferdous Alam**
- GitHub: [@mdferdousalam](https://github.com/mdferdousalam)
- Research: Post-Quantum Cryptography, Embedded Security, Rust

---

## License

This research repository is for educational and research purposes.

---

*Last Updated: February 2026*
