# Project Q-Secure: Doctoral Research Program

## Formal Research Title
**Design and Implementation of Hybrid Post-Quantum Cryptographic Primitives with Provable Security Guarantees for Resource-Constrained Embedded Systems**

## Research Abstract
This research investigates the design, implementation, and formal verification of hybrid cryptographic schemes combining classical elliptic curve cryptography with lattice-based post-quantum primitives, specifically targeting ARM Cortex-M class microcontrollers. The work addresses the critical challenge of achieving quantum resistance while maintaining computational efficiency within severe memory and power constraints typical of IoT devices. Novel contributions include optimized Number Theoretic Transform implementations in Rust, side-channel resistant masking schemes, and formal security proofs under the Module-LWE hardness assumption.

---

# Part I: Research Framework

## 1.1 Problem Statement

The advent of large-scale quantum computers poses an existential threat to currently deployed public-key cryptographic infrastructure. Shor's algorithm enables polynomial-time factorization of integers and computation of discrete logarithms, rendering RSA, DSA, ECDSA, and ECDH vulnerable. While NIST has standardized post-quantum algorithms (FIPS 203, 204, 205), their deployment on resource-constrained embedded systems presents significant challenges:

1. **Memory Constraints:** Kyber-768 requires ~2KB for keys vs ~64 bytes for ECDH
2. **Computational Overhead:** NTT operations dominate execution time
3. **Side-Channel Vulnerabilities:** Lattice operations leak timing/power information
4. **Hybrid Transition:** Need backward compatibility during migration period

## 1.2 Research Hypotheses

**H1:** A hybrid KEM combining X25519 and ML-KEM-768 can achieve IND-CCA2 security while maintaining sub-100ms key exchange on ARM Cortex-M4 @ 168MHz.

**H2:** First-order masking of NTT operations reduces DPA leakage by >90% with <2x performance overhead.

**H3:** Stack-only Rust implementations can match or exceed C reference performance while eliminating memory safety vulnerabilities.

## 1.3 Research Questions

| ID | Question | Methodology |
|----|----------|-------------|
| RQ1 | What is the minimum security level achievable for hybrid PQC on devices with <64KB RAM? | Empirical benchmarking + theoretical analysis |
| RQ2 | How does constant-time Rust code compare to constant-time C for NTT? | Comparative performance analysis |
| RQ3 | Can formal verification tools prove absence of timing side-channels? | Model checking with CBMC/Kani |
| RQ4 | What masking order is necessary for DPA resistance in practical attack scenarios? | Power analysis experiments |
| RQ5 | How to design hybrid protocols that degrade gracefully if one primitive is broken? | Security proof construction |

## 1.4 Novel Contributions (Expected)

1. **Theoretical:** Security proof for hybrid KEM under combined hardness assumptions
2. **Algorithmic:** Optimized NTT for ARM Cortex-M with SIMD-like instructions
3. **Implementation:** First formally verified PQC implementation in Rust for embedded
4. **Empirical:** Comprehensive side-channel evaluation methodology for lattice crypto

---

# Part II: Mathematical Foundations

## 2.1 Algebraic Structures

### 2.1.1 Polynomial Rings
```
R = Z[X]/(X^n + 1) where n = 256 (power of 2)
R_q = Z_q[X]/(X^n + 1) where q = 3329 (Kyber) or q = 8380417 (Dilithium)
```

**Study Topics:**
- [ ] Cyclotomic polynomials and their properties
- [ ] Chinese Remainder Theorem for polynomial rings
- [ ] NTT as isomorphism: R_q ≅ Z_q^n when q ≡ 1 (mod 2n)

### 2.1.2 Lattice Theory
```
Lattice L(B) = {Bx : x ∈ Z^n} for basis B ∈ R^(m×n)
```

**Fundamental Problems:**
| Problem | Definition | Hardness |
|---------|------------|----------|
| SVP | Find shortest non-zero vector in L | NP-hard under randomized reductions |
| CVP | Find closest lattice point to target | NP-hard |
| SIVP_γ | Find n linearly independent vectors within γ factor of λ_n | Believed quantum-hard for γ = poly(n) |
| GapSVP_γ | Distinguish if λ_1(L) ≤ 1 or λ_1(L) > γ | Basis of LWE security |

### 2.1.3 Learning With Errors (LWE)

**Definition (Search-LWE):**
Given (A, b = As + e) where:
- A ←$ Z_q^(m×n)
- s ←$ χ^n (secret distribution)
- e ←$ χ^m (error distribution)

Find s.

**Definition (Decision-LWE):**
Distinguish (A, As + e) from (A, u) where u ←$ Z_q^m

**Security Reduction (Regev 2005):**
```
GapSVP_γ ≤ LWE_{n,q,χ} for γ = Õ(n·q/α) and χ = D_{Z,αq}
```

### 2.1.4 Module-LWE (ML-LWE)

**Definition:**
Operating over R_q^k instead of Z_q^n:
- A ←$ R_q^(k×k)
- s ←$ R_q^k with small coefficients
- e ←$ R_q^k with small coefficients

**Advantages:** Smaller keys (factor of n reduction), structured for NTT optimization

## 2.2 Cryptographic Primitives

### 2.2.1 ML-KEM (FIPS 203, formerly Kyber)

**Parameters (ML-KEM-768):**
```
n = 256, k = 3, q = 3329
η₁ = 2, η₂ = 2, d_u = 10, d_v = 4
|pk| = 1184 bytes, |sk| = 2400 bytes, |ct| = 1088 bytes
```

**Algorithms:**
1. **KeyGen():** (pk, sk) where pk = (ρ, t = As + e)
2. **Encaps(pk):** (ct, K) using random m, derive K = H(K̄, H(ct))
3. **Decaps(sk, ct):** K using implicit rejection

**Security:** IND-CCA2 under ML-LWE in ROM

### 2.2.2 ML-DSA (FIPS 204, formerly Dilithium)

**Core Technique:** Fiat-Shamir with Aborts
```
σ = (z, h, c̃) where z = y + cs must satisfy ||z||∞ < γ₁ - β
```

**Rejection Sampling:** Ensures signature distribution independent of secret key

### 2.2.3 SLH-DSA (FIPS 205, formerly SPHINCS+)

**Hash-based signatures:** Security relies only on hash function properties
- Stateless variant of XMSS
- Conservative choice, larger signatures (~17KB for 128-bit security)

## 2.3 Number Theoretic Transform (NTT)

**Definition:**
For primitive 2n-th root of unity ζ in Z_q:
```
NTT(a)_i = Σ_{j=0}^{n-1} a_j · ζ^((2·bit_rev(i)+1)·j) mod q
```

**Cooley-Tukey Butterfly:**
```
(a', b') = (a + ζ^k·b, a - ζ^k·b)
```

**Montgomery Reduction:**
For R = 2^16, q = 3329:
```
MontReduce(a) = (a - ((a·q^(-1) mod R)·q)) / R
```

**Barrett Reduction:**
```
BarrettReduce(a) = a - ⌊a·μ/2^k⌋·q where μ = ⌊2^k/q⌋
```

---

# Part III: Detailed Curriculum

## Quarter 1: Theoretical Foundations (Months 1-3)

### Month 1: Quantum Computing & Cryptanalysis

#### Week 1: Quantum Mechanics for Cryptographers
**Theory:**
- [ ] Postulates of quantum mechanics (state vectors, observables, measurement)
- [ ] Dirac notation: |ψ⟩, ⟨φ|, ⟨φ|ψ⟩, |φ⟩⟨ψ|
- [ ] Tensor products and entanglement: |00⟩ + |11⟩ / √2

**Reading:**
- Nielsen & Chuang, "Quantum Computation and Quantum Information," Ch. 1-2
- Mermin, "Quantum Computer Science," Ch. 1

**Exercises:**
- [ ] Prove that {|0⟩, |1⟩} and {|+⟩, |-⟩} are both orthonormal bases
- [ ] Calculate action of Hadamard gate on computational basis states
- [ ] Show that CNOT is its own inverse

#### Week 2: Quantum Algorithms I - Fundamentals
**Theory:**
- [ ] Quantum parallelism and interference
- [ ] Deutsch-Jozsa algorithm (exponential speedup for oracle problem)
- [ ] Simon's algorithm (predecessor to Shor)

**Reading:**
- Vazirani lecture notes on quantum algorithms
- Original Simon paper (1994)

**Implementation:**
- [ ] Simulate Deutsch-Jozsa in Qiskit for n=3

#### Week 3: Shor's Algorithm - Deep Dive
**Theory:**
- [ ] Reduction: Factoring → Order Finding
- [ ] Quantum Fourier Transform: O(n²) gates vs O(n·2^n) classical FFT
- [ ] Period finding via phase estimation
- [ ] Success probability and continued fractions

**Mathematical Details:**
```
For N = p·q, find r such that a^r ≡ 1 (mod N)
If r is even and a^(r/2) ≢ -1 (mod N):
  gcd(a^(r/2) ± 1, N) gives factors
```

**Reading:**
- Shor, "Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer" (1994)
- Preskill lecture notes Ch. 6

**Exercises:**
- [ ] Trace through Shor's algorithm for N=15, a=7
- [ ] Prove QFT unitarity
- [ ] Analyze gate complexity for n-bit factoring

#### Week 4: Grover's Algorithm & Implications
**Theory:**
- [ ] Amplitude amplification
- [ ] Optimal O(√N) query complexity
- [ ] Lower bound proofs (BBBV theorem)

**Cryptographic Impact:**
| Primitive | Classical Security | Post-Grover Security |
|-----------|-------------------|---------------------|
| AES-128 | 2^128 | 2^64 |
| AES-256 | 2^256 | 2^128 |
| SHA-256 preimage | 2^256 | 2^128 |
| SHA-256 collision | 2^128 | 2^85 (BHT algorithm) |

**Reading:**
- Grover, "A Fast Quantum Mechanical Algorithm for Database Search" (1996)
- Brassard, Høyer, Tapp, "Quantum Cryptanalysis of Hash and Claw-Free Functions" (1998)

### Month 2: Lattice Cryptography Mathematics

#### Week 1: Lattice Fundamentals
**Theory:**
- [ ] Lattice definitions: full-rank, integral, q-ary
- [ ] Successive minima: λ₁(L) ≤ λ₂(L) ≤ ... ≤ λ_n(L)
- [ ] Minkowski's theorems
- [ ] Hermite normal form and basis reduction

**Key Inequalities:**
```
λ₁(L) ≤ √n · det(L)^(1/n)  (Minkowski)
λ₁(L) · λ₁(L*) ≥ 1         (Transference)
```

**Reading:**
- Micciancio & Goldwasser, "Complexity of Lattice Problems: A Cryptographic Perspective," Ch. 1-3
- Peikert, "A Decade of Lattice Cryptography" (2016 survey)

**Exercises:**
- [ ] Compute λ₁ for lattice with basis [[1,0],[0.5,√3/2]]
- [ ] Prove Minkowski's first theorem
- [ ] Implement Gram-Schmidt in Python, verify orthogonality

#### Week 2: Lattice Reduction Algorithms
**Theory:**
- [ ] LLL algorithm: achieves γ = 2^(n/2) approximation in poly time
- [ ] BKZ-β: better approximation γ = β^(n/β) but exponential in β
- [ ] Practical security estimates: Core-SVP model

**LLL Invariants:**
```
|b*_i|² ≥ (δ - μ²_{i,i-1}) · |b*_{i-1}|²  for δ = 3/4
```

**Reading:**
- Lenstra, Lenstra, Lovász, "Factoring Polynomials with Rational Coefficients" (1982)
- Chen & Nguyen, "BKZ 2.0: Better Lattice Security Estimates" (2011)

**Implementation:**
- [ ] Implement LLL in SageMath
- [ ] Attack low-dimension LWE (n=20) using lattice reduction

#### Week 3: LWE and Ring-LWE
**Theory:**
- [ ] Regev's reduction: worst-case to average-case
- [ ] Noise growth in homomorphic operations
- [ ] Ring-LWE: structure from ideal lattices
- [ ] NTRU vs Ring-LWE relationships

**Security Parameters:**
```
For 128-bit security against known attacks:
- LWE: n ≈ 600-800, q ≈ 2^14, σ ≈ 3.2
- Ring-LWE: n = 256-512, q ≈ 2^12, σ ≈ 3.2
```

**Reading:**
- Regev, "On Lattices, Learning with Errors, Random Linear Codes, and Cryptography" (2005)
- Lyubashevsky, Peikert, Regev, "On Ideal Lattices and Learning with Errors over Rings" (2010)

**Exercises:**
- [ ] Prove decision-LWE ≤ search-LWE for prime q
- [ ] Implement Ring-LWE encryption in SageMath
- [ ] Verify noise growth after multiplication

#### Week 4: Module-LWE and Kyber Construction
**Theory:**
- [ ] Module-LWE: interpolates between LWE and Ring-LWE
- [ ] Kyber.CPAPKE: IND-CPA from Module-LWE
- [ ] Fujisaki-Okamoto transform: CPA → CCA
- [ ] Decryption failure probability analysis

**Kyber Specifics:**
```
pk = (ρ, t = Compress_q(As + e, d_t))
ct = (Compress_q(A^T r + e₁, d_u), Compress_q(t^T r + e₂ + ⌈q/2⌋·m, d_v))
```

**Reading:**
- Bos et al., "CRYSTALS - Kyber: A CCA-Secure Module-Lattice-Based KEM" (2018)
- NIST FIPS 203 specification

### Month 3: Literature Review & Gap Analysis

#### Week 1: Systematic Literature Review
**Methodology:**
- [ ] Define inclusion/exclusion criteria
- [ ] Search databases: IEEE Xplore, ACM DL, IACR ePrint, arXiv
- [ ] Use PRISMA methodology for paper selection

**Search Queries:**
```
("post-quantum" OR "lattice-based") AND ("embedded" OR "IoT" OR "microcontroller")
("Kyber" OR "Dilithium") AND ("ARM" OR "Cortex" OR "RISC-V")
("side-channel" OR "DPA" OR "timing") AND ("lattice" OR "NTT")
```

**Key Papers to Analyze:**

| Paper | Contribution | Platform | Performance |
|-------|--------------|----------|-------------|
| Oder & Güneysu (2017) | First Kyber on Cortex-M4 | STM32F4 | 1.2M cycles |
| Botros et al. (2019) | Memory-optimized Kyber | Cortex-M4 | 0.8M cycles, 6KB RAM |
| Kannwischer et al. (2019) | pqm4 library | Cortex-M4 | Comprehensive benchmarks |
| Heinz et al. (2022) | Masked Kyber | Cortex-M4 | 2.8M cycles (1st order) |

#### Week 2: Gap Identification
**Analysis Framework:**

| Dimension | Current State | Gap | Opportunity |
|-----------|---------------|-----|-------------|
| Memory | 6-10KB minimum | Too large for Cortex-M0 | Streaming implementations |
| Speed | 0.5-1M cycles | Acceptable but improvable | SIMD-like optimizations |
| Side-channel | Few masked implementations | Unprotected = vulnerable | Efficient masking schemes |
| Verification | Informal testing only | No formal guarantees | Apply Rust/CBMC |
| Hybrid | Limited exploration | Migration challenge | Robust hybrid protocols |

#### Weeks 3-4: Literature Review Document
**Structure:**
1. Introduction and motivation
2. Background on PQC and embedded constraints
3. Taxonomy of existing approaches
4. Comparative analysis
5. Identified gaps and research opportunities
6. Proposed research directions

**Output:** 8000-word literature review suitable for journal publication

---

## Quarter 2: Implementation Foundations (Months 4-6)

### Month 4: Advanced Rust for Cryptographic Engineering

#### Week 1: Type-Level Security
**Concepts:**
- [ ] Phantom types for state machines
- [ ] Typestate pattern for protocol enforcement
- [ ] Const generics for compile-time size checking

**Code Pattern - Typestate KEM:**
```rust
struct KeyPair<S: KeyState> {
    secret: SecretKey,
    public: PublicKey,
    _state: PhantomData<S>,
}

trait KeyState {}
struct Unsealed;
struct Sealed;

impl KeyPair<Unsealed> {
    fn seal(self) -> KeyPair<Sealed> { ... }
}

impl KeyPair<Sealed> {
    fn encapsulate(&self, pk: &PublicKey) -> (Ciphertext, SharedSecret) { ... }
}
// Cannot call encapsulate on Unsealed - compile error!
```

**Reading:**
- "Type-Driven Development with Idris" (concepts applicable to Rust)
- Rust Embedded Book, "Static Guarantees" chapter

#### Week 2: Constant-Time Programming in Rust
**Threat Model:**
- Timing attacks: execution time depends on secret data
- Cache attacks: memory access patterns leak information

**Rules:**
1. No secret-dependent branches
2. No secret-dependent memory access indices
3. No early returns based on secret data

**Tools:**
- [ ] `subtle` crate: constant-time comparisons and conditional selection
- [ ] `dudect`: statistical timing leakage detection
- [ ] TIMECOP: compile-time analysis

**Implementation:**
```rust
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

fn constant_time_select(a: u32, b: u32, condition: bool) -> u32 {
    let choice = Choice::from(condition as u8);
    u32::conditional_select(&a, &b, choice)
}

fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
    a.ct_eq(b).into()
}
```

#### Week 3: Memory Safety Without Runtime Overhead
**Concepts:**
- [ ] Zero-copy deserialization with `zerocopy` crate
- [ ] Compile-time buffer size verification
- [ ] Stack-based allocation with `heapless`

**Critical Pattern - Zeroization:**
```rust
use zeroize::{Zeroize, ZeroizeOnDrop};

#[derive(Zeroize, ZeroizeOnDrop)]
struct SecretKey {
    bytes: [u8; 32],
}
// Automatically zeroed when dropped, even on panic
```

#### Week 4: Formal Verification with Kani
**Approach:**
- [ ] Bounded model checking for Rust
- [ ] Verify absence of panics, overflows
- [ ] Prove functional correctness of critical functions

**Example - Verifying NTT Butterfly:**
```rust
#[cfg(kani)]
#[kani::proof]
fn verify_butterfly() {
    let a: i16 = kani::any();
    let b: i16 = kani::any();
    let zeta: i16 = kani::any();

    kani::assume(a.abs() < 1665 && b.abs() < 1665);
    kani::assume(zeta.abs() < 1665);

    let (a_out, b_out) = butterfly(a, b, zeta);

    // Verify no overflow occurred
    assert!(a_out.abs() < 3329);
    assert!(b_out.abs() < 3329);
}
```

### Month 5: Embedded Systems & no_std Rust

#### Week 1: ARM Cortex-M Architecture Deep Dive
**Topics:**
- [ ] ARMv7-M instruction set relevant to crypto
- [ ] Pipeline and cycle counting
- [ ] SIMD-like instructions: `SMULL`, `SMLAL`, `UMAAL`

**Key Instructions for NTT:**
```assembly
@ Montgomery multiplication using UMAAL
@ Computes: (a * b * R^-1) mod q
umaal r4, r5, r0, r1    @ r5:r4 = r0*r1 + r4 + r5
```

**Reading:**
- ARM Cortex-M4 Technical Reference Manual
- Kannwischer et al., "Polynomial Multiplication on ARM Cortex-M4"

#### Week 2: Memory-Constrained Implementation
**Strategies:**
- [ ] On-the-fly matrix generation from seed (saves ~1KB)
- [ ] Streaming NTT: process coefficients in chunks
- [ ] Stack usage analysis with `-Z emit-stack-sizes`

**Memory Budget (Cortex-M4, 64KB RAM):**
```
Stack:        8KB  (NTT temporaries, function calls)
Public Key:   1.2KB
Secret Key:   2.4KB
Ciphertext:   1.1KB
Working:      ~4KB (matrix columns, intermediates)
Available:    ~47KB for application
```

#### Week 3: Interrupt Safety and Real-Time Constraints
**Challenges:**
- [ ] Side-channel risk from interrupts
- [ ] Preemption during crypto operations
- [ ] Timing guarantees for real-time systems

**Solutions:**
```rust
use cortex_m::interrupt;

fn secure_operation() {
    interrupt::free(|_| {
        // Critical crypto code runs without interruption
        // Prevents timing variations from preemption
    });
}
```

#### Week 4: Hardware Abstraction and Portability
**Design Pattern:**
```rust
pub trait RandomSource {
    fn fill_bytes(&mut self, dest: &mut [u8]);
}

// Hardware TRNG implementation
impl RandomSource for Stm32Trng { ... }

// Software fallback for testing
impl RandomSource for ChaChaRng { ... }

pub struct Kyber<R: RandomSource> {
    rng: R,
}
```

### Month 6: PQC Implementation

#### Week 1-2: Kyber Core Implementation
**Components:**
1. [ ] Polynomial arithmetic in R_q
2. [ ] NTT and inverse NTT
3. [ ] Compression/decompression
4. [ ] CBD (Centered Binomial Distribution) sampling
5. [ ] SHAKE128/256 for XOF operations

**NTT Implementation Strategy:**
```rust
/// Cooley-Tukey NTT, in-place, bit-reversed output
pub fn ntt(a: &mut [i16; 256]) {
    let mut k = 1;
    let mut len = 128;
    while len >= 2 {
        for start in (0..256).step_by(2 * len) {
            let zeta = ZETAS[k];
            k += 1;
            for j in start..(start + len) {
                let t = montgomery_mul(zeta, a[j + len]);
                a[j + len] = a[j] - t;
                a[j] = a[j] + t;
            }
        }
        len >>= 1;
    }
}
```

#### Week 3-4: Integration and Testing
**Test Vectors:**
- [ ] NIST KAT (Known Answer Tests)
- [ ] Intermediate value tests for each function
- [ ] Randomized testing with property-based approach

**Benchmarking Setup:**
```rust
#[cfg(feature = "benchmark")]
pub fn benchmark_kyber() {
    let start = cortex_m::peripheral::DWT::cycle_count();
    let (pk, sk) = kyber_keygen();
    let keygen_cycles = cortex_m::peripheral::DWT::cycle_count() - start;

    // ... similar for encaps/decaps
}
```

---

## Quarter 3: Novel Research (Months 7-9)

### Month 7: Hybrid Protocol Design

#### Week 1: Hybrid KEM Construction
**Design - X25519 || ML-KEM-768:**
```
HybridKEM.KeyGen():
    (pk_c, sk_c) ← X25519.KeyGen()
    (pk_q, sk_q) ← MLKEM768.KeyGen()
    return ((pk_c, pk_q), (sk_c, sk_q))

HybridKEM.Encaps(pk):
    (ct_c, ss_c) ← X25519.ECDH(pk_c)
    (ct_q, ss_q) ← MLKEM768.Encaps(pk_q)
    ss ← KDF(ss_c || ss_q || ct_c || ct_q)
    return ((ct_c, ct_q), ss)
```

**Security Theorem (Informal):**
If either X25519 (ECDH) or ML-KEM-768 (Module-LWE) is secure, the hybrid KEM achieves IND-CCA2 security.

#### Week 2: Security Proof
**Proof Strategy:**
1. Define hybrid games
2. Show indistinguishability between games
3. Reduce to hardness of underlying assumptions

**Game Sequence:**
```
G₀: Real security game
G₁: Replace ss_c with random (ECDH security)
G₂: Replace ss_q with random (ML-KEM security)
G₃: Final game with fully random ss
```

#### Week 3: Protocol Integration
**TLS 1.3 Hybrid Key Exchange:**
```
ClientHello:
    supported_groups: [x25519_mlkem768, x25519, mlkem768]
    key_share: [x25519_share, mlkem768_share]

ServerHello:
    selected_group: x25519_mlkem768
    key_share: [ecdh_response, mlkem_ciphertext]

Both parties compute:
    shared_secret = HKDF(ecdh_ss || mlkem_ss)
```

#### Week 4: Embedded Protocol Implementation
**State Machine:**
```rust
enum HandshakeState {
    Initial,
    SentClientHello { ephemeral_sk: HybridSecretKey },
    ReceivedServerHello { shared_secret: [u8; 32] },
    Established { keys: SessionKeys },
    Error(HandshakeError),
}
```

### Month 8: Side-Channel Countermeasures

#### Week 1: Masking Theory
**Boolean Masking:**
```
x = x₀ ⊕ x₁ ⊕ ... ⊕ x_d  (d-th order masking)
```

**Arithmetic Masking:**
```
x = x₀ + x₁ + ... + x_d (mod q)
```

**Conversion (Critical for NTT):**
- Boolean ↔ Arithmetic conversion is expensive
- Optimal strategy depends on operation mix

#### Week 2: Masked NTT Implementation
**Challenge:** NTT combines additions and multiplications

**Approach - Arithmetic Masking Throughout:**
```rust
struct MaskedPoly {
    shares: [[i16; 256]; 2],  // First-order masking
}

fn masked_ntt(a: &mut MaskedPoly) {
    // Process each share through NTT
    ntt(&mut a.shares[0]);
    ntt(&mut a.shares[1]);
    // Shares maintain arithmetic relationship
}

fn masked_multiply(a: &MaskedPoly, b: &MaskedPoly) -> MaskedPoly {
    // Requires secure multiplication protocol
    // ISW multiplication adapted for arithmetic masking
}
```

#### Week 3: Power Analysis Evaluation
**Experimental Setup:**
- ChipWhisperer Lite for power trace capture
- Target: STM32F4 @ 7.37MHz (for clean traces)
- 100,000 traces for DPA attack

**Metrics:**
- t-test for leakage detection (TVLA)
- Success rate of key recovery
- Traces to disclosure (TtD)

#### Week 4: Countermeasure Optimization
**Techniques:**
- [ ] Shuffling: randomize order of operations
- [ ] Blinding: multiply by random before NTT
- [ ] Noise injection: add controlled randomness

**Performance Budget:**
| Countermeasure | Overhead | Protection |
|----------------|----------|------------|
| 1st-order masking | 2-3x | Basic DPA |
| Shuffling | 1.2x | Horizontal attacks |
| Combined | 3-4x | State-of-the-art |

### Month 9: Verification & Validation

#### Week 1-2: Formal Verification
**Properties to Verify:**
1. [ ] Functional correctness (matches spec)
2. [ ] Memory safety (no buffer overflows)
3. [ ] Constant-time (no secret-dependent branches)
4. [ ] Arithmetic correctness (no overflows)

**Tools:**
- Kani for Rust bounded model checking
- ctverif for constant-time verification
- HACL* methodology adaptation

#### Week 3-4: Comprehensive Testing
**Test Categories:**
1. **Unit tests:** Each function in isolation
2. **Integration tests:** Full protocol flows
3. **Fuzzing:** AFL++ with custom harnesses
4. **Interoperability:** Test against reference implementations

**Fuzzing Harness:**
```rust
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() >= 32 {
        let mut sk = [0u8; 2400];
        sk[..data.len().min(2400)].copy_from_slice(&data[..data.len().min(2400)]);
        // Should not panic regardless of input
        let _ = kyber_decapsulate(&sk, &data[32..]);
    }
});
```

---

## Quarter 4: Analysis & Publication (Months 10-12)

### Month 10: Comprehensive Benchmarking

#### Performance Metrics
| Metric | Tool | Target |
|--------|------|--------|
| Cycle count | DWT cycle counter | All operations |
| Stack usage | `-Z emit-stack-sizes` | <8KB total |
| Code size | `cargo size` | <32KB .text |
| Energy | Oscilloscope + shunt | Per operation |

#### Comparative Analysis Framework
```
Platforms: STM32F407, STM32L476, nRF52840, ESP32-C3
Configurations: Reference, Optimized, Masked
Metrics: Cycles, RAM, Flash, Energy

Statistical analysis:
- Mean and standard deviation over 1000 runs
- Min/max for worst-case analysis
- Comparison with state-of-the-art (pqm4)
```

### Month 11: Dissertation Writing

#### Chapter Outline
1. **Introduction** (3000 words)
   - Motivation and problem statement
   - Research questions and contributions
   - Thesis structure

2. **Background** (5000 words)
   - Quantum computing threat model
   - Lattice cryptography foundations
   - Embedded systems constraints

3. **Literature Review** (5000 words)
   - Existing PQC implementations
   - Side-channel countermeasures
   - Hybrid cryptography approaches

4. **Methodology** (4000 words)
   - Research design
   - Implementation approach
   - Evaluation framework

5. **Hybrid Protocol Design** (6000 words)
   - Construction and security proof
   - Protocol specification
   - Security analysis

6. **Implementation** (6000 words)
   - Architecture decisions
   - Optimization techniques
   - Side-channel countermeasures

7. **Evaluation** (5000 words)
   - Performance benchmarks
   - Security validation
   - Comparative analysis

8. **Discussion** (3000 words)
   - Interpretation of results
   - Limitations
   - Future work

9. **Conclusion** (1000 words)
   - Summary of contributions
   - Implications

**Total: ~38,000 words (typical CS PhD thesis length)**

### Month 12: Publication & Release

#### Target Venues
| Venue | Type | Deadline | Focus |
|-------|------|----------|-------|
| CHES | Conference | March/Sept | Hardware security |
| CCS | Conference | May | Systems security |
| TCHES | Journal | Rolling | Implementation attacks |
| IEEE TCAD | Journal | Rolling | Hardware design |

#### Paper Structure (8-10 pages)
1. Abstract (200 words)
2. Introduction (1 page)
3. Preliminaries (1.5 pages)
4. Construction (2 pages)
5. Implementation (2 pages)
6. Evaluation (2 pages)
7. Related Work (0.5 pages)
8. Conclusion (0.5 pages)

#### Open Source Release
**Repository Structure:**
```
q-secure/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── kyber/
│   ├── hybrid/
│   ├── ntt/
│   └── masking/
├── examples/
├── tests/
├── benches/
├── docs/
│   ├── SPECIFICATION.md
│   ├── SECURITY.md
│   └── BENCHMARKS.md
└── verification/
    └── kani/
```

---

# Part IV: Resources & References

## Essential Reading List

### Textbooks
1. Galbraith, "Mathematics of Public Key Cryptography" (2012)
2. Peikert, "A Decade of Lattice Cryptography" (2016)
3. Micciancio & Goldwasser, "Complexity of Lattice Problems" (2002)
4. Nielsen & Chuang, "Quantum Computation and Quantum Information" (2010)
5. Mangard, Oswald, Popp, "Power Analysis Attacks" (2007)

### Key Papers
1. Regev, "On Lattices, Learning with Errors..." (2005) - LWE foundations
2. Lyubashevsky, Peikert, Regev, "On Ideal Lattices..." (2010) - Ring-LWE
3. Bos et al., "CRYSTALS-Kyber" (2018) - Kyber specification
4. Ducas et al., "CRYSTALS-Dilithium" (2018) - Dilithium specification
5. Oder & Güneysu, "Implementing Post-Quantum Cryptography..." (2017) - First embedded Kyber
6. Heinz et al., "First-Order Masked Kyber" (2022) - Masked implementation

### Standards
1. NIST FIPS 203 - ML-KEM
2. NIST FIPS 204 - ML-DSA
3. NIST FIPS 205 - SLH-DSA
4. IETF draft-ietf-tls-hybrid-design - Hybrid key exchange

## Tools & Software

| Category | Tool | Purpose |
|----------|------|---------|
| IDE | VS Code + rust-analyzer | Development |
| Compiler | rustc + LLVM | Compilation |
| Debugger | probe-rs | Embedded debugging |
| Profiler | cargo-flamegraph | Performance analysis |
| Fuzzer | cargo-fuzz (libFuzzer) | Security testing |
| Verifier | Kani | Formal verification |
| Side-channel | ChipWhisperer | Power analysis |
| Quantum sim | Qiskit | Algorithm simulation |
| Math | SageMath | Lattice computations |
| Writing | LaTeX + Overleaf | Documentation |
| Reference | Zotero | Bibliography management |

## Hardware Requirements

| Item | Specification | Purpose |
|------|---------------|---------|
| Dev Board 1 | STM32F407 Discovery | Primary target |
| Dev Board 2 | nRF52840 DK | BLE IoT target |
| Dev Board 3 | ESP32-C3 | RISC-V target |
| Logic Analyzer | Saleae Logic 8 | Debugging |
| Power Analysis | ChipWhisperer Lite | Side-channel testing |
| Oscilloscope | 100MHz+ | Energy measurement |

---

# Part V: Progress Tracking

## Weekly Log Template
```markdown
## Week [N] - [Date Range]

### Goals
- [ ] Goal 1
- [ ] Goal 2

### Accomplished
- Item 1
- Item 2

### Challenges
- Challenge and resolution

### Next Week
- Plan for next week

### Hours: [X] theory, [Y] implementation, [Z] writing
```

## Milestone Checkpoints

| Month | Milestone | Deliverable | Verified By |
|-------|-----------|-------------|-------------|
| 3 | Literature Review | 8000-word document | Self-review |
| 6 | Working Implementation | Passing NIST KATs | Automated tests |
| 9 | Security Validation | Formal proofs + tests | Code review |
| 12 | Publication | Submitted paper | Peer review |

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Scope creep | High | Medium | Strict milestone adherence |
| Implementation bugs | High | High | Extensive testing, verification |
| Side-channel vulnerabilities | High | Medium | Systematic evaluation methodology |
| Time constraints (job) | Medium | High | Realistic weekly goals |
| Hardware availability | Low | Low | Multiple backup platforms |

---

# Part VI: Success Metrics

## Quantitative Targets
- [ ] <1M cycles for Kyber keygen+encaps+decaps on Cortex-M4
- [ ] <8KB stack usage
- [ ] <32KB code size
- [ ] >90% reduction in DPA leakage (1st-order masking)
- [ ] 100% NIST KAT pass rate
- [ ] Zero memory safety issues (verified by Kani)

## Qualitative Targets
- [ ] Novel contribution to PQC embedded implementation
- [ ] Reproducible research (open source)
- [ ] At least 1 peer-reviewed publication submission
- [ ] Documentation suitable for practitioners

---

**Research Start Date:** _______________
**Target Completion:** _______________
**Primary Advisor/Mentor:** Self-directed with AI assistance

---

*"In cryptography, the attacker always has the advantage of time. Our defense must be mathematically certain."*

**Document Version:** 2.0 (Expert Level)
**Last Updated:** February 2026
