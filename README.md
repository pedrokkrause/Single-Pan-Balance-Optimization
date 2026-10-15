# Pascal Weighing Optimization

Improve the measurement precision of a single object on an existing digital balance without buying new hardware, using combinatorial weighing designs.

---

## How It Works (The Intuition)

### 1. The Core Idea: Canceling Noise with Helper Objects
Suppose you have a balance with measurement noise, and you want to measure object **A**. Introduce two helper objects, **B** and **C** (e.g., coins, keys, or calibration weights). Now perform these four distinct weighings:
1. $y_1 \approx A$
2. $y_2 \approx A + B$
3. $y_3 \approx A + C$
4. $y_4 \approx B + C$

Notice what happens if you take:
$$(A + B) + (A + C) - (B + C) = 2A$$

The unknown weights of helpers **B** and **C** cancel out completely. Combining all four readings via least squares averages out balance errors, yielding a significantly more precise estimate of **A**.

> **Why not just weigh *A* four times in a row?**
> Weighing the exact same object on a digital scale repeatedly is like measuring a block with a millimeter ruler four times—you will just read the same rounded number over and over. However, if you place the block alongside different helper objects of unknown lengths, the combined edges fall across different millimeter marks on the ruler. Measuring *distinct combinations* forces the scale to sample across different points of its internal sensor and rounding grid, extracting sub-step precision that single-object repeats cannot see.

### 2. The Single-Pan Constraint
On a two-pan balance, you could place objects on opposite pans to physically subtract them. On a modern single-pan digital scale, **you can only add weights** (every item is either on the pan or off).

To cancel the helper masses, the subtraction must be done mathematically after data collection via ordinary least squares (Gauss-Markov estimation).

### 3. The Common Trap
The intuitive instinct is: *"If I care about A, I should put A on the pan in almost every weighing."*

**This fails.** If you include **A** in 9 out of 10 weighings, the helper objects are never weighed enough among themselves. Their uncertain values leak into **A** through mathematical covariance, **inflating your final error**.

### 4. The Pascal Recipe
To prevent this noise leakage with $N$ total objects ($1$ target + $p = N - 1$ helpers), you split your weighings into two opposing blocks based on how many helpers ($r$) accompany the pan:
- **Block A (with A):** Use **light combinations** ($A$ alone, $A$ with $1$ helper, $A$ with $2$ helpers...).
- **Block B (without A):** Use **heavy combinations** (all $p$ helpers together, all helpers except $1$, all helpers except $2$...).

These combinations correspond to the opposite ends of **Pascal's Triangle**, where each layer contains $\binom{p}{r}$ combinations of $r$ helpers. When the number of combination layers in Block A ($k_A$) is kept exactly one layer ahead of Block B ($k_A = k_B + 1$), the helper covariance cancels out symmetrically. For intermediate budgets ($m$) that do not fill an entire layer, the algorithm decomposes the layer into **cyclic orbits** ($C_p$) to suppress parasitic spectral leakage.

---

## What This Software Does

You don't need to calculate combinations or invert matrices. Given:
1. Your target object ($m_1$),
2. Any number of available helper objects ($m_2, \dots, m_N$), and
3. How many total weighings you want to perform ($m$), from $N$ up to $2^N - 1$ possible combinations.

The script:
- Finds the mathematically optimal combinations of objects to place on the pan (using Pascal layer truncation and cyclic orbit decomposition).
- Orders the weighings to minimize physical manipulations on the bench (Nearest Neighbor + 2-Opt Hamming distance routing).
- Prompts you step-by-step: what to add (`[+]`) and what to remove (`[-]`), with undo support (`'z'`).
- Computes the true mass of $m_1$, its calibrated uncertainty $\pm u(m_1)$, simultaneously calibrated masses for all helpers, and the experimental standard deviation ($s$) of your balance.

---

## Quick Start

### 1. Requirements
Python 3.8+ and NumPy:
```bash
pip install numpy
```

### 2. Configure Items and Budget
Open `pascal_weighing_en.py` (or `pesagem_pascal_pt.py`):
```python
# The first item in the list MUST be the primary object of interest (m1).
# The remaining objects act as auxiliary masses (m2, ..., mN).
ITEMS = ['flash_drive', 'square_key', 'toy_llama', 'padlock', 'eraser', 'toy_frog']

# Desired total weighing budget (m >= len(ITEMS))
# Examples for N=6 (p=5): Star Clean Point = 7; Intermediate = 22; Full Factorial = 63
M_TARGET = 10
```

### 3. Run
```bash
python pascal_weighing_en.py
```

The script displays the spectral diagnostics of your chosen design and guides you step-by-step at the scale:
```text
=====================================================================
 PASCAL COMBINATORIAL WEIGHING PROCEDURE
=====================================================================
 Object of Interest (m1) : flash_drive
 Auxiliary Masses (p=5) : square_key, toy_llama, padlock, eraser, toy_frog
 Total Budget (m)        : 10 weighings (Block A: 8 | Block B: 2)
 Spectral Regime         : Sub-Orbital Fractional Budget (k < p)
 Net Information (Phi)   : 6.50000 (Collective Mode: -1.50000 | Leakage: -0.00000)
 Factor u(m1)/sigma      : 0.39223 x sigma
 Benchtop Movements      : 16 optimized add/remove actions (Gray + 2-Opt)
=====================================================================

WARNING: Ensure the balance is EMPTY and TARED before starting.

--- Weighing 1 of 10 ---
-> [+] PLACE : flash_drive
   Pan must contain: [ flash_drive ]
Enter balance reading (or 'z' to undo previous step): 8.413
```

Once all weighings are recorded, the solver outputs the calibrated masses and metrological diagnostics:
```text
=====================================================================
 METROLOGICAL RESULTS AND SPECTRAL DIAGNOSTICS
=====================================================================

>> OBJECT OF INTEREST (m1 - Gauss-Markov): flash_drive = 8.41292 ± 0.00029

>> SIMULTANEOUSLY CALIBRATED AUXILIARY MASSES:
   square_key        : 4.10418 ± 0.00057
   toy_llama         : 12.83118 ± 0.00057
   padlock           : 24.50885 ± 0.00057
   eraser            : 6.22085 ± 0.00057
   toy_frog          : 9.73951 ± 0.00057

>> EXPERIMENT METROLOGICAL DIAGNOSTICS:
   - Uncertainty factor u(m1)/sigma : 0.39223 (60.8% reduction vs. 1 direct weighing)
   - Degrees of freedom (m - N)     : 4
   - Balance standard deviation (s) : 0.00073 units
```
*(When operating in a 1st-order isotropic regime such as `M_TARGET = 7` or `22`, the output also prints the matrix-free `>> BENCHTOP CLOSED-FORM FORMULA` result.)*

---

## Repository Structure

```text
.
├── weighing_optimization_pascal_en.pdf    # Full paper (English)
├── otimizacao_pesagens_pascal_pt.pdf      # Artigo completo (Português)
├── pascal_weighing_en.py                  # CLI assistant & solver (English)
└── pesagem_pascal_pt.py                   # Assistente CLI e resolvedor (Português)
```

---

## License

MIT
