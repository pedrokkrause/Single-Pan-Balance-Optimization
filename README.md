# Pascal Weighing Optimization

Improve the measurement precision of a single object on an existing digital balance without buying new hardware, using combinatorial weighing designs.

---

## How It Works (The Intuition)

### 1. The Core Idea: Canceling Noise with Helper Objects
Suppose you have a balance with measurement noise, and you want to measure object **A**. Introduce two helper objects, **B** and **C** (e.g., coins, nuts, or calibration weights). Now perform these four weighings:
1. $y_1  \approx A$
2. $y_2  \approx A + B$
3. $y_3  \approx A + C$
4. $y_4  \approx B + C$

Notice what happens if you take:
$$(A + B) + (A + C) - (B + C) = 2A$$

The unknown weights of helpers **B** and **C** cancel out completely. Because four readings were combined, random balance errors average out, giving an estimate of **A** that is significantly more precise than direct weighings alone.

### 2. The Single-Pan Constraint
On a two-pan balance, you could place objects on opposite pans to physically subtract them. On a modern single-pan digital scale, **you can only add weights** (every item is either on the pan or off). 

To cancel the helper masses, the subtraction must be done mathematically after data collection via ordinary least squares (Gauss-Markov estimation).

### 3. The Common Trap
The intuitive instinct is: *"If I care about A, I should put A on the pan in almost every weighing."*

**This fails.** If you include **A** in 9 out of 10 weighings, the helper objects are never weighed enough among themselves. Their uncertain values leak into **A** through mathematical covariance, **inflating your final error by up to +95%**.

### 4. The Pascal Recipe
To prevent this noise leakage, you must split your weighings into two opposing blocks:
- **Block A (with A):** Use **light combinations** ($A$ alone, $A$ with 1 helper, $A$ with 2 helpers...).
- **Block B (without A):** Use **heavy combinations** (all helpers together, all helpers except one, all helpers except two...).

These combinations correspond to the opposite ends of **Pascal's Triangle** $\binom{p}{r}$. When Block A is kept exactly one layer ahead of Block B ($k_A = k_B + 1$), the helper noise cancels out identically.

---

## What This Software Does

You don't need to calculate combinations or invert matrices. Given:
1. Your target object ($m_1$),
2. Any number of available helper objects ($m_2, \dots, m_N$), and
3. How many total weighings you want to perform ($m$), up to a maximum of all $2^N-1$ possible combinations.

The script:
- Finds the mathematically optimal combinations of objects to place on the pan.
- Orders the weighings to minimize physical manipulations (using 2-Opt Hamming distance routing).
- Prompts you step-by-step: what to add (`[+]`) and what to remove (`[-]`).
- Computes the true mass of $m_1$, its calibrated uncertainty $\pm u(m_1)$, and the noise level ($s$) of your balance.

---

## Quick Start

### 1. Requirements
Python 3.8+ and NumPy:
```bash
pip install numpy
```

### 2. Configure Items and Budget
Open `src/pascal_weighing_en.py` (or `src/pesagem_pascal_pt.py`):
```python
# The FIRST item is your target object (m1); the rest are helpers
ITEMS = ['target_sample', 'brass_weight', 'hex_nut', 'glass_vial', 'steel_washer']

# Total weighings to perform (must be >= len(ITEMS))
M_TARGET = 16
```

### 3. Run
```bash
python src/pascal_weighing_en.py
```

Follow the terminal instructions:
```text
--- Weighing 1 of 16 ---
-> [+] PLACE: target_sample, brass_weight
   Pan must contain: [ target_sample, brass_weight ]
Enter balance reading (or 'z' to undo previous step): 42.158
```

At the end, you get the exact mass estimate and uncertainty.

---

## Repository Structure

```text
.
├── docs/
│   ├── weighing_optimization_pascal_en.pdf    # Full paper (English)
│   ├── otimizacao_pesagens_pascal_pt.pdf      # Artigo completo (Português)
└── src/
    ├── pascal_weighing_en.py                  # CLI assistant & solver (English)
    └── pesagem_pascal_pt.py                   # Assistente CLI e resolvedor (Português)
```

---

## License

MIT
