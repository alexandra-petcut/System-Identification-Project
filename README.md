# System Identification Project

Polynomial model-based system identification implemented in MATLAB. Two independent parts that tackle different types of systems using least-squares regression on polynomial regressors.

**Course:** System Identification — Technical University of Cluj-Napoca

---

## Part 1 — Static Function Approximation

Approximates an unknown **nonlinear static function** with two inputs and one output from noisy measurements.

- Builds polynomial models of increasing degree (1 to 35) using all monomial terms x₁ᵉ¹·x₂ᵉ² where e₁ + e₂ ≤ degree
- Estimates parameters via least-squares: θ = Φ \ Y
- Selects the optimal polynomial degree by minimizing validation MSE
- Visualizes results as 3D surface plots comparing true vs approximated outputs

**Optimal result:** degree ≈ 20, MSE_val ≈ 0.0052

➡️ [Part 1 details](Part1/README.md)

---

## Part 2 — Dynamic System Identification (Nonlinear ARX)

Models an unknown **dynamic system** (single-input, single-output) using a polynomial nonlinear AutoRegressive with eXogenous inputs (ARX) structure.

- Constructs delay vectors from past outputs and inputs: d(k) = [y(k−1), …, y(k−na), u(k−nk), …, u(k−nk−nb+1)]
- Generates polynomial regressors up to degree m from the delay vector
- Grid-searches over model orders (na = nb ∈ {1, 2, 3}) and polynomial degrees (1 to 10)
- Evaluates in two modes:
  - **Prediction** — uses real past outputs (one-step-ahead)
  - **Simulation** — uses the model's own past outputs (free-run)
- Selects the best model by minimizing simulation MSE on validation data
- Compares with MATLAB built-in ARX and Output-Error models

**Optimal result:** na = nb = 3, degree = 1 (linear), with ~94–95% simulation fit

➡️ [Part 2 details](Part2/README.md)

---

## Quick Start

**Requirements:** MATLAB R2018b or later. Part 2's comparison section optionally uses the System Identification Toolbox.

```matlab
% Part 1
cd Part1
ProjectPart1

% Part 2
cd Part2
ProjectPart2
```

---

## Repository Structure

```
├── Part1/
│   ├── ProjectPart1.m      % Polynomial function fitting script
│   ├── proj_fit_11.mat      % Dataset (identification + validation)
│   └── README.md
├── Part2/
│   ├── ProjectPart2.m      % Nonlinear ARX identification script
│   ├── iddata-13.mat        % Dataset (identification + validation)
│   └── README.md
└── README.md                % This file
```

