# Part 2: Nonlinear ARX Identification

## Table of Contents
1. [Overview](#overview)
2. [Background Concepts](#background-concepts)
3. [Problem Statement](#problem-statement)
4. [Mathematical Formulation](#mathematical-formulation)
5. [Implementation Details](#implementation-details)
6. [Model Selection Strategy](#model-selection-strategy)
7. [Results](#results)
8. [Conclusions](#conclusions)
9. [Usage](#usage)

---

## Overview

This project implements a **nonlinear AutoRegressive with eXogenous inputs (ARX)** model for system identification. The goal is to develop a black-box model that can predict and simulate the behavior of an unknown dynamic system based on measured input-output data.

**Key Features:**
- Polynomial nonlinear ARX model implementation
- Two operating modes: prediction and simulation
- Automated model order and polynomial degree selection
- Validation using Mean Squared Error (MSE) metrics
- Comparison with MATLAB built-in models (ARX and OE)

---

## Background Concepts

### Linear vs Nonlinear ARX

**Linear ARX:**
```
y(k) = a₁·y(k-1) + a₂·y(k-2) + ... + b₁·u(k-1) + b₂·u(k-2) + ...
```
The output is a simple weighted sum of past values.

**Nonlinear ARX (our project):**
```
y(k) = p(y(k-1), ..., y(k-na), u(k-nk), ..., u(k-nk-nb+1))
```
The output is a **polynomial function** of past values, which can include:
- Powers: y(k-1)², y(k-1)³, u(k-1)²
- Cross-terms: y(k-1)·u(k-1), y(k-1)²·u(k-2)

This allows the model to capture **nonlinear dynamics** (e.g., saturation, friction, chemical reactions) while still using **linear regression** for parameter estimation.

### Key Terminology

- **na**: Number of past output values used (output order)
- **nb**: Number of past input values used (input order)
- **nk**: Time delay between input and output
- **m**: Polynomial degree (maximum power/cross-term degree)
- **Identification dataset**: Data used to estimate model parameters
- **Validation dataset**: Independent data used to test model generalization
- **Prediction (one-step-ahead)**: Uses real past outputs y(k-1), y(k-2), ...
- **Simulation (free-run)**: Uses simulated past outputs ŷ(k-1), ŷ(k-2), ...

---

## Problem Statement

### Given Data

Two datasets measured from an unknown dynamic system:
1. **Identification dataset (id)**: Used to estimate model parameters
2. **Validation dataset (val)**: Used to validate model performance

Each dataset contains:
- **One input signal**: u(k) - the external input to the system
- **One output signal**: y(k) - the system response
- **Sampling time**: Ts - time between measurements

### Objectives

1. **Model the system** using a polynomial nonlinear ARX structure
2. **Find optimal model orders** (na, nb) and polynomial degree (m)
3. **Minimize validation simulation MSE** to select the best model
4. **Validate performance** using both prediction and simulation modes
5. **Compare results** with MATLAB built-in models

### Constraints

- System dynamics order ≤ 3 (na, nb ≤ 3)
- Fixed input delay: nk = 1
- Model must work in two modes: prediction and simulation

---

## Mathematical Formulation

### 1. Delay Vector

At time k, we construct a vector of delayed outputs and inputs:

```
d(k) = [y(k-1), ..., y(k-na), u(k-nk), u(k-nk-1), ..., u(k-nk-nb+1)]ᵀ
```

**Dimensions:** d(k) is a column vector of length Nd = na + nb

**Example** (na=2, nb=2, nk=1):
```
d(k) = [y(k-1), y(k-2), u(k-1), u(k-2)]ᵀ
```

### 2. Nonlinear ARX Model

The predicted output is a polynomial function of the delay vector:

```
ŷ(k) = p(d(k)) = φ(k) · θ
```

Where:
- **p**: Polynomial of degree m
- **φ(k)**: Regressor vector (contains polynomial terms of d(k))
- **θ**: Parameter vector to be estimated

### 3. Regressor Construction

The regressor φ contains:
1. **Constant term**: 1
2. **Individual powers** (q ≤ m): d₁^q, d₂^q, ..., d_Nd^q
3. **Cross-terms** with total degree ≤ m: d₁^i₁ · d₂^i₂ · ... where i₁+i₂+... ≤ m

**Example** (na=1, nb=1, nk=1, m=2):
```
d(k) = [y(k-1), u(k-1)]ᵀ
φ(k) = [1, y(k-1), u(k-1), y(k-1)², y(k-1)·u(k-1), u(k-1)²]ᵀ
```

**Avoiding duplicates:** We use non-decreasing index vectors to ensure terms like d₁·d₂ and d₂·d₁ are not both included.

### 4. Parameter Estimation

For the identification dataset, we stack all regressors into a matrix:

```
Φ_id = [φ(1), φ(2), ..., φ(N_id)]ᵀ    (N_id × n_params matrix)
Y_id = [y(1), y(2), ..., y(N_id)]ᵀ     (N_id × 1 vector)
```

**Linear least-squares solution:**
```
θ = Φ_id \ Y_id
```

This is equivalent to: θ = (Φ_id^T · Φ_id)^(-1) · Φ_id^T · Y_id

**Key insight:** Even though the model is nonlinear in the variables, it's **linear in the parameters θ**, so we can use fast linear regression.

### 5. Two Operating Modes

#### Prediction Mode (One-Step-Ahead)

Uses **real past outputs** from the measured data:
```
ŷ_pred(k) = φ_pred(k) · θ
where d(k) uses real y(k-1), y(k-2), ..., y(k-na)
```

**Use case:** Testing how well the model fits when it has access to true history.

#### Simulation Mode (Free-Run)

Uses **simulated past outputs** from the model itself:
```
ŷ_sim(k) = φ_sim(k) · θ
where d(k) uses ŷ_sim(k-1), ŷ_sim(k-2), ..., ŷ_sim(k-na)
```

**Use case:** Testing how well the model performs in autonomous operation (more realistic scenario).

**Initial conditions:** For negative and zero time indices, we assume y(0) = u(0) = 0.

### 6. Performance Metrics

**Prediction MSE:**
```
MSE_pred = (1/N) · Σ[ŷ_pred(k) - y(k)]²
```

**Simulation MSE:**
```
MSE_sim = (1/N) · Σ[ŷ_sim(k) - y(k)]²
```

We compute these for both ID and VAL datasets.

**Model selection criterion:** Choose (na, m) that minimizes **MSE_sim on the validation dataset**.

---

## Implementation Details

### Algorithm Overview

```
For each model order na = 1 to n (where na = nb):
    For each polynomial degree deg = 1 to m:

        // BUILD REGRESSORS AND ESTIMATE PARAMETERS
        1. Build delay vectors for ID dataset (prediction mode)
        2. Construct regressor matrix Φ_id_pred
        3. Estimate parameters: θ = Φ_id_pred \ Y_id

        // EVALUATE ON ID DATASET
        4. Compute ŷ_pred_id and MSE_pred_id
        5. Build delay vectors for simulation (using ŷ_sim)
        6. Compute ŷ_sim_id and MSE_sim_id

        // EVALUATE ON VAL DATASET (same θ!)
        7. Compute ŷ_pred_val and MSE_pred_val
        8. Compute ŷ_sim_val and MSE_sim_val

        // TRACK BEST MODEL
        9. If MSE_sim_val < MSE_min:
               Save this (na, deg) as best model
               Store outputs for plotting

Display results table and plots for best model
```

### Key Implementation Functions

#### 1. Delay Vector Construction

**Prediction mode:**
```matlab
for k = 1:N_id
    d_id_pred = []
    for i = 1:na
        if k-i > 0
            d_id_pred = [d_id_pred, y_id(k-i)]
        else
            d_id_pred = [d_id_pred, 0]  % Zero initial conditions
        end
    end
    for j = 1:nb
        if k-nk-j+1 > 0
            d_id_pred = [d_id_pred, u_id(k-nk-j+1)]
        else
            d_id_pred = [d_id_pred, 0]
        end
    end
    % Compute regressor for this time step
    phi_id_pred = computePhi(d_id_pred, deg)
end
```

**Simulation mode:**
```matlab
y_sim_id = zeros(N_id, 1)  % Initialize simulated output

for k = 1:N_id
    d_id_sim = []
    for i = 1:na
        if k-i > 0
            d_id_sim = [d_id_sim, y_sim_id(k-i)]  % Use simulated outputs!
        else
            d_id_sim = [d_id_sim, 0]
        end
    end
    for j = 1:nb
        if k-nk-j+1 > 0
            d_id_sim = [d_id_sim, u_id(k-nk-j+1)]
        else
            d_id_sim = [d_id_sim, 0]
        end
    end

    phi_id_sim = computePhi(d_id_sim, deg)
    y_sim_id(k) = phi_id_sim * theta  % Compute and store simulated output
end
```

#### 2. Polynomial Regressor Construction

The `computePhi` function generates all polynomial terms up to degree m:

```matlab
function phi = computePhi(d, m)
    Nd = length(d)
    phi = 1  % Start with constant term

    for p = 1:m  % For each degree
        idx = ones(1, p)  % Initialize index vector

        while true
            % Add the monomial d(idx(1)) * d(idx(2)) * ... * d(idx(p))
            monomial = prod(d(idx))
            phi = [phi; monomial]

            % Move to next non-decreasing index vector
            i = p
            while i >= 1 && idx(i) == Nd
                i = i - 1
            end

            if i == 0
                break  % Finished all monomials of degree p
            end

            idx(i) = idx(i) + 1
            idx(i+1:end) = idx(i)  % Set suffix equal to incremented value
        end
    end
end
```

**Index vector strategy:** By maintaining non-decreasing indices (i₁ ≤ i₂ ≤ ... ≤ iₚ), we:
- Include repeated indices for powers: [1,1] → d₁²
- Include different indices for cross-terms: [1,2] → d₁·d₂
- Avoid duplicates: d₁·d₂ appears once, not also as d₂·d₁

### Why This Works

1. **Linear in parameters:** Despite the nonlinear model, θ appears linearly, enabling fast least-squares
2. **Systematic enumeration:** Index vectors ensure all terms up to degree m are included
3. **Separate modes:** Computing prediction and simulation separately shows different aspects of model quality
4. **Validation-based selection:** Using independent data prevents overfitting

---

## Model Selection Strategy

### The Bias-Variance Tradeoff

**Underfitting (too simple):**
- Low na, low m
- Model cannot capture system complexity
- High error on both training and validation

**Good fit:**
- Appropriate na and m
- Model captures essential dynamics
- Low error on both training and validation

**Overfitting (too complex):**
- High na, high m
- Model fits training noise
- Low training error, high validation error
- Simulation may become unstable (NaN values)

### Selection Criterion

**Primary metric:** MSE_sim_val (simulation error on validation data)

**Why simulation MSE?**
- Prediction uses real outputs → easier, masks some problems
- Simulation uses model outputs → harder, reveals true model quality
- Simulation error accumulates → realistic test of model dynamics

**Why validation data?**
- Independent from training → tests generalization
- Prevents selecting models that memorize training data

### Observed Results

From the tuning results table:

| na | m=1 | m=2 | m=3 | ... |
|---|-----|-----|-----|-----|
| 1 | 0.5881 | 0.2791 | 0.8359 | ... |
| 2 | 0.3336 | NaN | NaN | ... |
| **3** | **0.2215** | NaN | NaN | ... |

**Best model: na=nb=3, m=1**

**Observations:**
- m=1 (linear model) with sufficient history (na=3) works best
- Higher polynomial degrees (m≥2) cause instability for na≥2
- Sweet spot balances model capacity with stability

---

## Results

### Optimal Model Performance

**Best model:** na = nb = 3, m = 1, nk = 1

**Validation MSE:**
- Prediction: Very low error (~99.77% fit)
- Simulation: Low error (~94-95% fit)

### Visualization Analysis

#### 1. Identification Dataset

**Prediction vs Real Output:**
- Predicted output (red) closely follows real output (blue)
- Demonstrates successful parameter estimation
- Model captures both trends and local variations

**Simulation vs Real Output:**
- Simulated output shows good agreement with real output
- Slightly larger deviations than prediction (expected)
- No divergence or instability

#### 2. Validation Dataset

**Prediction vs Real Output:**
- Excellent tracking of real output
- Similar quality to identification data
- Confirms good generalization

**Simulation vs Real Output:**
- Good overall tracking
- Errors remain bounded
- No significant drift over time

#### 3. Error Analysis

**Prediction Errors:**
- Zero-mean (unbiased model)
- Bounded within ±2 units
- Similar distribution on ID and VAL datasets

**Simulation Errors:**
- Slightly larger than prediction errors
- Still zero-mean and bounded
- Comparable behavior on both datasets

**Key insight:** Similar error characteristics on ID and VAL confirm the model generalizes well.

### Comparison with MATLAB Built-in Models

**Linear ARX (na=3, nb=3, nk=1):**
- Prediction: ~99.77% fit
- Simulation: ~94.07% fit

**Output-Error (OE) Model:**
- Prediction: ~99.65% fit
- Simulation: ~90.69% fit

**Our nonlinear ARX (na=3, m=1):**
- Comparable to linear ARX (which makes sense since m=1 is linear)
- Better simulation than OE model
- Validates correctness of implementation

### Understanding Prediction vs Simulation Errors

**Why is simulation harder?**
1. **Error accumulation:** Each error feeds back into future predictions
2. **No correction:** Cannot use real outputs to reset the model
3. **Stability test:** Reveals if model dynamics match true system

**Gap between prediction and simulation:**
- Natural and expected
- Shows the challenge of autonomous operation
- Larger gap indicates model dynamics don't perfectly match reality

---

## Conclusions

### Key Findings

1. **Polynomial nonlinear ARX models can capture nonlinear dynamics** while maintaining computational efficiency through linear parameter estimation.

2. **Linear-in-parameters structure** enables use of fast least-squares methods despite modeling nonlinear systems.

3. **Model order selection is critical:**
   - Too simple (low na, low m) → underfitting
   - Too complex (high na, high m) → overfitting and instability
   - Optimal: na=nb=3, m=1 for this system

4. **Validation-based selection using simulation MSE** provides robust model selection that balances:
   - Accuracy on training data
   - Generalization to new data
   - Stability in autonomous simulation

5. **Two operating modes serve different purposes:**
   - Prediction: Shows model's ability to correct errors with real data
   - Simulation: Tests model's intrinsic dynamics and long-term behavior

### Limitations

1. **Simulation degradation:** Errors accumulate over time in simulation mode
2. **Sensitivity to noise:** Unmodeled dynamics and measurement noise affect performance
3. **Instability risk:** High-degree polynomials (m≥2) with multiple lags can become unstable
4. **Fixed structure:** Polynomial basis may not be optimal for all nonlinearities

### Practical Applications

This approach is valuable for:
- **Process control:** Modeling chemical reactors, thermal systems
- **Robotics:** Joint dynamics with friction and saturation
- **Economics:** Nonlinear market models
- **Biology:** Population dynamics, enzyme kinetics

Anywhere you have:
- Input-output data from a dynamic system
- Unknown or complex nonlinear behavior
- Need for prediction or simulation capabilities

### Best Practices

1. **Always validate on independent data** - don't trust training error alone
2. **Test simulation mode** - it reveals problems prediction mode can hide
3. **Start simple** - try low-order models before increasing complexity
4. **Check for instability** - watch for NaN or diverging simulations
5. **Compare with baselines** - validate against known methods (linear ARX, OE)

---

## Usage

### Requirements

- MATLAB (tested on R2020b or later)
- System Identification Toolbox (optional, for comparison)
- Data file: `iddata-13.mat`

### Running the Code

1. Place `ProjectPart2.m` and `iddata-13.mat` in the same directory
2. Open MATLAB and navigate to the directory
3. Run the script:
   ```matlab
   ProjectPart2
   ```

### Expected Outputs

The script will generate:

1. **Input/Output plots** for both ID and VAL datasets
2. **MSE tables** showing performance for different (na, m) combinations
3. **Best model identification** with highlighted optimal parameters
4. **Performance plots** for the best model:
   - ID prediction vs real output
   - ID simulation vs real output
   - VAL prediction vs real output
   - VAL simulation vs real output
5. **Error plots** showing prediction and simulation errors
6. **Comparison plots** with MATLAB built-in models (if toolbox available)

### Customization

You can modify these parameters in the script:

```matlab
nk = 1            % Input delay (fixed for this project)
n = 3             % Maximum model order to test (na = nb ≤ n)
m = 10            % Maximum polynomial degree to test
```

**Warning:** High values of n and m significantly increase computation time and may cause numerical instability.

---

## File Structure

```
Part2/
├── README.md           # This file
├── ProjectPart2.m      # Main MATLAB implementation
└── iddata-13.mat       # Dataset (ID and VAL)
```

---

## References

1. Ljung, L. (1999). *System Identification: Theory for the User*. Prentice Hall.
2. Nelles, O. (2001). *Nonlinear System Identification*. Springer.
3. MATLAB System Identification Toolbox Documentation
4. Course materials: System Identification (Technical University of Cluj-Napoca)

---

## License

This project was completed as part of the System Identification course at the Technical University of Cluj-Napoca.
