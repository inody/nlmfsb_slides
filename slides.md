---
theme: default
background: null
fonts:
  sans: [ "Poppins", "Noto Sans JP" ]
  serif: "Roboto Slab"
  mono: 'Fira Code'
transition: fade
---

## Nonlocal Mean-Field Schrödinger Bridge with Learned Interactions

<br>

### Daisuke Inoue

<br>

### Outline

- Extended Mean-Field Schrödinger Bridge to handle nonlocal interactions
- Proposed a **four-stage alternating algorithm** reducing complexity from $O(N^2)$ to $O(N)$, where $N$ is the number of particles
- Provided **stability analysis** with Gronwall-type error bounds
- Demonstrated **~70% training time reduction** while maintaining trajectory fidelity

---

# Problem Statement

## Optimal transport of interacting particles

For $N$ agents with controls $\{\alpha_i\}$ and empirical measure $m(x,t) = \tfrac{1}{N}\sum_{j=1}^N \delta_{x-x_j(t)}$:
$$
\text{min}_{\{\alpha_i\}} \sum_{i=1}^N \bbE\!\left[\int_0^T \frac{1}{2\sigma^2}\|\alpha_i(t)\|^2 + F(x_i(t),t, m_t)\,dt\right]
$$
subject to
$$
dx_i(t) = \big(\alpha_i(t) + f(x_i(t),t, m_t)\big)\,dt + \sigma\, dW_i(t),\quad
x_i(0)\sim\rho_0,\; x_i(T)\sim\rho_T.
$$

## Applications

- **Control/Navigation** Swarm robotics control with collision avoidance / crowd navigation through complex environments
- **Reconstruction** Cell data interpolation from sparse observations

---

## Mathematical Motivation

- If $f=F=0$, this reduces to the classical Schrödinger Bridge, i.e., entropy-regularized OT minimizing $\mathrm{KL}(\rho_0\|\rho_T)$ along diffusion paths.

- Liu et al. (2022): DeepGSB for **local** interactions：$f(x,t,m_t)=f(x,t,m(x,t))$
- We want to build efficient solver for **nonlocal** interactions: 
   $$
   f(x,t,m_t) = f(x,t,m(\cdot,t)) = \frac{1}{N}\sum_{j=1}^N k(x,x_j(t))
   $$

- **Problem**: $O(N^2)$ cost -> Intractable for large-scale systems
- **Objective**: Develop an algorithm that remains fast even for large $N$


---

# Background

## Mean-Field Schrödinger Bridge (MFSB)

As $N\to\infty$, the empirical interaction $m_t$ converges to a density $\rho_t=\rho(\cdot,t)$, yielding the MFSB formulation:

$$
\begin{aligned}
&\min_{\alpha} \int_0^T \int \left[\frac{1}{2\sigma^2}\|\alpha(x,t)\|^2 + F(x,t,\rho_t) \right] \rho(x,t) \, dx \, dt \\
&\text{s.t.} \quad \partial_t \rho = -\nabla\cdot[\rho(x,t)\, (\alpha(x,t) + f(x,t,\rho_t))] + \frac{\sigma^2}{2}\Delta \rho(x,t) \\
&\qquad\quad \rho(\cdot, 0) = \rho_0, \quad \rho(\cdot, T) = \rho_T
\end{aligned}
$$

- The nonlocal interaction function is expressed as $f(x,t,\rho_t)=\int k(x,y)\rho(y,t)dy$.
- Convergence as $N\to\infty$ (propagation of chaos) is studied by Camilo et al. (2024).


---

## FBSDE Characterization

- For MFSB, Liu et al. (2022) derived FBSDE Characterization.
- From the stationarity conditions of Lagrangian, we obtain the HJB–FP system, and its solution is equivalent to a pair of FBSDEs:

**Forward** (from $X_0 \sim \rho_0$):
$$
\begin{aligned}
dX_t &= (f(X_t,t,\rho_t) + \sigma Z_t)dt + \sigma dW_t \\
dY_t &= \left(\tfrac{1}{2}\|Z_t\|^2 + F(X_t,t,\rho_t)\right)dt + Z_t^\top dW_t \\
d\hat{Y}_t &= \left(\tfrac{1}{2}\|\hat{Z}_t\|^2 + \nabla\cdot(\sigma\hat{Z}_t - f) + \hat{Z}_t^\top Z_t - F\right)dt + \hat{Z}_t^\top dW_t
\end{aligned}
$$

**Backward** (from $\bar{X}_0 \sim \rho_T$) derived similarly.

- $Y$ solves the HJB equation (Value function), and $\hat{Y}$ plays the dual role in the backward process.
- $Z = \sigma\nabla Y$, $\hat{Z} = \sigma\nabla\hat{Y}$: Gradients / $\alpha^* = \sigma Z$: Optimal control

---

## Neural Network Parametrization

- Replace unknown functions with neural networks: $Y_\theta(x,t)$, $Z_\theta(x,t)$, $\hat{Y}_\phi(x,t)$, $\hat{Z}_\phi(x,t)$

- Loss Functions: 
$$\mathcal{L}_{\text{total}} = \lambda_{\text{IPF}}\mathcal{L}_{\text{IPF}} + \lambda_{\text{TD}}\mathcal{L}_{\text{TD}} + \lambda_{\text{FK}}\mathcal{L}_{\text{FK}}$$



#### IPF Loss (Iterative Proportional Fitting)

$$
\mathcal{L}_{\text{IPF}}(\theta) = \int_0^T \mathbb{E}_{X_t}\left[\frac{1}{2}\|\hat{Z}_t^\phi + Z_t^\theta\|^2 + \nabla \cdot (\sigma Z_t^\theta - f)\right] dt
$$

- Ensures trajectories satisfy **boundary constraints** $\rho_0 \to \rho_T$
- Derived from the compatibility condition between forward and backward processes
- When $\hat{Z} + Z = 0$, the forward and backward densities match at all times

---

#### TD Loss (Temporal Difference)

$$
\begin{aligned}
\mathcal{L}_{\text{TD}}(\theta) &= \sum_{s=0}^{T-\delta s} \mathbb{E}_{\bar{X}_s}\left[\|Y_\theta(\bar{X}_{s+\delta s}, s+\delta s) - \text{TD}_s\|^2\right] \delta s,\\
\text{TD}_{s} &:= Y_s^\theta + \left(\frac{1}{2}\|Z_s^\theta\|^2 + \nabla \cdot (\sigma Z_s^\theta + f_s) + (Z_s^\theta)^T \hat{Z}_s^\phi - F_s\right) \delta s + (Z_s^\theta)^T \delta W_s
\end{aligned}
$$

- Enforces **temporal consistency** of the value function $Y$
- Ensures $Y$ satisfies the backward SDE equation

#### FK Loss (Feynman-Kac)

$$
\mathcal{L}_{\text{FK}}(\theta) = \sum_{s=0}^{T} \mathbb{E}_{\bar{X}_s}\left[\|\sigma \nabla Y_\theta(\bar{X}_s, s) - Z_\theta(\bar{X}_s, s)\|^2\right] \delta s
$$

- Enforces the **gradient relationship**: $Z = \sigma \nabla Y$

---

## DeepGSB Algorithm

**Two-stage cycle**:

1. **Backward drift update** ($\phi$)
   - Generate forward trajectories using $(Y_\theta, Z_\theta)$
   - Update $(\hat{Y}_\phi, \hat{Z}_\phi)$ to minimize $\mathcal{L}(\phi)$

2. **Forward drift update** ($\theta$)
   - Generate backward trajectories using $(\hat{Y}_\phi, \hat{Z}_\phi)$
   - Update $(Y_\theta, Z_\theta)$ to minimize $\mathcal{L}(\theta)$

**Limitation**: Each trajectory simulation requires $O(N^2 K)$ interaction evaluations.
   - $N$: Number of Agents
   - $K$: Time steps

---

# Our Contribution

## Key Idea: Surrogate Modeling

Instead of computing $\int k(x,y)\rho(y,t)dy$ repeatedly, **approximate** with neural networks:

$$
f(x,t,\rho_t) \approx \hat{f}_\psi(x,t), \quad F(x,t,\rho_t) \approx \hat{F}_\psi(x,t)
$$

### Training Strategy

- Compute analytical interactions per outer loop
- Train $\hat{f}_\psi, \hat{F}_\psi$ on cached trajectories using:

$$
\mathcal{L}_{\text{int}}(\psi) = \mathbb{E}\Big[\|f(x,t,\rho_t) - \hat{f}_\psi(x,t)\|^2 + \|F(x,t,\rho_t) - \hat{F}_\psi(x,t)\|^2\Big]
$$

**Benefit**: Inference uses $O(N)$ network evaluations instead of $O(N^2)$ summations.

---

## Four-Stage Alternating Algorithm

**Four-stage cycle**:

1. **Backward drift update** ($\phi$) (Repeat $M_{\text{SDE}}$ times)
   Simulate forward SDE with learned $Y_\theta, Z_\theta, \hat{f}_\psi, \hat{F}_\psi$ → Update $\hat{Y}_\phi, \hat{Z}_\phi$

2. **Forward interaction update** ($\psi$) (Repeat $M_{\text{int}}$ times)
   Compute analytical $f, F$ on forward trajectories → Update $\hat{f}_\psi, \hat{F}_\psi$

3. **Forward drift update** ($\theta$) (Repeat $M_{\text{SDE}}$ times)
   Simulate backward SDE with learned $\hat{Y}_\phi, \hat{Z}_\phi, \hat{f}_\psi, \hat{F}_\psi$ → Update $Y_\theta, Z_\theta$

4. **Backward interaction update** ($\psi$) (Repeat $M_{\text{int}}$ times)
   Compute analytical $f, F$ on backward trajectories → Update $\hat{f}_\psi, \hat{F}_\psi$

---

## Computational Complexity Analysis


### **Cost Components**

For $N$ = number of agents, $K$ = time steps, $H$ = width, $L$ = layers:
- Analytical interaction: $C_{\text{ana}} = O(K N^2 (d + d_{\mathrm{out}}))$
- NN forward/backward: $C_{\text{NN}} = O(K N (d H + (L-1)H^2 + H d_{\mathrm{out}}))$

### **Per Directional Pass**

- Proposed:
$C_{\text{prop}} = M_{\text{int}} C_{\text{ana}} + 2(M_{\text{SDE}} + M_{\text{int}}) C_{\text{NN}}$

- Baseline (direct evaluation):
$C_{\text{direct}} = M_{\text{SDE}} (C_{\text{ana}} + C_{\text{NN}})$

### **Efficiency Condition**

Define $\gamma = M_{\text{int}} / M_{\text{SDE}}$ and $R = C_{\text{ana}} / C_{\text{NN}}$, then efficiency condition is 
$
R > \frac{1 + 2\gamma}{1 - \gamma}
$

- **Large $N$**: $R = O(N) \implies$ Always efficient
- **High $d$**: Efficient if $N/H$ is large enough
- **Small $\gamma$**: Efficient if $C_{\text{ana}} > C_{\text{NN}}$

---

# Theoretical Analysis

## Setting

### **Assumption 1** (Strong Solution Setting)

- True solution $(X, Y, Z)$ and learned solution $(\tilde{X}, \tilde{Y}, \tilde{Z})$ are strong solutions
- Both driven by the **same Brownian motion** $W$ on the same probability space
- Share the **same initial condition**: $X_0 = \tilde{X}_0 \sim \rho_0$



### **Assumption 2** (Interaction Learning Error)

$$
\sup_{x,t} \|f(x,t,\rho_t) - \hat{f}(x,t)\| \le \varepsilon_f, \quad \sup_{x,t} |F(x,t,\rho_t) - \hat{F}(x,t)| \le \varepsilon_F
$$
where $\rho_t$ is a solution to the MFSB.

---


## Stability Question

### **How do approximation errors $\varepsilon_f, \varepsilon_F$ propagate to trajectory errors?**


Define error processes:
$$
\Delta X_t \coloneqq X_t - \tilde{X}_t, \quad \Delta Y_t \coloneqq Y_t - \tilde{Y}_t, \quad \Delta Z_t \coloneqq Z_t - \tilde{Z}_t
$$

### **Goal**: Bound the total energy error
$$
E(t) := \mathbb{E}\|\Delta X_t\|^2 + \mathbb{E}|\Delta Y_t|^2 + \mathbb{E}\|\Delta Z_t\|^2
$$


---

## Necessary Assumptions

### **Assumption 3** (Regularity of Interactions)

- Interactions $(f, F)$ and surrogates $(\hat{f}, \hat{F})$ are **Lipschitz continuous**:
  $$\|f(x,t,\rho) - f(x',t,\rho')\| \le L_x^{(f)}\|x-x'\| + L_\mu^{(f)} W_2(\rho, \rho')$$


### **Assumption 4** (Boundedness of Gradients)

- First-order: $\sup_{x,t} \|\nabla Y(x,t)\| \le C_{\nabla Y}$, $\sup_{x,t} \|\nabla \tilde{Y}(x,t)\| \le C_{\nabla \tilde{Y}}$
- Implies $\|Z_t\|, \|\tilde{Z}_t\| \le \sigma C_Z$ where $C_Z = \max\{C_{\nabla Y}, C_{\nabla \tilde{Y}}\}$

---

### **Assumption 5** (Regularity of Interaction Functions)

**(i) Uniform boundedness:**
$$
\sup_{x,t,\rho_t}\|f(x,t,\rho_t)\|\le L_f^\infty, \quad \sup_{x,t,\rho_t}|F(x,t,\rho_t)|\le L_F^\infty
$$
(same for $\hat{f}, \hat{F}$)

**(ii) Bounded spatial derivatives of cost $F$:**
$$
\sup_{x,t,\rho_t}\|\nabla^2 F(x,t,\rho_t)\|\le L_{\nabla^2 F}^\infty
$$
(same for $\hat{F}$)


**(iii) Bounded spatial derivatives of drift $f$:**

$$
\sup_{x,t,\rho_t}\|\nabla_x f(x,t,\rho_t)\|\le L_{\nabla f}^{\infty},\quad \sup_{x,t,\rho_t}\|\nabla_x^2 f(x,t,\rho_t)\|\le L_{\nabla^2 f}^{\infty}
$$
(same for $\hat{f}$)


---

### **Assumption 6** (Regularity of Value Functions $Y, \tilde{Y}$)

For all multi-indices $(k_t, k_x)$ with $k_t \in \{0,1\}$, $k_x \in \{0,1,2,3\}$:
$$
\sup_{x,t}\|\partial_t^{k_t}\nabla_x^{k_x}Y(x,t)\|\le C_{\nabla Y}^{(k_t,k_x)}, \quad
\sup_{x,t}\|\partial_t^{k_t}\nabla_x^{k_x}\tilde{Y}(x,t)\|\le C_{\nabla \tilde{Y}}^{(k_t,k_x)}
$$

- Implications for $Z = \sigma \nabla Y$: Define $C_{\nabla Z}^{(k)} := \max\{C_{\nabla Y}^{(k+1)}, C_{\nabla \tilde{Y}}^{(k+1)}\}$, then:
$$
\sup_{x,t}\|\nabla_x^{k} Z(x,t)\|,\ \sup_{x,t}\|\nabla_x^{k} \tilde{Z}(x,t)\| \le \sigma C_{\nabla Z}^{(k)}, \quad k=0,1,2,3
$$


### **Assumption 7** (Bounded difference in high-order gradients)

For $k \in \{2, 3\}$:
$$
\sup_{x,t}\|\nabla^k Y(x,t) - \nabla^k \tilde{Y}(x,t)\|\le \varepsilon_{Y}^{(k)}
$$

- Consequence for $Z$: 
$
\sup_{x,t}\|\nabla^{k-1} Z(x,t) - \nabla^{k-1} \tilde{Z}(x,t)\| \le \sigma \varepsilon_{Y}^{(k)}, \quad k=2,3
$


---

## Main Result: Gronwall-Type Bound

### **Theorem** (Energy Inequality and Global Bounds)

Total energy error $E(t) := \mathbb{E}\|\Delta X_t\|^2 + \mathbb{E}|\Delta Y_t|^2 + \mathbb{E}\|\Delta Z_t\|^2$ satisfies:

$$
\frac{\mathrm d}{\mathrm dt} E(t) \le C_E E(t) + C_{\varepsilon_f}\varepsilon_f^2 + C_{\varepsilon_F}\varepsilon_F^2 + C_{Z,2}(\varepsilon_Y^{(2)})^2 + C_{Z,3}(\varepsilon_Y^{(3)})^2
$$

By Gronwall's inequality:

$$
E(t) \le e^{C_E t} E(0) + \frac{e^{C_E t}-1}{C_E}\left(C_{\varepsilon_f}\varepsilon_f^2 + C_{\varepsilon_F}\varepsilon_F^2 + C_{Z,2}(\varepsilon_Y^{(2)})^2 + C_{Z,3}(\varepsilon_Y^{(3)})^2\right)
$$


1. $\varepsilon_f = \sup_{x,t} \|f(x,t,\rho_t) - \hat{f}(x,t)\|$：Error from surrogate drift $\hat{f}(x,t)$
  
2. $\varepsilon_F = \sup_{x,t} |F(x,t,\rho_t) - \hat{F}(x,t)|$：Error from surrogate cost $\hat{F}(x,t)$
  
3. $\varepsilon_Y^{(2)} = \sup_{x,t} \|\nabla^2 Y(x,t) - \nabla^2 \tilde{Y}(x,t)\|$：Second-order gradient gap
   
4. $\varepsilon_Y^{(3)} = \sup_{x,t} \|\nabla^3 Y(x,t) - \nabla^3 \tilde{Y}(x,t)\|$： Third-order gradient gap 

---

## Interpretation of the Bound


### **What we proved: Stability**
- **Error Propagation**: Small interaction errors ($\varepsilon_f, \varepsilon_F$) guarantee small trajectory errors ($E(t)$).
- **Computability**: The growth rate $C_E$ is explicitly determined by system parameters (Lipschitz constants, $\sigma$, $T$).

### **What remains open: Convergence**
- The theorem assumes the **existence** of the FBSDE solution.
- It does **not** prove that the alternating algorithm converges to this solution.
- **Future Work**: Rigorous convergence analysis of the coupled FBSDE + interaction learning process.

---

## Proof Sketch: Overview

The proof proceeds in **three main steps**:

1. **Derive differential inequalities for $\Delta X_t, \Delta Y_t$**
   - Use Itô's lemma on the error SDEs
   - Bound cross-terms using Young's inequality

2. **Derive differential inequality for $\Delta Z_t$**
   - Apply Itô-Wentzell formula (requires high-order regularity)
   - Linearize the error dynamics

3. **Combine and apply Gronwall**
   - Sum all three inequalities to get total energy $E(t)$
   - Integrate using Gronwall's lemma

---

### Step 1: Error Dynamics for $\Delta X_t$

True SDE:
$$dX_t = (f(X_t,t,\rho_t) + \sigma Z_t)dt + \sigma dW_t$$

Learned SDE:
$$d\tilde{X}_t = (\hat{f}(\tilde{X}_t,t) + \sigma \tilde{Z}_t)dt + \sigma dW_t$$

Subtract to get:
$$
d\Delta X_t = \underbrace{[f(X_t,t,\rho_t) - f(\tilde{X}_t,t,\rho_t)]}_{\text{Lipschitz term}} + \underbrace{[f(\tilde{X}_t,t,\rho_t) - \hat{f}(\tilde{X}_t,t)]}_{\text{approx. error } \varepsilon_f} + \sigma \Delta Z_t \, dt
$$

Apply Itô lemma to $\|\Delta X_t\|^2$ and use Lipschitz + Young's inequality:
$$
\frac{d}{dt}\mathbb{E}\|\Delta X_t\|^2 \le C_{XX}\mathbb{E}\|\Delta X_t\|^2 + C_{XZ}\mathbb{E}\|\Delta Z_t\|^2 + C_{X,\varepsilon_f}\varepsilon_f^2
$$

---

### Step 1 (cont.): Error Dynamics for $\Delta Y_t$

True SDE:
$dY_t = \left(\frac{1}{2}\|Z_t\|^2 + F(X_t,t,\rho_t)\right)dt + Z_t^\top dW_t$

Learned SDE:
$d\tilde{Y}_t = \left(\frac{1}{2}\|\tilde{Z}_t\|^2 + \hat{F}(\tilde{X}_t,t)\right)dt + \tilde{Z}_t^\top dW_t$


Apply Itô lemma to $|\Delta Y_t|^2$:
$$
\frac{d}{dt}\mathbb{E}|\Delta Y_t|^2 \le C_{YY}\mathbb{E}|\Delta Y_t|^2 + C_{YX}\mathbb{E}\|\Delta X_t\|^2 + C_{YZ}\mathbb{E}\|\Delta Z_t\|^2 + C_{Y,\varepsilon_F}\varepsilon_F^2
$$


- For the quadratic term $\|Z\|^2 - \|\tilde{Z}\|^2$, decompose the difference:
  $$
  \|Z\|^2 - \|\tilde{Z}\|^2 = (Z - \tilde{Z})^\top (Z + \tilde{Z}) = \Delta Z^\top (Z + \tilde{Z})
  $$

- Bound the cross-term in the drift $2\Delta Y (\frac{1}{2}(\|Z\|^2 - \|\tilde{Z}\|^2))$:
  $$
  |\Delta Y \Delta Z^\top (Z + \tilde{Z})| \le \underbrace{\frac{\beta}{4}|\Delta Y|^2}_{\text{absorb to } C_{YY}} + \underbrace{\frac{1}{\beta}\|\Delta Z\|^2 \cdot \sup \|Z+\tilde{Z}\|^2}_{\text{absorb to } C_{YZ}}
  $$


---

### Step 2: Error Dynamics for $\Delta Z_t$

Apply Itô-Wentzell formula to $Z(X_t, t)$:
$$
dZ_t = \underbrace{\left[-\sigma\nabla_x G^Z + P_t(f_t + \sigma Z_t) + \frac{\sigma^2}{2}\text{tr}\,Q_t\right]}_{\text{drift } b^Z(X_t, Z_t, P_t, Q_t, f_t, F_t, t)} dt + P_t \sigma dW_t
$$

where:
- $P_t = \nabla_x Z(X_t, t)$: Jacobian of $Z$
- $Q_t = \nabla_x^2 Z(X_t, t)$: Hessian of $Z$
- $G^Z = -\frac{1}{2}\|Z\|^2 + \frac{1}{\sigma}Z^\top f + \frac{\sigma}{2}\text{tr}\,P + F$

- We used the fact that $Y$ satisfies the HJB equation to express $\partial_t Z$.

- Assumptions 5--7 are required: Bounded derivatives up to order 3 for $Y, \tilde{Y}$

---

### Step 2 (cont.): Linearized Error Equation

Apply **Mean Value Theorem** to the drift difference $b^Z - \tilde{b}^Z$:
$$
d\Delta Z_t = \left[\Theta_t^{(X)}\Delta X_t + \Theta_t^{(Z)}\Delta Z_t + \Theta_t^{(P)}:\Delta P_t + \Theta_t^{(Q)}\!:\!\!:\Delta Q_t + \mathcal{R}_t\right]dt + \mathcal{N}_t dW_t
$$

where:
- $\Theta_t^{(\cdot)} = \partial_{(\cdot)} b^Z(\cdots)$ are sensitivities to $\Delta X, \Delta Z, \Delta P, \Delta Q$
- $\mathcal{R}_t$: remainder terms including approximation errors $\varepsilon_f, \varepsilon_F$ and gradient gaps $\varepsilon_Y^{(2)}, \varepsilon_Y^{(3)}$
- $\mathcal{N}_t = (P_t - \tilde{P}_t)\sigma$: diffusion difference term

Apply Itô lemma to $\|\Delta Z_t\|^2$ and bound carefully:
$$
\frac{d}{dt}\mathbb{E}\|\Delta Z_t\|^2 \le C_{ZZ}\mathbb{E}\|\Delta Z_t\|^2 + C_{ZX}\mathbb{E}\|\Delta X_t\|^2 + C_{Z,f}\varepsilon_f^2 + C_{Z,2}(\varepsilon_Y^{(2)})^2 + C_{Z,3}(\varepsilon_Y^{(3)})^2
$$

---

### Step 3: Combine and Apply Gronwall

Sum the three inequalities:
$$
\frac{d}{dt}E(t) = \frac{d}{dt}\left[\mathbb{E}\|\Delta X_t\|^2 + \mathbb{E}|\Delta Y_t|^2 + \mathbb{E}\|\Delta Z_t\|^2\right]
$$

Key cancellation: Cross-terms involving $\mathbb{E}\|\Delta X_t\|^2$, $\mathbb{E}|\Delta Y_t|^2$, $\mathbb{E}\|\Delta Z_t\|^2$ combine into:
$$
\frac{d}{dt}E(t) \le \underbrace{(C_{XX} + C_{YX} + C_{ZX} + C_{YY} + C_{XZ} + C_{YZ} + C_{ZZ})}_{C_E} E(t) + \text{error terms}
$$

Apply Gronwall's inequality:
$$
E(t) \le e^{C_E t}E(0) + \int_0^t e^{C_E(t-s)}[\text{error terms}]\,ds
$$

Evaluate integral to get final bound with $(e^{C_E t} - 1)/C_E$ factor.

---

# Numerical Experiments: GMM Navigation Task

### Problem Setup

- **N = 200** particles in $\mathbb{R}^2$
- **Initial**: Standard Gaussian $\mathcal{N}(0, I)$ -> **Target**: 8-component GMM on a circle
- **Horizon**: $T = 1.0$, $K = 20$ time steps

### Nonlocal Interactions

**Drift** (Gaussian attraction):
$$f(x,t,\rho) = w \int e^{-\|x-y\|^2/2\sigma^2} \rho(y,t) dy, \quad w=2.0, \sigma=2.0$$

**Cost** (soft obstacles):
$$F(x,t,\rho) = 1500 \sum_{k=1}^3 \max(0, 1.5-\|x-c_k\|)^6$$

---

## Results: Trajectory Fidelity

<div class="grid grid-cols-2 gap-4">
<div>

### Analytic Interactions

<img src="./gmm_nav_intdrift2/trajectory_forward.gif" width="80%">

</div>
<div>

### Learned Interactions

<img src="./gmm_nav_intdrift2/trajectory_learned_forward.gif" width="80%">

</div>
</div>

**Observation**: Trajectories are visually similar → high-fidelity surrogate modeling.

---

## Results: Training Loss Convergence

<div class="grid grid-cols-2 gap-4">
<div>

### Forward Losses

<img src="./gmm_nav_intdrift2/loss_forward_vs_iter.png" width="100%">

</div>
<div>

### Backward Losses

<img src="./gmm_nav_intdrift2/loss_backward_vs_iter.png" width="100%">

</div>
</div>

- IPF, TD, FK losses converge **similarly** for both methods
- **Stable training** despite surrogate approximation

---

## Results: Computational Speedup

<div class="grid grid-cols-2 gap-4">
<div>

###  $N$ vs Iteration Time

<img src="./gmm_nav_intdrift2/gmm_agents_sweep_time_per_iter.png" width="100%">

Becomes faster for **$N \gtrsim 80$**
</div>
<div>

### Total Training Time ($N=200$)

<img src="./gmm_nav_intdrift2/total_loss_vs_time.png" width="100%">

**~70% faster** convergence

</div>
</div>

$O(N^2) \to O(N)$ complexity reduction translates to practical wall-clock savings.

---

## Results: Interaction Approximation

<div class="flex justify-center">
<img src="./gmm_nav_intdrift2/interaction_loss_vs_iter.png" width="55%">
</div>

- Interaction loss $\mathcal{L}_{\text{int}}$ decreases rapidly and stabilizes
- Confirms $\hat{f}_\psi, \hat{F}_\psi$ accurately capture analytical interactions
- Low final loss validates surrogate modeling approach

---

# Summary

## Main Contributions

### 1. Algorithm: Four-stage alternating training with surrogate modeling
- Reduces $O(N^2) \to O(N)$ for nonlocal interactions
- Applicable to both training and inference

### 2. Theory: Gronwall-type stability bounds
- Explicitly quantifies error propagation
- Provides guidance for approximation quality requirements

### 3. Experiments: ~70% training time reduction on crowd navigation
- Maintains trajectory fidelity comparable to analytical baseline
- Validates computational efficiency and approximation accuracy

---


## Current Limitations / Future Works

1. The theorem does **not prove** convergence of alternating algorithm
   - Need to investigate convergence of IPF for standard Schrödinger Bridge
2. The theorem assumes **existence** of FBSDE solutions
   - Not well understood whether solutions exist for systems with strong interactions
3. Validation on **high-dimensional, large-scale** problems
   - Want to solve and compare results for d>10, N>1000. GPU memory is insufficient; will conduct experiments when new hardware arrives


---

## Network Architecture Details

<div class="grid grid-cols-2 gap-4">
<div>

### Value/Gradient Networks

```python
class YNet:
    input: (x, t) ∈ R^(d+1)
    Space: x → 128 → 128 → 128 (SiLU)
    Time: t → 128 → 128 (SiLU)
    Head: concat(256) → 128 → 1
    output: Y ∈ R
    lr: 1e-3
```

```python
class ZNet:
    input: (x, t) ∈ R^(d+1)
    Space: x → 128 → 128 → 128 (SiLU)
    Time: t → 128 → 128 (SiLU)
    Head: concat(256) → 128 → d
    output: Z ∈ R^d
    lr: 5e-4
```

</div>
<div>

### Interaction Networks

```python
class DriftNet:
    input: (x, t) ∈ R^(d+1)
    Space: x → 128 → 128 → 128 (SiLU)
    Time: t → 128 → 128 (SiLU)
    Head: concat(256) → 128 → d
    output: f ∈ R^d
    lr: 5e-4
```

```python
class CostNet:
    input: (x, t) ∈ R^(d+1)
    Space: x → 128 → 128 → 128 (SiLU)
    Time: t → 128 → 128 (SiLU)
    Head: concat(256) → 128 → 1
    output: F ∈ R
    lr: 5e-4
```

</div>
</div>

Architecture follows Liu et al.'s implementation with separate time/space branches.
