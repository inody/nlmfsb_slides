# Liu et al. (2022) Deep Generalized Schrödinger Bridge: 詳細分析

**最終更新**: 2025年10月23日

---

## 概要

このドキュメントは、Liu et al. (2022) "Deep Generalized Schrödinger Bridge" (NeurIPS 2022) の詳細な理論・アルゴリズム分析と、本プロジェクトとの関連をまとめたものです。

**論文リンク**: https://arxiv.org/abs/2209.09893
**GitHub**: https://github.com/ghliu/DeepGSB

---

## 1. 問題設定：Mean-Field Game (MFG) with Distributional Constraints

### 1.1. 従来のMFG

Hamilton-Jacobi-Bellman (HJB) 方程式とFokker-Planck (FP) 方程式の連立系：

$$
\begin{cases}
-\frac{\partial u(x,t)}{\partial t} + H(x, \nabla u, \rho) - \frac{\sigma^2}{2}\Delta u = F(x, \rho), & u(x, T) = G(x, \rho(\cdot, T)) \\
\frac{\partial \rho(x,t)}{\partial t} - \nabla \cdot (\rho \nabla_p H(x, \nabla u, \rho)) - \frac{\sigma^2}{2}\Delta \rho = 0, & \rho(x, 0) = \rho_0(x)
\end{cases}
$$

**課題**：
- $u(x,T)$ は $G(x, \rho(\cdot, T))$ として定義され、$\rho(x,T)$ に依存
- 目標分布 $\rho_{\text{target}}$ への厳密な収束を保証できない（ソフトペナルティのみ）

### 1.2. Liu et al. の拡張：Hard Distributional Constraints

Control-affine Hamiltonian $H(x, \nabla u, \rho) = \frac{1}{2}\|\sigma\nabla u\|^2 - \nabla u^\top f(x, \rho)$ を採用し、境界条件を変更：

$$
\begin{cases}
-\frac{\partial u}{\partial t} + \frac{1}{2}\|\sigma\nabla u\|^2 - \nabla u^\top f - \frac{\sigma^2}{2}\Delta u = F(x, \rho) \\
\frac{\partial \rho}{\partial t} - \nabla \cdot (\rho(\sigma^2\nabla u - f)) - \frac{\sigma^2}{2}\Delta \rho = 0, & \rho(x, 0) = \rho_0(x), \; \rho(x, T) = \rho_{\text{target}}(x)
\end{cases}
\tag{7}
$$

**重要な変更点**：
- HJBの終端条件 $u(x,T)$ がFP経由で**暗黙的**に定義される
- $\rho(x,T) = \rho_{\text{target}}(x)$ がハード制約となる
- これは最適輸送問題に類似（$\rho_0 \to \rho_{\text{target}}$ の変換）

---

## 2. Schrödinger Bridge との接続

### 2.1. Hopf-Cole 変換

以下の変換により、MFG PDEs (7) を Schrödinger Bridge PDEs に変換：

$$
\Psi(x, t) := \exp(-u(x, t)), \quad \hat{\Psi}(x, t) := \rho(x, t) \exp(u(x, t))
\tag{8}
$$

**導出**（詳細はAppendix A.4.1）：

$$
\begin{aligned}
\nabla \Psi &= -\exp(-u)\nabla u \\
\Delta \Psi &= \exp(-u)(\|\nabla u\|^2 - \Delta u) \\
\frac{\partial \Psi}{\partial t} &= \exp(-u)\left(-\frac{\partial u}{\partial t}\right) \\
&= \exp(-u)\left(-\frac{1}{2}\|\sigma\nabla u\|^2 + \nabla u^\top f + \frac{\sigma^2}{2}\Delta u + F\right) \\
&= -\nabla \Psi^\top f - \frac{\sigma^2}{2}\Delta \Psi + F\Psi
\end{aligned}
$$

同様に $\hat{\Psi}$ についても導出すると：

$$
\begin{cases}
\frac{\partial \Psi}{\partial t} = -\nabla \Psi^\top f - \frac{\sigma^2}{2}\Delta \Psi + F\Psi \\
\frac{\partial \hat{\Psi}}{\partial t} = -\nabla \cdot (\hat{\Psi}f) + \frac{\sigma^2}{2}\Delta \hat{\Psi} - F\hat{\Psi}
\end{cases}
\quad \text{s.t.} \quad
\begin{cases}
\Psi(\cdot, 0)\hat{\Psi}(\cdot, 0) = \rho_0 \\
\Psi(\cdot, T)\hat{\Psi}(\cdot, T) = \rho_{\text{target}}
\end{cases}
\tag{9}
$$

**従来のSB (F=0の場合、式4) との違い**：
- $+F\Psi$ と $-F\hat{\Psi}$ の項が追加される
- これが**Mean-Field interaction**を表現

### 2.2. エージェントダイナミクス

最適な制御により、各エージェントは以下のSDEに従う：

$$
dX_t = \left(f(X_t, t, \rho(\cdot, t)) + \sigma^2\nabla \log \Psi(X_t, t)\right)dt + \sigma dW_t, \quad X_0 \sim \rho_0
\tag{2}
$$

これは順方向SDE (3a) に一致：
$$
-\nabla_p H(X_t, \nabla u, \rho) = f - \sigma^2\nabla u = f + \sigma^2\nabla \log \Psi
$$

---

## 3. Generalized SB-FBSDEs (Theorem 2)

### 3.1. Nonlinear Feynman-Kac 変換

PDEs (9) に以下の変換を適用：

$$
\begin{aligned}
Y_t &\equiv Y(X_t, t) = \log \Psi(X_t, t), & Z_t &\equiv Z(X_t, t) = \sigma \nabla \log \Psi(X_t, t) \\
\hat{Y}_t &\equiv \hat{Y}(X_t, t) = \log \hat{\Psi}(X_t, t), & \hat{Z}_t &\equiv \hat{Z}(X_t, t) = \sigma \nabla \log \hat{\Psi}(X_t, t)
\end{aligned}
\tag{10}
$$

ここで $X_t$ は順方向SDE (3a) に従う：$X_0 \sim \rho_0$

### 3.2. Forward FBSDEs (w.r.t. forward SDE)

**Theorem 2 (Part 1)**：$\Psi, \hat{\Psi} \in C^{2,1}$ かつ $f, F$ が適切な正則性条件を満たすとき、以下のFBSDEsシステムが成立：

$$
\begin{cases}
dX_t = (f_t + \sigma Z_t)dt + \sigma dW_t \\[6pt]
dY_t = \left(\frac{1}{2}\|Z_t\|^2 + F_t\right)dt + Z_t^\top dW_t \\[6pt]
d\hat{Y}_t = \left(\frac{1}{2}\|\hat{Z}_t\|^2 + \nabla \cdot (\sigma\hat{Z}_t - f_t) + \hat{Z}_t^\top Z_t - F_t\right)dt + \hat{Z}_t^\top dW_t
\end{cases}
\tag{11}
$$

ここで：
- $f_t := f(X_t, \exp(Y_t + \hat{Y}_t))$
- $F_t := F(X_t, \exp(Y_t + \hat{Y}_t))$
- $Y_t + \hat{Y}_t = \log \rho(X_t, t)$ （構成より）

**導出のポイント**（Proof of Theorem 2, Appendix A.3.3）：
1. Itôの公式を $v = \log \Psi(X_t, t)$ に適用
2. PDE $\frac{\partial \log \Psi}{\partial t}$ を代入
3. Laplacian項を処理

### 3.3. Backward FBSDEs (w.r.t. backward SDE)

逆時間座標 $s := T - t$ を用いた backward SDE $\bar{X}_s \sim$ (3b), $\bar{X}_0 \sim \rho_{\text{target}}$ に対しても同様のFBSDEsが成立：

$$
\begin{cases}
d\bar{X}_s = (-f_s + \sigma\hat{Z}_s)ds + \sigma dW_s \\[6pt]
dY_s = \left(\frac{1}{2}\|Z_s\|^2 + \nabla \cdot (\sigma Z_s + f_s) + Z_s^\top \hat{Z}_s - F_s\right)ds + Z_s^\top dW_s \\[6pt]
d\hat{Y}_s = \left(\frac{1}{2}\|\hat{Z}_s\|^2 + F_s\right)ds + \hat{Z}_s^\top dW_s
\end{cases}
\tag{12}
$$

**従来のSB-FBSDE (F=0, 式22) との違い**：
- $Y_t$ のドリフトに $+F_t$
- $\hat{Y}_t$ のドリフトに $-F_t$
- これらが相殺しない → 新しい目的関数が必要

---

## 4. 損失関数の設計

### 4.1. Option 1: $\mathcal{L}_{\text{IPF}}$ のみ（不十分）

従来のSB-FBSDE [27] では、以下のIPF損失を使用：

**Forward IPF損失**：

$$
\mathcal{L}_{\text{IPF}}(\phi) := \int_0^T \mathbb{E}\left[\frac{1}{2}\|\hat{Z}^\phi_t + Z^\theta_t\|^2 + \nabla \cdot (\sigma\hat{Z}^\phi_t - f)\right]dt
\tag{6b}
$$

**Backward IPF損失**：

$$
\mathcal{L}_{\text{IPF}}(\theta) := \int_0^T \mathbb{E}\left[\frac{1}{2}\|Z^\theta_s + \hat{Z}^\phi_s\|^2 + \nabla \cdot (\sigma Z^\theta_s + f)\right]ds
\tag{6a}
$$

**問題点**：$dY^\theta_t + d\hat{Y}^\phi_t$ を積分すると $+F_t$ と $-F_t$ が相殺し、$F$ に依存しない！

$$
\mathcal{L}_{\text{IPF}}^{(11)}(\phi) = \int \mathbb{E}\left[dY^\theta_t + d\hat{Y}^\phi_t\right] = \int \mathbb{E}\left[\frac{1}{2}\|\hat{Z}^\phi_t + Z^\theta_t\|^2 + \nabla \cdot (\sigma\hat{Z}^\phi_t - f)\right]dt = \mathcal{L}_{\text{IPF}}^{(6b)}(\phi)
$$

→ Mean-field structure を捉えられない

**IPF損失のKL divergence解釈** (Lemma 1, Proposition 9)：

$$
\begin{aligned}
\mathcal{L}_{\text{IPF}}(\phi) &\propto D_{\text{KL}}(q^\theta \| q^\phi) \\
\mathcal{L}_{\text{IPF}}(\theta) &\propto D_{\text{KL}}(q^\phi \| q^\theta)
\end{aligned}
$$

ここで $q^\theta$, $q^\phi$ はそれぞれパラメータ化されたforward/backward SDEのpath-wise密度。

### 4.2. Option 2: $\mathcal{L}_{\text{IPF}} + \mathcal{L}_{\text{TD}}$（提案手法）

#### 4.2.1. Temporal Difference (TD) 目的関数

**重要な洞察**：$Y_t = \log \Psi(X_t, t) = -u(X_t, t)$ は**value function の確率的表現**

式 (11b) を離散化すると：

$$
Y^\theta_{t+\delta t} = Y^\theta_t + \left(\frac{1}{2}\|Z^\theta_t\|^2 + F_t\right)\delta t + Z^\theta_t{}^\top \delta W_t, \quad \delta W_t \sim \mathcal{N}(0, \delta t I)
\tag{13}
$$

これは**continuous-time Bellman equation** に対応！

**Proposition 3: TD objectives**

**Forward Process用のTD objectives** (式14a, 15, 16)：

Single-step TD target：

$$
\text{TD}^{\text{single}}_{t+\delta t} := \hat{Y}^\phi_t + \left(\frac{1}{2}\|\hat{Z}^\phi_t\|^2 + \nabla \cdot (\sigma\hat{Z}^\phi_t - f_t) + \hat{Z}^\phi_t{}^\top Z^\theta_t - F_t\right)\delta t + \hat{Z}^\phi_t{}^\top \delta W_t
\tag{14a}
$$

Multi-step TD target：

$$
\text{TD}^{\text{multi}}_{t+\delta t} := \widehat{\text{TD}}_0 + \sum_{\tau=\delta t}^t \delta \hat{Y}_\tau, \quad \delta \hat{Y}_t := \text{TD}^{\text{single}}_{t+\delta t} - \hat{Y}_t
\tag{15}
$$

TD損失：

$$
\mathcal{L}_{\text{TD}}(\phi) = \sum_{t=0}^T \mathbb{E}\left[\|\hat{Y}^\phi(X_t, t) - \widehat{\text{TD}}_t\|\right]\delta t
\tag{16}
$$

**Backward Process用のTD objectives** (式14b)：

Single-step TD target：

$$
\text{TD}^{\text{single}}_{s+\delta s} := Y^\theta_s + \left(\frac{1}{2}\|Z^\theta_s\|^2 + \nabla \cdot (\sigma Z^\theta_s + f_s) + Z^\theta_s{}^\top \hat{Z}^\phi_s - F_s\right)\delta s + Z^\theta_s{}^\top \delta W_s
\tag{14b}
$$

Multi-step TD target：

$$
\text{TD}^{\text{multi}}_{s+\delta s} := \text{TD}_0 + \sum_{\tau=\delta s}^s \delta Y_\tau, \quad \delta Y_s := \text{TD}^{\text{single}}_{s+\delta s} - Y_s
$$

TD損失：

$$
\mathcal{L}_{\text{TD}}(\theta) = \sum_{s=0}^T \mathbb{E}\left[\|Y^\theta(\bar{X}_s, s) - \text{TD}_s\|\right]\delta s
$$

**境界項の初期化**：

$$
\begin{aligned}
\widehat{\text{TD}}_0 &:= \log \rho_0 - Y^\theta_0 \quad \text{(forward用)} \\
\text{TD}_0 &:= \log \rho_{\text{target}} - \hat{Y}^\phi_0 \quad \text{(backward用)}
\end{aligned}
$$

これらの境界項により、TD targetがそれぞれの境界分布 $\rho_0$ と $\rho_{\text{target}}$ の対数密度に正しく接続される。

**実装上の利点**：
- TD targetが**regressand**として現れる → $F$ は微分可能である必要なし
- Replay buffer, Target network (EMA) などDeepRL技術が使える

#### 4.2.2. 必要十分性（Proposition 4）

**Proposition 4**：関数 $(Y^\theta, Z^\theta, \hat{Y}^\phi, \hat{Z}^\phi)$ がFBSDEs (11, 12) を満たす $\Leftrightarrow$ それらが結合損失 $\mathcal{L}(\theta, \phi) := \mathcal{L}_{\text{IPF}}(\phi) + \mathcal{L}_{\text{TD}}(\phi) + \mathcal{L}_{\text{IPF}}(\theta) + \mathcal{L}_{\text{TD}}(\theta)$ の最小化点である

**証明の概略**（Appendix A.3.5）：

**必要性** ($\Rightarrow$)：FBSDEs (11, 12) が成立 → 損失最小

1. $\mathcal{L}_{\text{TD}}$ は構成から明らかに0
2. (11) より $Y^\theta_T + \hat{Y}^\phi_T = Y^\theta_0 + \hat{Y}^\phi_0 + \int_0^T (dY^\theta_t + d\hat{Y}^\phi_t)$
3. 両辺に $\mathbb{E}_{q^\theta}$ を取ると、Proposition 9より $D_{\text{KL}}(q^\theta \| q^\phi) = 0$
4. Lemma 1より $\mathcal{L}_{\text{IPF}}(\phi) \propto D_{\text{KL}}(q^\theta \| q^\phi) = 0$

**十分性** ($\Leftarrow$)：損失最小 → FBSDEs (11, 12) が成立（より複雑、Appendix参照）

### 4.3. Option 3: $\mathcal{L}_{\text{IPF}} + \mathcal{L}_{\text{TD}} + \mathcal{L}_{\text{FK}}$

実用上、$(Z^\theta, \hat{Z}^\phi)$ を独立にパラメータ化し、以下のFK consistency lossを追加：

**Forward FK損失** (Backward trajectoryで評価)：

$$
\mathcal{L}_{\text{FK}}(\theta) = \sum_{s=0}^T \mathbb{E}\left[\|\sigma\nabla Y^\theta(\bar{X}_s, s) - Z^\theta(\bar{X}_s, s)\|\right]\delta s
$$

**Backward FK損失** (Forward trajectoryで評価)：

$$
\mathcal{L}_{\text{FK}}(\phi) = \sum_{t=0}^T \mathbb{E}\left[\|\sigma\nabla \hat{Y}^\phi(X_t, t) - \hat{Z}^\phi(X_t, t)\|\right]\delta t
$$

**目的**：Nonlinear Feynman-Kac変換（式10）の整合性を保証

$$
\begin{aligned}
Z_t &= \sigma \nabla \log \Psi(X_t, t) = \sigma \nabla Y(X_t, t) \\
\hat{Z}_t &= \sigma \nabla \log \hat{\Psi}(X_t, t) = \sigma \nabla \hat{Y}(X_t, t)
\end{aligned}
$$

**利点**：追加のロバスト性

**DeepGSBでは**：
- **Critic parametrization** (Option 2)：$(Y, \hat{Y})$ のみパラメータ化、$Z := \sigma\nabla Y$, $\hat{Z} := \sigma\nabla \hat{Y}$
- **Actor-critic parametrization** (Option 3)：$(Y, Z, \hat{Y}, \hat{Z})$ すべて独立にパラメータ化

---

## 5. DeepGSB アルゴリズム (Algorithm 1)

### 5.1. 交互最適化構造

```
repeat:
    # Backward drift learning (φ の更新)
    1. Sample forward trajectory: X^θ ~ (11a) with current θ
       → Add to replay buffer B

    2. for k = 1 to K:
           Sample on-policy X^θ_on and off-policy X^θ_off from X^θ and B
           Compute L(φ) = L_IPF(φ; X^θ_on) + L_TD(φ; X^θ_on) + L_TD(φ; X^θ_off) + L_FK(φ; X^θ_on)
           Update φ ← φ - η∇_φ L(φ)

    # Forward drift learning (θ の更新)
    3. Sample backward trajectory: X̄^φ ~ (12a) with current φ
       → Add to replay buffer B̄

    4. for k = 1 to K:
           Sample on-policy X̄^φ_on and off-policy X̄^φ_off from X̄^φ and B̄
           Compute L(θ) = L_IPF(θ; X̄^φ_on) + L_TD(θ; X̄^φ_on) + L_TD(θ; X̄^φ_off) + L_FK(θ; X̄^φ_on)
           Update θ ← θ - η∇_θ L(θ)

until convergence
```

### 5.2. 主要な技術的工夫

1. **Replay Buffer**: Off-policy TD学習で安定性向上
2. **Target Network (EMA)**: TD targetの計算に使用、安定性向上
3. **Multi-step TD**: より長期的な情報を活用
4. **Actor-critic**: $(Y, Z)$ を別々にパラメータ化してロバスト性向上

---

## 6. 従来手法との比較

### 6.1. 既存のMFGソルバーとの違い

| 手法 | 連続状態空間 | 確率的MF動力学 | 厳密な$\rho_{\text{target}}$収束 | 不連続MF相互作用$F$ | 最高次元 |
|------|------------|--------------|--------------------------|-------------------|---------|
| Ruthotto et al. [14] | ✓ | ✗ | ✗ | ✗ | 100 |
| Lin et al. [15] | ✓ | ✗ | ✗ | ✗ | 100 |
| Chen [16] | ✗ | ✓ | ✓ | ✗ | 2 |
| **DeepGSB (Liu et al.)** | ✓ | ✓ | ✓ | ✓ | **1000** |

### 6.2. 従来のSBソルバーとの違い

| 手法 | MF相互作用 $F$ | 計算量 | 技術 |
|------|--------------|--------|------|
| SB-FBSDE [27] | $F = 0$ のみ | $O(N)$ | IPF loss |
| **DeepGSB** | 一般の $F(x, \rho)$ | $O(N)$ | IPF + TD loss |

**本質的な拡張**：$F \neq 0$ に対応するためTD lossを導入

---

## 7. 実験結果の要約

### 7.1. 2D Crowd Navigation

**3つのMFG**：
- **GMM**: 障害物回避（$F_{\text{obstacle}}$）
- **V-neck**: エントロピー相互作用 + V字ボトルネック（$F_{\text{obstacle}} + F_{\text{entropy}}$）
- **S-tunnel**: 混雑相互作用 + S字トンネル（$F_{\text{obstacle}} + F_{\text{congestion}}$）

**MF interactions**：

$$
F_{\text{entropy}} := \log \rho(x, t) + 1
$$

$$
F_{\text{congestion}} := \mathbb{E}_{y\sim\rho}\left[\frac{2}{\|x-y\|^2 + 1}\right]
$$

$$
F_{\text{obstacle}} := 1500 \cdot \mathbb{1}_{\text{obs}}(x) \quad \text{(不連続！)}
$$

**結果**：
- DeepGSBは不連続な障害物コストでも動作
- 従来手法 [14, 15] は微分可能な$F$が必要
- Chen [16] は離散状態空間が必要（計算量$O(D^2)$、$D$=グリッド点数）

### 7.2. 1000次元 Opinion Depolarization

**Polarized dynamics**（式18）：

$$
f_{\text{polarize}}(x, \rho; \xi) := \mathbb{E}_{y\sim\rho}[a(x, y; \xi)\bar{y}]
$$

ここで

$$
a(x, y; \xi) := \begin{cases}
1 & \text{if } \text{sign}(\langle x, \xi \rangle) = \text{sign}(\langle y, \xi \rangle) \\
-1 & \text{otherwise}
\end{cases}
$$

**結果**：
- DeepGSBは1000次元でも偏向分布を穏健分布に誘導成功
- 従来手法の最高次元は100次元

---

## 8. 本プロジェクトとの関連

### 8.1. Option A (実装済み)：解析的相互作用

**Liu et al. の直接適用**：
- $f(x, t, \rho(\cdot, t))$ と $F(x, t, \rho(\cdot, t))$ を解析的に与える
- `analytical_interactions.py` で実装
- DeepGSBの2-step cycle を使用

**実装対応**：
- `FBSDESolver`: FBSDEs (11, 12) の実装
- `Trainer`: Algorithm 1 の実装
- Loss functions: $\mathcal{L}_{\text{IPF}}$, $\mathcal{L}_{\text{TD}}$, $\mathcal{L}_{\text{FK}}$

### 8.2. Option B (提案手法)：学習型相互作用

**Liu et al. の拡張**：
- $f(x, t, \rho(\cdot, t)) \approx f_\alpha(x, t)$ をNNで近似
- $F(x, t, \rho(\cdot, t)) \approx F_\beta(x, t)$ をNNで近似
- **4-step cycle** に拡張：
  1. Backward drift learning
  2. Interaction learning (backward)
  3. Forward drift learning
  4. Interaction learning (forward)

**利点**：
- 推論時の計算量：$O(N^2) \to O(N)$
- Han et al. (2024) のアプローチを参考

### 8.3. アルゴリズムの違い

| 側面 | DeepGSB (Liu et al.) | 本プロジェクト Option A | 本プロジェクト Option B |
|------|---------------------|---------------------|---------------------|
| **交互最適化** | 2-step (forward/backward drift) | 2-step | 4-step (+ interaction learning) |
| **相互作用** | 解析的 | 解析的 | NN学習 |
| **MF計算** | 直接MC (毎iteration) | 直接MC | MC (学習時), NN (推論時) |
| **推論時計算量** | $O(N^2)$ | $O(N^2)$ | **$O(N)$** |

---

## 9. 数学的ツールボックス

### 9.1. Itôの公式 (Lemma 6)

$X_t$ が $dX_t = f(X_t, t)dt + \sigma(X_t, t)dW_t$ に従うとき、$v(X_t, t) \in C^{2,1}$ は

$$
dv(X_t, t) = \frac{\partial v}{\partial t}dt + \nabla v^\top f dt + \frac{1}{2}\text{Tr}(\sigma^\top \nabla^2 v \sigma)dt + \nabla v^\top \sigma dW_t
$$

### 9.2. Hopf-Cole変換の公式 (Appendix A.4.1)

$$
\begin{aligned}
\nabla \Psi &= -\exp(-u)\nabla u \\
\Delta \Psi &= \exp(-u)(\|\nabla u\|^2 - \Delta u) \\
\nabla \hat{\Psi} &= \exp(u)(\rho\nabla u + \nabla \rho) \\
\Delta \hat{\Psi} &= \exp(u)[\rho\|\nabla u\|^2 + 2\nabla\rho^\top\nabla u + \Delta\rho + \rho\Delta u]
\end{aligned}
$$

### 9.3. KL divergenceの表現 (Proposition 9)

$$
D_{\text{KL}}(q^\theta \| q^\phi) = \int_0^T \mathbb{E}_{q^\theta_t}\left[\frac{1}{2}\|\hat{Z}^\phi + Z^\theta\|^2 + \nabla \cdot (\sigma\hat{Z}^\phi - f)\right]dt + \mathbb{E}_{q^\theta_0}[\log \rho_0] - \mathbb{E}_{q^\theta_T}[\log \rho_{\text{target}}]
$$

**導出**：Girsanovの定理 + Lemma 8 (Vargas [38])

---

## 10. 実装上の重要ポイント

### 10.1. TD targetの計算

**Single-step** (式14)：現在の状態から1ステップ先を予測

**Multi-step** (式15)：初期状態から累積的に予測

**実装推奨**：Multi-stepの方が性能良い（DeepRL文献と一致 [50-52]）

### 10.2. Divergence項の計算

$$
\nabla \cdot (\sigma\hat{Z}^\phi - f) = \sigma \nabla \cdot \hat{Z}^\phi - \nabla \cdot f
$$

**Hutchinson's trace estimator**：
$$
\nabla \cdot \hat{Z}^\phi = \text{Tr}(\nabla \hat{Z}^\phi) \approx \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}[\epsilon^\top (\nabla \hat{Z}^\phi) \epsilon]
$$

### 10.3. 境界条件の扱い

#### 10.3.1. 境界分布の設定

**PDEレベルでの境界条件**（式9）：

$$
\begin{cases}
\Psi(\cdot, 0)\hat{\Psi}(\cdot, 0) = \rho_0 \\
\Psi(\cdot, T)\hat{\Psi}(\cdot, T) = \rho_{\text{target}}
\end{cases}
$$

これは以下のように解釈される：

$$
\begin{aligned}
\rho(x, 0) &= \Psi(x, 0) \cdot \hat{\Psi}(x, 0) = \rho_0(x) \\
\rho(x, T) &= \Psi(x, T) \cdot \hat{\Psi}(x, T) = \rho_{\text{target}}(x)
\end{aligned}
$$

#### 10.3.2. FBSDEsレベルでの境界条件

**Forward FBSDEs**（式11）：
- $X_0 \sim \rho_0$（初期分布からサンプリング）
- $X_T$ の分布が自動的に $\rho_{\text{target}}$ に収束

**Backward FBSDEs**（式12）：
- $\bar{X}_0 \sim \rho_{\text{target}}$（目標分布からサンプリング）
- $\bar{X}_T$ の分布が自動的に $\rho_0$ に収束

#### 10.3.3. TD目的関数における境界項

**境界項の初期値**：

$$
\begin{aligned}
\widehat{\text{TD}}_0 &= \log \rho_0(X_0) - Y^\theta_0 \quad \text{(forward multi-step TD用)} \\
\text{TD}_0 &= \log \rho_{\text{target}}(\bar{X}_0) - \hat{Y}^\phi_0 \quad \text{(backward multi-step TD用)}
\end{aligned}
$$

**重要な性質**：
- $Y_t + \hat{Y}_t = \log \rho(X_t, t)$（構成より）
- この関係により、境界条件が暗黙的に満たされる

#### 10.3.4. DeepGSBの重要な発見

**DeepGSBの画期的な特性**（Figure 7, Section 4.3）：

$\rho_0$, $\rho_{\text{target}}$ の**密度関数が未知でもサンプリング可能なら学習可能**！

**学習可能な理由**：
1. **TD objectives における self-consistency**：
   - Single-step TD target（式14）は現在の状態のみに依存
   - 境界での密度値がなくても、相対的な値関数の変化を学習可能

2. **IPF objective における KL-matching**：
   - $\mathcal{L}_{\text{IPF}}(\phi) \propto D_{\text{KL}}(q^\theta \| q^\phi)$
   - Path-wise密度のKL divergenceは相対的な確率のみに依存

3. **交互最適化による境界条件の暗黙的満足**：
   - Forward/backwardの交互更新により、両方向からの制約が自然に満たされる

**従来手法との対比**：

| 手法 | 境界条件の扱い | 密度関数の必要性 |
|------|--------------|----------------|
| Ruthotto et al. [14] | $D_{\text{KL}}(\rho \| \rho_{\text{target}})$ を直接微分 | ✓ 必要 |
| Lin et al. [15] | 全状態空間で密度を回帰 | ✓ 必要 |
| **DeepGSB** | Self-consistency + KL-matching | ✗ 不要（サンプリングのみ） |

**実装上の注意**：
- 密度関数なしで学習する場合、single-step TD（式14）を使用
- 密度関数が利用可能な場合、multi-step TD（式15）でより良い性能
- Figure 7 では密度なしでも収束することを実証

---

## 11. 理論的保証と限界

### 11.1. 収束性について

**Proposition 4** は必要十分条件を提供するが、大域的収束は保証しない

**類似性**：Trust Region Policy Optimization (TRPO) [55]
- 両方とも $\pi^{(i+1)} = \arg\min_\pi D_{\text{KL}}(\pi^{(i)} \| \pi) + \mathbb{E}_{\pi^{(i)}}[\mathcal{L}(\pi)]$
- Monotonic improvement と局所収束が期待できる

### 11.2. 制約状態空間

**制限**：DeepGSBは主に $\mathbb{R}^d$ の制約なし状態空間向け

**拡張方向**：ドメイン固有の構造（制約状態空間）への適応

### 11.3. 次元の呪い

**IPF lossの発散**：次元が増えるとスケールが悪化する可能性

**緩和策**：De Bortoli et al. [24] の単純な回帰を採用

---

## 12. まとめ

### 12.1. Liu et al. の主要貢献

1. **理論**：Schrödinger BridgeをMean-Field Gameに拡張（Hopf-Cole変換経由）
2. **アルゴリズム**：TD lossによりMF interaction $F$ を扱える
3. **実装**：不連続・非微分可能な$F$にも対応
4. **実験**：1000次元までスケール、SOTA達成

### 12.2. 本プロジェクトの位置づけ

**Option A**：Liu et al. の再現・検証

**Option B**：さらなる拡張
- Interaction learningを追加した4-step cycle
- 推論時計算量を $O(N^2) \to O(N)$ に削減
- Han et al. (2024) のアイデアを統合

---

## 参考文献

**主論文**:
- Liu, G.-H., Chen, T., So, O., & Theodorou, E. A. (2022). Deep Generalized Schrödinger Bridge. NeurIPS 2022.

**関連論文**:
- [24] De Bortoli, V., et al. (2021). Diffusion Schrödinger Bridge with applications to score-based generative modeling.
- [27] Chen, T., Liu, G.-H., & Theodorou, E. A. (2021). Likelihood training of Schrödinger bridge using forward-backward SDEs theory.
- [38] Vargas, F. (2021). Machine-learning approaches for the empirical Schrödinger bridge problem.
- [55] Schulman, J., et al. (2015). Trust region policy optimization.

**本プロジェクト参照**:
- Han, J., et al. (2024). Learning High-Dimensional McKean-Vlasov Forward-Backward Stochastic Differential Equations with General Distribution Dependence.

---
