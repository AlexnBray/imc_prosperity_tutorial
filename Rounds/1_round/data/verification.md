Peer Review Request: High-Frequency Trading (HFT) Alpha Validation

Objective: Validate the statistical methodology and strategy logic for a mean-reversion model on a discrete-tick financial asset (ASH_COATED_OSMIUM).
1. Initial Data Observation & Topology

    Asset Behavior: The price action is highly discrete, constrained by integer tick sizes.

    Distribution Analysis: Initial residuals (Price - Rolling Median) displayed a trimodal distribution with massive central spikes at 0 and symmetric "walls" at ±1,±2 ticks.

    Kurtosis: Excess Kurtosis was measured at ~8.5, indicating an extremely leptokurtic (fat-tailed) distribution.

    Initial Hypothesis: The asset exists in two regimes: a "Quiet" regime (bid-ask bounce around a stable fair value) and a "Storm" regime (violent price "teleportation" or jumps).

2. Noise Separation Methodology

    Model Selection: Switched from a standard Moving Median to a Hampel Filter (W=17,n_sigma=3).

    Reasoning: The Hampel Filter uses the Median Absolute Deviation (MAD) as a robust estimator of volatility. This prevents the "Storm" regime outliers from inflating the threshold, allowing for precise isolation of the "Fair Value" (FV).

    Verification: The residuals (ϵt​=Pt​−FVt​) passed the Ljung-Box Test (p>0.05 across all test days).

    Conclusion: The Hampel Filter successfully separated the predictable signal from the noise, resulting in White Noise residuals, a prerequisite for mean reversion anchors.

3. Advanced Statistical Modeling

To define the structure of the "Storms," the following tests were conducted:

    Student’s t-Distribution Fit: Yielded a Degrees of Freedom (ν) of 2.12. This confirms "Infinite Variance" characteristics where outliers are the primary drivers of volatility.

    Runs Test for Randomness: Yielded p=0.00. This proves that outliers are temporally clustered. A deviation at t significantly increases the probability of a deviation at t+1 (Regime Clustering).

    Variance Ratio Test: Yielded V≈1.006. This indicates that, globally, the asset follows a Random Walk.

4. The "Local Mean Reversion" Paradox

Despite the Global Random Walk (VR = 1.0), we conducted a Conditional Edge Test (Expectancy given a threshold).

    Finding: At a threshold of 4 ticks, the Conditional Edge was ~7.4 ticks over a 10-tick horizon.

    Interpretation: While the asset is efficient most of the time, it becomes highly inefficient at the extremes. The "rubber band" of the Fair Value only exerts force once the deviation exceeds ±3 ticks.

    Signal Decay: The Information Coefficient (IC) peaked at Horizon 3 (predictability), while the Edge (Ticks) peaked at Horizon 10 (profit magnitude).

5. Proposed Trading Strategy: "Hampel-Storm Fade"

    Anchor: Real-time Hampel Fair Value (W=17).

    Entry Logic: "Wait-and-Fade." Enter when ∣ϵt​∣≥3 ticks AND the residual begins to contract (∣ϵt​∣<∣ϵt−1​∣). This mitigates the risk of clustered storms.

    Exit Logic: Exit at ϵ=0 or after 10 ticks (Alpha decay).

    Stop Loss: Placed at the 95th percentile of the Maximum Adverse Excursion (MAE).

Questions for Verification:

    Regime Switching: Given V≈1.0 and ν≈2.12, is the "Wait-and-Fade" entry logic sufficient to prevent "catching a falling knife" during a permanent regime shift?

    Filter Selection: Is there a more responsive robust estimator than the Hampel (MAD-based) filter that could reduce the 10-tick decay horizon?

    Overfitting Risk: Does the high IC (-0.49) suggest a structural exchange artifact (like a matching engine quirk) that can be reliably exploited, or is it likely a result of backtest overfitting on the discrete peaks?

    Capacity: Given the trade frequency (~900/day) and the 7.6-tick expectancy, what is the impact of a non-stationary Fair Value on the risk-of-ruin?