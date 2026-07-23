# MNQ PCS Regime Research — True North

## 1. Purpose

Determine whether market information available at 3:45 p.m. each Friday, beyond the absolute VIX level, can improve the decision process for the MNQ Weekly Put Credit Spread System.

The existing VIX strike ladder remains the baseline strategy. Regime research is intended to determine whether specific Friday conditions justify a different treatment of the new weekly cohort.

## 2. Primary Research Question

Can a simple, reproducible, and statistically defensible market regime identify when the standard VIX-ladder PCS has a materially different:

- Expiration failure probability
- Short-strike touch probability
- Downside severity
- Failure-clustering risk
- Expected strategy outcome

## 3. Potential Friday Actions

A validated regime may eventually map to one of four actions:

1. **Normal Entry**
   - Standard VIX ladder
   - Full calculated position size

2. **Reduced Size**
   - Standard VIX ladder
   - Reduced contract quantity

3. **Defensive Strike**
   - Short strike placed farther below spot
   - Credit floor must remain achievable
   - Position sizing must remain within the official risk cap

4. **Skip**
   - No new weekly cohort opened

The research must determine the appropriate action. Actions will not be assigned merely because a regime appears uncomfortable.

## 4. Baseline Strategy

The control strategy is the existing mechanical MNQ PCS system:

- Entry each Friday afternoon
- Approximately 26–35 days to expiration
- VIX-based short-strike distance
- 150-point spread width
- Credit floor enforced
- Hold to expiration
- No rolling or discretionary defense
- Four overlapping weekly cohorts at steady state

Historical baseline from the current research dataset:

- Eligible Friday entries: 1,154
- Short-strike touch rate: 5.55%
- Expiration failure rate: 1.73%
- Historical expiration failures: 20
- Many losses occur in overlapping market-stress clusters

All regime results must be compared with this baseline.

## 5. What We Are Testing

The research may evaluate information available by the Friday entry time, including:

### Trend
- Position relative to weekly moving averages
- Moving-average slope
- Trend deterioration or recovery

### Momentum
- Four-week and twelve-week returns
- Consecutive negative periods
- Momentum acceleration or reversal

### Drawdown
- Distance below prior 20-week or 52-week highs
- Drawdown depth and direction

### Volatility Behavior
- VIX rate of change
- VIX relative to its moving average
- VIX historical percentile
- Realized range expansion or contraction

### Price Structure
- Weekly range relative to prior ATR
- Rolling four-week range relative to prior ATR
- Close location within the recent range
- Reversal and range-expansion conditions

### Relative Strength
- NDX performance relative to SPY
- Nasdaq-specific deterioration or leadership

### Later Research
- Market breadth
- Volatility term structure
- Scheduled economic and earnings events
- Credit availability and historical option pricing, if reliable data becomes available

## 6. What We Are Not Testing

This project is not intended to:

- Predict the next four-week market direction
- Build a general Nasdaq trading model
- Optimize an indicator until it explains past crashes
- Replace the VIX ladder without strong evidence
- Improve average NDX returns while ignoring PCS tail risk
- Use information unavailable at the Friday entry time
- Use intraday NQ features from the separate one-minute research project
- Alter the official strategy based on a small number of attractive historical examples
- Model the credit floor or option premium without appropriate historical options data

## 7. Primary Evaluation Metrics

The most important measures are:

1. Expiration failure rate
2. Percentage of total failures captured
3. Failure-rate lift versus baseline
4. Short-strike touch rate
5. First- and fifth-percentile expiration outcomes
6. Maximum adverse movement during the holding period
7. Number and severity of overlapping failure clusters
8. Sample size
9. Stability across historical periods
10. Economic effect of the proposed trade treatment

Average market return is secondary.

## 8. Research Standards

A candidate regime must:

- Use only information available at entry
- Have a clear economic or market rationale
- Use simple and stable thresholds
- Contain a meaningful number of observations
- Identify multiple independent market episodes
- Survive subperiod analysis
- Survive crash-cluster analysis
- Survive walk-forward or out-of-sample testing
- Remain useful under modest threshold changes
- Improve the actual strategy outcome after applying the proposed trade action

A result that explains only one crash episode is not sufficient.

## 9. Evidence Classifications

### Exploratory

An interesting relationship identified during initial testing. It cannot alter the trading strategy.

### Candidate

A relationship that shows meaningful tail-risk separation, adequate sample size, and reasonable consistency across periods.

### Validated

A candidate that survives walk-forward testing, cluster-adjusted analysis, threshold sensitivity, and economic simulation.

### Approved

A validated regime formally added to the operational strategy rules.

Only an Approved regime can change a Friday trade.

## 10. Current Preliminary Finding

Three individual factors currently qualify as candidate risk flags:

1. Rolling four-week range at least 120% of prior ATR
2. NDX below its 40-week moving average
3. Current weekly range at least 120% of prior ATR

Preliminary combined results:

- Zero active flags: 1.12% expiration failure rate
- One active flag: 1.37%
- Two active flags: 4.00%
- Three active flags: 6.38%

Current working hypothesis:

> Fridays with zero or one active flag represent the normal regime. Fridays with two or three active flags may represent an elevated-risk regime.

This hypothesis is not yet approved for operational use.

## 11. Immediate Research Sequence

The next stages are:

1. Validate the candidate score across historical subperiods
2. Test sensitivity to reasonable threshold changes
3. Remove individual crash clusters and rerun the analysis
4. Conduct walk-forward and out-of-sample validation
5. Compare the regime score with simpler alternatives
6. Simulate potential actions:
   - Full size
   - Reduced size
   - Wider strike
   - Skip
7. Evaluate the effect on:
   - Compounded returns
   - Maximum drawdown
   - Failure clustering
   - Capital efficiency
   - Trade count
8. Recommend whether the regime should be rejected, retained for monitoring, or promoted

## 12. Project Decision Standard

The project succeeds only if it produces one of two defensible conclusions:

1. A validated regime materially improves the Friday PCS decision process; or
2. No non-VIX regime provides sufficient improvement, and the mechanical VIX-ladder strategy should remain unchanged.

A negative result is a successful research conclusion if it prevents an unsupported strategy modification.

## 13. True North Statement

> Build the simplest defensible Friday decision framework that improves the MNQ PCS strategy’s tail-risk management without sacrificing its core expectancy through overfitting, discretion, or unnecessary trade avoidance.
