# 2b. Feature Selection

Investigation notebook: `4_feature_investigation.ipynb`.

Starting point: the 43 candidate columns produced by `2_feature_engineer.ipynb`
(all columns of `Synthetic_Financial_datasets_features.parquet` minus the ID/target
columns `step`, `type`, `nameOrig`, `nameDest`, `isFraud` — `type` itself is retained
via its one-hot `type_*` encodings, which are among the 43). Split via
`app.data.time_based_split` (`TEST_FRAC=0.2`), giving 5,090,096 training rows.

## Method

1. Fit a probe `RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42, n_jobs=-1)`
   on the full 43-feature training set and rank features by `feature_importances_`.
2. Correlation-prune: for every pair of features with `|corr| > 0.9` (computed on
   `X_train`), drop whichever of the two has the lower RF importance.
3. Re-rank the surviving features and look for the importance elbow.
4. Take everything above the elbow as `SELECTED_FEATURES`.

## Results

### Step 2 — RF probe importances (all 43 features)

Full ranked importances are printed in `4_feature_investigation.ipynb` (cell 3, real
output from the fitted probe).

### Step 3 — Correlation pruning (|corr| > 0.9)

9 features were dropped for being highly collinear with a more important feature:

```
['amount', 'amount_to_dest_avg_ratio', 'amount_to_orig_avg_ratio', 'dest_cum_count',
 'errorBalanceOrig', 'isNewBalanceDestZero', 'is_full_balance_transfer',
 'is_merchant_dest', 'orig_cum_avg']
```

34 features remain after pruning.

### Step 4 — Importance elbow

Cumulative importance of the 34 surviving features (real numbers from the probe RF):

| Rank | Feature | Cumulative importance |
|---|---|---|
| 9  | `is_transfer_or_cashout` | 91.92% |
| 19 | `type_TRANSFER` | 98.03% |
| **20** | `dest_balance_change_abs_error` | **99.53%** |
| 21 | `isDestBalanceZero` | 99.63% |
| 30 | `day_of_week` | 100.00% (of the surviving mass) |
| 31 | `is_first_dest_tx` | importance ≈ 1.0e-5, floating-point noise from here down |

The top 20 surviving features already capture 99.53% of total RF importance mass.
From rank 21 onward each feature contributes < 0.09 percentage points individually,
and the last four (ranks 31–34) are effectively zero (~1e-5 to 1e-7) — clearly noise,
not signal. There is no additional large cliff between rank 20 and rank 30 (the drop
is gradual and small), but the sharpest large relative drop after the top cluster is
right at rank 20→21 combined with the >99.5% cumulative-mass threshold already being
crossed, so 20 was chosen as the cutoff: it is comfortably inside the brief's
"roughly 15–25" target range and is the point past which additional features add
negligible explanatory power.

### Full table: importance, kept/dropped, reason

| Rank | Feature | Importance | Status | Reason |
|---|---|---|---|---|
| 1 | `is_amount_equal_oldbalanceOrig` | 0.245788 | Kept | selected (top 20 by importance after pruning) |
| 2 | `orig_balance_change_abs_error` | 0.156128 | Kept | selected (top 20 by importance after pruning) |
| 3 | `isOrigBalanceEnough` | 0.076348 | Kept | selected (top 20 by importance after pruning) |
| 4 | `errorBalanceOrig` | 0.066609 | Dropped | correlation-pruned (\|corr\|>0.9 with a higher-importance feature) |
| 5 | `amount_to_orig_ratio` | 0.066282 | Kept | selected (top 20 by importance after pruning) |
| 6 | `balance_drop_ratio` | 0.058530 | Kept | selected (top 20 by importance after pruning) |
| 7 | `isNewBalanceOrigZero` | 0.057248 | Kept | selected (top 20 by importance after pruning) |
| 8 | `orig_balance_after_expected` | 0.050281 | Kept | selected (top 20 by importance after pruning) |
| 9 | `orig_balance_change` | 0.047872 | Kept | selected (top 20 by importance after pruning) |
| 10 | `is_transfer_or_cashout` | 0.043066 | Kept | selected (top 20 by importance after pruning) |
| 11 | `is_full_balance_transfer` | 0.026247 | Dropped | correlation-pruned (\|corr\|>0.9 with a higher-importance feature) |
| 12 | `amount_to_dest_cum_avg_ratio` | 0.017646 | Kept | selected (top 20 by importance after pruning) |
| 13 | `amount_to_dest_ratio` | 0.014747 | Kept | selected (top 20 by importance after pruning) |
| 14 | `amount` | 0.011630 | Dropped | correlation-pruned (\|corr\|>0.9 with a higher-importance feature) |
| 15 | `type_PAYMENT` | 0.010612 | Kept | selected (top 20 by importance after pruning) |
| 16 | `is_merchant_dest` | 0.009867 | Dropped | correlation-pruned (\|corr\|>0.9 with a higher-importance feature) |
| 17 | `amount_to_orig_avg_ratio` | 0.006840 | Dropped | correlation-pruned (\|corr\|>0.9 with a higher-importance feature) |
| 18 | `errorBalanceDest` | 0.005485 | Kept | selected (top 20 by importance after pruning) |
| 19 | `type_TRANSFER` | 0.004822 | Kept | selected (top 20 by importance after pruning) |
| 20 | `amount_to_dest_avg_ratio` | 0.003517 | Dropped | correlation-pruned (\|corr\|>0.9 with a higher-importance feature) |
| 21 | `isNewBalanceDestZero` | 0.003023 | Dropped | correlation-pruned (\|corr\|>0.9 with a higher-importance feature) |
| 22 | `type_CASH_OUT` | 0.002826 | Kept | selected (top 20 by importance after pruning) |
| 23 | `is_large_amount` | 0.002473 | Kept | selected (top 20 by importance after pruning) |
| 24 | `hour_of_day` | 0.002391 | Kept | selected (top 20 by importance after pruning) |
| 25 | `dest_balance_change` | 0.002263 | Kept | selected (top 20 by importance after pruning) |
| 26 | `dest_balance_after_expected` | 0.002043 | Kept | selected (top 20 by importance after pruning) |
| 27 | `dest_balance_change_abs_error` | 0.001111 | Kept | selected (top 20 by importance after pruning) |
| 28 | `isDestBalanceZero` | 0.000846 | Dropped | below-elbow (importance negligible after rank 20) |
| 29 | `type_CASH_IN` | 0.000584 | Dropped | below-elbow (importance negligible after rank 20) |
| 30 | `dest_cum_sum` | 0.000509 | Dropped | below-elbow (importance negligible after rank 20) |
| 31 | `dest_amount_last_24h` | 0.000445 | Dropped | below-elbow (importance negligible after rank 20) |
| 32 | `dest_cum_avg` | 0.000398 | Dropped | below-elbow (importance negligible after rank 20) |
| 33 | `day_of_month` | 0.000365 | Dropped | below-elbow (importance negligible after rank 20) |
| 34 | `isOrigBalanceZero` | 0.000305 | Dropped | below-elbow (importance negligible after rank 20) |
| 35 | `dest_unique_orig_count` | 0.000247 | Dropped | below-elbow (importance negligible after rank 20) |
| 36 | `dest_cum_count` | 0.000228 | Dropped | correlation-pruned (\|corr\|>0.9 with a higher-importance feature) |
| 37 | `dest_txn_last_24h` | 0.000226 | Dropped | below-elbow (importance negligible after rank 20) |
| 38 | `day_of_week` | 0.000141 | Dropped | below-elbow (importance negligible after rank 20) |
| 39 | `is_first_dest_tx` | 0.0000103 | Dropped | below-elbow (importance negligible after rank 20) |
| 40 | `type_DEBIT` | 0.0000014 | Dropped | below-elbow (importance negligible after rank 20) |
| 41 | `orig_cum_count` | 0.0000011 | Dropped | below-elbow (importance negligible after rank 20) |
| 42 | `orig_cum_sum` | 0.0000007 | Dropped | below-elbow (importance negligible after rank 20) |
| 43 | `orig_cum_avg` | 0.0000006 | Dropped | correlation-pruned (\|corr\|>0.9 with a higher-importance feature) |

(43 candidate features total, not 44 — `type` contributes only its one-hot
`type_*` columns among the 43 feature columns produced by `time_based_split`, one
fewer than the brief's estimate; this does not change the method.)

## Final `SELECTED_FEATURES`

```python
SELECTED_FEATURES = [
    'is_amount_equal_oldbalanceOrig',
    'orig_balance_change_abs_error',
    'isOrigBalanceEnough',
    'amount_to_orig_ratio',
    'balance_drop_ratio',
    'isNewBalanceOrigZero',
    'orig_balance_after_expected',
    'orig_balance_change',
    'is_transfer_or_cashout',
    'amount_to_dest_cum_avg_ratio',
    'amount_to_dest_ratio',
    'type_PAYMENT',
    'errorBalanceDest',
    'type_TRANSFER',
    'type_CASH_OUT',
    'is_large_amount',
    'hour_of_day',
    'dest_balance_change',
    'dest_balance_after_expected',
    'dest_balance_change_abs_error',
]
```

## Elbow rationale

After correlation pruning, 34 features remain. Looking at the ratio between
consecutive importances (rather than just the raw values), the single sharpest
drop in the whole ranking is between rank 9 and rank 10 — `is_transfer_or_cashout`
(0.0431) to `amount_to_dest_cum_avg_ratio` (0.0176), a 2.4x fall — and by rank 9
the top features already account for 91.9% of cumulative importance. That is the
*primary* elbow: on importance alone, a 9-feature model would already capture
almost all of the signal this probe RF found.

The brief's target of roughly 15–25 features overrides that primary elbow, so the
cutoff was extended past it. Ranks 10–19 form a second, much flatter plateau
(0.0176 down to 0.0048, each step a mild ~1.2–1.9x decline, no sharp break), and
that plateau ends at rank 19→20 with a 1.8x drop (0.0020 → 0.0011) followed by a
long, nearly flat tail: ranks 21–34 individually contribute ≤0.00085 each and sum
to under 0.5 cumulative percentage points, with the bottom four
(`is_first_dest_tx`, `type_DEBIT`, `orig_cum_count`, `orig_cum_sum`) at ~1e-5–1e-7
— indistinguishable from floating-point noise. There is no second sharp cliff
between rank 20 and rank 30; the decline past 20 is just a long negligible tail.

The cutoff was therefore drawn at rank 20 (`dest_balance_change_abs_error`,
importance 0.0011): it is the end of the second plateau, sits inside the brief's
15–25 target range, captures 99.53% of cumulative importance, and excludes a long
tail of features (`dest_cum_*`, `orig_cum_*`, low-cardinality calendar features
like `day_of_week`) whose individual contribution is too small to justify the
added complexity/leakage surface. In short: the data's own elbow is at 9; 20 is
the honest stopping point once that primary elbow is extended to satisfy the
brief's range requirement.
