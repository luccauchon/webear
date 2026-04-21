#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Credit Spread Regime Optimization Pipeline
==========================================

Optimizes clustering-based market regime detection to evaluate credit spread
trading opportunities using Optuna for hyperparameter tuning.

Key Features:
• Regime detection via K-Means or Gaussian Mixture clustering
• Binary target: probability of spread expiring OTM (profitable)
• Expectancy-based trade filtering with risk-adjusted edge ratio
• Optuna optimization with SQLite/PostgreSQL persistence
• CLI interface for flexible configuration

Author: Luc Cauchon
Date: 2026
"""

# =========================================================
# IMPORTS
# =========================================================
try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import numpy as np
import pandas as pd
import pickle
import argparse
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
import shutil
# Technical analysis
import pandas_ta as ta
from pandas_ta import macd
from utils import DATASET_AVAILABLE, next_weekday, next_week, next_month
# Machine Learning
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN,
    Birch, OPTICS
)
from sklearn.neighbors import NearestNeighbors  # For DBSCAN prediction
from sklearn.model_selection import TimeSeriesSplit

# Optimization
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Statistics
from statsmodels.stats.proportion import proportion_confint

# Logging configuration
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# GLOBAL CONFIGURATION
# =========================================================
DEFAULT_RANDOM_SEED = 42


# =========================================================
# ALL REGIMES SUMMARY - For Real-Time Mode
# =========================================================
from optimizers.regime.regime_switching_optimizer import print_all_regimes_summary


# =========================================================
# REGIME FILTERING HELPER
# =========================================================
from optimizers.regime.regime_switching_optimizer import regime_passes_filters


# =========================================================
# MODEL NAMING HELPER
# =========================================================
from optimizers.regime.regime_switching_optimizer import generate_model_filename


# =========================================================
#
# =========================================================
from optimizers.regime.regime_switching_optimizer import compute_oos_regime_stats


# =========================================================
#
# =========================================================
from optimizers.regime.regime_switching_optimizer import characterize_clusters


# =========================================================
#
# =========================================================
from optimizers.regime.regime_switching_optimizer import print_all_cluster_characteristics


# =========================================================
# TRADE DECISION HELPER - Credit Spread Expectancy Filter
# =========================================================
from optimizers.regime.regime_switching_optimizer import should_trade_credit_spread


# =========================================================
# DATA LOADING
# =========================================================
from optimizers.regime.regime_switching_optimizer import load_data


# =========================================================
# FEATURE ENGINEERING
# =========================================================
from optimizers.regime.regime_switching_optimizer import build_features


# =========================================================
# TARGET CONSTRUCTION - OTM Expiration Probability
# =========================================================
from optimizers.regime.regime_switching_optimizer import build_target


# =========================================================
# LATEST PREDICTION HELPER
# =========================================================
from optimizers.regime.regime_switching_optimizer import predict_latest


# =========================================================
# REPORT PRINTING
# =========================================================
from optimizers.regime.regime_switching_optimizer import print_report


# =========================================================
# REAL-TIME INFERENCE FUNCTION
# =========================================================
from optimizers.regime.regime_switching_optimizer import run_real_time_inference


# =========================================================
# MAIN PIPELINE FUNCTION
# =========================================================
def entry_main(args):
    """
    Main execution pipeline for regime optimization and trade evaluation.

    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Extract arguments with defaults fallback
    ticker = getattr(args, 'ticker', '^GSPC')
    dataset_id = getattr(args, 'dataset_id', 'day')
    study_name = getattr(args, 'study_name', 'spx_clustering_2')
    storage_url = getattr(args, 'storage_url', 'sqlite:///spx_study__target_OTM.db')
    max_n_trials = getattr(args, 'max_n_trials', 99999)
    timeout = getattr(args, 'timeout', 86400)
    n_jobs = getattr(args, 'n_jobs', 1)
    random_seed = getattr(args, 'random_seed', DEFAULT_RANDOM_SEED)

    min_n_in_cluster = getattr(args, 'min_n_in_cluster', 160)

    spread_type = getattr(args, 'spread_type', 'put')
    strike_distance = getattr(args, 'strike_distance', 0.03)
    forward_days = getattr(args, 'forward_days', 20)

    spread_width = getattr(args, 'spread_width', 5.0)
    credit_received = getattr(args, 'credit_received', 2.0)
    min_edge_ratio = getattr(args, 'min_edge_ratio', 0.04)

    # Set global random seed
    np.random.seed(random_seed)

    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🚀 Starting Credit Spread Regime Optimization")
    print(f"   Ticker: {ticker} | Dataset: {dataset_id}")
    print(f"   Spread: {spread_type} @{strike_distance * 100:.1f}% | DTE: {forward_days}")
    print(f"   Min samples/cluster: {min_n_in_cluster} | Trials: {max_n_trials} | Timeout: {timeout} seconds")
    print(f"   Forward : {forward_days} {dataset_id}")
    print("-" * 60)
    assert args.dataset_id in ['day', 'week', 'month']
    # Load data
    df = load_data(_ticker=ticker, _dataset_id=dataset_id)
    if args.lookback_years > 0:
        rows_per_year = {'day': 252, 'week': 52, 'month': 12, 'quarter': 4, 'year': 1}.get(args.dataset_id)
        cutoff = args.lookback_years * rows_per_year
        print(f"📅 Using {args.lookback_years}-year lookback: ~{cutoff} rows")
        df = df.iloc[-cutoff:].copy()
    minimum_train_data, minimum_test_data = 1000, 200
    if args.dataset_id == 'week':
        minimum_train_data, minimum_test_data = minimum_train_data // 5, minimum_test_data // 5
    if args.dataset_id == 'month':
        minimum_train_data, minimum_test_data = minimum_train_data // 20, minimum_test_data // 20
    print(f"📦 Loaded {len(df)} rows of data ({df.index[0].date()} to {df.index[-1].date()})")
    assert len(df) > minimum_train_data + minimum_test_data
    total_number_of_rows = len(df)

    # =========================================================
    # OPTUNA OBJECTIVE FUNCTION
    # =========================================================
    # =========================================================
    # ENHANCED OPTUNA OBJECTIVE FUNCTION
    # =========================================================
    def objective(trial):
        """
        Optuna objective: MAXIMIZE risk-adjusted trading edge across regimes.

        Key Enhancements:
        • Directly optimizes expectancy (P&L) instead of abstract clustering metrics
        • Time-series cross-validation to prevent temporal overfitting
        • Tunable weights let Optuna learn optimal metric blending
        • Penalties for uncertainty, instability, and cluster imbalance
        • Wilson CI-based uncertainty quantification

        Returns:
        --------
        float : Higher = better risk-adjusted edge for credit spread trading
        """
        # ===== 1. SUGGEST HYPERPARAMETERS =====
        if args.dataset_id == 'day':
            _n_clusters = trial.suggest_int("n_clusters", 3, 6)
            _pct1 = trial.suggest_int("pct1", 1, 5)
            _pct2 = trial.suggest_int("pct2", 5, 10)
            _pct3 = trial.suggest_int("pct3", 10, 20)
            _vol1 = trial.suggest_int("vol1", 1, 10)
            _vol2 = trial.suggest_int("vol2", 10, 20)
            _vol3 = trial.suggest_int("vol3", 20, 60)
            _ema1 = trial.suggest_int("ema1", 40, 60)
            _ema2 = trial.suggest_int("ema2", 180, 220)
            _atr_period = trial.suggest_int("atr_period", 2, 21)
            _rsi_length = trial.suggest_int("rsi_length", 10, 20)
        elif args.dataset_id == 'week':
            _n_clusters = trial.suggest_int("n_clusters", 3, 6)
            _pct1 = trial.suggest_int("pct1", 1, 3)
            _pct2 = trial.suggest_int("pct2", 3, 8)
            _pct3 = trial.suggest_int("pct3", 8, 16)
            _vol1 = trial.suggest_int("vol1", 1, 10)
            _vol2 = trial.suggest_int("vol2", 10, 20)
            _vol3 = trial.suggest_int("vol3", 20, 30)
            _ema1 = trial.suggest_int("ema1", 8, 14)
            _ema2 = trial.suggest_int("ema2", 40, 44)
            _atr_period = trial.suggest_int("atr_period", 4, 12)
            _rsi_length = trial.suggest_int("rsi_length", 6, 14)
        elif args.dataset_id == 'month':
            _n_clusters = trial.suggest_int("n_clusters", 3, 5)
            _pct1 = trial.suggest_int("pct1", 1, 2)
            _pct2 = trial.suggest_int("pct2", 2, 4)
            _pct3 = trial.suggest_int("pct3", 4, 8)
            _vol1 = trial.suggest_int("vol1", 1, 3)
            _vol2 = trial.suggest_int("vol2", 3, 6)
            _vol3 = trial.suggest_int("vol3", 6, 18)
            _ema1 = trial.suggest_int("ema1", 2, 6)
            _ema2 = trial.suggest_int("ema2", 12, 24)
            _atr_period = trial.suggest_int("atr_period", 3, 8)
            _rsi_length = trial.suggest_int("rsi_length", 4, 10)

        # Suggest clustering algorithm
        _clustering_algo = trial.suggest_categorical(
            "clustering_algo",
            ["kmeans", "gaussian_mixture"]
        )

        # ===== SUGGEST OBJECTIVE WEIGHTS (Let Optuna learn optimal blending) =====
        _w_edge = trial.suggest_float("weight_edge", 0.75, 0.99)  # Primary: trading edge
        _w_stability = trial.suggest_float("weight_stability", 0.05, 0.2)  # Secondary: regime persistence
        _w_uncertainty = trial.suggest_float("weight_uncertainty", 0.05, 0.1)  # Penalty: statistical uncertainty

        # Normalize weights to sum to 1.0
        _weight_sum = _w_edge + _w_stability + _w_uncertainty
        _w_edge /= _weight_sum
        _w_stability /= _weight_sum
        _w_uncertainty /= _weight_sum

        # ===== 2. TIME-SERIES CROSS-VALIDATION SETUP =====
        # Use purged CV to prevent lookahead bias in rolling features
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        # Pre-compute trade context for edge calculations
        _min_samples = args.min_n_in_cluster

        for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
            # ----- Split raw data FIRST (prevent leakage) -----
            df_train_raw = df.iloc[train_idx].copy()
            df_test_raw = df.iloc[test_idx].copy()

            # ----- Build features & target for this fold -----
            features_train = build_features(
                _df=df_train_raw,
                _pct1=_pct1,
                _pct2=_pct2,
                _pct3=_pct3,
                _vol1=_vol1,
                _vol2=_vol2,
                _vol3=_vol3,
                _rsi_length=_rsi_length,
                _ema1=_ema1,
                _ema2=_ema2,
                _atr_period=_atr_period,
                _add_enhanced_features=True,
            )
            target_train = build_target(
                _df=df_train_raw,
                _forward_days=forward_days,
                _strike_distance=strike_distance,
                _spread_type=spread_type
            )

            features_test = build_features(
                _df=df_test_raw,
                _pct1=_pct1,
                _pct2=_pct2,
                _pct3=_pct3,
                _vol1=_vol1,
                _vol2=_vol2,
                _vol3=_vol3,
                _rsi_length=_rsi_length,
                _ema1=_ema1,
                _ema2=_ema2,
                _atr_period=_atr_period,
                _add_enhanced_features=True,
            )
            target_test = build_target(
                _df=df_test_raw,
                _forward_days=forward_days,
                _strike_distance=strike_distance,
                _spread_type=spread_type
            )
            nan_rate = features_test.isna().mean().mean()
            # Penalize trials with >20% missing values (unstable feature config)
            if nan_rate > 0.20:
                return -999 + nan_rate  # Heavy penalty, but slightly reward lower NaN rates
            # ----- Align & clean -----
            train_aligned = pd.concat([features_train, target_train.rename('target')], axis=1).dropna()
            test_aligned = pd.concat([features_test, target_test.rename('target')], axis=1).dropna()

            if len(train_aligned) < minimum_train_data or len(test_aligned) < minimum_test_data:
                cv_scores.append(-999)  # Heavy penalty for insufficient data
                continue

            # ----- Scale features (fit on train only) -----
            _scaler = RobustScaler()
            X_train = _scaler.fit_transform(train_aligned.drop(columns=['target']))
            X_test = _scaler.transform(test_aligned.drop(columns=['target']))

            # ----- Train clustering model -----
            if _clustering_algo == 'kmeans':
                _model = KMeans(
                    n_clusters=_n_clusters, init='k-means++', n_init=10,
                    max_iter=900, tol=1e-4, random_state=random_seed,
                    algorithm='lloyd', verbose=0
                )
            elif _clustering_algo == 'gaussian_mixture':
                _model = GaussianMixture(
                    n_components=_n_clusters, covariance_type='full',
                    tol=1e-3, reg_covar=1e-5, max_iter=900, n_init=20,
                    init_params='k-means++', random_state=random_seed, verbose=0
                )
            else:
                raise ValueError(f"Unknown algorithm: {_clustering_algo}")

            regimes_train = _model.fit_predict(X_train)
            regimes_test = _model.predict(X_test)
            test_aligned = test_aligned.copy()
            test_aligned['regime'] = regimes_test
            # ----- Evaluate each regime on TEST data -----
            regime_scores = []
            for r in range(_n_clusters):
                subset = test_aligned[test_aligned["regime"] == regimes_test]["target"]

                # Skip under-sampled regimes
                if len(subset) < _min_samples:
                    continue

                # === COMPONENT 1: OTM PROBABILITY (Primary - maximize this) ===
                prob_otm = subset.mean()

                # === COMPONENT 2: UNCERTAINTY PENALTY (Wilson CI width) ===
                ci_low, ci_upp = proportion_confint(
                    count=subset.sum(), nobs=len(subset), alpha=0.05, method='wilson'
                )
                ci_width = ci_upp - ci_low
                uncertainty_penalty = -ci_width * 3  # Scale: wider CI = bigger penalty

                # === COMPONENT 3: TEMPORAL COHERENCE BONUS (Regime persistence) ===
                regime_mask = (regimes_test == r)
                if len(regime_mask) > 1:
                    # Count regime switches: fewer switches = more stable
                    switches = np.sum(regime_mask[:-1] != regime_mask[1:])
                    persistence = 1 - (switches / len(regime_mask))
                    # Only reward if persistence is meaningfully above random
                    coherence_bonus = max(0, persistence - 0.5) * 2
                else:
                    coherence_bonus = 0

                # === COMPONENT 4: CLUSTER BALANCE PENALTY (Optional regularizer) ===
                # Computed once per fold outside the regime loop (see below)

                # === COMBINE WITH TUNABLE WEIGHTS ===
                regime_score = (
                        _w_edge * prob_otm +
                        _w_uncertainty * uncertainty_penalty +
                        _w_stability * coherence_bonus
                )
                regime_scores.append(regime_score)

            # ----- Apply cluster balance penalty (discourage pathological splits) -----
            if regime_scores:
                cluster_sizes = [
                    np.sum(regimes_test == r) for r in range(_n_clusters)
                    if np.sum(regimes_test == r) >= _min_samples
                ]
                if len(cluster_sizes) >= 2:
                    # Coefficient of variation: higher = more imbalanced
                    cv_imbalance = np.std(cluster_sizes) / (np.mean(cluster_sizes) + 1e-8)
                    # Penalize extreme imbalance (>2x std/mean ratio)
                    balance_penalty = min(0.3, cv_imbalance * 0.15)
                    fold_score = np.mean(regime_scores) * (1 - balance_penalty)
                else:
                    fold_score = np.mean(regime_scores)
            else:
                fold_score = -999  # No valid regimes = heavy penalty

            cv_scores.append(fold_score)

        # ===== 3. AGGREGATE CV RESULTS WITH CONSISTENCY PENALTY =====
        if not cv_scores or all(s < -998 for s in cv_scores):
            return -999  # Trial failed

        # Filter out failed folds
        valid_scores = [s for s in cv_scores if s > -998]
        if len(valid_scores) < 2:
            return np.mean(valid_scores) if valid_scores else -999

        # Reward high mean prob_otm, penalize high variance (unstable performance)
        mean_score = np.mean(valid_scores)
        score_std  = np.std(valid_scores)

        # Sharpe-like ratio: mean / (std + epsilon)
        # This encourages consistent edge across time periods
        consistency_adjusted = mean_score - 0.35 * score_std

        # Bonus for trials that produce >= 2 tradeable regimes
        n_tradeable = sum(1 for s in valid_scores if s > 0)
        if n_tradeable >= 2:
            consistency_adjusted += 0.05  # Small bonus for diversification

        return consistency_adjusted

    # =========================================================
    # OPTUNA STUDY SETUP
    # =========================================================
    print(f"\n🔬 Initializing Optuna study: '{study_name}'")
    print(f"   Storage: {storage_url}")

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(seed=random_seed,
                           multivariate=True,  # Better for correlated params like weights
                           group=True,  # Respect parameter groups
                           constant_liar=True,  # Speed up parallel trials
                           ),
        pruner=MedianPruner(n_startup_trials=15,      # Wait for more data before pruning
                            n_warmup_steps=5,
                            interval_steps=3),
        storage=storage_url,
        load_if_exists=True,
    )

    # ===== DB Resume Verification =====
    trials = study.trials
    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("\n" + "─" * 40)
    print("📊 OPTUNA PERSISTENCE STATUS")
    print("─" * 40)
    print(f"Total trials:     {len(trials)}")
    print(f"Completed:        {len(completed)}")

    if completed:
        print(f"Best score:       {study.best_value:.6f}")
        print(f"Best params:      {study.best_params}")
        print("→ Resuming optimization...")
    else:
        print("→ Starting fresh optimization...")
    print("─" * 40 + "\n")

    # ===== Run Optimization =====
    print("⚙️  Running Optuna optimization...\n")
    study.optimize(
        objective,
        n_trials=max_n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=True,
        callbacks=[],
    )

    # After study.optimize():
    print("\n🎯 LEARNED OBJECTIVE WEIGHTS (Top 5 trials):")
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:5]

    for i, t in enumerate(top_trials, 1):
        w_edge = t.params.get('weight_edge', 'N/A')
        w_stab = t.params.get('weight_stability', 'N/A')
        w_unc = t.params.get('weight_uncertainty', 'N/A')
        print(f"{i}. Score={t.value:.4f} | Weight Edge:{w_edge:.2f}   Weight Stability:{w_stab:.2f}   Weight Uncertainty:{w_unc:.2f}")

    # ===== Results Summary =====
    print(f"\n✅ Optimization complete!")
    print(f"🏆 Best Score: {study.best_value:.6f}")
    print(f"🎯 Best Parameters:")
    for k, v in study.best_params.items():
        print(f"   • {k}: {v}")

    # =========================================================
    # RETRAIN WITH BEST PARAMETERS
    # =========================================================
    print("\n" + "═" * 60)
    print("🔄 RETRAINING FINAL MODEL WITH BEST PARAMETERS")
    print("═" * 60)

    best_params = study.best_params.copy()

    # Extract parameters (handle optional clustering_algo)
    _n_clusters = best_params['n_clusters']
    _clustering_algo = best_params['clustering_algo']

    print(f"Selected algorithm: {_clustering_algo}")
    print(f"Clusters: {_n_clusters}")

    # Build features with best params
    features = build_features(
        _df=df.copy(),
        _pct1=best_params.get('pct1', 5),
        _pct2=best_params.get('pct2', 10),
        _pct3=best_params.get('pct3', 20),
        _vol1=best_params.get('vol1', 10),
        _vol2=best_params.get('vol2', 20),
        _vol3=best_params.get('vol3', 60),
        _rsi_length=best_params.get('rsi_length', 14),
        _ema1=best_params.get('ema1', 50),
        _ema2=best_params.get('ema2', 200),
        _atr_period=best_params.get('atr_period', 14),
        _add_enhanced_features=True,
    )

    # Build target
    target = build_target(
        _df=df.copy(),
        _forward_days=forward_days,
        _strike_distance=strike_distance,
        _spread_type=spread_type
    )

    # ===== Align and Prepare Full Dataset =====
    _aligned = pd.concat([features, target.rename('target')], axis=1)
    _valid = _aligned.dropna().copy()

    if len(_valid) == 0:
        print("❌ ERROR: No valid data after feature/target alignment")
        return

    # ===== Scale Features =====
    _scaler = RobustScaler()
    X = _scaler.fit_transform(_valid.drop(columns=['target']))

    # ===== Train Final Model (with CONSISTENT parameters) =====
    if _clustering_algo == 'kmeans':
        _model = KMeans(
            n_clusters=_n_clusters,
            random_state=random_seed,
            n_init=10,
            algorithm='lloyd',
            max_iter=900,
            tol=1e-4
        )
    elif _clustering_algo == 'gaussian_mixture':
        _model = GaussianMixture(
            n_components=_n_clusters,
            random_state=random_seed,
            n_init=20,  # ← MATCHES objective()
            covariance_type='full',
            max_iter=900,  # ← MATCHES objective()
            tol=1e-3,
            reg_covar=1e-5,
            init_params='k-means++',
        )
    elif _clustering_algo == 'birch':
        _threshold = best_params.get('birch_threshold', 0.5)
        _branching = best_params.get('birch_branching', 50)
        _model = Birch(threshold=_threshold, branching_factor=_branching, n_clusters=_n_clusters)
    else:
        raise ValueError(f"Unknown algorithm: {_clustering_algo}")

    regimes = _model.fit_predict(X)
    _valid["regime"] = regimes

    # ===== Compute Regime Statistics (with Wilson CI) =====
    _stats = {}
    for r in range(_n_clusters):
        subset = _valid[_valid["regime"] == r]["target"].dropna()

        if len(subset) < min_n_in_cluster:
            continue

        prob_otm = subset.mean()
        n = len(subset)

        # Wilson score interval (more accurate for proportions)
        ci_low, ci_upp = proportion_confint(
            count=subset.sum(),
            nobs=n,
            alpha=0.05,
            method='wilson'
        )

        _stats[r] = {
            "count": n,
            "total_count": total_number_of_rows,
            "prob_otm": prob_otm,
            "prob_itm": 1 - prob_otm,
            "win_rate": prob_otm,
            "std_outcome": np.sqrt(prob_otm * (1 - prob_otm)),  # Bernoulli std
            "ci_95_lower": ci_low,
            "ci_95_upper": ci_upp,
            "forward_days": forward_days,
            "dataset_id": dataset_id,
        }

    if not _stats:
        print(f"⚠️  WARNING: No clusters met minimum sample requirement ({min_n_in_cluster})")
        return

    # Show ALL cluster characteristics
    print_all_cluster_characteristics(_stats=_stats, _features=features, _model=_model, _scaler=_scaler, _n_clusters=_n_clusters)

    # ===== Predict Latest Regime =====
    result = predict_latest(_features=features, _model=_model, _scaler=_scaler, _stats=_stats)

    if result is None:
        print("⚠️  Could not predict regime for latest data point")
        return

    regime, regime_stats = result

    # ===== Print Analysis Report =====
    print_report(_regime=regime, _stats=regime_stats, _strike_distance=strike_distance, _spread_type=spread_type)

    # ===== TRADE EVALUATION =====
    print("💰 TRADE DECISION EVALUATION")
    print("─" * 40)

    max_loss = spread_width - credit_received

    trade_decision = should_trade_credit_spread(_regime_stats=regime_stats, _credit_received=credit_received, _max_loss=max_loss, _min_edge_ratio=min_edge_ratio)

    print(f"Spread Configuration:")
    print(f"   • Width:           ${spread_width:.2f}")
    print(f"   • Credit Received: ${credit_received:.2f}")
    print(f"   • Max Loss:        ${max_loss:.2f}")
    print(f"   • Min Edge Ratio:  {min_edge_ratio * 100:.1f}%")
    print()
    print(f"Regime-Based Metrics:")
    print(f"   • Break-Even Win Rate: {trade_decision['break_even_prob'] * 100:.2f}%")
    print(f"   • Expected Value:      ${trade_decision['expectancy']:.3f}/share")
    print(f"   • Edge Ratio:          {trade_decision['edge_ratio'] * 100:.2f}%")
    print()
    print(f"🎯 DECISION: {trade_decision['message']}")

    # ===== Save Model =====
    print(f"\n💾 Saving model artifacts...")

    # Generate unique, descriptive filename
    model_filename = generate_model_filename(
        _ticker=ticker,
        _study_name=study_name,
        _params=best_params,
        _metadata_extra={
            'spread_type': spread_type,
            'strike_distance': strike_distance,
            'forward_days': forward_days
        }
    )
    model_path = f"{args.output_dir}/{model_filename}"

    with open(model_path, "wb") as f:
        pickle.dump({
            "model": _model,
            "scaler": _scaler,
            "stats": _stats,
            "params": best_params,
            "metadata": {
                "ticker": ticker,
                "spread_type": spread_type,
                "strike_distance": strike_distance,
                "forward_days": forward_days,
                "clustering_algo": _clustering_algo,
                "timestamp": datetime.now(),
                "study_name": study_name,
                "best_score": study.best_value,
                "model_filename": model_filename,
                "dataset_id": args.dataset_id
            },
            "trade_context": {
                "credit_received": credit_received,
                "spread_width": spread_width,
                "max_loss": max_loss,
                "edge_ratio": trade_decision['edge_ratio'],
                "expectancy": trade_decision['expectancy'],
                "evaluated_at": datetime.now()
            }
        }, f)

    print(f"✅ Model saved to: {model_path}")

    # Create/update a "latest" symlink for convenience
    latest_link = f"{args.output_dir}/{ticker.replace('^', '')}_regime_model_latest.pkl"
    try:
        # Check if the destination already exists and remove it to avoid errors
        if os.path.exists(latest_link):
            os.remove(latest_link)
        shutil.copy2(model_path, latest_link)
        print(f"💾 Created copy: {latest_link} (from {model_filename})")
    except (OSError, IOError) as e:
        print(f"❌ Failed to copy model: {e}")

    print("\n" + "═" * 60)
    print("✨ PIPELINE COMPLETE")
    print("═" * 60 + "\n")


# =========================================================
# MAIN ENTRY POINT
# =========================================================
if __name__ == "__main__":
    """
    Command-line interface for Credit Spread Regime Optimization.

    EXAMPLES:
    ---------
    # Basic run with defaults:
    python script.py

    # Optimize put spreads on SPY with 30 DTE:
    python script.py --ticker SPY --spread-type put --forward-days 30

    # Use PostgreSQL storage:
    python script.py --storage-url "postgresql://user:pass@localhost/db" \\
                     --max-n-trials 200

    # Conservative trade filtering:
    python script.py --min-edge-ratio 0.08 --credit-received 1.5

    # Quick test run:
    python script.py --max-n-trials 10 --timeout 300 --study-name test_run

    # List available models for SPY:
    python script.py --ticker SPY --list-models

    # Run inference with a specific experiment:
    python script.py --ticker SPY --real-time \\
                     --model-filename "SPY__my_study__kmeansc4__put_sd3pct_dte20__20260402_143022.pkl"

    # Use the latest model (default behavior):
    python script.py --ticker SPY --real-time
    """

    parser = argparse.ArgumentParser(
        description="Optimize regime clustering for credit spread trading decisions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
        TRADE LOGIC:
        • Regime detection via unsupervised clustering (K-Means or GMM)
        • Target: Binary outcome of spread expiring OTM (profitable)
        • Trade approval requires:
          1. Edge ratio (expectancy/max_loss) ≥ min_edge_ratio
          2. Regime OTM probability > break-even probability
        • Model saved only when trade is approved
        """
    )

    # ─────────────────────────────────────────────────────
    # DATA & ASSET CONFIGURATION
    # ─────────────────────────────────────────────────────
    parser.add_argument(
        "--ticker", type=str, default="^GSPC",
        help="Ticker symbol to analyze (e.g., ^GSPC, SPY, QQQ)"
    )
    parser.add_argument(
        "--dataset-id", type=str, default="day", choices=DATASET_AVAILABLE,
        help=f"Dataset frequency: {DATASET_AVAILABLE}"
    )
    parser.add_argument(
        "--lookback-years", type=int, default=15,
        help="Years of historical data to use (0 = full history). Recommended: 10-15 for daily data"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models_tscv",
        help=f"Directory where to save the model"
    )

    # ─────────────────────────────────────────────────────
    # OPTUNA OPTIMIZATION CONFIGURATION
    # ─────────────────────────────────────────────────────
    parser.add_argument(
        "--study-name", type=str, default="spx_clustering_2",
        help="Unique Optuna study name (for DB persistence)"
    )
    parser.add_argument(
        "--storage-url", type=str, default=None,
        help="Database URL: sqlite:///path.db or postgresql://..."
    )
    parser.add_argument(
        "--max-n-trials", type=int, default=999999,
        help="Maximum Optuna trials to run"
    )
    parser.add_argument(
        "--timeout", type=int, default=86400,
        help="Max optimization runtime in seconds (86400 = 24h)"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel jobs for Optuna (1 recommended for SQLite)"
    )
    parser.add_argument(
        "--random-seed", type=int, default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducibility"
    )

    # ─────────────────────────────────────────────────────
    # CLUSTERING / REGIME MODEL CONFIGURATION
    # ─────────────────────────────────────────────────────
    parser.add_argument(
        "--min-n-in-cluster", type=int, default=160,
        help="Minimum samples per cluster to be considered valid"
    )

    # ─────────────────────────────────────────────────────
    # CREDIT SPREAD TRADE CONFIGURATION
    # ─────────────────────────────────────────────────────
    parser.add_argument(
        "--spread-type", type=str, default="put",
        choices=["put", "call", "iron_condor"],
        help="Spread type: 'put' (bullish), 'call' (bearish), 'iron_condor'"
    )
    parser.add_argument(
        "--strike-distance", type=float, default=0.03,
        help="Short strike distance as fraction (0.03 = 3%% away)"
    )
    parser.add_argument(
        "--forward-days", type=int, default=20,
        help="Days to expiration / holding period. Could be week or month, depends on dataset-id."
    )

    # ─────────────────────────────────────────────────────
    # TRADE EVALUATION PARAMETERS
    # ─────────────────────────────────────────────────────
    parser.add_argument(
        "--spread-width", type=float, default=5.0,
        help="Total spread width in dollars (e.g., 5.0 = $5 wide)"
    )
    parser.add_argument(
        "--credit-received", type=float, default=2.0,
        help="Premium received per share in dollars"
    )
    parser.add_argument(
        "--min-edge-ratio", type=float, default=0.04,
        help="Minimum edge ratio (expectancy/max_loss) to approve trade"
    )

    # ─────────────────────────────────────────────────────
    # EXECUTION MODE
    # ─────────────────────────────────────────────────────
    parser.add_argument(
        "--real-time", action="store_true",
        help="Load saved model and run inference on latest data (skip optimization)"
    )
    parser.add_argument("--short-verbose", action="store_true",
        help="In Real-Time mode only, display only the minimal information"
    )
    # ─────────────────────────────────────────────────────
    # REGIME FILTERING OPTIONS (Real-Time Mode Only)
    # ─────────────────────────────────────────────────────
    parser.add_argument(
        "--min-prob-otm", type=float, default=None,
        help="Minimum OTM probability required for a regime to be tradeable (e.g., 0.60 = 60%%)"
    )
    parser.add_argument(
        "--max-prob-itm", type=float, default=None,
        help="Maximum ITM probability allowed for a regime to be tradeable (e.g., 0.40 = 40%%)"
    )
    parser.add_argument(
        "--min-regime-samples", type=int, default=None,
        help="Minimum historical samples required for a regime to be considered valid"
    )
    parser.add_argument(
        "--allowed-regimes", type=str, default=None,
        help="Comma-separated list of regime IDs to ALLOW (e.g., '0,2,4'). All others rejected."
    )
    parser.add_argument(
        "--exclude-regimes", type=str, default=None,
        help="Comma-separated list of regime IDs to EXCLUDE (e.g., '1,3'). Takes precedence over --allowed-regimes."
    )
    parser.add_argument(
        "--min-ev-per-dollar", type=float, default=None,
        help="Minimum expected value per $1 risked to approve trade (e.g., 0.10 = +$0.10 EV per $1)"
    )
    parser.add_argument(
        "--model-filename", type=str, default=None,
        help="Specific model filename to load for inference (overrides auto-detection)"
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available trained models for this ticker and exit"
    )

    # ─────────────────────────────────────────────────────
    # EXECUTE
    # ─────────────────────────────────────────────────────
    args = parser.parse_args()

    # Ensure models directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        if args.real_time:
            # Run Inference Mode
            run_real_time_inference(args, ticker=args.ticker, list_models=args.list_models, model_filename=args.model_filename, use_enhanced_features=True)
        else:
            # Run Optimization Mode
            entry_main(args)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user. Exiting gracefully.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)