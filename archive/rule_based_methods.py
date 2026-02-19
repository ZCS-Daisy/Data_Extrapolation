"""
Rule-Based Methods Module
==========================
VERSION: 1.1 (Feb 2026) — fixes: Kendall/point-biserial feed into effective_r2,
  AnnualFeature_x_Seasonal prediction formula, features field split on ' + '

Tests statistical relationships between VQ and available features using:
  - Pearson r, R², p-value
  - Spearman ρ, p-value
  - Kendall τ, p-value
  - Point-biserial r for categorical features

Rule-based methods discovered and tested:
  1. Feature intensity:  VQ ~ Feature  (every numeric feature)
  2. Seasonal average:   VQ ~ Site average × seasonal factor
  3. Site average:       VQ ~ mean of known site VQ
  4. Historic adjusted:  VQ ~ VQ_historic × (Feature_current / Feature_historic)
  5. Annual + seasonal:  VQ ~ (AnnualFeature / 12) × SeasonalFactor

Selection:
  - Effective_R² = whichever of Pearson R² / Spearman ρ² is higher, but ONLY if
    the corresponding p-value < 0.05.  Otherwise effective_R² = 0.
  - All methods enter the selection pool on equal footing with ML models.

Principle alignment:
  P1 – training pool masked to match each scenario's feature availability
  P2 – handles monthly / annual / mixed / historic feature formats
  P3 – scenarios established first; methods tested per scenario
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_pearson(x, y):
    """Return (r, r2, p) or (None, None, None) if insufficient data."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return None, None, None
    try:
        r, p = stats.pearsonr(x, y)
        return float(r), float(r ** 2), float(p)
    except Exception:
        return None, None, None


def _safe_spearman(x, y):
    """Return (rho, rho2, p) or (None, None, None)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return None, None, None
    try:
        r, p = stats.spearmanr(x, y)
        return float(r), float(r ** 2), float(p)
    except Exception:
        return None, None, None


def _safe_kendall(x, y):
    """Return (tau, tau2, p) or (None, None, None)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return None, None, None
    try:
        tau, p = stats.kendalltau(x, y)
        return float(tau), float(tau ** 2), float(p)
    except Exception:
        return None, None, None


def _safe_pointbiserial(binary_x, y):
    """Return (r, r2, p) for a binary/categorical feature vs continuous VQ."""
    mask = np.isfinite(binary_x) & np.isfinite(y)
    binary_x, y = binary_x[mask], y[mask]
    if len(binary_x) < 4 or len(np.unique(binary_x)) < 2:
        return None, None, None
    try:
        r, p = stats.pointbiserialr(binary_x, y)
        return float(r), float(r ** 2), float(p)
    except Exception:
        return None, None, None


def _effective_r2(pearson_r2, pearson_p, spearman_r2, spearman_p,
                  kendall_r2=None, kendall_p=None,
                  pointbiserial_r2=None, pointbiserial_p=None):
    """
    Best significant R² across all tests (p < 0.05).
    Considers Pearson, Spearman, Kendall τ², and point-biserial r².
    Returns 0 if no test is significant.
    """
    candidates = []
    if pearson_r2 is not None and pearson_p is not None and pearson_p < 0.05:
        candidates.append(pearson_r2)
    if spearman_r2 is not None and spearman_p is not None and spearman_p < 0.05:
        candidates.append(spearman_r2)
    if kendall_r2 is not None and kendall_p is not None and kendall_p < 0.05:
        candidates.append(kendall_r2)
    if pointbiserial_r2 is not None and pointbiserial_p is not None and pointbiserial_p < 0.05:
        candidates.append(pointbiserial_r2)
    return max(candidates) if candidates else 0.0


# ── Core class ───────────────────────────────────────────────────────────────

class RuleBasedMethodTester:
    """
    Discovers, tests, and registers rule-based prediction methods.
    Results feed directly into the scenario-based model selection pool.
    """

    METADATA_COLS = {
        'Site identifier', 'Location', 'GHG Category', 'Date from', 'Date to',
        'Volumetric Quantity', 'Timeframe', 'Month', 'Data Timeframe',
        'Data integrity', 'Estimation Method', 'Data Quality', 'Data Quality Score',
        'Year', 'Years_Since_Baseline', 'Client', 'FY', 'Financial Year',
        'Data_Months', 'Data_Has_Features', 'Data_Has_Historic', 'Data_Availability',
    }

    def __init__(self, data_df, blank_df, historic_manager=None, seasonal_patterns=None):
        self.data_df = data_df.copy()
        self.blank_df = blank_df.copy()
        self.hdm = historic_manager
        self.seasonal_patterns = seasonal_patterns or {}

        # Output
        self.rule_based_models = {}   # same schema as ML models dict
        self.test_results = []        # rows for the Rule_Based_Methods sheet

        self._numeric_features = self._detect_numeric_features()
        self._categorical_features = self._detect_categorical_features()

        print("\n" + "=" * 70)
        print("RULE-BASED METHOD TESTING")
        print("=" * 70)
        print(f"  Numeric features:     {self._numeric_features}")
        print(f"  Categorical features: {self._categorical_features}")

    # ── Feature detection ─────────────────────────────────────────────────────

    def _detect_numeric_features(self):
        feats = []
        for col in self.data_df.columns:
            if col in self.METADATA_COLS:
                continue
            if pd.api.types.is_numeric_dtype(self.data_df[col]):
                non_null = self.data_df[col].notna().sum()
                if non_null >= 4:
                    feats.append(col)
        return feats

    def _detect_categorical_features(self):
        feats = []
        for col in self.data_df.columns:
            if col in self.METADATA_COLS or col in self._numeric_features:
                continue
            non_null = self.data_df[col].notna().sum()
            unique = self.data_df[col].nunique()
            if non_null >= 4 and 2 <= unique <= 20:
                feats.append(col)
        return feats

    def _is_annual_feature(self, feature):
        """
        Returns True if feature is constant within each site
        (same value every month → annual/repeated).
        """
        monthly = self.data_df[self.data_df['Timeframe'] == 'Monthly']
        if len(monthly) == 0:
            return True
        for site in monthly['Site identifier'].unique():
            site_vals = monthly[monthly['Site identifier'] == site][feature].dropna()
            if len(site_vals) > 1 and site_vals.nunique() > 1:
                return False
        return True

    # ── Public entry point ────────────────────────────────────────────────────

    def run_all_tests(self):
        """Discover and test all rule-based methods. Returns model dict."""
        print("\n📊 Testing feature intensity methods...")
        self._test_intensity_methods()

        print("\n📊 Testing seasonal/site average methods...")
        self._test_seasonal_methods()

        if self.hdm and len(self.hdm.site_periods) > 0:
            print("\n📊 Testing historic-adjusted intensity methods...")
            self._test_historic_adjusted_methods()

        print(f"\n✅ Rule-based methods tested: {len(self.rule_based_models)}")
        viable = sum(1 for m in self.rule_based_models.values() if m['r2'] > 0)
        print(f"   Viable (R²>0, p<0.05): {viable}")
        return self.rule_based_models

    # ── Method 1: Feature intensity ───────────────────────────────────────────

    def _test_intensity_methods(self):
        """
        For each numeric feature, test VQ ~ Feature across all scenarios
        (feature availability profiles matching blank rows).
        """
        rows_with_vq = self.data_df[self.data_df['Volumetric Quantity'].notna()].copy()

        # Identify unique feature-availability scenarios from blank rows
        scenarios = self._get_blank_row_scenarios()

        for feature in self._numeric_features:
            # Global test first (all rows that have both VQ and the feature)
            pool = rows_with_vq[rows_with_vq[feature].notna()].copy()
            if len(pool) < 4:
                continue

            vq = pool['Volumetric Quantity'].values.astype(float)
            feat_vals = pool[feature].values.astype(float)

            pr, pr2, pp = _safe_pearson(feat_vals, vq)
            sr, sr2, sp = _safe_spearman(feat_vals, vq)
            kr, kr2, kp = _safe_kendall(feat_vals, vq)
            eff_r2 = _effective_r2(pr2, pp, sr2, sp, kr2, kp)

            method_name = f'RuleBased_Intensity_{feature}'
            is_annual = self._is_annual_feature(feature)

            self._register_method(
                name=method_name,
                feature=feature,
                method_type='Feature_Intensity',
                is_annual_feature=is_annual,
                n=len(pool),
                pearson_r=pr, pearson_r2=pr2, pearson_p=pp,
                spearman_r=sr, spearman_r2=sr2, spearman_p=sp,
                kendall_tau=kr, kendall_r2=kr2, kendall_p=kp,
                pointbiserial_r=None, pointbiserial_r2=None, pointbiserial_p=None,
                effective_r2=eff_r2,
                scenario='Global',
                intensity_slope=self._calc_intensity_slope(feat_vals, vq),
            )

            # Per-scenario tests (masked to match blank row availability)
            for scenario_key, scenario_info in scenarios.items():
                scene_pool = self._get_scenario_pool(pool, scenario_info, feature)
                if len(scene_pool) < 4:
                    continue

                svq = scene_pool['Volumetric Quantity'].values.astype(float)
                sfeat = scene_pool[feature].values.astype(float)

                s_pr, s_pr2, s_pp = _safe_pearson(sfeat, svq)
                s_sr, s_sr2, s_sp = _safe_spearman(sfeat, svq)
                s_kr, s_kr2, s_kp = _safe_kendall(sfeat, svq)
                s_eff = _effective_r2(s_pr2, s_pp, s_sr2, s_sp, s_kr2, s_kp)

                scene_name = f'RuleBased_Intensity_{feature}_{scenario_key}'
                self._register_method(
                    name=scene_name,
                    feature=feature,
                    method_type='Feature_Intensity',
                    is_annual_feature=is_annual,
                    n=len(scene_pool),
                    pearson_r=s_pr, pearson_r2=s_pr2, pearson_p=s_pp,
                    spearman_r=s_sr, spearman_r2=s_sr2, spearman_p=s_sp,
                    kendall_tau=s_kr, kendall_r2=s_kr2, kendall_p=s_kp,
                    pointbiserial_r=None, pointbiserial_r2=None, pointbiserial_p=None,
                    effective_r2=s_eff,
                    scenario=scenario_key,
                    intensity_slope=self._calc_intensity_slope(sfeat, svq),
                )

            # Annual feature + seasonal: test on monthly data
            if is_annual and len(self.seasonal_patterns.get('Overall', {})) > 0:
                self._test_annual_feature_plus_seasonal(feature, rows_with_vq)

        # Categorical: point-biserial
        for feature in self._categorical_features:
            pool = rows_with_vq[rows_with_vq[feature].notna()].copy()
            if len(pool) < 4:
                continue
            vq = pool['Volumetric Quantity'].values.astype(float)
            # Encode to numeric
            codes = pd.Categorical(pool[feature]).codes.astype(float)

            pbr, pbr2, pbp = _safe_pointbiserial(codes, vq)
            eff_r2 = _effective_r2(None, None, None, None,
                                   pointbiserial_r2=pbr2, pointbiserial_p=pbp)

            self._register_method(
                name=f'RuleBased_Categorical_{feature}',
                feature=feature,
                method_type='Categorical_PointBiserial',
                is_annual_feature=self._is_annual_feature(feature),
                n=len(pool),
                pearson_r=None, pearson_r2=None, pearson_p=None,
                spearman_r=None, spearman_r2=None, spearman_p=None,
                kendall_tau=None, kendall_r2=None, kendall_p=None,
                pointbiserial_r=pbr, pointbiserial_r2=pbr2, pointbiserial_p=pbp,
                effective_r2=eff_r2,
                scenario='Global',
                intensity_slope=None,
            )

    def _calc_intensity_slope(self, feat_vals, vq):
        """VQ-per-unit-feature (median intensity ratio)."""
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(feat_vals > 0, vq / feat_vals, np.nan)
        valid = ratios[np.isfinite(ratios)]
        return float(np.median(valid)) if len(valid) > 0 else None

    def _test_annual_feature_plus_seasonal(self, feature, rows_with_vq):
        """
        Test: VQ_predicted = (AnnualFeature / 12) × SeasonalFactor
        Only valid for monthly rows with an annual (constant) feature.
        """
        monthly = rows_with_vq[
            (rows_with_vq['Timeframe'] == 'Monthly') &
            (rows_with_vq['Month'].notna()) &
            (rows_with_vq[feature].notna())
        ].copy()

        if len(monthly) < 4:
            return

        predictions, actuals = [], []
        for _, row in monthly.iterrows():
            month = row.get('Month')
            feat_val = row.get(feature)
            if not month or pd.isna(feat_val):
                continue
            seasonal_factor = self.seasonal_patterns.get('Overall', {}).get(month, 1.0)
            pred = (float(feat_val) / 12.0) * seasonal_factor
            predictions.append(pred)
            actuals.append(row['Volumetric Quantity'])

        if len(predictions) < 4:
            return

        p_arr = np.array(predictions, dtype=float)
        a_arr = np.array(actuals, dtype=float)

        pr, pr2, pp = _safe_pearson(p_arr, a_arr)
        sr, sr2, sp = _safe_spearman(p_arr, a_arr)
        kr, kr2, kp = _safe_kendall(p_arr, a_arr)
        eff = _effective_r2(pr2, pp, sr2, sp, kr2, kp)

        self._register_method(
            name=f'RuleBased_AnnualFeatureSeasonal_{feature}',
            feature=feature,
            method_type='AnnualFeature_x_Seasonal',
            is_annual_feature=True,
            n=len(predictions),
            pearson_r=pr, pearson_r2=pr2, pearson_p=pp,
            spearman_r=sr, spearman_r2=sr2, spearman_p=sp,
            kendall_tau=kr, kendall_r2=kr2, kendall_p=kp,
            pointbiserial_r=None, pointbiserial_r2=None, pointbiserial_p=None,
            effective_r2=eff,
            scenario='Global_Monthly',
            intensity_slope=None,
        )

    # ── Method 2: Seasonal / site average ─────────────────────────────────────

    def _test_seasonal_methods(self):
        """Test seasonal average and site average methods."""
        monthly = self.data_df[
            (self.data_df['Timeframe'] == 'Monthly') &
            (self.data_df['Volumetric Quantity'].notna()) &
            (self.data_df['Month'].notna())
        ].copy()

        if len(monthly) < 4:
            return

        # Site average (no seasonality)
        site_avgs = monthly.groupby('Site identifier')['Volumetric Quantity'].mean()
        preds_site = monthly['Site identifier'].map(site_avgs).values.astype(float)
        actuals = monthly['Volumetric Quantity'].values.astype(float)

        pr, pr2, pp = _safe_pearson(preds_site, actuals)
        sr, sr2, sp = _safe_spearman(preds_site, actuals)
        kr, kr2, kp = _safe_kendall(preds_site, actuals)
        eff = _effective_r2(pr2, pp, sr2, sp, kr2, kp)

        self._register_method(
            name='RuleBased_SiteAverage',
            feature='Site identifier',
            method_type='Site_Average',
            is_annual_feature=False,
            n=len(monthly),
            pearson_r=pr, pearson_r2=pr2, pearson_p=pp,
            spearman_r=sr, spearman_r2=sr2, spearman_p=sp,
            kendall_tau=kr, kendall_r2=kr2, kendall_p=kp,
            pointbiserial_r=None, pointbiserial_r2=None, pointbiserial_p=None,
            effective_r2=eff,
            scenario='Global_Monthly',
            intensity_slope=None,
        )

        # Seasonal average (site avg × seasonal factor)
        if len(self.seasonal_patterns.get('Overall', {})) > 0:
            preds_seasonal = []
            for _, row in monthly.iterrows():
                site_avg = site_avgs.get(row['Site identifier'],
                                         monthly['Volumetric Quantity'].mean())
                sf = self.seasonal_patterns['Overall'].get(row['Month'], 1.0)
                preds_seasonal.append(site_avg * sf)

            p_arr = np.array(preds_seasonal, dtype=float)
            pr2s, pr2s2, pp2s = _safe_pearson(p_arr, actuals)
            sr2s, sr2s2, sp2s = _safe_spearman(p_arr, actuals)
            kr2s, kr2s2, kp2s = _safe_kendall(p_arr, actuals)
            eff2 = _effective_r2(pr2s2, pp2s, sr2s2, sp2s, kr2s2, kp2s)

            self._register_method(
                name='RuleBased_SiteSeasonal',
                feature='Site identifier + Month',
                method_type='Site_Seasonal_Average',
                is_annual_feature=False,
                n=len(monthly),
                pearson_r=pr2s, pearson_r2=pr2s2, pearson_p=pp2s,
                spearman_r=sr2s, spearman_r2=sr2s2, spearman_p=sp2s,
                kendall_tau=kr2s, kendall_r2=kr2s2, kendall_p=kp2s,
                pointbiserial_r=None, pointbiserial_r2=None, pointbiserial_p=None,
                effective_r2=eff2,
                scenario='Global_Monthly',
                intensity_slope=None,
            )

    # ── Method 3: Historic-adjusted intensity ──────────────────────────────────

    def _test_historic_adjusted_methods(self):
        """
        Test: VQ_pred = VQ_historic × (Feature_current / Feature_historic)
        for each numeric feature where historic data is available.
        """
        rows_with_vq = self.data_df[
            (self.data_df['Volumetric Quantity'].notna()) &
            (self.data_df['Timeframe'] == 'Monthly')
        ].copy()

        for feature in self._numeric_features:
            predictions, actuals = [], []

            for _, row in rows_with_vq.iterrows():
                site = row['Site identifier']
                month = row.get('Month')
                current_feat = row.get(feature)
                current_vq = row['Volumetric Quantity']

                if pd.isna(current_feat) or not month:
                    continue

                row_date = pd.to_datetime(row['Date from'], dayfirst=True)
                period = self.hdm.get_most_recent_period(site, row_date)
                if not period:
                    continue

                hist_vq = self.hdm.get_period_vq_for_month(period, month)
                if not hist_vq:
                    annual = self.hdm.get_period_annual_total(period)
                    if annual:
                        hist_vq = annual / period.months_covered

                hist_feat = self.hdm.get_period_feature(period, feature, month=month)
                if not hist_feat and hasattr(self.hdm, 'get_historic_feature'):
                    hist_feat = self.hdm.get_historic_feature(site, period.year, feature)

                if hist_vq and hist_feat and float(hist_feat) != 0:
                    ratio = float(current_feat) / float(hist_feat)
                    pred = hist_vq * ratio
                    predictions.append(pred)
                    actuals.append(current_vq)

            if len(predictions) < 4:
                continue

            p_arr = np.array(predictions, dtype=float)
            a_arr = np.array(actuals, dtype=float)

            pr, pr2, pp = _safe_pearson(p_arr, a_arr)
            sr, sr2, sp = _safe_spearman(p_arr, a_arr)
            kr, kr2, kp = _safe_kendall(p_arr, a_arr)
            eff = _effective_r2(pr2, pp, sr2, sp, kr2, kp)

            self._register_method(
                name=f'RuleBased_HistoricAdjusted_{feature}',
                feature=feature,
                method_type='Historic_Adjusted_Intensity',
                is_annual_feature=self._is_annual_feature(feature),
                n=len(predictions),
                pearson_r=pr, pearson_r2=pr2, pearson_p=pp,
                spearman_r=sr, spearman_r2=sr2, spearman_p=sp,
                kendall_tau=kr, kendall_r2=kr2, kendall_p=kp,
                pointbiserial_r=None, pointbiserial_r2=None, pointbiserial_p=None,
                effective_r2=eff,
                scenario='Global_Monthly_Historic',
                intensity_slope=None,
            )

    # ── Scenario helpers ──────────────────────────────────────────────────────

    def _get_blank_row_scenarios(self):
        """
        Returns dict of unique feature-availability profiles from blank rows.
        Key = tuple of available feature names.
        Value = dict with profile info.
        """
        scenarios = {}
        for _, row in self.blank_df.iterrows():
            available = []
            for feat in self._numeric_features:
                try:
                    val = row.get(feat)
                    if pd.notna(val) and float(val) > 0:
                        available.append(feat)
                except (TypeError, ValueError):
                    pass
            key = tuple(sorted(available))
            if key not in scenarios:
                scenarios[key] = {
                    'features': list(key),
                    'count': 0,
                }
            scenarios[key]['count'] += 1
        return scenarios

    def _get_scenario_pool(self, pool, scenario_info, feature):
        """
        Restrict training pool to rows that have the same feature availability
        as the scenario (masking to simulate limited data availability).
        Includes the target feature only if the scenario has it.
        """
        required_features = scenario_info['features']
        if not required_features:
            return pool  # No features available — use full pool

        # Keep only rows that have ALL features this scenario requires
        mask = pd.Series([True] * len(pool), index=pool.index)
        for req_feat in required_features:
            if req_feat in pool.columns:
                mask = mask & pool[req_feat].notna()
        return pool[mask]

    # ── Registration ──────────────────────────────────────────────────────────

    def _register_method(self, name, feature, method_type, is_annual_feature,
                         n, pearson_r, pearson_r2, pearson_p,
                         spearman_r, spearman_r2, spearman_p,
                         kendall_tau, kendall_r2, kendall_p,
                         pointbiserial_r, pointbiserial_r2, pointbiserial_p,
                         effective_r2, scenario, intensity_slope):
        """Register method in both the model dict and the reporting list."""
        sig_flag = (
            (pearson_p is not None and pearson_p < 0.05) or
            (spearman_p is not None and spearman_p < 0.05) or
            (pointbiserial_p is not None and pointbiserial_p < 0.05)
        )

        # Build prediction function metadata for selection pool
        # (actual prediction applied via predict_for_row)
        model_entry = {
            'model': 'rule_based_statistical',
            'type': method_type,
            'features': [f.strip() for f in feature.split(' + ')] if ' + ' in feature else [feature],
            'r2': effective_r2,
            'r2_full': effective_r2,
            'r2_imputed': effective_r2,
            'has_derived': False,
            'is_rule_based': True,
            'is_annual_feature': is_annual_feature,
            'intensity_slope': intensity_slope,
            'pearson_r': pearson_r,
            'pearson_r2': pearson_r2,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_r2': spearman_r2,
            'spearman_p': spearman_p,
            'kendall_tau': kendall_tau,
            'kendall_p': kendall_p,
            'pointbiserial_r': pointbiserial_r,
            'pointbiserial_p': pointbiserial_p,
            'significant': sig_flag,
            'scenario': scenario,
            'description': (
                f"{method_type} using {feature}. "
                f"n={n}. "
                f"Pearson R²={pearson_r2:.3f}(p={pearson_p:.3f}) "
                f"Spearman ρ²={spearman_r2:.3f}(p={spearman_p:.3f}) "
                f"Effective R²={effective_r2:.3f}"
                if pearson_r2 is not None and spearman_r2 is not None
                else f"{method_type} using {feature}. n={n}. Effective R²={effective_r2:.3f}"
            ),
            'predictions_made': 0,
        }

        # Only register in selection pool if meaningful effective R²
        self.rule_based_models[name] = model_entry

        sig_str = "✓" if sig_flag else "✗"
        print(f"  {sig_str} {name}: EffR²={effective_r2:.4f} "
              f"(Pearson p={pearson_p:.3f}, Spearman p={spearman_p:.3f})"
              if pearson_p is not None and spearman_p is not None
              else f"  {sig_str} {name}: EffR²={effective_r2:.4f}")

        # Reporting row
        self.test_results.append({
            'Method': name,
            'Feature': feature,
            'Type': method_type,
            'Scenario': scenario,
            'N': n,
            'Annual_Feature': 'Yes' if is_annual_feature else 'No',
            'Intensity_Slope': round(intensity_slope, 6) if intensity_slope else None,
            'Pearson_r': round(pearson_r, 4) if pearson_r is not None else None,
            'Pearson_R2': round(pearson_r2, 4) if pearson_r2 is not None else None,
            'Pearson_p': round(pearson_p, 4) if pearson_p is not None else None,
            'Spearman_r': round(spearman_r, 4) if spearman_r is not None else None,
            'Spearman_R2': round(spearman_r2, 4) if spearman_r2 is not None else None,
            'Spearman_p': round(spearman_p, 4) if spearman_p is not None else None,
            'Kendall_tau': round(kendall_tau, 4) if kendall_tau is not None else None,
            'Kendall_R2': round(kendall_r2, 4) if kendall_r2 is not None else None,
            'Kendall_p': round(kendall_p, 4) if kendall_p is not None else None,
            'PointBiserial_r': round(pointbiserial_r, 4) if pointbiserial_r is not None else None,
            'PointBiserial_R2': round(pointbiserial_r2, 4) if pointbiserial_r2 is not None else None,
            'PointBiserial_p': round(pointbiserial_p, 4) if pointbiserial_p is not None else None,
            'Significant_p05': 'Yes' if sig_flag else 'No',
            'Effective_R2_for_Selection': round(effective_r2, 4),
            'Used_In_Prediction': '',      # filled in post-run
            'Predictions_Made': 0,          # filled in post-run
        })

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict_for_row(self, row, method_name):
        """
        Apply a registered rule-based method to make a prediction for a single row.
        Returns float or None.
        """
        if method_name not in self.rule_based_models:
            return None

        info = self.rule_based_models[method_name]
        method_type = info.get('type')
        feature = info.get('features', [None])[0]

        try:
            if method_type == 'Feature_Intensity':
                slope = info.get('intensity_slope')
                if slope is None or feature is None:
                    return None
                feat_val = row.get(feature)
                if pd.isna(feat_val):
                    return None
                return float(feat_val) * slope

            elif method_type == 'AnnualFeature_x_Seasonal':
                feat_val = row.get(feature)
                month = row.get('Month')
                if pd.isna(feat_val) or not month:
                    return None
                sf = self.seasonal_patterns.get('Overall', {}).get(month, 1.0)
                # Annual feature / 12 gives monthly average, then apply seasonal factor
                # intensity_slope = median(VQ / feature) across monthly rows,
                # so VQ_monthly = feature_monthly * slope = (feat_annual/12) * sf * slope
                slope = info.get('intensity_slope') or 1.0
                return (float(feat_val) / 12.0) * sf

            elif method_type == 'Site_Average':
                monthly = self.data_df[
                    (self.data_df['Timeframe'] == 'Monthly') &
                    (self.data_df['Volumetric Quantity'].notna())
                ]
                site = row.get('Site identifier')
                site_data = monthly[monthly['Site identifier'] == site]
                if len(site_data) > 0:
                    return float(site_data['Volumetric Quantity'].mean())
                return float(monthly['Volumetric Quantity'].mean())

            elif method_type == 'Site_Seasonal_Average':
                monthly = self.data_df[
                    (self.data_df['Timeframe'] == 'Monthly') &
                    (self.data_df['Volumetric Quantity'].notna())
                ]
                site = row.get('Site identifier')
                month = row.get('Month')
                site_data = monthly[monthly['Site identifier'] == site]
                site_avg = (float(site_data['Volumetric Quantity'].mean())
                            if len(site_data) > 0
                            else float(monthly['Volumetric Quantity'].mean()))
                sf = self.seasonal_patterns.get('Overall', {}).get(month, 1.0)
                return site_avg * sf

            elif method_type == 'Historic_Adjusted_Intensity':
                if not self.hdm:
                    return None
                site = row.get('Site identifier')
                month = row.get('Month')
                current_feat = row.get(feature)
                if pd.isna(current_feat) or not month:
                    return None
                row_date = pd.to_datetime(row['Date from'], dayfirst=True)
                period = self.hdm.get_most_recent_period(site, row_date)
                if not period:
                    return None
                hist_vq = self.hdm.get_period_vq_for_month(period, month)
                if not hist_vq:
                    annual = self.hdm.get_period_annual_total(period)
                    if annual:
                        hist_vq = annual / period.months_covered
                hist_feat = self.hdm.get_period_feature(period, feature, month=month)
                if hist_vq and hist_feat and float(hist_feat) != 0:
                    return hist_vq * (float(current_feat) / float(hist_feat))
                return None

            elif method_type == 'Categorical_PointBiserial':
                # No meaningful way to predict from a categorical code —
                # fall back to the group mean
                if feature not in self.data_df.columns:
                    return None
                cat_val = row.get(feature)
                group = self.data_df[
                    (self.data_df[feature] == cat_val) &
                    (self.data_df['Volumetric Quantity'].notna())
                ]
                if len(group) > 0:
                    return float(group['Volumetric Quantity'].mean())
                return None

        except Exception:
            return None

    # ── Reporting sheet ───────────────────────────────────────────────────────

    def create_report_dataframe(self, scenario_best_models=None):
        """
        Build the Rule_Based_Methods DataFrame for the Excel tab.
        Optionally annotates which methods were actually selected for predictions.
        """
        if not self.test_results:
            return None

        # Annotate usage
        if scenario_best_models:
            selected_methods = set()
            for scenario, (model_name, _) in scenario_best_models.items():
                if model_name and model_name.startswith('RuleBased_'):
                    selected_methods.add(model_name)

            for row in self.test_results:
                used = 'Yes' if row['Method'] in selected_methods else 'No'
                row['Used_In_Prediction'] = used
                if model_name := row['Method']:
                    if model_name in self.rule_based_models:
                        row['Predictions_Made'] = self.rule_based_models[model_name].get('predictions_made', 0)

        df = pd.DataFrame(self.test_results)

        # Sort: significant first, then by effective R² descending
        df['_sig_sort'] = (df['Significant_p05'] == 'Yes').astype(int)
        df = df.sort_values(['_sig_sort', 'Effective_R2_for_Selection'],
                            ascending=[False, False])
        df = df.drop(columns=['_sig_sort'])
        df = df.reset_index(drop=True)

        return df