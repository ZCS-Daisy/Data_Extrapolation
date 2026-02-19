"""
Simple Scenario-Based Model Selection
======================================

VERSION: 3.0 - Honest per-scenario testing (Feb 2026)

Groups blank rows by their data availability scenario.
Tests all models on each scenario's specific data subset.
Uses the best model for all rows in that scenario.

CHANGES v3.0:
- Per-scenario R² testing re-enabled (was deprecated/disabled in v2.1)
- No R²>0 gate: every model competes on its actual tested R² for that scenario
- Annual_Average is tested and scored honestly rather than used as a hardcoded fallback
- Blanks always get filled: highest R² wins, even if R² is very low
- Data Quality Score reflects actual scenario-tested R², not global training R²
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import r2_score

VERSION = "3.0_honest_scenario_testing"


class ScenarioBasedModelSelector:

    def __init__(self, data_df, blank_df, models, historic_manager, seasonal_patterns):
        self.data_df = data_df
        self.blank_df = blank_df
        self.models = models
        self.hdm = historic_manager
        self.seasonal_patterns = seasonal_patterns
        self.scenario_best_models = {}

        print("\n" + "=" * 70)
        print("SCENARIO-BASED MODEL SELECTION v3.0")
        print("=" * 70)

    def identify_scenario(self, row):
        """
        Identify the data availability scenario for this row.
        Returns: (timeframe_type, months_bucket, features_key, has_historic)
        """
        site_id = row['Site identifier']
        row_timeframe = row.get('Timeframe', 'Monthly')
        timeframe_type = 'annual' if row_timeframe == 'Annual' else 'monthly'

        site_monthly_data = self.data_df[
            (self.data_df['Site identifier'] == site_id) &
            (self.data_df['Timeframe'] == 'Monthly') &
            (self.data_df['Volumetric Quantity'].notna())
        ]
        site_months = len(site_monthly_data)

        available_features = []
        for col in row.index:
            if col in ('Site identifier', 'Location', 'Volumetric Quantity',
                       'Timeframe', 'Month', 'Date from', 'Date to',
                       'GHG Category', 'Data integrity', 'Estimation Method',
                       'Data Quality', 'Data Quality Score', 'Year',
                       'Years_Since_Baseline', 'Client'):
                continue
            try:
                val = row.get(col)
                if pd.notna(val) and float(val) > 0:
                    available_features.append(col)
            except (TypeError, ValueError):
                pass
        features_key = tuple(sorted(available_features))

        has_historic = False
        if self.hdm and site_id in self.hdm.site_periods:
            has_historic = len(self.hdm.site_periods[site_id]) > 0

        if site_months == 0:
            months_bucket = 0
        elif site_months <= 3:
            months_bucket = 3
        elif site_months <= 6:
            months_bucket = 6
        else:
            months_bucket = 12

        return (timeframe_type, months_bucket, features_key, has_historic)

    def find_best_models_for_all_scenarios(self):
        """For each unique scenario in blank rows, find the best model."""
        scenario_groups = defaultdict(list)

        for idx, row in self.blank_df.iterrows():
            scenario = self.identify_scenario(row)
            scenario_groups[scenario].append(idx)

        print(f"\nFound {len(scenario_groups)} unique scenarios:")
        for scenario, rows in scenario_groups.items():
            tf, months, features_key, has_hist = scenario
            if not isinstance(features_key, (tuple, list)):
                features_key = tuple()
            print(f"  [{tf}] {months} months, Features:{list(features_key)}, Historic:{has_hist} -> {len(rows)} rows")

        self.scenario_test_results = {}

        print("\nTesting models for each scenario...")

        for scenario in scenario_groups.keys():
            best_model, best_r2, test_results = self._find_best_model_for_scenario(scenario)
            self.scenario_best_models[scenario] = (best_model, best_r2)
            self.scenario_test_results[scenario] = test_results

        print("\n" + "=" * 70)
        print("DATA AVAILABILITY ANALYSIS")
        print("=" * 70)
        for scenario, rows in sorted(scenario_groups.items(), key=lambda x: x[0]):
            tf, months, _, _ = scenario
            print(f"  [{tf}] {months:2d} months: {len(rows):3d} blank rows")
        print("=" * 70)

    def _find_best_model_for_scenario(self, scenario):
        """
        Find best model for this scenario by actually testing each model
        on the scenario-specific data subset.

        v3.0: No R²>0 gate. Highest tested R² wins. Blanks always get filled.
        """
        timeframe_type, months_bucket, features_key, has_historic = scenario

        if not isinstance(features_key, (tuple, list)):
            features_key = tuple()

        test_sites = self._get_test_sites_for_scenario(scenario)

        print(f"\n  Scenario: [{timeframe_type}] {months_bucket}mo, Features:{list(features_key)}, Historic:{has_historic}")
        print(f"    Test pool: {len(test_sites)} sites")

        if len(test_sites) < 3:
            print(f"    → Too few test sites — using global training R²")
            return self._find_best_by_training_r2(scenario, test_sites)

        # Test every compatible model on this scenario's data
        scenario_r2s = {}

        for model_name, model_info in self.models.items():
            # Check feature compatibility
            model_features = set(model_info.get('features', []))
            model_features.discard('Month')
            scenario_features = set(features_key)

            if model_features and not model_features.issubset(scenario_features):
                continue

            # Block month-dependent models for annual rows
            if timeframe_type == 'annual' and 'Month' in model_info.get('features', []):
                continue

            # Block historic models if no historic data available
            model_uses_historic = any(
                x in str(model_info.get('features', []))
                for x in ['Historic', 'YoY', 'Growth', 'Trend', 'Volatility']
            ) or model_info.get('type', '') in [
                'LastYear_Direct', 'LastYear_TurnoverAdjusted',
                'MultiYear_Average', 'LastYear_GrowthAdjusted'
            ]
            if model_uses_historic and not has_historic:
                continue

            # Actually test this model on the scenario's data subset
            tested_r2 = self._test_model_on_scenario(model_name, model_info, scenario, test_sites)

            if tested_r2 is not None:
                scenario_r2s[model_name] = tested_r2
                status = "✓" if tested_r2 > 0.3 else ("~" if tested_r2 > 0 else "✗")
                print(f"      {status} {model_name}: Scenario R² = {tested_r2:.4f}")
            else:
                # Fall back to training R² if can't test (e.g. rule-based with no test data)
                training_r2 = model_info.get('r2', None)
                if training_r2 is not None:
                    scenario_r2s[model_name] = training_r2
                    print(f"      ~ {model_name}: Training R² = {training_r2:.4f} (no scenario test)")

        if not scenario_r2s:
            print(f"    → No compatible models found")
            test_results = {'test_pool_size': len(test_sites), 'model_r2s': {}}
            return None, 0, test_results

        # Best model = highest R², no floor — blanks get filled no matter what
        best_model = max(scenario_r2s, key=scenario_r2s.get)
        best_r2 = scenario_r2s[best_model]

        quality_flag = "" if best_r2 >= 0.3 else " ⚠️  [low R² — best available]"
        print(f"    → BEST: {best_model} (R² = {best_r2:.4f}){quality_flag}")

        test_results = {'test_pool_size': len(test_sites), 'model_r2s': scenario_r2s}
        return best_model, best_r2, test_results

    def _find_best_by_training_r2(self, scenario, test_sites):
        """
        Fallback when test pool is too small: use global training R².
        Still no R²>0 gate — best available wins.
        """
        timeframe_type, months_bucket, features_key, has_historic = scenario

        if not isinstance(features_key, (tuple, list)):
            features_key = tuple()

        compatible_r2s = {}

        for model_name, model_info in self.models.items():
            model_features = set(model_info.get('features', []))
            model_features.discard('Month')
            scenario_features = set(features_key)

            if model_features and not model_features.issubset(scenario_features):
                continue
            if timeframe_type == 'annual' and 'Month' in model_info.get('features', []):
                continue

            model_uses_historic = any(
                x in str(model_info.get('features', []))
                for x in ['Historic', 'YoY', 'Growth', 'Trend', 'Volatility']
            ) or model_info.get('type', '') in [
                'LastYear_Direct', 'LastYear_TurnoverAdjusted',
                'MultiYear_Average', 'LastYear_GrowthAdjusted'
            ]
            if model_uses_historic and not has_historic:
                continue

            training_r2 = model_info.get('r2', None)
            if training_r2 is not None:
                compatible_r2s[model_name] = training_r2

        if not compatible_r2s:
            test_results = {'test_pool_size': len(test_sites), 'model_r2s': {}}
            return None, 0, test_results

        best_model = max(compatible_r2s, key=compatible_r2s.get)
        best_r2 = compatible_r2s[best_model]

        test_results = {'test_pool_size': len(test_sites), 'model_r2s': compatible_r2s}
        return best_model, best_r2, test_results

    def _get_test_sites_for_scenario(self, scenario):
        """Get sites that match this scenario's data availability profile."""
        timeframe_type, months_bucket, features_key, has_historic = scenario

        if not isinstance(features_key, (tuple, list)):
            features_key = tuple()

        test_sites = []

        if timeframe_type == 'annual':
            candidate_sites = self.data_df['Site identifier'].unique()
            for site in candidate_sites:
                site_data = self.data_df[self.data_df['Site identifier'] == site]
                if site_data['Volumetric Quantity'].notna().sum() == 0:
                    continue
                row = site_data.iloc[0]
                if any(pd.isna(row.get(f)) for f in features_key):
                    continue
                if has_historic:
                    if not self.hdm or site not in self.hdm.site_periods:
                        continue
                    if len(self.hdm.site_periods[site]) == 0:
                        continue
                test_sites.append(site)
        else:
            monthly_data = self.data_df[self.data_df['Timeframe'] == 'Monthly']
            site_month_counts = monthly_data.groupby('Site identifier').size()
            for site in site_month_counts.index:
                if site_month_counts[site] < max(months_bucket, 1):
                    continue
                site_rows = self.data_df[self.data_df['Site identifier'] == site]
                if len(site_rows) == 0:
                    continue
                row = site_rows.iloc[0]
                if any(pd.isna(row.get(f)) for f in features_key):
                    continue
                if has_historic:
                    if not self.hdm or site not in self.hdm.site_periods:
                        continue
                    if len(self.hdm.site_periods[site]) == 0:
                        continue
                test_sites.append(site)

        return test_sites

    def _test_model_on_scenario(self, model_name, model_info, scenario, test_sites):
        """
        Test a model on the specific data subset for this scenario.
        Returns R² score or None if insufficient data.

        This gives a genuine per-scenario performance score rather than
        reusing the global training R².
        """
        timeframe_type, months_bucket, features_key, has_historic = scenario

        if not isinstance(features_key, (tuple, list)):
            features_key = tuple()

        predictions = []
        actuals = []

        for site in test_sites:
            try:
                site_data = self.data_df[self.data_df['Site identifier'] == site]

                if timeframe_type == 'annual':
                    annual_rows = site_data[site_data['Timeframe'] == 'Annual']
                    monthly_rows = site_data[site_data['Timeframe'] == 'Monthly']

                    if len(annual_rows) > 0 and annual_rows['Volumetric Quantity'].notna().sum() > 0:
                        actual_annual = annual_rows['Volumetric Quantity'].sum()
                    elif len(monthly_rows) >= 6:
                        monthly_total = monthly_rows['Volumetric Quantity'].sum()
                        actual_annual = monthly_total * (12 / len(monthly_rows))
                    else:
                        continue

                    test_row = site_data.iloc[0].copy()
                    if not self._can_model_predict(model_name, model_info, test_row):
                        continue

                    pred = self._apply_model(model_name, model_info, test_row, pd.DataFrame())
                    if pred is None or pred <= 0:
                        continue

                    model_type = model_info.get('type', '')
                    if (model_type not in ['Annual_Average'] and
                            'Annual' not in model_name and
                            model_info.get('features') and
                            'Month' in model_info.get('features', [])):
                        pred = pred * 12

                    predictions.append(pred)
                    actuals.append(actual_annual)

                else:
                    site_monthly = site_data[site_data['Timeframe'] == 'Monthly'].copy()
                    if len(site_monthly) == 0:
                        continue

                    # Simulate the scenario: only use the first `months_bucket` months as context
                    site_data_limited = site_monthly.head(months_bucket) if months_bucket > 0 else pd.DataFrame()

                    if months_bucket == 0:
                        # No prior months — predict one row and compare to annual total
                        test_row = site_monthly.iloc[0].copy()
                        if not self._can_model_predict(model_name, model_info, test_row):
                            continue
                        pred = self._apply_model(model_name, model_info, test_row, site_data_limited)
                        if pred is None or pred <= 0:
                            continue
                        actual_annual = site_monthly['Volumetric Quantity'].sum()
                        model_type = model_info.get('type', '')
                        if model_type != 'Annual_Average':
                            pred = pred * 12
                        predictions.append(pred)
                        actuals.append(actual_annual)
                    else:
                        # Test on each monthly row individually
                        for idx, row in site_monthly.iterrows():
                            if not self._can_model_predict(model_name, model_info, row):
                                continue
                            pred = self._apply_model(model_name, model_info, row, site_data_limited)
                            if pred is None or pred <= 0:
                                continue
                            predictions.append(pred)
                            actuals.append(row['Volumetric Quantity'])

            except Exception:
                continue

        if len(predictions) >= 3:
            try:
                return float(r2_score(actuals, predictions))
            except Exception:
                return None
        return None

    def _can_model_predict(self, model_name, model_info, row):
        """Check if model can predict this row."""
        model_type = model_info.get('type', '')
        row_timeframe = row.get('Timeframe', 'Monthly')

        if model_type in ['LastYear_Direct', 'LastYear_TurnoverAdjusted',
                          'MultiYear_Average', 'LastYear_GrowthAdjusted']:
            site = row['Site identifier']
            if self.hdm and site in self.hdm.site_periods:
                return len(self.hdm.site_periods[site]) > 0
            return False

        if model_type == 'Annual_Average':
            return True

        # Rule-based models: check their required feature
        if model_info.get('is_rule_based'):
            features = model_info.get('features', [])
            for feat in features:
                if feat in ('Site identifier', 'Month'):
                    continue
                if pd.isna(row.get(feat, None)):
                    return False
            return True

        features = model_info.get('features', [])
        if 'Month' in features:
            if row_timeframe == 'Annual':
                return False
            month = row.get('Month', '')
            if pd.isna(month) or month == '':
                return False

        for feat in features:
            if feat == 'Month':
                continue
            if pd.isna(row.get(feat, None)):
                return False

        return True

    def _apply_model(self, model_name, model_info, row, site_data_limited):
        """Apply a model to predict a row."""
        model_type = model_info.get('type', '')

        # Rule-based models delegate to the rule_based_tester on the main tool
        # For scenario testing purposes, we use the stored intensity slope / method
        if model_info.get('is_rule_based'):
            return self._apply_rule_based(model_name, model_info, row)

        if model_type == 'Site_Seasonal':
            return self._predict_site_seasonal(row, site_data_limited)
        elif model_type in ['LastYear_Direct', 'LastYear_TurnoverAdjusted',
                            'MultiYear_Average', 'LastYear_GrowthAdjusted']:
            return self._predict_historic(model_name, row)
        elif model_type == 'Annual_Average':
            return model_info.get('average')
        else:
            return self._predict_ml(model_info, row)

    def _apply_rule_based(self, model_name, model_info, row):
        """Apply rule-based model using stored statistics."""
        method_type = model_info.get('type', '')
        features = model_info.get('features', [])
        feature = features[0] if features else None

        try:
            if method_type == 'Feature_Intensity':
                slope = model_info.get('intensity_slope')
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

    def _predict_site_seasonal(self, row, site_data_limited):
        """Predict using site average + seasonal pattern."""
        month = row.get('Month')
        if not month:
            return None

        if len(site_data_limited) > 0:
            site_avg = site_data_limited['Volumetric Quantity'].mean()
        else:
            monthly_data = self.data_df[self.data_df['Timeframe'] == 'Monthly']
            site_avg = monthly_data['Volumetric Quantity'].mean()

        if 'Overall' in self.seasonal_patterns and month in self.seasonal_patterns['Overall']:
            return site_avg * self.seasonal_patterns['Overall'][month]

        return site_avg

    def _predict_historic(self, model_name, row):
        """Historic rule-based prediction — delegates to main tool."""
        return None

    def _predict_ml(self, model_info, row):
        """Predict using ML model."""
        model = model_info.get('model')
        if not model or model == 'rule_based_statistical':
            return None

        features = model_info.get('features', [])
        X_features = []

        months_list = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']

        for feat in features:
            if feat == 'Month':
                month_val = row.get('Month', '')
                if month_val in months_list:
                    X_features.append(months_list.index(month_val))
                else:
                    return None
            elif feat in row.index:
                val = row.get(feat)
                if pd.notna(val):
                    X_features.append(float(val))
                else:
                    return None
            else:
                return None

        if len(X_features) == len(features):
            try:
                return model.predict(np.array(X_features).reshape(1, -1))[0]
            except Exception:
                return None

        return None

    def get_best_model_for_row(self, row):
        """
        Get the best model for this row based on its scenario.
        If exact scenario not seen, relax constraints progressively.

        v3.0: Returns whatever model has the highest R², even if R² is low.
        A filled blank with a weak model is better than no prediction.
        """
        scenario = self.identify_scenario(row)

        if scenario in self.scenario_best_models:
            model_name, r2 = self.scenario_best_models[scenario]
            if model_name is not None:
                return model_name, r2

        # Relax constraints progressively to find closest match
        tf, months, features_key, has_hist = scenario

        if not isinstance(features_key, (tuple, list)):
            features_key = tuple()

        for fallback_scenario in [
            (tf, months, features_key, False),
            (tf, months, tuple(), has_hist),
            (tf, months, tuple(), False),
            (tf, 0, tuple(), False),
            ('monthly', 0, tuple(), False),
        ]:
            if fallback_scenario in self.scenario_best_models:
                model_name, r2 = self.scenario_best_models[fallback_scenario]
                if model_name is not None:
                    return model_name, r2

        # Last resort: Annual_Average or any model
        if 'Annual_Average' in self.models:
            r2 = self.models['Annual_Average'].get('r2', 0)
            return 'Annual_Average', r2

        # Pick any model
        for name, info in self.models.items():
            return name, info.get('r2', 0)

        return None, 0

    def print_scenario_summary(self):
        """Print summary of scenarios and their best models."""
        print("\n" + "=" * 70)
        print("SCENARIO MODEL ASSIGNMENTS")
        print("=" * 70)
        for scenario, (model_name, r2) in sorted(self.scenario_best_models.items()):
            tf, months, features_key, has_hist = scenario
            if not isinstance(features_key, (tuple, list)):
                features_key = tuple()
            quality = "✓" if r2 >= 0.3 else ("~" if r2 > 0 else "⚠️ ")
            print(f"\n[{tf}] {months} months, Features:{list(features_key)}, Historic:{has_hist}")
            print(f"  -> {quality} {model_name} (R² = {r2:.4f})")