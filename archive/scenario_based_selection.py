"""
Simple Scenario-Based Model Selection
======================================

VERSION: 2.0 - Features_key implementation (Feb 2026)

Groups blank rows by their data availability scenario.
Tests all models on each scenario.
Uses the best model for all rows in that scenario.

FIXED:
- identify_scenario() detects annual vs monthly rows correctly
- _get_test_sites_for_scenario() includes annual sites in test pool
- _test_model_on_scenario() handles annual rows properly
- _can_model_predict() blocks Site_Seasonal/Month-dependent models for annual rows
- Zero hardcoded fallbacks -- best available model always wins
- get_best_model_for_row() gracefully relaxes constraints rather than hardcoding
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import r2_score

VERSION = "2.0_features_key"  # Check this to verify you're running the updated file


class ScenarioBasedModelSelector:

    def __init__(self, data_df, blank_df, models, historic_manager, seasonal_patterns):
        self.data_df = data_df
        self.blank_df = blank_df
        self.models = models
        self.hdm = historic_manager
        self.seasonal_patterns = seasonal_patterns
        self.scenario_best_models = {}

        print("\n" + "=" * 70)
        print("SCENARIO-BASED MODEL SELECTION")
        print("=" * 70)

    def identify_scenario(self, row):
        """
        Identify the data availability scenario for this row.

        Returns: (timeframe_type, months_bucket, features_key, has_historic)
        timeframe_type = 'annual' or 'monthly'
        features_key = tuple of available feature names (e.g. ('Electricity', 'Turnover'))
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

        # Detect ALL available numeric features for this row (not just Turnover)
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

            # Defensive: ensure features_key is iterable
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
        Test ALL models on sites matching this scenario.
        Pick highest R². No hardcoded fallbacks -- best available wins.
        If nothing scores positively, pick least-bad. If nothing testable, pick
        highest training R² from registered models.
        """
        timeframe_type, months_bucket, features_key, has_historic = scenario
        test_sites = self._get_test_sites_for_scenario(scenario)

        print(
            f"\n  Scenario: [{timeframe_type}] {months_bucket}mo, Features:{list(features_key)}, Historic:{has_historic}")
        print(f"    Test pool: {len(test_sites)} sites")

        test_results = {'test_pool_size': len(test_sites), 'model_r2s': {}}
        model_r2s = {}

        for model_name, model_info in self.models.items():
            r2 = self._test_model_on_scenario(model_name, model_info, scenario, test_sites)
            test_results['model_r2s'][model_name] = r2
            if r2 is not None:
                model_r2s[model_name] = r2
                status = "+" if r2 > 0 else "-"
                print(f"      [{status}] {model_name}: R2 = {r2:.4f}")

        # Best positive R2
        viable = {n: r for n, r in model_r2s.items() if r > 0}
        if viable:
            best = max(viable, key=viable.get)
            print(f"    -> BEST: {best} (R2 = {viable[best]:.4f})")
            return best, viable[best], test_results

        # Least bad
        if model_r2s:
            best = max(model_r2s, key=model_r2s.get)
            print(f"    -> LEAST BAD: {best} (R2 = {model_r2s[best]:.4f})")
            return best, model_r2s[best], test_results

        # Nothing testable -- pick highest training R2 from registered models
        if self.models:
            best = max(self.models, key=lambda k: self.models[k].get('r2', 0))
            best_r2 = self.models[best].get('r2', 0)
            print(f"    -> TRAINING R2 FALLBACK: {best} (training R2 = {best_r2:.4f})")
            return best, best_r2, test_results

        return None, 0, test_results

    def _get_test_sites_for_scenario(self, scenario):
        """
        Get sites for testing this scenario.

        Annual scenarios: any site with actual VQ data to validate against.
        Monthly scenarios: sites with >= months_bucket monthly rows (recycling).
        """
        timeframe_type, months_bucket, features_key, has_historic = scenario

        # Defensive: ensure features_key is iterable
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
                # Check all required features are present for this site
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
                if site_month_counts[site] < months_bucket:
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
        Test a model against known actuals for this scenario.
        Annual scenarios compare predicted annual total vs actual annual total.
        Monthly scenarios compare predicted monthly VQ vs actual monthly VQ.
        Minimum 3 predictions required to compute R2.

        CRITICAL: Only test models whose features are available in this scenario.
        E.g., don't test Intensity_Poly_All (which uses Electricity) on a scenario
        where Electricity is not available.
        """
        timeframe_type, months_bucket, features_key, has_historic = scenario

        # Defensive: ensure features_key is iterable
        if not isinstance(features_key, (tuple, list)):
            features_key = tuple()

        # Filter: Don't test models that require features not in this scenario
        model_features = set(model_info.get('features', []))
        model_features.discard('Month')  # Month is timeframe-specific, not a real feature
        scenario_features = set(features_key)

        if model_features and not model_features.issubset(scenario_features):
            # Model requires features this scenario doesn't have - skip testing
            return None

        predictions = []
        actuals = []

        for site in test_sites:
            try:
                site_data = self.data_df[self.data_df['Site identifier'] == site]

                if timeframe_type == 'annual':
                    # Derive actual annual total
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

                    # Scale monthly models to annual
                    model_type = model_info.get('type', '')
                    if (model_type not in ['Annual_Average'] and
                            'Annual' not in model_name and
                            model_info.get('features') and
                            'Month' in model_info.get('features', [])):
                        pred = pred * 12

                    predictions.append(pred)
                    actuals.append(actual_annual)

                else:
                    site_data_actual = site_data[site_data['Timeframe'] == 'Monthly'].copy()
                    if len(site_data_actual) == 0:
                        continue

                    site_data_limited = site_data_actual.head(months_bucket) if months_bucket > 0 else pd.DataFrame()

                    if months_bucket == 0:
                        test_row = site_data_actual.iloc[0].copy()
                        if not self._can_model_predict(model_name, model_info, test_row):
                            continue
                        pred = self._apply_model(model_name, model_info, test_row, site_data_limited)
                        if pred is None or pred <= 0:
                            continue
                        actual_annual = site_data_actual['Volumetric Quantity'].sum()
                        if model_info.get('type') != 'Annual_Average':
                            pred = pred * 12
                        predictions.append(pred)
                        actuals.append(actual_annual)
                    else:
                        for idx, row in site_data_actual.iterrows():
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
            return r2_score(actuals, predictions)
        return None

    def _can_model_predict(self, model_name, model_info, row):
        """
        Check if model can predict this row based purely on what it needs.
        No hardcoding -- derived from model metadata.
        """
        model_type = model_info.get('type', '')
        row_timeframe = row.get('Timeframe', 'Monthly')

        # Historic rule-based models need historic data
        if model_type in ['LastYear_Direct', 'LastYear_TurnoverAdjusted',
                          'MultiYear_Average', 'LastYear_GrowthAdjusted']:
            site = row['Site identifier']
            if self.hdm and site in self.hdm.site_periods:
                return len(self.hdm.site_periods[site]) > 0
            return False

        # Annual_Average works for any row
        if model_type == 'Annual_Average':
            return True

        # Any model with 'Month' in features cannot predict annual rows
        features = model_info.get('features', [])
        if 'Month' in features:
            if row_timeframe == 'Annual':
                return False
            month = row.get('Month', '')
            if pd.isna(month) or month == '':
                return False

        # Check all other required features are present
        for feat in features:
            if feat == 'Month':
                continue  # Already checked above
            if pd.isna(row.get(feat, None)):
                return False

        return True

    def _apply_model(self, model_name, model_info, row, site_data_limited):
        """Apply a model to predict a row"""
        model_type = model_info.get('type', '')

        if model_type == 'Site_Seasonal':
            return self._predict_site_seasonal(row, site_data_limited)
        elif model_type in ['LastYear_Direct', 'LastYear_TurnoverAdjusted',
                            'MultiYear_Average', 'LastYear_GrowthAdjusted']:
            return self._predict_historic(model_name, row)
        elif model_type == 'Annual_Average':
            return model_info.get('average')
        else:
            return self._predict_ml(model_info, row)

    def _predict_site_seasonal(self, row, site_data_limited):
        """Predict using site average + seasonal pattern"""
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
        """Historic rule-based prediction -- delegates to main tool"""
        return None

    def _predict_ml(self, model_info, row):
        """Predict using ML model"""
        model = model_info.get('model')
        if not model or model == 'rule_based':
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
        """
        scenario = self.identify_scenario(row)

        if scenario in self.scenario_best_models:
            return self.scenario_best_models[scenario]

        # Relax constraints progressively to find closest match
        tf, months, features_key, has_hist = scenario

        # Defensive: ensure features_key is iterable
        if not isinstance(features_key, (tuple, list)):
            features_key = tuple()
        for fallback in [
            (tf, months, features_key, False),
            (tf, months, tuple(), has_hist),
            (tf, months, tuple(), False),
            (tf, 0, tuple(), False),
            ('monthly', 0, tuple(), False),
        ]:
            if fallback in self.scenario_best_models:
                return self.scenario_best_models[fallback]

        # Absolute last resort -- best model by scenario R2
        if self.scenario_best_models:
            return max(self.scenario_best_models.values(), key=lambda x: x[1])

        # Nothing -- pick highest training R2
        if self.models:
            best = max(self.models, key=lambda k: self.models[k].get('r2', 0))
            return best, self.models[best].get('r2', 0)

        return None, 0

    def print_scenario_summary(self):
        """Print summary of scenarios and their best models"""
        print("\n" + "=" * 70)
        print("SCENARIO MODEL ASSIGNMENTS")
        print("=" * 70)
        for scenario, (model_name, r2) in sorted(self.scenario_best_models.items()):
            tf, months, features_key, has_hist = scenario

            # Defensive: ensure features_key is iterable
            if not isinstance(features_key, (tuple, list)):
                features_key = tuple()
            print(f"\n[{tf}] {months} months, Features:{list(features_key)}, Historic:{has_hist}")
            print(f"  -> {model_name} (R2 = {r2:.4f})")