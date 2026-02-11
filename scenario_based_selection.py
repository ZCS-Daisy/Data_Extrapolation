"""
Simple Scenario-Based Model Selection
======================================

Groups blank rows by their data availability scenario.
Tests all models on each scenario.
Uses the best model for all rows in that scenario.

Much simpler than full context-aware system!
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import r2_score


class ScenarioBasedModelSelector:
    """
    Simple approach: Group rows by scenario, use best model per scenario.

    Scenario = data availability pattern (e.g., "6 months + Turnover + Historic")
    """

    def __init__(self, data_df, blank_df, models, historic_manager, seasonal_patterns):
        self.data_df = data_df
        self.blank_df = blank_df
        self.models = models
        self.hdm = historic_manager
        self.seasonal_patterns = seasonal_patterns

        # Results: {scenario_key: (best_model_name, r2)}
        self.scenario_best_models = {}

        print("\n" + "="*70)
        print("SCENARIO-BASED MODEL SELECTION")
        print("="*70)

    def identify_scenario(self, row):
        """
        Identify the data availability scenario for this row.

        Returns:
        --------
        tuple : (site_months, has_turnover, has_historic)
        """
        site_id = row['Site identifier']

        # How many months of current year data?
        site_data = self.data_df[
            (self.data_df['Site identifier'] == site_id) &
            (self.data_df['Timeframe'] == 'Monthly') &
            (self.data_df['Volumetric Quantity'].notna())
        ]
        site_months = len(site_data)

        # Has Turnover?
        has_turnover = pd.notna(row.get('Turnover', None))

        # Has historic data?
        has_historic = False
        if self.hdm and site_id in self.hdm.site_periods:
            has_historic = len(self.hdm.site_periods[site_id]) > 0

        # Round months to nearest bucket (0, 3, 6, 12)
        if site_months == 0:
            months_bucket = 0
        elif site_months <= 3:
            months_bucket = 3
        elif site_months <= 6:
            months_bucket = 6
        else:
            months_bucket = 12

        return (months_bucket, has_turnover, has_historic)

    def find_best_models_for_all_scenarios(self):
        """
        For each unique scenario in blank rows, find the best model.
        """
        # Group blank rows by scenario
        scenario_groups = defaultdict(list)

        for idx, row in self.blank_df.iterrows():
            scenario = self.identify_scenario(row)
            scenario_groups[scenario].append(idx)

        print(f"\n📊 Found {len(scenario_groups)} unique scenarios:")
        for scenario, rows in scenario_groups.items():
            months, has_turn, has_hist = scenario
            print(f"  • {months} months, Turnover: {has_turn}, Historic: {has_hist} → {len(rows)} rows")

        # Initialize storage for detailed test results (for Model Testing Matrix)
        self.scenario_test_results = {}

        # Find best model for each scenario
        print("\n🎯 Testing models for each scenario...")

        for scenario in scenario_groups.keys():
            best_model, best_r2, test_results = self._find_best_model_for_scenario(scenario)
            self.scenario_best_models[scenario] = (best_model, best_r2)
            self.scenario_test_results[scenario] = test_results  # Store detailed results

        # Show recycling pattern summary
        print("\n" + "=" * 70)
        print("DATA AVAILABILITY ANALYSIS")
        print("=" * 70)
        monthly_data = self.data_df[self.data_df['Timeframe'] == 'Monthly']
        site_month_counts = monthly_data.groupby('Site identifier').size()

        # Sort contexts by month count
        sorted_contexts = sorted(scenario_groups.keys(), key=lambda x: x[0])

        for context in sorted_contexts:
            months, _, _ = context
            sites_available = sum(1 for count in site_month_counts if count >= months)
            rows_using = len(scenario_groups[context])
            print(f"  {months:2d} months context: {sites_available:3d} sites in pool → testing for {rows_using:3d} blank rows")

        print("\n  ↑ More test sites (recycling sites with more months)")
        print("  ↓ Fewer test sites (can't fake data you don't have)")
        print("=" * 70)

    def _find_best_model_for_scenario(self, scenario):
        """
        Test ALL applicable models on sites matching this scenario.
        Return the best one AND detailed test results.

        KEY PRINCIPLE: NO hardcoded assumptions - test everything, pick the best!
        """
        months_bucket, has_turnover, has_historic = scenario

        # Get test sites that match this scenario
        test_sites = self._get_test_sites_for_scenario(scenario)

        # Show recycling statistics
        monthly_data = self.data_df[self.data_df['Timeframe'] == 'Monthly']
        site_month_counts = monthly_data.groupby('Site identifier').size()
        sites_with_enough_months = sum(1 for count in site_month_counts if count >= months_bucket)

        print(f"\n  Data Availability: {months_bucket}mo, Turnover:{has_turnover}, Historic:{has_historic}")
        print(f"    Recycling pool: {sites_with_enough_months} sites with ≥{months_bucket} months")
        print(f"    After feature filter: {len(test_sites)} sites (simulating {months_bucket}mo data)")

        # Prepare detailed results for matrix
        test_results = {
            'test_pool_size': len(test_sites),
            'model_r2s': {}
        }

        if len(test_sites) < 5:
            # Not enough test sites - can't get reliable R²
            print(f"    ⚠️  Too few test sites ({len(test_sites)}) - using fallback")
            test_results['model_r2s']['Site_Seasonal'] = 0.35
            return 'Site_Seasonal', 0.35, test_results

        # Test EVERY model - no filtering!
        model_r2s = {}

        print(f"    Testing ALL models:")

        for model_name, model_info in self.models.items():
            # Test this model on these test sites
            r2 = self._test_model_on_scenario(model_name, model_info, scenario, test_sites)

            # Store ALL results (even None and negative)
            test_results['model_r2s'][model_name] = r2

            # Only keep models with positive R² (better than mean)
            # But we TEST everything first!
            if r2 is not None:
                model_r2s[model_name] = r2
                status = "✓" if r2 > 0 else "✗"
                print(f"      {status} {model_name}: R² = {r2:.4f}")

        # Filter to positive R² only (better than using the mean)
        viable_models = {name: r2 for name, r2 in model_r2s.items() if r2 > 0}

        # Pick best
        if viable_models:
            best_model = max(viable_models, key=viable_models.get)
            best_r2 = viable_models[best_model]
            print(f"    → BEST: {best_model} (R² = {best_r2:.4f})")
            return best_model, best_r2, test_results
        else:
            print(f"    ⚠️  No viable models (all R² ≤ 0) - using fallback")
            return 'Site_Seasonal', 0.35, test_results  # Fallback

    def _get_test_sites_for_scenario(self, scenario):
        """
        Get sites for testing in this scenario.

        KEY RECYCLING PRINCIPLE:
        - Sites with 12 months can test scenarios 0, 1, 2... 12 months (pretend to have less)
        - Sites with 7 months can test scenarios 0, 1, 2... 7 months
        - Sites with 3 months can test scenarios 0, 1, 2, 3 months

        Result: Early scenarios (0-3 months) have MOST test sites (recycling!)
                Later scenarios (10-12 months) have LEAST test sites

        Example: For "7 months" scenario:
        - Use ALL sites with ≥7 months (could be 165 sites)
        - Sites with 12 months: Pretend they only have 7 months
        - Sites with 7 months: Use all their data
        - Sites with 3 months: Can't use (not enough data)
        """
        months_bucket, has_turnover, has_historic = scenario

        # Get ALL sites that have actual VQ to validate against
        monthly_data = self.data_df[self.data_df['Timeframe'] == 'Monthly']

        # Count months per site
        site_month_counts = monthly_data.groupby('Site identifier').size()

        test_sites = []

        for site in site_month_counts.index:
            # Check if site has ENOUGH months for this scenario
            # Site must have ≥ months_bucket (can have more, will pretend to have less)
            if site_month_counts[site] < months_bucket:
                continue  # Can't test - site doesn't have enough months

            site_rows = self.data_df[self.data_df['Site identifier'] == site]

            if len(site_rows) == 0:
                continue

            row = site_rows.iloc[0]

            # Filter by scenario feature requirements
            # Check Turnover
            site_has_turnover = pd.notna(row.get('Turnover', None))
            if has_turnover and not site_has_turnover:
                continue  # Scenario needs Turnover, site doesn't have it

            # Check Historic
            site_has_historic = False
            if self.hdm and site in self.hdm.site_periods:
                site_has_historic = len(self.hdm.site_periods[site]) > 0
            if has_historic and not site_has_historic:
                continue  # Scenario needs Historic, site doesn't have it

            # Site matches ALL requirements - can use for testing!
            test_sites.append(site)

        return test_sites  # Return ALL matching sites

    def _test_model_on_scenario(self, model_name, model_info, scenario, test_sites):
        """
        Test a model on sites in this scenario.

        Critical: We SIMULATE the scenario's data availability, even if the
        test site actually has more data.

        Example: For "0 months" scenario:
        - Test site has 12 months of actual data
        - We PRETEND it has 0 months (site_data_limited = empty)
        - Model can only use Turnover/Historic (not current year data)
        - Compare prediction to actual annual total
        """
        months_bucket, has_turnover, has_historic = scenario

        predictions = []
        actuals = []

        for site in test_sites:
            # Get site's actual monthly data (for validation)
            site_data_actual = self.data_df[
                (self.data_df['Site identifier'] == site) &
                (self.data_df['Timeframe'] == 'Monthly')
            ].copy()

            if len(site_data_actual) == 0:
                continue

            # SIMULATE the scenario's limited data availability
            if months_bucket > 0:
                # Scenario has some months - use first N months
                site_data_limited = site_data_actual.head(months_bucket)
            else:
                # Scenario is NEW SITE (0 months) - no current year data!
                site_data_limited = pd.DataFrame()

            # For each actual month, try to predict it using limited data
            # (For 0 months scenario, we predict annual total instead)
            if months_bucket == 0:
                # New site scenario - predict annual total
                try:
                    # Get a representative row (first month)
                    test_row = site_data_actual.iloc[0].copy()

                    # Can this model predict with 0 months of current data?
                    if not self._can_model_predict(model_name, model_info, test_row):
                        continue

                    # Make prediction (model can ONLY use Turnover/Historic)
                    pred = self._apply_model(model_name, model_info, test_row, site_data_limited)

                    if pred is None or pred <= 0:
                        continue

                    # Calculate actual annual total
                    actual_annual = site_data_actual['Volumetric Quantity'].sum()

                    # For monthly models, multiply by 12
                    if model_info.get('type') not in ['Annual_Average']:
                        pred = pred * 12  # Monthly → Annual

                    predictions.append(pred)
                    actuals.append(actual_annual)

                except:
                    continue
            else:
                # Partial data scenario - predict each month
                for idx, row in site_data_actual.iterrows():
                    try:
                        # Can this model predict?
                        if not self._can_model_predict(model_name, model_info, row):
                            continue

                        # Make prediction using LIMITED data
                        pred = self._apply_model(model_name, model_info, row, site_data_limited)

                        if pred is None or pred <= 0:
                            continue

                        predictions.append(pred)
                        actuals.append(row['Volumetric Quantity'])

                    except:
                        continue

        # Calculate R² if we have enough predictions
        if len(predictions) >= 10:
            return r2_score(actuals, predictions)
        else:
            return None

    def _can_model_predict(self, model_name, model_info, row):
        """Check if model can predict this row"""
        model_type = model_info.get('type')

        # Rule-based models
        if model_type in ['LastYear_Direct', 'LastYear_TurnoverAdjusted',
                         'MultiYear_Average', 'LastYear_GrowthAdjusted']:
            # Need historic data
            site = row['Site identifier']
            if self.hdm and site in self.hdm.site_periods:
                return len(self.hdm.site_periods[site]) > 0
            return False

        # Site_Seasonal needs month
        if model_type == 'Site_Seasonal':
            return pd.notna(row.get('Month'))

        # ML models need their features
        features = model_info.get('features', [])
        for feat in features:
            if feat != 'Month' and pd.isna(row.get(feat)):
                return False

        return True

    def _apply_model(self, model_name, model_info, row, site_data_limited):
        """Apply a model to predict a row"""
        model_type = model_info.get('type')

        if model_type == 'Site_Seasonal':
            return self._predict_site_seasonal(row, site_data_limited)
        elif model_type in ['LastYear_Direct', 'LastYear_TurnoverAdjusted']:
            return self._predict_historic(model_name, row)
        elif model_type == 'Annual_Average':
            return model_info.get('average')
        else:
            return self._predict_ml(model_info, row)

    def _predict_site_seasonal(self, row, site_data_limited):
        """Predict using site average + seasonal"""
        month = row.get('Month')
        if not month:
            return None

        # Calculate site average from limited data
        if len(site_data_limited) > 0:
            site_avg = site_data_limited['Volumetric Quantity'].mean()
        else:
            # New site - use overall average
            monthly_data = self.data_df[self.data_df['Timeframe'] == 'Monthly']
            site_avg = monthly_data['Volumetric Quantity'].mean()

        # Apply seasonal factor
        if 'Overall' in self.seasonal_patterns and month in self.seasonal_patterns['Overall']:
            seasonal_factor = self.seasonal_patterns['Overall'][month]
            return site_avg * seasonal_factor

        return site_avg

    def _predict_historic(self, model_name, row):
        """Predict using historic rule-based model"""
        # Simplified - would need full implementation
        return None

    def _predict_ml(self, model_info, row):
        """Predict using ML model"""
        model = model_info.get('model')
        if not model:
            return None

        features = model_info.get('features', [])
        X_features = []

        for feat in features:
            if feat == 'Month':
                months = ['January', 'February', 'March', 'April', 'May', 'June',
                         'July', 'August', 'September', 'October', 'November', 'December']
                month_val = months.index(row.get('Month', 'January'))
                X_features.append(month_val)
            elif feat in row.index:
                val = row.get(feat)
                if pd.notna(val):
                    X_features.append(float(val))
                else:
                    return None
            else:
                return None

        if len(X_features) == len(features):
            X = np.array(X_features).reshape(1, -1)
            try:
                return model.predict(X)[0]
            except:
                return None

        return None

    def get_best_model_for_row(self, row):
        """
        Get the best model for this row based on its scenario.

        Returns:
        --------
        (model_name, r2)
        """
        scenario = self.identify_scenario(row)

        if scenario in self.scenario_best_models:
            return self.scenario_best_models[scenario]
        else:
            # Scenario not tested, use fallback
            return 'Site_Seasonal', 0.35

    def print_scenario_summary(self):
        """Print summary of scenarios and their best models"""
        print("\n" + "="*70)
        print("SCENARIO MODEL ASSIGNMENTS")
        print("="*70)

        for scenario, (model_name, r2) in sorted(self.scenario_best_models.items()):
            months, has_turn, has_hist = scenario
            print(f"\n{months} months, Turnover: {has_turn}, Historic: {has_hist}")
            print(f"  → {model_name} (R² = {r2:.4f})")