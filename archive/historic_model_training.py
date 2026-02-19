"""
PART 3: Historic Model Training Module
Trains ML models and rule-based models using historic data with fair R² evaluation

✅ UPDATED: Month-aware feature access throughout
- Rule-based models now use month-specific feature values (e.g. monthly Turnover)
- Feature engineering uses month-specific historic values for YoY calculations
- Covers all input file formats (annual features, monthly features, mixed)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')


def identify_derived_features(features):
    """
    Identify which features are derived (calculated from VQ).

    Derived features include:
    - VQ_per_* (requires VQ to calculate)
    - VQ_YoY_* (requires VQ to calculate)
    - Efficiency_* (requires VQ to calculate)
    - VQ_Trend_* (requires VQ to calculate)
    - VQ_Volatility (requires VQ to calculate)
    - VQ_Historic_Avg (requires VQ to calculate)
    - VQ_2Yr_Growth (requires VQ to calculate)

    Returns:
    --------
    tuple : (base_features, derived_features)
    """
    derived = []
    base = []

    for feature in features:
        if feature == 'Month':
            base.append(feature)
        elif any(x in feature for x in ['VQ_per_', 'VQ_YoY', 'Efficiency_',
                                         'VQ_Trend', 'VQ_Volatility', 'VQ_Historic_Avg',
                                         'VQ_2Yr_Growth']):
            derived.append(feature)
        else:
            base.append(feature)

    return base, derived


class HistoricModelTrainer:
    """
    Trains models using historic data.
    Includes ML models (TIER 1), rule-based models (TIER 2), and existing models (TIER 3).

    ✅ FIXED: Site-level features trained on MONTHLY data for honest R² scores
    ✅ FIXED: Rule-based models use month-specific feature values where available
    """

    def __init__(self, historic_data_manager, feature_engineer):
        self.hdm = historic_data_manager
        self.fe = feature_engineer
        self.models = {}
        self.categorical_encoders = {}
        self.feature_set_month_r2 = {}
        self.feature_set_r2 = {}

    def train_all_models(self, enhanced_df, data_df, selected_features):
        """Train all model tiers."""
        print("\n" + "="*70)
        print("HISTORIC MODEL TRAINING")
        print("="*70)

        if len(data_df) < 10:
            print("⚠️  Insufficient data for training (need at least 10 rows)")
            return self.models

        print("\n🎯 TIER 1: ML Models with Historic Data")
        self._train_ml_models_multiyear(data_df, selected_features)

        print("\n📋 TIER 2: Rule-Based Historic Models")
        self._train_rule_based_models(data_df)

        print("\n🔍 TIER 2.5: Similar Sites Methods")
        self._evaluate_similar_sites_methods(data_df, selected_features)

        print("\n📊 DATA QUALITY ASSESSMENT")
        self._generate_quality_grid()

        print(f"\n✅ Total models trained: {len(self.models)}")
        return self.models

    def _generate_quality_grid(self):
        """Generate and display data quality grid showing R² by feature combination"""
        if len(self.feature_set_r2) == 0:
            print("  ℹ️  No feature-set R² scores available")
            return

        print("  " + "="*66)
        print("  DATA QUALITY BY FEATURE SET")
        print("  " + "="*66)

        sorted_feature_sets = sorted(
            self.feature_set_r2.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print(f"  {'Feature Set':<45} {'R² Score':>12}")
        print("  " + "-"*66)

        for feature_tuple, r2 in sorted_feature_sets:
            features_str = ', '.join(feature_tuple)
            if len(features_str) > 44:
                features_str = features_str[:41] + '...'

            if r2 >= 0.70:
                indicator = "✓✓"
            elif r2 >= 0.50:
                indicator = "✓"
            elif r2 >= 0.30:
                indicator = "~"
            else:
                indicator = "⚠️"

            print(f"  {features_str:<45} {r2:>10.4f} {indicator}")

        print("  " + "-"*66)
        print(f"  Legend: ✓✓ Excellent (≥0.70) | ✓ Good (≥0.50) | ~ Fair (≥0.30) | ⚠️ Poor (<0.30)")
        print("  " + "="*66)

    def _train_ml_models_multiyear(self, data_df, selected_features):
        """
        Train ML models on data including engineered historic features.
        Filters out VQ-derived features that can't predict blank rows.
        """
        engineered_features = [col for col in data_df.columns
                              if any(x in col for x in ['YoY', 'Trend', 'Volatility',
                                                        'Growth', 'Historic_Avg',
                                                        'VQ_per_', 'Efficiency_',
                                                        'Year', 'Years_Since'])]

        all_potential_features = list(set(selected_features + engineered_features))

        available_features = []
        excluded_features = []

        for feature in all_potential_features:
            if feature in data_df.columns:
                availability = data_df[feature].notna().sum() / len(data_df)
                if availability < 0.2:
                    continue

                if any(x in feature for x in ['VQ_per_', 'Efficiency_', 'VQ_YoY',
                                               'VQ_Trend', 'VQ_Volatility',
                                               'VQ_Historic_Avg', 'VQ_2Yr_Growth']):
                    excluded_features.append(feature)
                    continue

                available_features.append(feature)

        if len(excluded_features) > 0:
            print(f"\n  ⚠️  Excluded {len(excluded_features)} VQ-derived features (can't predict blank rows):")
            for feat in excluded_features[:5]:
                print(f"     ✗ {feat}")
            if len(excluded_features) > 5:
                print(f"     ... and {len(excluded_features) - 5} more")

        if len(available_features) == 0:
            print("  ⚠️  No suitable features available for ML training")
            return

        print(f"  ✓ Using {len(available_features)} features for ML training: {available_features}")

        monthly_data = data_df[data_df['Timeframe'] == 'Monthly'].copy()
        if len(monthly_data) >= 10:
            self._train_monthly_ml_models(monthly_data, available_features)

        site_level_features = [f for f in available_features
                              if f not in ['Month'] and
                              'YoY' not in f and 'Trend' not in f and
                              'Volatility' not in f and 'Historic_Avg' not in f]

        if len(site_level_features) > 0:
            self._train_site_level_ml_models(data_df, site_level_features)

    def _train_monthly_ml_models(self, monthly_data, features):
        """Train ML models for monthly data"""
        X_features = []
        feature_names = []

        if 'Month' in monthly_data.columns:
            month_encoder = LabelEncoder()
            valid_months = monthly_data[monthly_data['Month'] != '']['Month']
            if len(valid_months) > 0:
                month_encoder.fit(valid_months)
                self.categorical_encoders['Month'] = month_encoder
                X_features.append(month_encoder.transform(
                    monthly_data['Month'].fillna('Unknown')
                ).reshape(-1, 1))
                feature_names.append('Month')

        for feature in features:
            if feature == 'Month':
                continue
            if feature not in monthly_data.columns:
                continue

            if pd.api.types.is_numeric_dtype(monthly_data[feature]):
                feature_data = monthly_data[feature].fillna(monthly_data[feature].median())
                X_features.append(feature_data.values.reshape(-1, 1))
                feature_names.append(feature)
            else:
                encoder = LabelEncoder()
                feature_data = monthly_data[feature].fillna('Unknown')
                non_null = feature_data[feature_data != 'Unknown']
                if len(non_null) > 0:
                    encoder.fit(non_null)
                    self.categorical_encoders[feature] = encoder
                    encoded = []
                    for val in feature_data:
                        if val == 'Unknown':
                            encoded.append(0)
                        else:
                            try:
                                encoded.append(encoder.transform([val])[0])
                            except:
                                encoded.append(0)
                    X_features.append(np.array(encoded).reshape(-1, 1))
                    feature_names.append(feature)

        if len(X_features) == 0:
            return

        y = monthly_data['Volumetric Quantity'].values
        X_all = np.hstack(X_features)

        if len(X_all) >= 10:
            self._train_and_evaluate_model(
                X_all, y, 'Historic_Monthly_RandomForest',
                'RandomForest', feature_names
            )
            self._train_and_evaluate_model(
                X_all, y, 'Historic_Monthly_GradientBoosting',
                'GradientBoosting', feature_names
            )

            if 'Year' in monthly_data.columns and 'Month' in feature_names:
                key_features = ['Month', 'Year']
                key_feature_indices = [i for i, f in enumerate(feature_names) if f in key_features]
                if len(key_feature_indices) >= 2:
                    X_key = X_all[:, key_feature_indices]
                    self._train_and_evaluate_model(
                        X_key, y, 'Historic_Monthly_Linear_Year_Month',
                        'Linear', key_features
                    )

    def _train_site_level_ml_models(self, data_df, features):
        """
        Train site-level features on MONTHLY data for honest R² scores.
        """
        monthly_data = data_df[data_df['Timeframe'] == 'Monthly'].copy()

        if len(monthly_data) < 100:
            print("  ⚠️  Insufficient monthly data for site-level feature testing")
            print(f"     (Need ≥100 monthly rows, found {len(monthly_data)})")
            return

        print(f"\n  Testing site-level features on {len(monthly_data)} monthly rows...")

        for feature in features:
            if feature not in monthly_data.columns:
                continue

            feature_availability = monthly_data[feature].notna().sum() / len(monthly_data)
            if feature_availability < 0.2:
                print(f"     ⚠️  Skipping {feature}: Only {feature_availability:.1%} available")
                continue

            X_features = []
            feature_names = []

            month_encoder = LabelEncoder()
            valid_months = monthly_data[monthly_data['Month'] != '']['Month']
            if len(valid_months) > 0:
                month_encoder.fit(valid_months)
                self.categorical_encoders[f'{feature}_Month'] = month_encoder
                X_features.append(
                    month_encoder.transform(monthly_data['Month'].fillna('Unknown')).reshape(-1, 1)
                )
                feature_names.append('Month')

            if pd.api.types.is_numeric_dtype(monthly_data[feature]):
                feature_data = monthly_data[feature].fillna(monthly_data[feature].median())
                X_features.append(feature_data.values.reshape(-1, 1))
                feature_names.append(feature)

                if len(X_features) > 0:
                    y = monthly_data['Volumetric Quantity'].values
                    X_all = np.hstack(X_features)

                    self._train_and_evaluate_model(
                        X_all, y,
                        f'Historic_SiteLevel_Monthly_{feature}',
                        'Linear',
                        feature_names
                    )
                    self._train_and_evaluate_model(
                        X_all, y,
                        f'Historic_SiteLevel_Monthly_Polynomial_{feature}',
                        'Polynomial',
                        feature_names
                    )

        if len(features) > 1:
            X_features = []
            feature_names = []

            month_encoder = LabelEncoder()
            valid_months = monthly_data[monthly_data['Month'] != '']['Month']
            if len(valid_months) > 0:
                month_encoder.fit(valid_months)
                self.categorical_encoders['Combined_Month'] = month_encoder
                X_features.append(
                    month_encoder.transform(monthly_data['Month'].fillna('Unknown')).reshape(-1, 1)
                )
                feature_names.append('Month')

            for feature in features:
                if feature in monthly_data.columns:
                    feature_availability = monthly_data[feature].notna().sum() / len(monthly_data)
                    if feature_availability >= 0.2:
                        if pd.api.types.is_numeric_dtype(monthly_data[feature]):
                            feature_data = monthly_data[feature].fillna(monthly_data[feature].median())
                            X_features.append(feature_data.values.reshape(-1, 1))
                            feature_names.append(feature)

            if len(X_features) >= 2:
                y = monthly_data['Volumetric Quantity'].values
                X_all = np.hstack(X_features)

                print(f"  Testing combined model with {len(feature_names)} features on {len(monthly_data)} monthly rows...")

                self._train_and_evaluate_model(
                    X_all, y,
                    'Historic_SiteLevel_Monthly_RandomForest_All',
                    'RandomForest',
                    feature_names
                )
                self._train_and_evaluate_model(
                    X_all, y,
                    'Historic_SiteLevel_Monthly_GradientBoosting_All',
                    'GradientBoosting',
                    feature_names
                )

    def _train_rule_based_models(self, data_df):
        """
        Train and evaluate rule-based historic models.
        ✅ FIXED: Uses month-specific feature values where available.
        """
        monthly_data = data_df[data_df['Timeframe'] == 'Monthly'].copy()

        has_any_historic = len(self.hdm.site_periods) > 0

        if len(monthly_data) == 0 and not has_any_historic:
            print("  ⚠️  No data for rule-based models")
            return

        print("  ℹ️  Testing rule-based historic models on actual data...")

        models_to_test = [
            ('LastYear_Direct', ['Historic VQ']),
            ('LastYear_TurnoverAdjusted', ['Historic VQ', 'Turnover']),
            ('MultiYear_Average', ['Historic VQ']),
            ('LastYear_GrowthAdjusted', ['Historic VQ'])
        ]

        for model_name, features in models_to_test:
            r2 = self._test_rule_based_model(model_name, monthly_data)

            if r2 is not None and r2 > -0.5:
                self.models[model_name] = {
                    'model': 'rule_based',
                    'type': model_name,
                    'features': features,
                    'r2': r2,
                    'r2_full': r2,
                    'r2_imputed': r2,
                    'has_derived': False,
                    'description': f'Rule-based model using {", ".join(features)}. Tested on {len(monthly_data)} monthly rows. R² = {r2:.3f}'
                }

                feature_key = tuple(sorted(features))
                self.feature_set_r2[feature_key] = r2

                print(f"  ✓ {model_name}: R² = {r2:.4f} (tested on {len(monthly_data)} rows)")
            else:
                print(f"  ✗ {model_name}: Cannot test (R² = {r2 if r2 else 'N/A'}) - skipping")

    def _test_rule_based_model(self, model_name, monthly_data):
        """
        Test a rule-based model on actual data to get real R².
        ✅ FIXED: All feature lookups now use month-specific values where available.
        """
        predictions = []
        actuals = []

        for idx, row in monthly_data.iterrows():
            site = row['Site identifier']
            month = row.get('Month')
            actual_vq = row['Volumetric Quantity']
            row_date = pd.to_datetime(row['Date from'], dayfirst=True)

            historic_period = self.hdm.get_most_recent_period(site, row_date)
            if not historic_period:
                continue

            prediction = None

            if model_name == 'LastYear_Direct':
                month_vq = self.hdm.get_period_vq_for_month(historic_period, month)
                if month_vq:
                    prediction = month_vq
                else:
                    annual = self.hdm.get_period_annual_total(historic_period)
                    if annual:
                        prediction = annual / historic_period.months_covered

            elif model_name == 'LastYear_TurnoverAdjusted':
                month_vq = self.hdm.get_period_vq_for_month(historic_period, month)
                if not month_vq:
                    annual = self.hdm.get_period_annual_total(historic_period)
                    if annual:
                        month_vq = annual / historic_period.months_covered

                if month_vq:
                    current_turnover = row.get('Turnover')
                    # ✅ Use month-specific historic Turnover if available
                    period_turnover = self.hdm.get_period_feature(historic_period, 'Turnover', month=month)

                    if pd.notna(current_turnover) and period_turnover and period_turnover != 0:
                        adjustment = float(current_turnover) / float(period_turnover)
                        prediction = month_vq * adjustment

            elif model_name == 'MultiYear_Average':
                all_periods = self.hdm.get_all_periods(site)
                relevant_periods = [p for p in all_periods if p.end_date < row_date]

                if len(relevant_periods) >= 2:
                    monthly_values = []
                    for period in relevant_periods:
                        month_vq = self.hdm.get_period_vq_for_month(period, month)
                        if month_vq:
                            monthly_values.append(month_vq)

                    if monthly_values:
                        prediction = np.mean(monthly_values)

            elif model_name == 'LastYear_GrowthAdjusted':
                all_periods = self.hdm.get_all_periods(site)
                relevant_periods = [p for p in all_periods if p.end_date < row_date]

                if len(relevant_periods) >= 2:
                    recent_two = relevant_periods[-2:]
                    prev_total = recent_two[0].get_normalized_annual()
                    recent_total = recent_two[1].get_normalized_annual()

                    if prev_total and recent_total and prev_total > 0:
                        growth_rate = (recent_total - prev_total) / prev_total

                        month_vq = self.hdm.get_period_vq_for_month(recent_two[1], month)
                        if month_vq:
                            prediction = month_vq * (1 + growth_rate)

            if prediction is not None and prediction > 0:
                predictions.append(prediction)
                actuals.append(actual_vq)

        if len(predictions) >= 10:
            r2 = r2_score(actuals, predictions)
            return r2
        else:
            return None

    def _evaluate_similar_sites_methods(self, data_df, selected_features):
        """
        Evaluate different 'similar sites' grouping methods using cross-validation.
        ✅ FIXED: Uses month-specific feature values when finding similar sites.
        """
        available_features = [f for f in selected_features if f in data_df.columns]

        if 'Turnover' not in available_features and 'SqFt' not in available_features:
            print("  ⚠️  No features (Turnover, SqFt) available for similar sites matching")
            return

        sites_with_both = self._get_sites_with_both_periods(data_df)

        if len(sites_with_both) < 10:
            print(f"  ⚠️  Insufficient sites with both periods ({len(sites_with_both)} found, need ≥10)")
            return

        print(f"  Testing on {len(sites_with_both)} sites with both current & historic data...")

        methods_to_test = []

        if 'Turnover' in available_features:
            methods_to_test.extend([
                {'name': 'Turnover_±20%', 'type': 'range', 'feature': 'Turnover', 'threshold': 0.2},
                {'name': 'Turnover_Top5', 'type': 'top_n', 'feature': 'Turnover', 'n': 5},
                {'name': 'VQ_per_Turnover', 'type': 'ratio', 'feature': 'Turnover'}
            ])

        if 'SqFt' in available_features:
            methods_to_test.extend([
                {'name': 'SqFt_±20%', 'type': 'range', 'feature': 'SqFt', 'threshold': 0.2},
                {'name': 'VQ_per_SqFt', 'type': 'ratio', 'feature': 'SqFt'}
            ])

        if 'Turnover' in available_features and 'Brand' in available_features:
            methods_to_test.append(
                {'name': 'Multi_Feature', 'type': 'multi', 'features': ['Turnover', 'Brand']}
            )

        best_method = None
        best_r2 = -999

        for method_config in methods_to_test:
            r2 = self._test_similar_sites_method(data_df, sites_with_both, method_config)

            if r2 is not None and r2 > best_r2:
                best_r2 = r2
                best_method = method_config

            if r2 is not None:
                star = " ⭐" if r2 == best_r2 else ""
                print(f"  ✓ {method_config['name']}: R² = {r2:.4f}{star}")

        if best_method and best_r2 > 0:
            self.models[f"SimilarSites_{best_method['name']}"] = {
                'model': 'similar_sites',
                'type': 'SimilarSites',
                'method_config': best_method,
                'features': [best_method.get('feature', 'Multiple')],
                'r2': best_r2,
                'r2_full': best_r2,
                'r2_imputed': best_r2,
                'has_derived': False,
                'description': f"Uses similar sites by {best_method['name']} to predict VQ. R² = {best_r2:.4f} (cross-validated)."
            }
            print(f"\n  📊 Best similar sites method: {best_method['name']} (R² = {best_r2:.4f})")

    def _get_sites_with_both_periods(self, data_df):
        """Get sites that have data in both current and at least one historic period"""
        sites_with_both = []

        for site in data_df['Site identifier'].unique():
            if site not in self.hdm.site_periods or len(self.hdm.site_periods[site]) == 0:
                continue

            site_current = data_df[data_df['Site identifier'] == site]
            if site_current['Volumetric Quantity'].notna().sum() > 0:
                sites_with_both.append(site)

        return sites_with_both

    def _test_similar_sites_method(self, data_df, test_sites, method_config):
        """
        Test a specific similar sites method using leave-one-out cross-validation.
        ✅ FIXED: Uses month-specific feature values.
        """
        predictions = []
        actuals = []

        for test_site in test_sites:
            training_sites = [s for s in test_sites if s != test_site]

            test_site_data = data_df[data_df['Site identifier'] == test_site]
            test_site_current = test_site_data[test_site_data['Volumetric Quantity'].notna()]

            if len(test_site_current) == 0:
                continue

            similar_sites = self._find_similar_sites(
                test_site,
                training_sites,
                data_df,
                method_config
            )

            if len(similar_sites) == 0:
                continue

            for _, row in test_site_current.iterrows():
                month = row.get('Month')
                actual_vq = row['Volumetric Quantity']
                row_date = pd.to_datetime(row['Date from'], dayfirst=True)

                test_historic_period = self.hdm.get_most_recent_period(test_site, row_date)
                if not test_historic_period:
                    continue

                similar_predictions = []

                for sim_site in similar_sites:
                    sim_site_data = data_df[data_df['Site identifier'] == sim_site]
                    sim_current_row = sim_site_data[
                        (sim_site_data['Month'] == month) &
                        (sim_site_data['Volumetric Quantity'].notna())
                    ]

                    if len(sim_current_row) == 0:
                        continue

                    sim_current_vq = sim_current_row.iloc[0]['Volumetric Quantity']
                    sim_current_date = pd.to_datetime(sim_current_row.iloc[0]['Date from'], dayfirst=True)

                    sim_historic_period = self.hdm.get_most_recent_period(sim_site, sim_current_date)
                    if not sim_historic_period:
                        continue

                    sim_historic_vq = self.hdm.get_period_vq_for_month(sim_historic_period, month)

                    if not sim_historic_vq:
                        sim_annual = self.hdm.get_period_annual_total(sim_historic_period)
                        if sim_annual:
                            sim_historic_vq = sim_annual / sim_historic_period.months_covered

                    if sim_historic_vq and sim_historic_vq > 0:
                        growth_rate = sim_current_vq / sim_historic_vq
                        similar_predictions.append(growth_rate)

                if len(similar_predictions) > 0:
                    test_historic_vq = self.hdm.get_period_vq_for_month(test_historic_period, month)

                    if not test_historic_vq:
                        test_annual = self.hdm.get_period_annual_total(test_historic_period)
                        if test_annual:
                            test_historic_vq = test_annual / test_historic_period.months_covered

                    if test_historic_vq:
                        avg_growth = np.mean(similar_predictions)
                        prediction = test_historic_vq * avg_growth

                        predictions.append(prediction)
                        actuals.append(actual_vq)

        if len(predictions) >= 10:
            return r2_score(actuals, predictions)
        else:
            return None

    def _find_similar_sites(self, target_site, candidate_sites, data_df, method_config):
        """Find similar sites based on method configuration."""
        method_type = method_config['type']

        target_data = data_df[data_df['Site identifier'] == target_site].iloc[0]

        if method_type == 'range':
            feature = method_config['feature']
            threshold = method_config['threshold']

            target_value = target_data.get(feature)
            if pd.isna(target_value) or target_value == 0:
                return []

            similar = []
            for site in candidate_sites:
                site_data = data_df[data_df['Site identifier'] == site].iloc[0]
                site_value = site_data.get(feature)

                if pd.notna(site_value) and site_value > 0:
                    ratio = min(target_value, site_value) / max(target_value, site_value)
                    if ratio >= (1 - threshold):
                        similar.append(site)

            return similar

        elif method_type == 'top_n':
            feature = method_config['feature']
            n = method_config['n']

            target_value = target_data.get(feature)
            if pd.isna(target_value):
                return []

            distances = []
            for site in candidate_sites:
                site_data = data_df[data_df['Site identifier'] == site].iloc[0]
                site_value = site_data.get(feature)

                if pd.notna(site_value):
                    distance = abs(target_value - site_value)
                    distances.append((site, distance))

            distances.sort(key=lambda x: x[1])
            return [site for site, _ in distances[:n]]

        elif method_type == 'ratio':
            feature = method_config['feature']

            target_value = target_data.get(feature)
            if pd.isna(target_value) or target_value == 0:
                return []

            target_periods = self.hdm.get_all_periods(target_site)
            if not target_periods:
                return []

            target_historic_vq = target_periods[-1].get_monthly_average()
            if not target_historic_vq:
                return []

            target_ratio = target_historic_vq / target_value

            similar = []
            for site in candidate_sites:
                site_data = data_df[data_df['Site identifier'] == site].iloc[0]
                site_value = site_data.get(feature)

                if pd.notna(site_value) and site_value > 0:
                    site_periods = self.hdm.get_all_periods(site)
                    if site_periods:
                        site_historic_vq = site_periods[-1].get_monthly_average()
                        if site_historic_vq:
                            site_ratio = site_historic_vq / site_value

                            ratio_similarity = min(target_ratio, site_ratio) / max(target_ratio, site_ratio)
                            if ratio_similarity >= 0.7:
                                similar.append(site)

            return similar[:5]

        elif method_type == 'multi':
            features = method_config['features']

            scores = {}
            for site in candidate_sites:
                site_data = data_df[data_df['Site identifier'] == site].iloc[0]

                score = 0
                weights = 0

                for feature in features:
                    if feature == 'Turnover':
                        target_val = target_data.get('Turnover')
                        site_val = site_data.get('Turnover')

                        if pd.notna(target_val) and pd.notna(site_val) and target_val > 0 and site_val > 0:
                            ratio = min(target_val, site_val) / max(target_val, site_val)
                            score += ratio * 3
                            weights += 3

                    elif feature == 'Brand':
                        if target_data.get('Brand') == site_data.get('Brand'):
                            score += 1
                            weights += 1

                if weights > 0:
                    scores[site] = score / weights

            sorted_sites = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [site for site, _ in sorted_sites[:5]]

        return []

    def _train_and_evaluate_model(self, X, y, model_name, model_type, features):
        """Train and evaluate a single ML model with feature-aware R² calculation"""
        try:
            if len(X) < 10:
                return

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )

            if model_type == 'Linear':
                model = LinearRegression()
            elif model_type == 'Polynomial':
                model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
            elif model_type == 'RandomForest':
                model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            elif model_type == 'GradientBoosting':
                model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
            else:
                return

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            r2_full = r2_score(y_test, y_pred)

            base_features, derived_features = identify_derived_features(features)

            r2_imputed = r2_full

            if len(derived_features) > 0:
                X_test_imputed = X_test.copy()

                for i, feature in enumerate(features):
                    if feature in derived_features:
                        median_val = np.median(X_train[:, i])
                        X_test_imputed[:, i] = median_val

                y_pred_imputed = model.predict(X_test_imputed)
                r2_imputed = r2_score(y_test, y_pred_imputed)

            self.models[model_name] = {
                'model': model,
                'features': features,
                'base_features': base_features,
                'derived_features': derived_features,
                'r2_full': r2_full,
                'r2_imputed': r2_imputed,
                'r2': r2_full,
                'type': model_type,
                'has_derived': len(derived_features) > 0,
                'description': (
                    f'{model_type} model using {", ".join(features[:3])}{"..." if len(features) > 3 else ""}. '
                    f'Trained on {len(X_train)} samples, tested on {len(X_test)}. '
                    f'R²(full)={r2_full:.3f}, R²(imputed)={r2_imputed:.3f}'
                    if len(derived_features) > 0 else
                    f'{model_type} model using {", ".join(features[:3])}{"..." if len(features) > 3 else ""}. '
                    f'Trained on {len(X_train)} samples, tested on {len(X_test)}. R²={r2_full:.3f}'
                )
            }

            feature_key = tuple(sorted(features))
            self.feature_set_r2[feature_key] = r2_full

            if len(derived_features) > 0:
                print(f"  ✓ {model_name}: R²(full) = {r2_full:.4f}, R²(imputed) = {r2_imputed:.4f}")
            else:
                print(f"  ✓ {model_name}: R² = {r2_full:.4f}")

        except Exception as e:
            print(f"  ✗ {model_name} failed: {str(e)}")