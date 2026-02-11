"""
Context-Aware Model Selection System
=====================================
Tests models in the SPECIFIC context of each blank row to get honest R² scores.

Key Concept:
- Different rows have different data availability (6 months vs 12 months, historic vs not, etc.)
- Models should be tested in the SAME context they'll be used
- Each row gets a custom R² based on what data IT has available

Example:
- Row A has 6 months of site data + Turnover
- Test models on sites with full data, but PRETEND they only have 6 months
- This gives honest R² for "predicting with 6 months of data"
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
from sklearn.metrics import r2_score


@dataclass
class RowContext:
    """Defines what data is available for a specific blank row"""

    # Basic info
    site_id: str
    month: Optional[str]
    timeframe: str

    # Feature availability
    features_available: Set[str]  # e.g., {'Turnover', 'Electricity Annual'}

    # Site's current year data
    site_months_available: int  # How many months of 2025 data does this site have?
    site_has_any_current_data: bool

    # Historic data
    has_historic_data: bool
    historic_features: Set[str]  # e.g., {'Historic VQ', 'Turnover'}
    historic_periods_count: int

    # Derived attributes
    brand: Optional[str] = None

    def __hash__(self):
        """Make context hashable for caching"""
        return hash((
            self.timeframe,
            frozenset(self.features_available),
            self.site_months_available,
            self.has_historic_data,
            frozenset(self.historic_features)
        ))

    def get_signature(self):
        """Get a unique signature for this context (for caching R² scores)"""
        return (
            self.timeframe,
            tuple(sorted(self.features_available)),
            self.site_months_available,
            self.has_historic_data,
            tuple(sorted(self.historic_features))
        )


class ContextAwareModelSelector:
    """
    Selects the best model for each blank row based on its specific context.
    Tests all models in the same context to get honest R² scores.
    """

    def __init__(self, data_df, blank_df, models, historic_manager, seasonal_patterns):
        """
        Parameters:
        -----------
        data_df : DataFrame
            Rows with actual VQ (for testing)
        blank_df : DataFrame
            Rows with missing VQ (to predict)
        models : dict
            All trained models
        historic_manager : HistoricDataManager
            Historic data access
        seasonal_patterns : dict
            Seasonal adjustment factors
        """
        self.data_df = data_df
        self.blank_df = blank_df
        self.models = models
        self.hdm = historic_manager
        self.seasonal_patterns = seasonal_patterns

        # Cache for context-specific R² scores
        self.context_r2_cache = {}  # {(context_sig, model_name): r2}

        # Pre-calculate some useful lookups
        self._prepare_test_infrastructure()

    def _prepare_test_infrastructure(self):
        """Prepare data structures for efficient context testing"""
        print("\n" + "="*70)
        print("PREPARING CONTEXT-AWARE MODEL SELECTION")
        print("="*70)

        # Group sites by how many months of data they have
        self.sites_by_month_count = defaultdict(list)

        monthly_data = self.data_df[self.data_df['Timeframe'] == 'Monthly']
        for site in monthly_data['Site identifier'].unique():
            site_data = monthly_data[monthly_data['Site identifier'] == site]
            month_count = len(site_data)
            self.sites_by_month_count[month_count].append(site)

        print(f"✓ Analyzed {len(monthly_data['Site identifier'].unique())} sites")
        print(f"  Sites with 12 months: {len(self.sites_by_month_count[12])}")
        print(f"  Sites with 6 months: {len(self.sites_by_month_count[6])}")
        print(f"  Sites with 3 months: {len(self.sites_by_month_count[3])}")

        # Identify which features are available broadly
        self.available_features = set()
        for col in self.data_df.columns:
            if col not in ['Site identifier', 'Location', 'GHG Category', 'Date from', 'Date to',
                          'Volumetric Quantity', 'Timeframe', 'Month', 'Data Timeframe',
                          'Data integrity', 'Estimation Method', 'Data Quality', 'Data Quality Score']:
                if self.data_df[col].notna().sum() / len(self.data_df) > 0.1:
                    self.available_features.add(col)

        print(f"✓ Identified {len(self.available_features)} broadly available features")

    def analyze_row_context(self, row):
        """
        Analyze what data is available for this specific blank row.

        Returns:
        --------
        RowContext
        """
        site_id = row['Site identifier']
        month = row.get('Month')
        timeframe = row.get('Timeframe')

        # Check which features this row has
        features_available = set()
        for feature in self.available_features:
            if feature in row.index and pd.notna(row[feature]):
                features_available.add(feature)

        # Check site's current year data
        site_current_data = self.data_df[
            (self.data_df['Site identifier'] == site_id) &
            (self.data_df['Timeframe'] == 'Monthly')
        ]
        site_months_available = len(site_current_data)
        site_has_any_current_data = site_months_available > 0

        # Check historic data
        has_historic_data = False
        historic_features = set()
        historic_periods_count = 0

        if self.hdm and site_id in self.hdm.site_periods:
            periods = self.hdm.site_periods[site_id]
            has_historic_data = len(periods) > 0
            historic_periods_count = len(periods)

            # Check which historic features are available
            if periods:
                most_recent = periods[-1]
                if most_recent.monthly_data or most_recent.annual_total:
                    historic_features.add('Historic VQ')

                for feature_name in most_recent.features:
                    historic_features.add(feature_name)

        return RowContext(
            site_id=site_id,
            month=month,
            timeframe=timeframe,
            features_available=features_available,
            site_months_available=site_months_available,
            site_has_any_current_data=site_has_any_current_data,
            has_historic_data=has_historic_data,
            historic_features=historic_features,
            historic_periods_count=historic_periods_count,
            brand=row.get('Brand')
        )

    def get_best_model_for_row(self, row):
        """
        Find the best model for this specific row based on context-specific R² testing.

        Returns:
        --------
        (model_name, context_r2, prediction)
        """
        context = self.analyze_row_context(row)

        # Test all applicable models in this context
        model_scores = {}

        for model_name, model_info in self.models.items():
            # Skip models with negative training R² (worse than mean)
            if model_info.get('r2', 0) < 0:
                continue

            # Check if model can be applied to this row
            if not self._model_can_predict(model_name, model_info, context):
                continue

            # Get context-specific R² for this model
            context_r2 = self.get_context_specific_r2(model_name, model_info, context)

            # ONLY use models with positive R² (better than mean)
            if context_r2 is not None and context_r2 > 0:
                model_scores[model_name] = context_r2

        # Pick best model
        if model_scores:
            best_model_name = max(model_scores, key=model_scores.get)
            best_r2 = model_scores[best_model_name]
            return best_model_name, best_r2
        else:
            return None, None

    def _model_can_predict(self, model_name, model_info, context):
        """Check if a model can predict given this context"""
        model_features = set(model_info.get('features', []))

        # Rule-based models
        if model_info.get('type') in ['LastYear_Direct', 'LastYear_TurnoverAdjusted',
                                      'MultiYear_Average', 'LastYear_GrowthAdjusted']:
            return context.has_historic_data

        # Site_Seasonal needs month
        if model_info.get('type') == 'Site_Seasonal':
            return context.month is not None

        # Annual_Average works for annual rows
        if model_info.get('type') == 'Annual_Average':
            return context.timeframe == 'Annual'

        # ML models need their features
        required_features = model_features - {'Month'}  # Month is always available
        available_features = context.features_available | context.historic_features

        return required_features.issubset(available_features)

    def get_context_specific_r2(self, model_name, model_info, context):
        """
        Calculate R² for this model in THIS specific context.

        Tests the model on sites with full data, but simulating the same
        data availability as the context.
        """
        # Check cache first
        cache_key = (context.get_signature(), model_name)
        if cache_key in self.context_r2_cache:
            return self.context_r2_cache[cache_key]

        # Calculate context-specific R²
        r2 = self._calculate_context_r2(model_name, model_info, context)

        # Cache it
        self.context_r2_cache[cache_key] = r2

        return r2

    def _calculate_context_r2(self, model_name, model_info, context):
        """
        Actually calculate the context-specific R² by testing on sites with full data
        """
        model_type = model_info.get('type')

        # Get test sites (sites with full data that we can test on)
        test_sites = self._get_test_sites(context)

        if len(test_sites) < 10:
            # Not enough test sites - can't calculate context R²
            # Return None instead of general R² (which might be negative!)
            return None

        predictions = []
        actuals = []

        for test_site in test_sites:
            try:
                # Get test site's data, limited to context
                limited_data = self._limit_site_to_context(test_site, context)

                if limited_data is None:
                    continue

                # Make prediction using limited data
                pred = self._predict_with_model(
                    model_name, model_info, limited_data, context
                )

                if pred is None or pred <= 0:
                    continue

                # Get actual value
                actual = self._get_actual_for_test_site(test_site, context)

                if actual is None or actual <= 0:
                    continue

                predictions.append(pred)
                actuals.append(actual)

            except Exception as e:
                continue

        # Calculate R²
        if len(predictions) >= 10:
            r2 = r2_score(actuals, predictions)
            return r2
        else:
            # Not enough predictions - can't calculate reliable R²
            # Return None instead of using general R² (which might be negative!)
            return None

    def _get_test_sites(self, context):
        """Get sites suitable for testing in this context"""
        # We want sites with MORE data than the context
        # (so we can simulate having less)

        if context.timeframe == 'Monthly':
            # Get sites with at least as many months as context requires
            min_months = max(context.site_months_available, 6)

            test_sites = []
            for month_count in range(min_months, 13):
                test_sites.extend(self.sites_by_month_count.get(month_count, []))

            return test_sites[:100]  # Limit to 100 for performance
        else:
            # Annual - use all sites with annual data
            annual_sites = self.data_df[
                self.data_df['Timeframe'] == 'Annual'
            ]['Site identifier'].unique()
            return list(annual_sites)[:100]

    def _limit_site_to_context(self, test_site, context):
        """
        Limit a test site's data to match the context.

        E.g., if context has 6 months, only use first 6 months of test site.
        """
        site_data = self.data_df[
            (self.data_df['Site identifier'] == test_site) &
            (self.data_df['Timeframe'] == 'Monthly')
        ].copy()

        if len(site_data) == 0:
            return None

        # Limit to same number of months as context
        if context.site_months_available > 0:
            site_data = site_data.head(context.site_months_available)

        # Get a representative row (use the month we're predicting if available)
        if context.month:
            month_rows = site_data[site_data['Month'] == context.month]
            if len(month_rows) > 0:
                return month_rows.iloc[0]

        # Otherwise use first row
        return site_data.iloc[0] if len(site_data) > 0 else None

    def _predict_with_model(self, model_name, model_info, limited_row, context):
        """Make a prediction using a model with limited data"""
        # This will call the appropriate prediction method based on model type
        # For now, returning None to be implemented with actual model types

        model_type = model_info.get('type')

        if model_type == 'Site_Seasonal':
            return self._predict_site_seasonal(limited_row, context)
        elif model_type in ['LastYear_Direct', 'LastYear_TurnoverAdjusted']:
            return self._predict_historic_rule(model_name, limited_row, context)
        elif model_type in ['Linear', 'RandomForest', 'GradientBoosting', 'Polynomial']:
            return self._predict_ml_model(model_info, limited_row, context)
        else:
            return None

    def _predict_site_seasonal(self, limited_row, context):
        """Predict using Site_Seasonal with limited data"""
        if 'Site_Seasonal' not in self.models or not context.month:
            return None

        site_id = limited_row['Site identifier']

        # Calculate site average from limited data
        site_data = self.data_df[
            (self.data_df['Site identifier'] == site_id) &
            (self.data_df['Timeframe'] == 'Monthly')
        ]

        if context.site_months_available > 0:
            site_data = site_data.head(context.site_months_available)

        if len(site_data) == 0:
            # Use overall average
            site_avg = self.data_df[
                self.data_df['Timeframe'] == 'Monthly'
            ]['Volumetric Quantity'].mean()
        else:
            site_avg = site_data['Volumetric Quantity'].mean()

        # Apply seasonal factor
        brand = context.brand or 'Overall'
        pattern_key = brand if brand in self.seasonal_patterns else 'Overall'

        if pattern_key in self.seasonal_patterns and context.month in self.seasonal_patterns[pattern_key]:
            seasonal_factor = self.seasonal_patterns[pattern_key][context.month]
            return site_avg * seasonal_factor

        return site_avg

    def _predict_historic_rule(self, model_name, limited_row, context):
        """Predict using historic rule-based model"""
        # Simplified - actual implementation would use the full logic
        if not context.has_historic_data:
            return None

        # Get historic VQ (placeholder - actual implementation needed)
        return None

    def _predict_ml_model(self, model_info, limited_row, context):
        """Predict using ML model"""
        # Build feature vector from limited row
        features = model_info.get('features', [])
        feature_values = []

        for feature in features:
            if feature == 'Month':
                # Encode month
                months = ['January', 'February', 'March', 'April', 'May', 'June',
                         'July', 'August', 'September', 'October', 'November', 'December']
                month_val = months.index(limited_row.get('Month', 'January'))
                feature_values.append(month_val)
            elif feature in limited_row.index:
                val = limited_row[feature]
                if pd.notna(val):
                    feature_values.append(float(val))
                else:
                    return None  # Missing required feature
            else:
                return None  # Missing required feature

        if len(feature_values) != len(features):
            return None

        X = np.array(feature_values).reshape(1, -1)

        try:
            return model_info['model'].predict(X)[0]
        except:
            return None

    def _get_actual_for_test_site(self, test_site, context):
        """Get the actual value for a test site"""
        if context.timeframe == 'Monthly' and context.month:
            # Get actual VQ for this specific month
            month_data = self.data_df[
                (self.data_df['Site identifier'] == test_site) &
                (self.data_df['Month'] == context.month) &
                (self.data_df['Timeframe'] == 'Monthly')
            ]

            if len(month_data) > 0:
                return month_data.iloc[0]['Volumetric Quantity']

        elif context.timeframe == 'Annual':
            # Get annual total
            annual_data = self.data_df[
                (self.data_df['Site identifier'] == test_site) &
                (self.data_df['Timeframe'] == 'Annual')
            ]

            if len(annual_data) > 0:
                return annual_data.iloc[0]['Volumetric Quantity']

        return None

    def generate_context_r2_report(self):
        """Generate a report showing R² scores for different contexts"""
        print("\n" + "="*70)
        print("CONTEXT-SPECIFIC R² ANALYSIS")
        print("="*70)

        # Group by context signatures
        context_groups = defaultdict(list)

        for (context_sig, model_name), r2 in self.context_r2_cache.items():
            context_groups[context_sig].append((model_name, r2))

        # Display top contexts
        for i, (context_sig, model_r2s) in enumerate(list(context_groups.items())[:10]):
            print(f"\nContext {i+1}: {context_sig[0]}, {context_sig[2]} months data")
            print("  Features:", ", ".join(context_sig[1]) if context_sig[1] else "None")
            print("  Has historic:", context_sig[3])
            print("\n  Model Performance:")

            # Sort by R²
            sorted_models = sorted(model_r2s, key=lambda x: x[1], reverse=True)
            for model_name, r2 in sorted_models[:5]:
                print(f"    {model_name}: R² = {r2:.4f}")