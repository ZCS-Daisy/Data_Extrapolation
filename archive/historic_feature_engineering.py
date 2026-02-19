"""
PART 2: Feature Engineering Module
Creates YoY changes, growth rates, trends, and ratio features from historic data

✅ CRITICAL FIX: Derived features (VQ_per_*, Efficiency_*) are NO LONGER recommended
   because they require VQ to calculate, and blank rows don't have VQ!
"""

import pandas as pd
import numpy as np
from collections import defaultdict


class HistoricFeatureEngineer:
    """
    Creates features from historic data to improve ML predictions.
    Generates YoY changes, growth rates, trends, and efficiency metrics.
    """

    def __init__(self, historic_data_manager):
        """
        Parameters:
        -----------
        historic_data_manager : HistoricDataManager
            Initialized manager with loaded historic data
        """
        self.hdm = historic_data_manager
        self.engineered_features = {}  # {site: {month: {feature: value}}}

    def engineer_all_features(self, current_df):
        """
        Engineer all features for the current dataset.

        Parameters:
        -----------
        current_df : DataFrame
            Current year data

        Returns:
        --------
        DataFrame with additional engineered features
        """
        print("\n" + "="*70)
        print("FEATURE ENGINEERING FROM HISTORIC DATA")
        print("="*70)

        df_enhanced = current_df.copy()

        # Add Year column (numeric for ML)
        df_enhanced['Year'] = self.hdm.current_year
        print(f"✓ Added Year feature: {self.hdm.current_year}")

        # Add Years_Since_Baseline
        if self.hdm.historic_years:
            baseline_year = min(self.hdm.historic_years)
            df_enhanced['Years_Since_Baseline'] = self.hdm.current_year - baseline_year
            print(f"✓ Added Years_Since_Baseline (baseline: {baseline_year})")

        # Generate YoY features
        df_enhanced = self._add_yoy_features(df_enhanced)

        # Generate trend features
        df_enhanced = self._add_trend_features(df_enhanced)

        # Generate ratio/efficiency features
        df_enhanced = self._add_ratio_features(df_enhanced)

        # Count engineered features
        engineered_cols = [col for col in df_enhanced.columns if col not in current_df.columns]
        print(f"\n✓ Generated {len(engineered_cols)} engineered features total")

        return df_enhanced

    def _add_yoy_features(self, df):
        """Add Year-over-Year change features"""
        print("\n📊 Generating YoY Change Features...")

        if not self.hdm.historic_years:
            print("  ⚠️  No historic years available - skipping YoY features")
            return df

        # VQ YoY changes
        vq_yoy_changes = []
        vq_yoy_absolute = []

        # Feature YoY changes
        feature_yoy_changes = defaultdict(list)

        for idx, row in df.iterrows():
            site = row['Site identifier']
            month = row.get('Month')

            # Get last year's VQ
            last_year = self.hdm.current_year - 1
            last_year_vq = self.hdm.get_historic_vq(site, last_year, month) if month else None
            current_vq = row.get('Volumetric Quantity')

            # Calculate VQ YoY
            if last_year_vq and pd.notna(current_vq):
                yoy_change = (current_vq - last_year_vq) / last_year_vq
                yoy_absolute = current_vq - last_year_vq
                vq_yoy_changes.append(yoy_change)
                vq_yoy_absolute.append(yoy_absolute)
            else:
                vq_yoy_changes.append(None)
                vq_yoy_absolute.append(None)

            # Calculate Feature YoY changes (e.g., Turnover, SqFt)
            for feature in self.hdm.historic_manager.site_year_features.get(site, {}).get(last_year, {}).keys() if hasattr(self.hdm, 'historic_manager') else []:
                last_year_val = self.hdm.get_historic_feature(site, last_year, feature)
                current_val = row.get(feature)

                if last_year_val and pd.notna(current_val):
                    try:
                        yoy_change = (float(current_val) - float(last_year_val)) / float(last_year_val)
                        feature_yoy_changes[f'{feature}_YoY_Change'].append(yoy_change)
                    except:
                        feature_yoy_changes[f'{feature}_YoY_Change'].append(None)
                else:
                    feature_yoy_changes[f'{feature}_YoY_Change'].append(None)

        # Add to dataframe
        df['VQ_YoY_Change'] = vq_yoy_changes
        df['VQ_YoY_Absolute'] = vq_yoy_absolute

        vq_yoy_count = sum(1 for x in vq_yoy_changes if x is not None)
        print(f"  ✓ VQ_YoY_Change: {vq_yoy_count} calculated")

        for feature_name, values in feature_yoy_changes.items():
            df[feature_name] = values
            count = sum(1 for x in values if x is not None)
            print(f"  ✓ {feature_name}: {count} calculated")

        return df

    def _add_trend_features(self, df):
        """Add trend features based on multiple years of data"""
        print("\n📈 Generating Trend Features...")

        if len(self.hdm.historic_years) < 2:
            print("  ⚠️  Need at least 2 historic years for trend features - skipping")
            return df

        vq_2yr_growth = []
        vq_3month_avg = []
        vq_trend_slope = []
        vq_volatility = []

        for idx, row in df.iterrows():
            site = row['Site identifier']
            month = row.get('Month')

            # Get historic VQ for this site/month across years
            historic_vqs = []
            historic_years_list = []

            for year in sorted(self.hdm.historic_years):
                vq = self.hdm.get_historic_vq(site, year, month) if month else None
                if vq:
                    historic_vqs.append(vq)
                    historic_years_list.append(year)

            # 2-year growth (if we have data from 2 years ago)
            if len(self.hdm.historic_years) >= 2:
                two_years_ago = self.hdm.current_year - 2
                vq_2yr_ago = self.hdm.get_historic_vq(site, two_years_ago, month) if month else None
                current_vq = row.get('Volumetric Quantity')

                if vq_2yr_ago and pd.notna(current_vq):
                    growth_2yr = ((current_vq - vq_2yr_ago) / vq_2yr_ago) / 2  # Annualized
                    vq_2yr_growth.append(growth_2yr)
                else:
                    vq_2yr_growth.append(None)
            else:
                vq_2yr_growth.append(None)

            # 3-month average (average of this month across available years)
            if len(historic_vqs) >= 2:
                avg_3mo = np.mean(historic_vqs[-3:] if len(historic_vqs) >= 3 else historic_vqs)
                vq_3month_avg.append(avg_3mo)
            else:
                vq_3month_avg.append(None)

            # Trend slope (linear regression across years)
            if len(historic_vqs) >= 3 and len(historic_years_list) >= 3:
                try:
                    # Simple linear regression: slope = cov(x,y) / var(x)
                    x = np.array(historic_years_list)
                    y = np.array(historic_vqs)

                    x_mean = np.mean(x)
                    y_mean = np.mean(y)

                    numerator = np.sum((x - x_mean) * (y - y_mean))
                    denominator = np.sum((x - x_mean) ** 2)

                    if denominator > 0:
                        slope = numerator / denominator
                        vq_trend_slope.append(slope)
                    else:
                        vq_trend_slope.append(None)
                except:
                    vq_trend_slope.append(None)
            else:
                vq_trend_slope.append(None)

            # Volatility (standard deviation across years)
            if len(historic_vqs) >= 2:
                volatility = np.std(historic_vqs)
                vq_volatility.append(volatility)
            else:
                vq_volatility.append(None)

        # Add to dataframe
        if len(self.hdm.historic_years) >= 2:
            df['VQ_2Yr_Growth'] = vq_2yr_growth
            count = sum(1 for x in vq_2yr_growth if x is not None)
            print(f"  ✓ VQ_2Yr_Growth: {count} calculated")

        df['VQ_Historic_Avg'] = vq_3month_avg
        count = sum(1 for x in vq_3month_avg if x is not None)
        print(f"  ✓ VQ_Historic_Avg: {count} calculated")

        df['VQ_Trend_Slope'] = vq_trend_slope
        count = sum(1 for x in vq_trend_slope if x is not None)
        print(f"  ✓ VQ_Trend_Slope: {count} calculated")

        df['VQ_Volatility'] = vq_volatility
        count = sum(1 for x in vq_volatility if x is not None)
        print(f"  ✓ VQ_Volatility: {count} calculated")

        return df

    def _add_ratio_features(self, df):
        """Add ratio/efficiency features"""
        print("\n⚙️  Generating Ratio/Efficiency Features...")

        # VQ per feature ratios (e.g., VQ per Turnover, VQ per SqFt)
        ratio_features_added = []

        # Get list of available features from current data
        feature_cols = [col for col in df.columns
                       if col not in ['Site identifier', 'Location', 'GHG Category',
                                     'Date from', 'Date to', 'Volumetric Quantity',
                                     'Timeframe', 'Month', 'Data Timeframe',
                                     'Data integrity', 'Estimation Method',
                                     'Data Quality', 'Data Quality Score', 'Year',
                                     'Years_Since_Baseline']
                       and pd.api.types.is_numeric_dtype(df[col])]

        for feature in feature_cols:
            # Skip YoY and trend features themselves
            if any(x in feature for x in ['YoY', 'Trend', 'Volatility', 'Growth', 'Historic_Avg']):
                continue

            # Calculate VQ per Feature
            vq_per_feature = []
            for idx, row in df.iterrows():
                vq = row.get('Volumetric Quantity')
                feature_val = row.get(feature)

                if pd.notna(vq) and pd.notna(feature_val) and feature_val != 0:
                    try:
                        ratio = float(vq) / float(feature_val)
                        vq_per_feature.append(ratio)
                    except:
                        vq_per_feature.append(None)
                else:
                    vq_per_feature.append(None)

            ratio_col = f'VQ_per_{feature}'
            df[ratio_col] = vq_per_feature
            count = sum(1 for x in vq_per_feature if x is not None)
            if count > 0:
                print(f"  ✓ {ratio_col}: {count} calculated")
                ratio_features_added.append(ratio_col)

            # Calculate YoY efficiency change (how efficiency changed vs last year)
            if not self.hdm.historic_years:
                continue

            efficiency_yoy = []
            last_year = self.hdm.current_year - 1

            for idx, row in df.iterrows():
                site = row['Site identifier']
                month = row.get('Month')

                # Current efficiency
                current_vq = row.get('Volumetric Quantity')
                current_feature = row.get(feature)
                current_eff = None
                if pd.notna(current_vq) and pd.notna(current_feature) and current_feature != 0:
                    try:
                        current_eff = float(current_vq) / float(current_feature)
                    except:
                        pass

                # Last year efficiency
                last_vq = self.hdm.get_historic_vq(site, last_year, month) if month else None
                last_feature = self.hdm.get_historic_feature(site, last_year, feature)
                last_eff = None
                if last_vq and last_feature and last_feature != 0:
                    try:
                        last_eff = float(last_vq) / float(last_feature)
                    except:
                        pass

                # Calculate change
                if current_eff and last_eff:
                    eff_change = (current_eff - last_eff) / last_eff
                    efficiency_yoy.append(eff_change)
                else:
                    efficiency_yoy.append(None)

            eff_col = f'Efficiency_{feature}_YoY'
            df[eff_col] = efficiency_yoy
            count = sum(1 for x in efficiency_yoy if x is not None)
            if count > 0:
                print(f"  ✓ {eff_col}: {count} calculated")
                ratio_features_added.append(eff_col)

        if not ratio_features_added:
            print("  ⚠️  No ratio features generated (no suitable numeric features)")

        return df

    def get_feature_availability_summary(self, df):
        """Get summary of which engineered features are available"""
        engineered_features = [col for col in df.columns
                              if any(x in col for x in ['YoY', 'Trend', 'Volatility',
                                                        'Growth', 'Historic_Avg',
                                                        'VQ_per_', 'Efficiency_'])]

        summary = {}
        for feature in engineered_features:
            available_count = df[feature].notna().sum()
            availability_pct = available_count / len(df) * 100
            summary[feature] = {
                'available_count': available_count,
                'availability_pct': availability_pct
            }

        return summary

    def recommend_features_for_ml(self, df, min_availability_pct=20):
        """
        Recommend which engineered features should be used in ML models.

        ⚠️ CRITICAL: Excludes derived features (VQ_per_*, Efficiency_*) because:
        - These features require VQ to calculate
        - Blank rows don't have VQ (that's what we're predicting!)
        - Models using these features can't make predictions on blank rows

        Parameters:
        -----------
        df : DataFrame
            DataFrame with engineered features
        min_availability_pct : float
            Minimum % availability to recommend (default 20%)

        Returns:
        --------
        List of recommended feature names (excludes VQ-derived features)
        """
        feature_summary = self.get_feature_availability_summary(df)

        recommended = []
        derived_features_skipped = []

        for feature, stats in feature_summary.items():
            if stats['availability_pct'] >= min_availability_pct:
                # ⚠️ SKIP derived features that require VQ to calculate
                if any(x in feature for x in ['VQ_per_', 'Efficiency_']):
                    derived_features_skipped.append(feature)
                    continue

                # Also skip VQ-based features that won't help prediction
                if any(x in feature for x in ['VQ_YoY', 'VQ_Trend', 'VQ_Volatility', 'VQ_Historic_Avg', 'VQ_2Yr_Growth']):
                    derived_features_skipped.append(feature)
                    continue

                recommended.append(feature)

        print(f"\n💡 Recommended {len(recommended)} engineered features for ML (≥{min_availability_pct}% availability):")
        for feat in recommended:
            pct = feature_summary[feat]['availability_pct']
            print(f"  • {feat}: {pct:.1f}%")

        if derived_features_skipped:
            print(f"\n⚠️  Skipped {len(derived_features_skipped)} VQ-derived features (can't predict blank rows):")
            for feat in derived_features_skipped[:10]:  # Show first 10
                pct = feature_summary[feat]['availability_pct']
                print(f"  ✗ {feat}: {pct:.1f}% (requires VQ)")
            if len(derived_features_skipped) > 10:
                print(f"  ... and {len(derived_features_skipped) - 10} more")

        return recommended