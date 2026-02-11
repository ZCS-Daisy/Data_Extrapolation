"""
Volumetric Data Extrapolation Tool with Historic Data Support
Fills in missing volumetric quantities using regression models, seasonal patterns, and historic data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Force reload of modules to avoid caching issues
import sys

if 'historic_data_module' in sys.modules:
    del sys.modules['historic_data_module']
if 'feature_engineering_module' in sys.modules:
    del sys.modules['feature_engineering_module']
if 'historic_model_training' in sys.modules:
    del sys.modules['historic_model_training']
if 'scenario_based_selection' in sys.modules:
    del sys.modules['scenario_based_selection']

# Import historic data modules
from historic_data_handler import HistoricDataManager
from historic_feature_engineering import HistoricFeatureEngineer
from historic_model_training import HistoricModelTrainer
from scenario_based_selection import ScenarioBasedModelSelector

# Core columns that should never be used as features
CORE_COLUMNS = {
    'Client', 'Site identifier', 'Location', 'Date from', 'Date to',
    'GHG Category', 'Volumetric Quantity', 'Timeframe', 'Month',
    'Data Timeframe', 'Data integrity', 'Estimation Method',
    'Data Quality', 'Data Quality Score', 'Year', 'Years_Since_Baseline'
}


def get_effective_r2_for_row(model_info, row):
    """
    Determine which R² to use based on row's feature availability.

    If row is missing any derived features → use r2_imputed
    If row has all features → use r2_full

    Parameters:
    -----------
    model_info : dict
        Model information with both R² scores
    row : Series
        Row being predicted

    Returns:
    --------
    float : Appropriate R² score for this row
    """
    # Rule-based models don't have derived features issue
    if model_info.get('type') in ['LastYear_Direct', 'LastYear_TurnoverAdjusted',
                                  'MultiYear_Average', 'LastYear_GrowthAdjusted',
                                  'Site_Seasonal', 'Annual_Average']:
        return model_info.get('r2', 0)

    # Check if model has derived features
    if not model_info.get('has_derived', False):
        return model_info.get('r2_full', model_info.get('r2', 0))

    # Check if any derived features are missing in this row
    derived_features = model_info.get('derived_features', [])

    for feature in derived_features:
        if feature not in row.index or pd.isna(row[feature]):
            # Missing a derived feature, use imputed R²
            return model_info.get('r2_imputed', model_info.get('r2', 0))

    # All derived features present, use full R²
    return model_info.get('r2_full', model_info.get('r2', 0))


class VolumetricExtrapolationTool:
    """
    Tool to extrapolate missing volumetric quantities using regression models,
    seasonal patterns, and optionally historic data.
    """

    def __init__(self, extrapolate_file, factor_database_file=None, selected_features=None,
                 historic_records_file=None, historic_features_file=None):
        self.extrapolate_file = extrapolate_file
        self.factor_database_file = factor_database_file
        self.selected_features = selected_features or []
        self.historic_records_file = historic_records_file
        self.historic_features_file = historic_features_file

        # Data
        self.df = None
        self.blank_df = None
        self.data_df = None

        # Models and patterns
        self.seasonal_patterns = {}
        self.models = {}
        self.site_level_features = []
        self.row_level_features = []
        self.categorical_features = []
        self.categorical_encoders = {}

        # Historic data components
        self.historic_manager = None
        self.feature_engineer = None
        self.historic_trainer = None
        self.has_historic_data = False

    def load_data(self):
        """Load the extrapolate file and identify columns"""
        if self.extrapolate_file.endswith('.csv'):
            self.df = pd.read_csv(self.extrapolate_file)
        else:
            self.df = pd.read_excel(self.extrapolate_file)

        # CRITICAL FIX: Strip whitespace from column names
        self.df.columns = self.df.columns.str.strip()

        print(f"Loaded {len(self.df)} rows with {len(self.df.columns)} columns")

        # Force numeric conversion for columns that should be numeric
        self._convert_numeric_columns()

        return self.df

    def _convert_numeric_columns(self):
        """
        Convert columns to numeric if they contain numeric data stored as text.
        This handles cases where Excel/CSV stores numbers as strings.
        """
        # Known numeric columns that should always be numeric
        always_numeric = ['Volumetric Quantity', 'Turnover', 'Transactions',
                          'Square Footage', 'Floor Space', 'Revenue', 'Sales']

        for col in self.df.columns:
            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(self.df[col]):
                continue

            # Check if column name suggests it should be numeric
            should_be_numeric = any(keyword.lower() in col.lower()
                                    for keyword in always_numeric)

            if should_be_numeric or col in always_numeric:
                # Try to convert
                try:
                    converted = pd.to_numeric(self.df[col], errors='coerce')
                    successful = converted.notna().sum()
                    total = self.df[col].notna().sum()

                    if total > 0 and successful / total > 0.8:  # 80% successful
                        self.df[col] = converted
                        print(f"  ✓ Converted '{col}' to numeric ({successful}/{total} values)")
                except:
                    pass

            # Also try converting any column with many unique numeric-like values
            elif not should_be_numeric:
                try:
                    non_null = self.df[col].dropna()
                    if len(non_null) > 0:
                        # Try conversion
                        converted = pd.to_numeric(non_null, errors='coerce')
                        successful = converted.notna().sum()
                        conversion_rate = successful / len(non_null)
                        unique_ratio = non_null.nunique() / len(non_null)

                        # If >80% convert successfully AND high uniqueness (not categorical)
                        if conversion_rate > 0.8 and unique_ratio > 0.1:
                            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                            print(f"  ✓ Auto-detected numeric column '{col}' (was text format)")
                except:
                    pass

    def parse_dates(self, date_str):
        """Parse date string handling both standard format and Excel serial dates"""
        try:
            if isinstance(date_str, (int, float)):
                excel_epoch = datetime(1899, 12, 30)
                return excel_epoch + pd.Timedelta(days=date_str)
            return pd.to_datetime(date_str, format='%d/%m/%Y')
        except:
            return pd.to_datetime(date_str, dayfirst=True)

    def determine_timeframe(self, row):
        """Determine if data is Monthly, Annual, or custom days"""
        date_from = self.parse_dates(row['Date from'])
        date_to = self.parse_dates(row['Date to'])

        # Check if monthly (1st to end of month)
        if date_from.day == 1:
            next_month = date_from + relativedelta(months=1)
            last_day = next_month - pd.Timedelta(days=1)

            if date_to.date() == last_day.date():
                return 'Monthly', date_from.strftime('%B')

        # Check if annual (full year period)
        days_diff = (date_to - date_from).days + 1
        if days_diff >= 365 and days_diff <= 366:
            return 'Annual', None

        # Custom period
        return f'{days_diff} days', None

    def add_timeframe_columns(self):
        """Add Timeframe and Month columns to dataframe"""
        timeframes = []
        months = []

        for idx, row in self.df.iterrows():
            timeframe, month = self.determine_timeframe(row)
            timeframes.append(timeframe)
            months.append(month if month else '')

        self.df['Timeframe'] = timeframes
        self.df['Month'] = months
        print(f"Detected timeframes: {self.df['Timeframe'].value_counts().to_dict()}")

    def load_historic_data(self):
        """Load and process historic data if provided"""
        if not self.historic_records_file:
            print("\nℹ️  No historic data provided - using standard models only")
            return False

        print("\n" + "=" * 70)
        print("LOADING HISTORIC DATA")
        print("=" * 70)

        try:
            # Initialize historic data manager
            self.historic_manager = HistoricDataManager()

            # Detect current period from current data
            self.historic_manager.detect_current_period(self.df)

            # Get GHG categories for filtering
            ghg_categories = self.df['GHG Category'].unique().tolist()

            # Load and organize historic records into periods
            self.historic_manager.load_historic_records(
                self.historic_records_file,
                ghg_categories
            )

            # Load historic features if provided
            if self.historic_features_file:
                self.historic_manager.load_historic_features(self.historic_features_file)

            # Create availability matrix
            current_sites = self.df['Site identifier'].unique().tolist()
            self.historic_manager.create_availability_matrix(self.df)

            # Identify new sites
            self.historic_manager.identify_new_sites(current_sites)

            # CRITICAL VALIDATION: Check if any historic periods were actually created
            total_periods = sum(len(periods) for periods in self.historic_manager.site_periods.values())
            historic_years = self.historic_manager.historic_years if hasattr(self.historic_manager,
                                                                             'historic_years') else []

            print("\n" + "=" * 70)
            print("HISTORIC DATA VALIDATION")
            print("=" * 70)

            if total_periods == 0:
                print("⚠️  WARNING: Historic data loaded but NO HISTORIC PERIODS detected!")
                print("    Possible causes:")
                print("    • Historic records are from current year (not previous years)")
                print("    • Missing Year/FY column in historic files")
                print("    • Date format issues preventing year extraction")
                print("    • No matching GHG categories")
                print("\n❌ HISTORIC MODELS DISABLED - Using standard models only")
                print("=" * 70)
                self.has_historic_data = False
                return False

            if len(historic_years) == 0:
                print("⚠️  WARNING: Historic periods found but NO HISTORIC YEARS detected!")
                print("    Cannot use year-over-year features or trend analysis")
                print("\n❌ HISTORIC MODELS DISABLED - Using standard models only")
                print("=" * 70)
                self.has_historic_data = False
                return False

            # Success!
            print(f"✅ VALIDATION PASSED:")
            print(f"    • {total_periods} historic periods across {len(self.historic_manager.site_periods)} sites")
            print(f"    • {len(historic_years)} historic years detected: {sorted(historic_years)}")
            print(f"    • Historic models ENABLED")
            print("=" * 70)

            self.has_historic_data = True
            return True

        except Exception as e:
            print(f"\n⚠️  Error loading historic data: {e}")
            import traceback
            traceback.print_exc()
            print("  Continuing with standard models only...")
            self.has_historic_data = False
            return False

    def engineer_historic_features(self):
        """Create features from historic data"""
        if not self.has_historic_data:
            return self.df

        try:
            self.feature_engineer = HistoricFeatureEngineer(self.historic_manager)
            self.df = self.feature_engineer.engineer_all_features(self.df)

            # Get recommended features
            recommended = self.feature_engineer.recommend_features_for_ml(self.df)

            # Add to selected features (avoiding duplicates)
            for feat in recommended:
                if feat not in self.selected_features:
                    self.selected_features.append(feat)

            return self.df

        except Exception as e:
            print(f"\n⚠️  Error engineering features: {e}")
            return self.df

    def identify_blank_rows(self):
        """Separate rows with blank volumetric quantities from those with data"""
        self.blank_df = self.df[self.df['Volumetric Quantity'].isna()].copy()
        self.data_df = self.df[self.df['Volumetric Quantity'].notna()].copy()

        print(f"\nFound {len(self.blank_df)} rows with missing volumetric quantities")
        print(f"Found {len(self.data_df)} rows with data")

    def detect_available_features(self):
        """Detect all non-core columns that could be used as features"""
        available_features = []

        for col in self.df.columns:
            if col not in CORE_COLUMNS:
                non_null_count = self.df[col].notna().sum()
                if non_null_count > 0:
                    available_features.append(col)

        return available_features

    def classify_features(self):
        """Classify features as site-level (constant per site) or row-level (varies)"""
        print("\n" + "=" * 60)
        print("FEATURE CLASSIFICATION")
        print("=" * 60)

        self.site_level_features = []
        self.row_level_features = []
        self.categorical_features = []

        if 'Timeframe' in self.df.columns:
            self.categorical_features.append('Timeframe')

        for feature in self.selected_features:
            if feature not in self.df.columns:
                continue

            # Check if feature is the same for all rows with the same site
            is_site_level = True
            sites = self.df['Site identifier'].unique()

            for site in sites:
                site_data = self.df[self.df['Site identifier'] == site][feature]
                site_data_clean = site_data.dropna()

                if len(site_data_clean) > 1:
                    if pd.api.types.is_numeric_dtype(site_data_clean):
                        if site_data_clean.nunique() > 1:
                            is_site_level = False
                            break
                    else:
                        if len(set(site_data_clean)) > 1:
                            is_site_level = False
                            break

            # Classify feature
            if is_site_level:
                self.site_level_features.append(feature)
                print(f"✓ {feature}: Site-level (constant per site)")
            else:
                self.row_level_features.append(feature)
                print(f"✓ {feature}: Row-level (varies by month)")

            # Check if categorical
            if not pd.api.types.is_numeric_dtype(self.df[feature]):
                if feature not in self.categorical_features:
                    self.categorical_features.append(feature)

        print(f"\nSite-level features: {len(self.site_level_features)}")
        print(f"Row-level features: {len(self.row_level_features)}")
        print(f"Categorical features: {len(self.categorical_features)}")

    def calculate_seasonal_patterns(self):
        """Calculate average seasonal patterns by month"""
        monthly_data = self.data_df[self.data_df['Timeframe'] == 'Monthly'].copy()

        if len(monthly_data) == 0:
            print("\nNo monthly data for seasonal pattern calculation")
            return

        print("\n" + "=" * 60)
        print("SEASONAL PATTERN ANALYSIS")
        print("=" * 60)

        # Overall seasonal pattern
        month_avgs = monthly_data.groupby('Month')['Volumetric Quantity'].mean()
        overall_avg = month_avgs.mean()

        self.seasonal_patterns['Overall'] = {}
        for month in month_avgs.index:
            self.seasonal_patterns['Overall'][month] = month_avgs[month] / overall_avg

        print("\nOverall seasonal pattern:")
        months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                        'July', 'August', 'September', 'October', 'November', 'December']
        for month, factor in sorted(self.seasonal_patterns['Overall'].items(),
                                    key=lambda x: months_order.index(x[0]) if x[0] in months_order else 99):
            print(f"  {month}: {factor:.2%}")

        # Brand-specific seasonal patterns if Brand exists
        if 'Brand' in self.df.columns:
            brands = monthly_data['Brand'].dropna().unique()
            if len(brands) > 1:
                print(f"\nDetected {len(brands)} brands - calculating brand-specific patterns...")
                for brand in brands:
                    brand_data = monthly_data[monthly_data['Brand'] == brand]
                    if len(brand_data) >= 6:
                        brand_month_avgs = brand_data.groupby('Month')['Volumetric Quantity'].mean()
                        brand_overall_avg = brand_month_avgs.mean()

                        self.seasonal_patterns[brand] = {}
                        for month in brand_month_avgs.index:
                            self.seasonal_patterns[brand][month] = brand_month_avgs[month] / brand_overall_avg
                        print(f"  ✓ {brand}: {len(brand_month_avgs)} months of data")

        # Handle missing months
        self._handle_missing_seasonal_months()

    def _handle_missing_seasonal_months(self):
        """
        Detect and handle months with no data in current year.
        Tries: Historic → Interpolation → Equal distribution
        """
        all_months = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']

        # Check Overall pattern for missing months
        if 'Overall' not in self.seasonal_patterns or len(self.seasonal_patterns['Overall']) == 0:
            print("\n⚠️  WARNING: No seasonal patterns calculated - insufficient monthly data")
            return

        current_months = set(self.seasonal_patterns['Overall'].keys())
        missing_months = [m for m in all_months if m not in current_months]

        if len(missing_months) == 0:
            return  # All months present, nothing to do

        print(f"\n⚠️  MISSING SEASONAL DATA DETECTED")
        print(f"   Missing months: {', '.join(missing_months)}")

        for missing_month in missing_months:
            pattern = self._calculate_missing_month_pattern(missing_month, all_months)

            if pattern:
                self.seasonal_patterns['Overall'][missing_month] = pattern['factor']
                print(f"   ✓ {missing_month}: {pattern['factor']:.2%} ({pattern['source']})")

                # Mark as having lower confidence
                if not hasattr(self, 'seasonal_warnings'):
                    self.seasonal_warnings = {}
                self.seasonal_warnings[missing_month] = pattern['source']

    def _calculate_missing_month_pattern(self, missing_month, all_months):
        """
        Calculate seasonal pattern for a missing month.
        Priority: Historic → Interpolation → Equal distribution
        """
        # STRATEGY 1: Use historic data if available
        if self.has_historic_data and self.historic_manager:
            historic_pattern = self._get_historic_seasonal_pattern(missing_month)
            if historic_pattern:
                return {
                    'factor': historic_pattern,
                    'source': 'historic data'
                }

        # STRATEGY 2: Interpolate from adjacent months
        month_index = all_months.index(missing_month)
        prev_month = all_months[(month_index - 1) % 12]
        next_month = all_months[(month_index + 1) % 12]

        prev_pattern = self.seasonal_patterns['Overall'].get(prev_month)
        next_pattern = self.seasonal_patterns['Overall'].get(next_month)

        if prev_pattern and next_pattern:
            interpolated = (prev_pattern + next_pattern) / 2
            return {
                'factor': interpolated,
                'source': f'interpolated from {prev_month} & {next_month}'
            }

        # STRATEGY 3: Equal distribution (fallback)
        return {
            'factor': 1.0,
            'source': 'equal distribution (no data available)'
        }

    def _get_historic_seasonal_pattern(self, month):
        """
        Calculate seasonal pattern from historic data for a specific month.
        Returns factor or None if not available.
        """
        # Collect VQ values for this month across all historic periods
        month_vqs = []
        all_vqs = []

        for site, periods in self.historic_manager.site_periods.items():
            for period in periods:
                # Get VQ for this specific month
                month_vq = self.historic_manager.get_period_vq_for_month(period, month)
                if month_vq:
                    month_vqs.append(month_vq)

                # Get all monthly VQs for calculating average
                if period.monthly_data:
                    all_vqs.extend(period.monthly_data.values())
                elif period.annual_total:
                    # Use monthly average from annual
                    monthly_avg = period.annual_total / period.months_covered
                    all_vqs.extend([monthly_avg] * period.months_covered)

        if len(month_vqs) >= 3 and len(all_vqs) >= 12:
            # Calculate pattern: month_avg / overall_avg
            month_avg = np.mean(month_vqs)
            overall_avg = np.mean(all_vqs)

            if overall_avg > 0:
                return month_avg / overall_avg

        return None

    def train_models(self):
        """Train various regression models on available data"""
        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)

        if len(self.data_df) < 10:
            print("Insufficient data for model training (need at least 10 rows)")
            return

        # Train historic models if available
        if self.has_historic_data and self.historic_manager and self.feature_engineer:
            self.historic_trainer = HistoricModelTrainer(
                self.historic_manager,
                self.feature_engineer
            )
            historic_models = self.historic_trainer.train_all_models(
                self.df,
                self.data_df,
                self.selected_features
            )
            self.models.update(historic_models)

        # Train standard models
        print("\n🎯 TIER 3: Standard Models (No Historic Data Required)")

        # Train models for monthly data with row-level features
        if len(self.row_level_features) > 0:
            monthly_data = self.data_df[self.data_df['Timeframe'] == 'Monthly'].copy()
            if len(monthly_data) >= 10:
                print("\n📊 Training models for MONTHLY data with row-level features...")
                self.train_monthly_models(monthly_data)

        # Train models for site-level features
        if len(self.site_level_features) > 0:
            print("\n📊 Training models for SITE-LEVEL features...")
            self.train_site_level_models()

        # Train site seasonal model
        self.train_site_seasonal_model()

        # Create seasonal variants of site-level models
        self.create_seasonal_variants()

        # Train annual average model
        self.train_annual_average_model()

        print(f"\n✓ Trained {len(self.models)} models total")

    def train_monthly_models(self, monthly_data):
        """Train models for monthly data with row-level features"""
        X_features = []
        feature_names = []

        # Add Month as a feature
        month_encoder = LabelEncoder()
        valid_months = monthly_data[monthly_data['Month'] != '']['Month']
        if len(valid_months) > 0:
            month_encoder.fit(valid_months)
            X_features.append(month_encoder.transform(monthly_data['Month'].fillna('Unknown')).reshape(-1, 1))
            feature_names.append('Month')

        # Add row-level features
        for feature in self.row_level_features:
            if feature in monthly_data.columns:
                if pd.api.types.is_numeric_dtype(monthly_data[feature]):
                    feature_data = monthly_data[feature].fillna(monthly_data[feature].median())
                    X_features.append(feature_data.values.reshape(-1, 1))
                    feature_names.append(feature)
                else:
                    encoder = LabelEncoder()
                    feature_data = monthly_data[feature].fillna('Unknown')
                    encoder.fit(feature_data)
                    X_features.append(encoder.transform(feature_data).reshape(-1, 1))
                    feature_names.append(feature)

        y = monthly_data['Volumetric Quantity'].values

        if len(X_features) > 0:
            # Test individual features
            for i, feature_name in enumerate(feature_names):
                X = X_features[i]
                if len(np.unique(X)) > 1:
                    self.train_and_test_model(X, y, f'Monthly_Linear_{feature_name}',
                                              'Linear', [feature_name])

            # Test all features together
            if len(X_features) > 1:
                X_all = np.hstack(X_features)
                self.train_and_test_model(X_all, y, 'Monthly_RandomForest_All',
                                          'RandomForest', feature_names)
                self.train_and_test_model(X_all, y, 'Monthly_GradientBoosting_All',
                                          'GradientBoosting', feature_names)

    def train_site_level_models(self):
        """
        Train models using site-level features tested on monthly data.

        Tests: Do sites with higher Turnover/SqFt/etc have higher monthly VQ?
        Includes Month to account for seasonality.
        """
        monthly_data = self.data_df[self.data_df['Timeframe'] == 'Monthly'].copy()

        if len(monthly_data) < 100:
            print("  ⚠️  Insufficient monthly data for site-level feature testing")
            print(f"     (Need ≥100 monthly rows, found {len(monthly_data)})")
            return

        print(f"  Testing site-level features on {len(monthly_data)} monthly rows...")

        # Test individual site-level features
        for feature in self.site_level_features:
            if feature not in monthly_data.columns:
                continue

            # Check feature availability
            feature_availability = monthly_data[feature].notna().sum() / len(monthly_data)
            if feature_availability < 0.2:
                print(f"     ⚠️  Skipping {feature}: Only {feature_availability:.1%} available")
                continue

            # Build feature matrix: Month + Site-level Feature
            X_features = []
            feature_names = []

            # Add Month (categorical - important for seasonality!)
            month_encoder = LabelEncoder()
            valid_months = monthly_data[monthly_data['Month'] != '']['Month']
            if len(valid_months) > 0:
                month_encoder.fit(valid_months)
                self.categorical_encoders[f'{feature}_Month'] = month_encoder
                X_features.append(
                    month_encoder.transform(monthly_data['Month'].fillna('Unknown')).reshape(-1, 1)
                )
                feature_names.append('Month')

            # Add site-level feature (same for all months at a site)
            if pd.api.types.is_numeric_dtype(monthly_data[feature]):
                feature_data = monthly_data[feature].fillna(monthly_data[feature].median())
                X_features.append(feature_data.values.reshape(-1, 1))
                feature_names.append(feature)

                if len(X_features) > 0:
                    y = monthly_data['Volumetric Quantity'].values
                    X_all = np.hstack(X_features)

                    # Train Linear model
                    self.train_and_test_model(
                        X_all, y,
                        f'SiteLevel_Monthly_{feature}',
                        'Linear',
                        feature_names
                    )

                    # Also train Polynomial for potentially non-linear relationships
                    self.train_and_test_model(
                        X_all, y,
                        f'SiteLevel_Monthly_Polynomial_{feature}',
                        'Polynomial',
                        feature_names
                    )

        # Test all site-level features together
        if len(self.site_level_features) > 1:
            X_features = []
            feature_names = []

            # Add Month
            month_encoder = LabelEncoder()
            valid_months = monthly_data[monthly_data['Month'] != '']['Month']
            if len(valid_months) > 0:
                month_encoder.fit(valid_months)
                self.categorical_encoders['Combined_Month'] = month_encoder
                X_features.append(
                    month_encoder.transform(monthly_data['Month'].fillna('Unknown')).reshape(-1, 1)
                )
                feature_names.append('Month')

            # Add all numeric site-level features
            for feature in self.site_level_features:
                if feature in monthly_data.columns:
                    feature_availability = monthly_data[feature].notna().sum() / len(monthly_data)
                    if feature_availability >= 0.2:
                        if pd.api.types.is_numeric_dtype(monthly_data[feature]):
                            feature_data = monthly_data[feature].fillna(monthly_data[feature].median())
                            X_features.append(feature_data.values.reshape(-1, 1))
                            feature_names.append(feature)

            if len(X_features) >= 2:  # At least Month + 1 site feature
                y = monthly_data['Volumetric Quantity'].values
                X_all = np.hstack(X_features)

                print(
                    f"  Testing combined model with {len(feature_names)} features on {len(monthly_data)} monthly rows...")

                # Train ensemble models
                self.train_and_test_model(
                    X_all, y,
                    'SiteLevel_Monthly_RandomForest_All',
                    'RandomForest',
                    feature_names
                )
                self.train_and_test_model(
                    X_all, y,
                    'SiteLevel_Monthly_GradientBoosting_All',
                    'GradientBoosting',
                    feature_names
                )

    def train_and_test_model(self, X, y, model_name, model_type, features):
        """Train and test a model with feature-aware R² calculation"""
        try:
            if len(X) < 10:
                return

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=42
            )

            if model_type == 'Linear':
                model = LinearRegression()
                description = f"Linear Regression using {', '.join(features)}"
            elif model_type == 'Polynomial':
                model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
                description = f"Polynomial Regression (degree 2) using {', '.join(features)}"
            elif model_type == 'RandomForest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                description = f"Random Forest (100 trees) using {', '.join(features)}"
            elif model_type == 'GradientBoosting':
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                description = f"Gradient Boosting (100 estimators) using {', '.join(features)}"
            else:
                return

            model.fit(X_train, y_train)

            # Calculate standard R² (with all features)
            y_pred = model.predict(X_test)
            r2_full = r2_score(y_test, y_pred)

            # ⚠️ CRITICAL: Don't create models with negative R² (worse than mean)
            if r2_full < 0:
                print(f"  ✗ {model_name}: R² = {r2_full:.4f} (negative - skipping)")
                return

            # Identify derived features
            from historic_model_training import identify_derived_features
            base_features, derived_features = identify_derived_features(features)

            # Calculate imputed R² if there are derived features
            r2_imputed = r2_full

            if len(derived_features) > 0:
                X_test_imputed = X_test.copy()

                for i, feature in enumerate(features):
                    if feature in derived_features:
                        median_val = np.median(X_train[:, i])
                        X_test_imputed[:, i] = median_val

                y_pred_imputed = model.predict(X_test_imputed)
                r2_imputed = r2_score(y_test, y_pred_imputed)

                # Also check imputed R²
                if r2_imputed < 0:
                    print(f"  ✗ {model_name}: R²(imputed) = {r2_imputed:.4f} (negative - skipping)")
                    return

            self.models[model_name] = {
                'model': model,
                'features': features,
                'base_features': base_features,
                'derived_features': derived_features,
                'r2': r2_full,
                'r2_full': r2_full,
                'r2_imputed': r2_imputed,
                'type': model_type,
                'has_derived': len(derived_features) > 0,
                'description': f"{description}. R²(full)={r2_full:.3f}, R²(imputed)={r2_imputed:.3f}" if len(
                    derived_features) > 0 else f"{description}. R²={r2_full:.3f}"
            }

            if len(derived_features) > 0:
                print(f"  ✓ {model_name}: R²(full) = {r2_full:.4f}, R²(imputed) = {r2_imputed:.4f}")
            else:
                print(f"  ✓ {model_name}: R² = {r2_full:.4f}")

        except Exception as e:
            print(f"  ✗ {model_name} failed: {str(e)}")

    def train_site_seasonal_model(self):
        """Train model using site averages with seasonal patterns"""
        monthly_data = self.data_df[self.data_df['Timeframe'] == 'Monthly'].copy()

        if len(monthly_data) == 0:
            return

        site_avgs = monthly_data.groupby('Site identifier')['Volumetric Quantity'].mean().to_dict()

        predictions = []
        actuals = []

        for idx, row in monthly_data.iterrows():
            site_id = row['Site identifier']
            month = row['Month']

            if site_id in site_avgs and month:
                brand = row.get('Brand', 'Overall')
                pattern_key = brand if brand in self.seasonal_patterns else 'Overall'

                if pattern_key in self.seasonal_patterns and month in self.seasonal_patterns[pattern_key]:
                    seasonal_factor = self.seasonal_patterns[pattern_key][month]
                    prediction = site_avgs[site_id] * seasonal_factor
                    predictions.append(prediction)
                    actuals.append(row['Volumetric Quantity'])

        if len(predictions) > 0:
            r2 = r2_score(actuals, predictions)
            self.models['Site_Seasonal'] = {
                'model': 'site_seasonal',
                'features': ['Site identifier', 'Month'],
                'r2': r2,
                'site_avgs': site_avgs,
                'type': 'Site_Seasonal',
                'description': f'Site-specific average adjusted by seasonal patterns. Tested on {len(predictions)} monthly data points with actual values.'
            }
            print(f"\n  ✓ Site_Seasonal: R² = {r2:.4f} (tested on {len(predictions)} points)")

    def create_seasonal_variants(self):
        """
        Create seasonal variants of site-level models that DON'T already include Month.

        Note: New SiteLevel_Monthly_* models already include Month, so they don't need variants.
        This is for backwards compatibility with any old annual-only models.
        """
        monthly_data = self.data_df[self.data_df['Timeframe'] == 'Monthly'].copy()

        if len(monthly_data) == 0 or len(self.seasonal_patterns.get('Overall', {})) == 0:
            return

        # Only create variants for site-level models that DON'T already have Month
        site_level_models = {k: v for k, v in self.models.items()
                             if k.startswith('SiteLevel_')
                             and 'Month' not in v.get('features', [])}

        if len(site_level_models) == 0:
            return  # All site-level models already include Month

        print("\n📊 Creating seasonal variants for annual-only site-level models...")

        for base_model_name, base_model_info in site_level_models.items():
            try:
                predictions = []
                actuals = []

                for idx, row in monthly_data.iterrows():
                    month = row.get('Month')
                    if not month:
                        continue

                    base_pred = self.apply_model_to_row(row, base_model_info)
                    if base_pred is None or base_pred <= 0:
                        continue

                    brand = row.get('Brand', 'Overall')
                    pattern_key = brand if brand in self.seasonal_patterns else 'Overall'

                    if pattern_key in self.seasonal_patterns and month in self.seasonal_patterns[pattern_key]:
                        seasonal_factor = self.seasonal_patterns[pattern_key][month]
                        # Divide by 12 to get monthly average, then apply seasonal factor
                        monthly_pred = (base_pred / 12) * seasonal_factor

                        predictions.append(monthly_pred)
                        actuals.append(row['Volumetric Quantity'])

                if len(predictions) >= 10:
                    r2 = r2_score(actuals, predictions)
                    seasonal_model_name = f"{base_model_name}_Seasonal"

                    self.models[seasonal_model_name] = {
                        'model': base_model_info['model'],
                        'features': base_model_info['features'] + ['Month'],
                        'r2': r2,
                        'r2_full': r2,
                        'r2_imputed': r2,
                        'type': f"{base_model_info['type']}_Seasonal",
                        'base_model': base_model_name,
                        'has_derived': False,
                        'description': f"{base_model_info.get('description', '')} with seasonal adjustment. Predicts annual total then applies monthly seasonality. Tested on {len(predictions)} monthly data points."
                    }
                    print(f"  ✓ {seasonal_model_name}: R² = {r2:.4f} (tested on {len(predictions)} monthly points)")

            except Exception as e:
                print(f"  ✗ Failed to create seasonal variant of {base_model_name}: {str(e)}")

    def train_annual_average_model(self):
        """Train model using annual averages"""
        annual_data = self.data_df[self.data_df['Timeframe'] == 'Annual'].copy()

        monthly_data = self.data_df[self.data_df['Timeframe'] == 'Monthly'].copy()
        site_month_counts = monthly_data.groupby('Site identifier').size()
        complete_sites = site_month_counts[site_month_counts == 12].index

        annual_values = list(annual_data['Volumetric Quantity'].values)

        for site in complete_sites:
            site_data = monthly_data[monthly_data['Site identifier'] == site]
            annual_total = site_data['Volumetric Quantity'].sum()
            annual_values.append(annual_total)

        if len(annual_values) > 0:
            avg_annual = np.mean(annual_values)

            predictions = [avg_annual] * len(annual_values)
            actuals = annual_values

            ss_res = sum((actual - pred) ** 2 for actual, pred in zip(actuals, predictions))
            ss_tot = sum((actual - np.mean(actuals)) ** 2 for actual in actuals)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            self.models['Annual_Average'] = {
                'model': 'annual_average',
                'features': ['Annual'],
                'r2': r2,
                'average': avg_annual,
                'type': 'Annual_Average',
                'description': f'Simple average of annual totals. Calculated from {len(annual_values)} annual data points (including sites with complete 12-month data).'
            }
            print(f"  ✓ Annual_Average: R² = {r2:.4f} (calculated from {len(annual_values)} annual totals)")

    def get_quality_score_for_row(self, row, model_name):
        """
        Get data quality score (R²) for a prediction based on available features.

        Parameters:
        -----------
        row : pandas Series
            Row being predicted
        model_name : str
            Name of model used for prediction

        Returns:
        --------
        float : Quality score (R²)
        """
        if model_name not in self.models:
            return 0.3  # Default low quality

        model_info = self.models[model_name]
        model_r2 = model_info.get('r2', 0.3)

        # Get features used by this model
        model_features = model_info.get('features', [])

        # Determine which features are available in this row
        available_features = []
        for feature in model_features:
            if feature in ['Month', 'Historic VQ']:
                # Always consider these available
                available_features.append(feature)
            elif feature in row.index and pd.notna(row[feature]):
                available_features.append(feature)

        # Look up feature-set-specific R²
        if hasattr(self, 'historic_trainer') and self.historic_trainer:
            feature_key = tuple(sorted(available_features))

            # Try to get feature-set-specific R²
            feature_r2 = self.historic_trainer.feature_set_r2.get(feature_key)

            if feature_r2 is not None:
                return feature_r2

        # Fallback to model's overall R²
        return model_r2

    def extrapolate_blank_rows(self):
        """Fill in blank rows using scenario-based model selection"""
        print("\n" + "=" * 60)
        print("MAKING PREDICTIONS (SCENARIO-BASED)")
        print("=" * 60)

        # Initialize scenario-based selector
        self.scenario_selector = ScenarioBasedModelSelector(
            data_df=self.data_df,
            blank_df=self.blank_df,
            models=self.models,
            historic_manager=self.historic_manager,
            seasonal_patterns=self.seasonal_patterns
        )

        # Find best model for each scenario
        self.scenario_selector.find_best_models_for_all_scenarios()

        results = []

        for idx, row in self.blank_df.iterrows():
            # Identify row's scenario
            scenario = self.scenario_selector.identify_scenario(row)

            # Get best model for this scenario
            best_model_name, scenario_r2 = self.scenario_selector.get_best_model_for_row(row)

            # Make prediction
            if best_model_name:
                prediction = self._apply_model_to_row_safe(row, best_model_name)

                # UPDATE: Increment predictions_made counter
                if best_model_name in self.models:
                    if 'predictions_made' not in self.models[best_model_name]:
                        self.models[best_model_name]['predictions_made'] = 0
                    self.models[best_model_name]['predictions_made'] += 1
            else:
                best_model_name = None
                scenario_r2 = 0.3
                prediction = None

            # Ultimate fallback if still no prediction
            if prediction is None or prediction <= 0:
                if row['Timeframe'] == 'Monthly':
                    monthly_data = self.data_df[self.data_df['Timeframe'] == 'Monthly']
                    prediction = monthly_data['Volumetric Quantity'].mean()
                    best_model_name = 'Overall_Monthly_Average'
                    scenario_r2 = 0.3
                elif row['Timeframe'] == 'Annual':
                    annual_data = self.data_df[self.data_df['Timeframe'] == 'Annual']
                    if len(annual_data) > 0:
                        prediction = annual_data['Volumetric Quantity'].mean()
                    else:
                        prediction = self.data_df['Volumetric Quantity'].mean()
                    best_model_name = 'Overall_Annual_Average'
                    scenario_r2 = 0.3
                else:
                    prediction = self.data_df['Volumetric Quantity'].mean()
                    best_model_name = 'Overall_Average'
                    scenario_r2 = 0.3

            months_bucket, has_turnover, has_historic = scenario

            results.append({
                'index': idx,
                'prediction': prediction,
                'model': best_model_name,
                'r2': scenario_r2,  # Scenario-specific R²!
                'context_months': months_bucket,
                'has_turnover': has_turnover,
                'context_has_historic': has_historic
            })

        # Print summary
        self.scenario_selector.print_scenario_summary()
        print(f"\n✓ Made {len(results)} predictions using scenario-based selection")

        return results

    def predict_monthly_row(self, row):
        """Predict a single monthly row using feature-aware R² selection"""
        site_id = row.get('Site identifier')
        month = row.get('Month')
        best_r2 = -float('inf')
        prediction = None
        best_model_name = None

        # Try all applicable models and select best by EFFECTIVE R²
        candidates = []

        # Historic models (if available)
        if self.has_historic_data and self.historic_manager:
            # Try rule-based historic models
            for model_name in ['LastYear_TurnoverAdjusted', 'LastYear_Direct',
                               'MultiYear_Average', 'LastYear_GrowthAdjusted']:
                if model_name in self.models:
                    pred = self.apply_historic_model(row, model_name)
                    if pred and pred > 0:
                        effective_r2 = self.models[model_name]['r2']
                        candidates.append((pred, effective_r2, model_name))

            # Try ML models with historic features
            for model_name in ['Historic_Monthly_RandomForest', 'Historic_Monthly_GradientBoosting']:
                if model_name in self.models:
                    pred = self.apply_ml_model_historic(row, model_name)
                    if pred and pred > 0:
                        effective_r2 = get_effective_r2_for_row(self.models[model_name], row)
                        candidates.append((pred, effective_r2, model_name))

        # Standard models
        # Use Site_Seasonal model (works for both existing and new sites)
        if 'Site_Seasonal' in self.models and month:
            model_info = self.models['Site_Seasonal']
            site_avgs = model_info.get('site_avgs', {})

            # Use site-specific average if available, otherwise use overall average
            if site_id in site_avgs:
                site_avg = site_avgs[site_id]
            elif len(site_avgs) > 0:
                # New site - use overall average from all sites
                site_avg = np.mean(list(site_avgs.values()))
            else:
                site_avg = None

            if site_avg:
                brand = row.get('Brand', 'Overall')
                pattern_key = brand if brand in self.seasonal_patterns else 'Overall'

                if pattern_key in self.seasonal_patterns and month in self.seasonal_patterns[pattern_key]:
                    seasonal_factor = self.seasonal_patterns[pattern_key][month]
                    pred = site_avg * seasonal_factor
                    effective_r2 = model_info['r2']
                    candidates.append((pred, effective_r2, 'Site_Seasonal'))

        # Try seasonal variants of site-level models
        for model_name, model_info in self.models.items():
            if model_name.endswith('_Seasonal') and model_name.startswith('SiteLevel_'):
                base_model_info = self.models.get(model_info.get('base_model'))
                if base_model_info:
                    base_pred = self.apply_model_to_row(row, base_model_info)
                    if base_pred and base_pred > 0 and month:
                        brand = row.get('Brand', 'Overall')
                        pattern_key = brand if brand in self.seasonal_patterns else 'Overall'
                        if pattern_key in self.seasonal_patterns and month in self.seasonal_patterns[pattern_key]:
                            seasonal_factor = self.seasonal_patterns[pattern_key][month]
                            pred = (base_pred / 12) * seasonal_factor
                            effective_r2 = get_effective_r2_for_row(model_info, row)
                            candidates.append((pred, effective_r2, model_name))

        # Try monthly models with EFFECTIVE R²
        for model_name, model_info in self.models.items():
            if model_name.startswith('Monthly_'):
                pred = self.apply_model_to_row(row, model_info)
                if pred and pred > 0:
                    # Use effective R² based on feature availability
                    effective_r2 = get_effective_r2_for_row(model_info, row)
                    candidates.append((pred, effective_r2, model_name))

        # Try historic site-level models
        for model_name, model_info in self.models.items():
            if model_name.startswith('Historic_SiteLevel_Monthly_'):
                pred = self.apply_model_to_row(row, model_info)
                if pred and pred > 0:
                    effective_r2 = get_effective_r2_for_row(model_info, row)
                    candidates.append((pred, effective_r2, model_name))

        # Monthly average fallback
        if month:
            month_data = self.data_df[
                (self.data_df['Month'] == month) &
                (self.data_df['Timeframe'] == 'Monthly')
                ]
            if len(month_data) > 0:
                pred = month_data['Volumetric Quantity'].mean()
                candidates.append((pred, 0.6, f'Monthly_Average_{month}'))

        # Select best by EFFECTIVE R²
        if candidates:
            best = max(candidates, key=lambda x: x[1])
            return best[0], best[2], best[1]

        return None, None, -float('inf')

    def predict_annual_row(self, row):
        """Predict a single annual row using feature-aware R² selection"""
        candidates = []

        # Try all applicable models
        for model_name, model_info in self.models.items():
            if (model_name.startswith('SiteLevel_') or
                    model_name.startswith('Historic_SiteLevel_') or
                    model_name == 'Annual_Average'):

                if model_name.endswith('_Seasonal'):
                    continue

                if model_info.get('type') == 'rule_based':
                    continue  # Rule-based are for monthly

                pred = self.apply_model_to_row(row, model_info)
                if pred and pred > 0:
                    # Use effective R² based on feature availability
                    effective_r2 = get_effective_r2_for_row(model_info, row)
                    candidates.append((pred, effective_r2, model_name))

        # Select best by EFFECTIVE R²
        if candidates:
            best = max(candidates, key=lambda x: x[1])
            return best[0], best[2], best[1]

        return None, None, -float('inf')

    def apply_historic_model(self, row, model_name):
        """Apply a rule-based historic model using period-based matching"""
        if not self.has_historic_data:
            return None

        site = row['Site identifier']
        month = row['Month']
        row_date = pd.to_datetime(row['Date from'])

        # Get most recent historic period ending before this row's date
        recent_period = self.historic_manager.get_most_recent_period(site, row_date)

        if not recent_period:
            return None  # No historic data for this site

        if model_name == 'LastYear_TurnoverAdjusted':
            # Step 1: Get VQ from period (monthly if available, annual otherwise)
            period_vq = self.historic_manager.get_period_vq_for_month(recent_period, month)

            if not period_vq:
                # Convert annual to monthly
                annual = self.historic_manager.get_period_annual_total(recent_period)
                if annual and month:
                    # Calculate monthly average
                    monthly_avg = annual / recent_period.months_covered

                    # Apply seasonal pattern
                    brand = row.get('Brand', 'Overall')
                    pattern_key = brand if brand in self.seasonal_patterns else 'Overall'
                    if pattern_key in self.seasonal_patterns and month in self.seasonal_patterns[pattern_key]:
                        seasonal_factor = self.seasonal_patterns[pattern_key][month]
                        period_vq = monthly_avg * seasonal_factor

            if not period_vq:
                return None

            # Step 2: Adjust by turnover change
            current_turnover = row.get('Turnover')
            period_turnover = self.historic_manager.get_period_feature(recent_period, 'Turnover')

            if pd.notna(current_turnover) and period_turnover and period_turnover != 0:
                adjustment = float(current_turnover) / float(period_turnover)
                return period_vq * adjustment
            else:
                # No turnover data, return unadjusted
                return period_vq

        elif model_name == 'LastYear_Direct':
            # Try to get specific month from period
            month_vq = self.historic_manager.get_period_vq_for_month(recent_period, month)

            if month_vq:
                return month_vq

            # Convert annual to monthly
            annual = self.historic_manager.get_period_annual_total(recent_period)
            if annual and month:
                monthly_avg = annual / recent_period.months_covered

                # Apply seasonal pattern
                brand = row.get('Brand', 'Overall')
                pattern_key = brand if brand in self.seasonal_patterns else 'Overall'
                if pattern_key in self.seasonal_patterns and month in self.seasonal_patterns[pattern_key]:
                    seasonal_factor = self.seasonal_patterns[pattern_key][month]
                    return monthly_avg * seasonal_factor

            return None

        elif model_name == 'MultiYear_Average':
            # Get all periods for this site
            all_periods = self.historic_manager.get_all_periods(site)

            # Filter to periods ending before current row
            relevant_periods = [p for p in all_periods if p.end_date < row_date]

            if len(relevant_periods) < 1:
                return None

            # Try to get monthly values from each period
            monthly_values = []
            for period in relevant_periods:
                month_vq = self.historic_manager.get_period_vq_for_month(period, month)

                if month_vq:
                    monthly_values.append(month_vq)
                else:
                    # Convert annual to monthly
                    annual = self.historic_manager.get_period_annual_total(period)
                    if annual and month:
                        monthly_avg = annual / period.months_covered

                        brand = row.get('Brand', 'Overall')
                        pattern_key = brand if brand in self.seasonal_patterns else 'Overall'
                        if pattern_key in self.seasonal_patterns and month in self.seasonal_patterns[pattern_key]:
                            seasonal_factor = self.seasonal_patterns[pattern_key][month]
                            monthly_values.append(monthly_avg * seasonal_factor)

            if monthly_values:
                return sum(monthly_values) / len(monthly_values)

            return None

        elif model_name == 'LastYear_GrowthAdjusted':
            # Need at least 2 periods for growth calculation
            all_periods = self.historic_manager.get_all_periods(site)
            relevant_periods = [p for p in all_periods if p.end_date < row_date]

            if len(relevant_periods) < 2:
                return None

            # Get most recent two periods
            recent_two = relevant_periods[-2:]

            # Calculate growth rate
            prev_total = recent_two[0].get_normalized_annual()
            recent_total = recent_two[1].get_normalized_annual()

            if not prev_total or not recent_total or prev_total == 0:
                return None

            growth_rate = (recent_total - prev_total) / prev_total

            # Get base prediction from most recent period
            month_vq = self.historic_manager.get_period_vq_for_month(recent_two[1], month)

            if not month_vq:
                annual = self.historic_manager.get_period_annual_total(recent_two[1])
                if annual and month:
                    monthly_avg = annual / recent_two[1].months_covered

                    brand = row.get('Brand', 'Overall')
                    pattern_key = brand if brand in self.seasonal_patterns else 'Overall'
                    if pattern_key in self.seasonal_patterns and month in self.seasonal_patterns[pattern_key]:
                        seasonal_factor = self.seasonal_patterns[pattern_key][month]
                        month_vq = monthly_avg * seasonal_factor

            if month_vq:
                # Apply growth rate
                return month_vq * (1 + growth_rate)

            return None

        elif model_name.startswith('SimilarSites_'):
            # Use similar sites method
            return self.apply_similar_sites_prediction(row, model_name)

        return None

    def apply_similar_sites_prediction(self, row, model_name):
        """
        Apply similar sites method to predict VQ.
        Uses growth rates from similar sites to adjust historic VQ.
        """
        if model_name not in self.models:
            return None

        model_info = self.models[model_name]
        method_config = model_info.get('method_config')

        if not method_config:
            return None

        site = row['Site identifier']
        month = row['Month']
        row_date = pd.to_datetime(row['Date from'], dayfirst=True)

        # Get this site's historic period
        historic_period = self.historic_manager.get_most_recent_period(site, row_date)
        if not historic_period:
            return None  # No historic data for this site

        # Find similar sites
        similar_sites = self.historic_trainer._find_similar_sites(
            site,
            [s for s in self.df['Site identifier'].unique() if s != site],
            self.df,
            method_config
        )

        if len(similar_sites) == 0:
            return None

        # Get growth rates from similar sites
        growth_rates = []

        for sim_site in similar_sites:
            # Get similar site's current and historic VQ for this month
            sim_site_data = self.df[
                (self.df['Site identifier'] == sim_site) &
                (self.df['Month'] == month) &
                (self.df['Volumetric Quantity'].notna())
                ]

            if len(sim_site_data) == 0:
                continue

            sim_current_vq = sim_site_data.iloc[0]['Volumetric Quantity']
            sim_current_date = pd.to_datetime(sim_site_data.iloc[0]['Date from'], dayfirst=True)

            # Get similar site's historic period
            sim_historic_period = self.historic_manager.get_most_recent_period(sim_site, sim_current_date)
            if not sim_historic_period:
                continue

            # Get historic VQ for this month
            sim_historic_vq = self.historic_manager.get_period_vq_for_month(sim_historic_period, month)

            if not sim_historic_vq:
                # Convert annual to monthly
                sim_annual = self.historic_manager.get_period_annual_total(sim_historic_period)
                if sim_annual:
                    sim_historic_vq = sim_annual / sim_historic_period.months_covered

            if sim_historic_vq and sim_historic_vq > 0:
                growth_rate = sim_current_vq / sim_historic_vq
                growth_rates.append(growth_rate)

        if len(growth_rates) == 0:
            return None

        # Get target site's historic VQ
        target_historic_vq = self.historic_manager.get_period_vq_for_month(historic_period, month)

        if not target_historic_vq:
            # Convert annual to monthly
            target_annual = self.historic_manager.get_period_annual_total(historic_period)
            if target_annual:
                # Calculate monthly average
                monthly_avg = target_annual / historic_period.months_covered

                # Apply seasonal pattern
                brand = row.get('Brand', 'Overall')
                pattern_key = brand if brand in self.seasonal_patterns else 'Overall'
                if pattern_key in self.seasonal_patterns and month in self.seasonal_patterns[pattern_key]:
                    seasonal_factor = self.seasonal_patterns[pattern_key][month]
                    target_historic_vq = monthly_avg * seasonal_factor

        if not target_historic_vq:
            return None

        # Apply average growth from similar sites
        avg_growth = np.mean(growth_rates)
        prediction = target_historic_vq * avg_growth

        return prediction

    def apply_ml_model_historic(self, row, model_name):
        """Apply an ML model trained with historic features"""
        if model_name not in self.models:
            return None

        model_info = self.models[model_name]

        try:
            # Build feature vector
            feature_values = []

            for feature in model_info['features']:
                if feature == 'Month':
                    if 'Month' in self.historic_trainer.categorical_encoders:
                        encoder = self.historic_trainer.categorical_encoders['Month']
                        month_val = row.get('Month', 'Unknown')
                        try:
                            encoded = encoder.transform([month_val])[0]
                        except:
                            encoded = 0
                        feature_values.append(encoded)
                elif feature in row.index:
                    val = row[feature]
                    if pd.notna(val):
                        feature_values.append(float(val))
                    else:
                        # Use median from training data
                        feature_values.append(0)
                else:
                    return None

            if len(feature_values) != len(model_info['features']):
                return None

            X = np.array(feature_values).reshape(1, -1)
            return model_info['model'].predict(X)[0]

        except:
            return None

    def apply_model_to_row(self, row, model_info):
        """Apply a trained model to make a prediction for a single row"""
        if model_info['type'] == 'Annual_Average':
            return model_info['average']

        features = model_info['features']
        feature_values = []

        for feature in features:
            if feature == 'Month':
                if pd.notna(row.get('Month')) and row.get('Month') != '':
                    months_list = ['January', 'February', 'March', 'April', 'May', 'June',
                                   'July', 'August', 'September', 'October', 'November', 'December']
                    month_val = months_list.index(row['Month']) if row['Month'] in months_list else 0
                    feature_values.append(month_val)
                else:
                    feature_values.append(0)
            elif feature in row.index:
                val = row[feature]
                if pd.notna(val):
                    if feature in self.categorical_features:
                        unique_vals = self.data_df[feature].dropna().unique()
                        if val in unique_vals:
                            feature_values.append(list(unique_vals).index(val))
                        else:
                            feature_values.append(0)
                    else:
                        feature_values.append(float(val))
                else:
                    median_val = self.data_df[feature].median()
                    feature_values.append(median_val if pd.notna(median_val) else 0)
            else:
                return None

        if len(feature_values) != len(features):
            return None

        X = np.array(feature_values).reshape(1, -1)
        return model_info['model'].predict(X)[0]

    def _apply_model_to_row_safe(self, row, model_name):
        """
        Safely apply a model to a row, handling different model types.
        Returns prediction or None if failed.
        """
        if model_name not in self.models:
            return None

        model_info = self.models[model_name]

        try:
            # Rule-based models
            if model_info.get('type') in ['LastYear_Direct', 'LastYear_TurnoverAdjusted',
                                          'MultiYear_Average', 'LastYear_GrowthAdjusted']:
                return self.apply_historic_model(row, model_name)

            # Site_Seasonal
            elif model_info.get('type') == 'Site_Seasonal':
                return self._predict_site_seasonal_safe(row, model_info)

            # Annual_Average
            elif model_info.get('type') == 'Annual_Average':
                return model_info.get('average')

            # ML models
            else:
                return self.apply_model_to_row(row, model_info)

        except Exception as e:
            return None

    def _predict_site_seasonal_safe(self, row, model_info):
        """Safely predict using Site_Seasonal model"""
        site_id = row.get('Site identifier')
        month = row.get('Month')

        if not site_id or not month:
            return None

        site_avgs = model_info.get('site_avgs', {})

        # Use site average if available, otherwise overall average
        if site_id in site_avgs:
            site_avg = site_avgs[site_id]
        elif len(site_avgs) > 0:
            site_avg = np.mean(list(site_avgs.values()))
        else:
            return None

        # Apply seasonal factor
        brand = row.get('Brand', 'Overall')
        pattern_key = brand if brand in self.seasonal_patterns else 'Overall'

        if pattern_key in self.seasonal_patterns and month in self.seasonal_patterns[pattern_key]:
            seasonal_factor = self.seasonal_patterns[pattern_key][month]
            return site_avg * seasonal_factor

        return site_avg

    def create_output_dataframe(self, extrapolation_results):
        """Combine original data with extrapolated values (with data availability context)"""
        output_df = self.df.copy()

        # Initialize columns with proper dtypes
        output_df['Data integrity'] = 'Actual'
        output_df['Estimation Method'] = ''
        output_df['Data Quality'] = ''
        output_df['Data Quality Score'] = np.nan
        output_df['Data_Months'] = np.nan
        output_df['Data_Has_Turnover'] = ''
        output_df['Data_Has_Historic'] = ''
        output_df['Data_Availability'] = ''

        for result in extrapolation_results:
            idx = result['index']
            output_df.at[idx, 'Volumetric Quantity'] = result['prediction']
            output_df.at[idx, 'Data integrity'] = 'Estimated'
            output_df.at[idx, 'Estimation Method'] = result['model']
            output_df.at[idx, 'Data Quality'] = f"{result['r2']:.3f}"
            output_df.at[idx, 'Data Quality Score'] = result['r2']

            # Data availability details
            months = result.get('context_months', 0)
            has_turnover = result.get('has_turnover', False)
            has_historic = result.get('context_has_historic', False)

            output_df.at[idx, 'Data_Months'] = months
            output_df.at[idx, 'Data_Has_Turnover'] = 'Yes' if has_turnover else 'No'
            output_df.at[idx, 'Data_Has_Historic'] = 'Yes' if has_historic else 'No'

            # Create human-readable description
            # Check if site is truly new (0 months AND not in historic data)
            site_id = self.df.at[idx, 'Site identifier'] if 'Site identifier' in self.df.columns else None
            is_new_site = False

            if months == 0 and site_id:
                # Check if site exists in historic data
                is_new_site = True
                if self.historic_manager and hasattr(self.historic_manager, 'site_periods'):
                    if site_id in self.historic_manager.site_periods:
                        if len(self.historic_manager.site_periods[site_id]) > 0:
                            is_new_site = False

            # Build description with ACTUAL month count
            parts = []
            if is_new_site:
                parts.append("New site")
            else:
                # Calculate actual month count for this site
                site_id = self.df.at[idx, 'Site identifier']
                actual_months = 0
                if site_id in self.data_df['Site identifier'].values:
                    site_monthly = self.data_df[
                        (self.data_df['Site identifier'] == site_id) &
                        (self.data_df['Timeframe'] == 'Monthly')
                        ]
                    actual_months = len(site_monthly)

                parts.append(f"{actual_months} months current data")

            if has_turnover:
                parts.append("Turnover")
            if has_historic:
                parts.append("Historic")

            output_df.at[idx, 'Data_Availability'] = ', '.join(parts)

        return output_df

    def create_model_testing_matrix(self):
        """
        Create matrix showing R² for every model tested in every data availability context.

        Format:
        Rows = Data availability contexts (exact months, not bucketed)
        Columns = Models
        Values = R² scores from scenario testing

        Purpose: See at a glance which models work best in which contexts
        """
        if not hasattr(self, 'scenario_selector') or not self.scenario_selector:
            return None

        # Get all scenarios that were tested
        if not hasattr(self.scenario_selector, 'scenario_test_results'):
            return None

        matrix_data = []

        # Sort scenarios by months for logical ordering
        sorted_scenarios = sorted(
            self.scenario_selector.scenario_test_results.keys(),
            key=lambda x: (x[0], not x[1], not x[2])  # months, turnover, historic
        )

        for scenario in sorted_scenarios:
            months, has_turnover, has_historic = scenario
            test_results = self.scenario_selector.scenario_test_results[scenario]

            # Build data availability description
            parts = [f"{months} months"]
            if has_turnover:
                parts.append("Turnover")
            if has_historic:
                parts.append("Historic")

            data_availability = ", ".join(parts)

            # Get test pool size
            test_pool = test_results.get('test_pool_size', 0)

            # Start row with description and test pool
            row = {
                'Data_Availability': data_availability,
                'Months': months,
                'Test_Pool': test_pool
            }

            # Add R² for each model
            model_results = test_results.get('model_r2s', {})
            for model_name, r2 in model_results.items():
                row[model_name] = round(r2, 4) if r2 is not None else ''

            matrix_data.append(row)

        if not matrix_data:
            return None

        # Create DataFrame
        matrix_df = pd.DataFrame(matrix_data)

        # Reorder columns: Data_Availability, Months, Test_Pool, then all models
        base_cols = ['Data_Availability', 'Months', 'Test_Pool']
        model_cols = [col for col in matrix_df.columns if col not in base_cols]
        matrix_df = matrix_df[base_cols + sorted(model_cols)]

        return matrix_df

    def create_model_performance_sheet(self):
        """
        Create detailed model performance sheet showing ALL models
        (including rule-based and those with negative R²)
        """
        performance_data = []

        for model_name, model_info in self.models.items():
            r2 = model_info.get('r2', 0)
            r2_imputed = model_info.get('r2_imputed', r2)
            features = model_info.get('features', [])
            description = model_info.get('description', '')
            model_type = model_info.get('type', 'ML')
            predictions_made = model_info.get('predictions_made', 0)

            # Determine type label
            if model_type in ['LastYear_Direct', 'LastYear_TurnoverAdjusted',
                              'MultiYear_Average', 'LastYear_GrowthAdjusted']:
                type_label = 'Rule-Based (Historic)'
            elif model_type == 'Site_Seasonal':
                type_label = 'Rule-Based (Seasonal)'
            elif model_type == 'Annual_Average':
                type_label = 'Rule-Based (Average)'
            else:
                type_label = 'ML Model'

            # Count how many data availability contexts this model won/was viable in
            scenarios_matched = 0
            scenarios_viable = 0

            if hasattr(self, 'scenario_selector') and self.scenario_selector:
                for scenario, (best_model, _) in self.scenario_selector.scenario_best_models.items():
                    if best_model == model_name:
                        scenarios_matched += 1

                # Count viable scenarios (R² > 0)
                if hasattr(self.scenario_selector, 'scenario_test_results'):
                    for scenario, results in self.scenario_selector.scenario_test_results.items():
                        model_r2s = results.get('model_r2s', {})
                        if model_name in model_r2s and model_r2s[model_name] is not None:
                            if model_r2s[model_name] > 0:
                                scenarios_viable += 1

            # Format R² values
            r2_display = r2

            # Format features
            features_str = ', '.join(features) if features else 'N/A'

            performance_data.append({
                'Model': model_name,
                'Type': type_label,
                'Training_R²': r2_display,
                'R²_Imputed': r2_imputed if r2_imputed != r2 else '',
                'Scenarios_Matched': scenarios_matched,
                'Scenarios_Viable': scenarios_viable,
                'Features_Used': features_str,
                'Description': description,
                'Predictions_Made': predictions_made
            })

        # Create DataFrame
        performance_df = pd.DataFrame(performance_data)

        # Sort: by Scenarios_Matched (descending), then by Training_R²
        performance_df = performance_df.sort_values(
            ['Scenarios_Matched', 'Training_R²'],
            ascending=[False, False]
        )

        return performance_df

    def create_data_availability_sheet(self):
        """
        Create data availability summary showing which model was chosen for each context
        """
        if not hasattr(self, 'scenario_selector') or not self.scenario_selector:
            return None

        availability_data = []

        for scenario, (model_name, r2) in self.scenario_selector.scenario_best_models.items():
            months, has_turnover, has_historic = scenario

            # Count how many rows used this data availability context
            context_row_count = sum(
                1 for _, row in self.blank_df.iterrows()
                if self.scenario_selector.identify_scenario(row) == scenario
            )

            # Get test pool size
            test_pool = 0
            if hasattr(self.scenario_selector, 'scenario_test_results'):
                test_results = self.scenario_selector.scenario_test_results.get(scenario, {})
                test_pool = test_results.get('test_pool_size', 0)

            # Build data availability description
            parts = [f"{months} months"]
            if has_turnover:
                parts.append("Turnover")
            if has_historic:
                parts.append("Historic")

            availability_data.append({
                'Data_Availability': ', '.join(parts),
                'Months': months,
                'Has_Turnover': 'Yes' if has_turnover else 'No',
                'Has_Historic': 'Yes' if has_historic else 'No',
                'Test_Pool_Sites': test_pool,
                'Best_Model': model_name,
                'R²_Score': r2,
                'Rows_Using_This': context_row_count
            })

        availability_df = pd.DataFrame(availability_data)
        availability_df = availability_df.sort_values(['Months', 'Has_Turnover', 'Has_Historic'])

        return availability_df

    def create_training_data_sheet(self):
        """
        Create training data sheet showing which models can use each training row.

        Format: Each row shows the training data and TRUE/FALSE for each model indicating
        whether that model can use this row (has all required features).
        """
        # Get actual training data (non-blank rows)
        training_rows = self.data_df[self.data_df['Timeframe'] == 'Monthly'].copy()

        if len(training_rows) == 0:
            return None

        # Start with core columns
        output_data = []

        for idx, row in training_rows.iterrows():
            row_data = {
                'Site_Identifier': row.get('Site identifier', ''),
                'Month': row.get('Month', ''),
                'VQ': row.get('Volumetric Quantity', ''),
            }

            # Add feature columns that exist
            if 'Turnover' in row.index:
                row_data['Turnover'] = row.get('Turnover', '')
            if 'Brand' in row.index:
                row_data['Brand'] = row.get('Brand', '')

            # Add any selected features
            if hasattr(self, 'selected_features') and self.selected_features:
                for feat in self.selected_features:
                    if feat in row.index and feat not in row_data:
                        row_data[feat] = row.get(feat, '')

            # For each model, check if it can use this row
            for model_name, model_info in self.models.items():
                required_features = model_info.get('features', [])

                # Check if row has all required features
                can_use = True

                for feature in required_features:
                    # Handle special cases
                    if feature == 'Month':
                        if pd.isna(row.get('Month')) or row.get('Month') == '':
                            can_use = False
                            break
                    elif feature == 'Site identifier':
                        if pd.isna(row.get('Site identifier')) or row.get('Site identifier') == '':
                            can_use = False
                            break
                    elif feature in row.index:
                        if pd.isna(row.get(feature)):
                            can_use = False
                            break
                    else:
                        # Feature doesn't exist in row
                        can_use = False
                        break

                # Also check if model needs historic data
                model_type = model_info.get('type', '')
                if model_type in ['LastYear_Direct', 'LastYear_TurnoverAdjusted', 'MultiYear_Average',
                                  'LastYear_GrowthAdjusted']:
                    site_id = row.get('Site identifier')
                    if self.historic_manager and site_id in self.historic_manager.site_periods:
                        if len(self.historic_manager.site_periods[site_id]) == 0:
                            can_use = False
                    else:
                        can_use = False

                row_data[model_name] = 'TRUE' if can_use else 'FALSE'

            output_data.append(row_data)

        training_df = pd.DataFrame(output_data)

        # Reorder columns: Site, Month, VQ, features, then models
        base_cols = ['Site_Identifier', 'Month', 'VQ']
        feature_cols = [col for col in training_df.columns if col not in base_cols and col not in self.models]
        model_cols = [col for col in training_df.columns if col in self.models]

        final_cols = base_cols + feature_cols + sorted(model_cols)
        training_df = training_df[[col for col in final_cols if col in training_df.columns]]

        return training_df

    def create_historic_analysis_sheet(self, output_df=None):
        """
        Create historic data analysis showing YoY comparisons.

        SMART FORMATTING:
        - If BOTH years have annual turnover → 1 row per site (annual comparison)
        - If either year has monthly data → 13 rows per site (12 months + TOTAL)

        Column names: Uses "Estimation Method" and "Data Quality Score" (consistent with other sheets)
        """
        if not self.has_historic_data or not self.historic_manager:
            return None

        data_source = output_df if output_df is not None else self.df

        required_cols = ['Site identifier', 'Volumetric Quantity']
        missing_cols = [col for col in required_cols if col not in data_source.columns]
        if missing_cols:
            print(f"⚠️  Cannot create historic analysis - missing columns: {missing_cols}")
            return None

        if 'Month' not in data_source.columns:
            print(f"⚠️  Cannot create historic analysis - 'Month' column not found")
            return None

        analysis_data = []
        bold_cells = []

        try:
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December']

            sites_with_both = []
            for site_id in data_source['Site identifier'].unique():
                if site_id in self.historic_manager.site_periods:
                    if len(self.historic_manager.site_periods[site_id]) > 0:
                        sites_with_both.append(site_id)

            if not sites_with_both:
                print("⚠️  No sites have both current and historic data for comparison")
                return None

            row_idx = 0

            for site_id in sorted(sites_with_both):
                historic_periods = self.historic_manager.site_periods[site_id]
                if len(historic_periods) == 0:
                    continue

                period = historic_periods[0]
                period_year = period.year
                site_data = data_source[data_source['Site identifier'] == site_id].copy()
                current_year = self.historic_manager.current_year

                # Check if historic turnover is annual
                historic_turnover_annual, historic_is_repeated = period.get_annual_feature('Turnover')

                # Check if current turnover is annual
                current_turnover_list = []
                for month in month_order:
                    month_data = site_data[site_data['Month'] == month]
                    if len(month_data) > 0 and 'Turnover' in month_data.columns:
                        turnover_val = month_data['Turnover'].iloc[0]
                        if pd.notna(turnover_val):
                            current_turnover_list.append(turnover_val)

                current_is_repeated = False
                current_turnover_annual = None
                if len(current_turnover_list) > 0:
                    unique_current = set(round(v, 2) for v in current_turnover_list if v is not None)
                    if len(unique_current) == 1:
                        current_turnover_annual = current_turnover_list[0]
                        current_is_repeated = True
                    else:
                        current_turnover_annual = sum(current_turnover_list)
                        current_is_repeated = False

                # DECISION: Annual vs Monthly comparison
                if historic_is_repeated and current_is_repeated:
                    self._create_annual_comparison_row(
                        site_id, period, period_year, current_year,
                        site_data, historic_turnover_annual, current_turnover_annual,
                        analysis_data, bold_cells, row_idx
                    )
                    row_idx += 1
                else:
                    row_idx = self._create_monthly_comparison_rows(
                        site_id, period, period_year, current_year,
                        site_data, month_order,
                        historic_turnover_annual, historic_is_repeated,
                        current_turnover_annual, current_is_repeated,
                        analysis_data, bold_cells, row_idx
                    )

            if not analysis_data:
                return None

            analysis_df = pd.DataFrame(analysis_data)
            analysis_df._bold_cells = bold_cells

            return analysis_df

        except KeyError as e:
            print(f"\n⚠️  ERROR in Historic Data Analysis:")
            print(f"   Missing column: {e}")
            return None
        except Exception as e:
            print(f"\n⚠️  ERROR in Historic Data Analysis:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_annual_comparison_row(self, site_id, period, period_year, current_year,
                                      site_data, historic_turnover_annual, current_turnover_annual,
                                      analysis_data, bold_cells, row_idx):
        """Create a single annual comparison row when both years have annual turnover"""

        historic_vq_annual = period.get_normalized_annual()
        current_vq_annual = site_data['Volumetric Quantity'].sum()

        has_extrapolated = False
        if 'Data integrity' in site_data.columns:
            has_extrapolated = (site_data['Data integrity'] == 'Estimated').any()

        estimation_method = ''
        r2_score = None
        if has_extrapolated:
            extrapolated_rows = site_data[site_data['Data integrity'] == 'Estimated']
            if len(extrapolated_rows) > 0:
                if 'Estimation Method' in extrapolated_rows.columns:
                    estimation_method = extrapolated_rows['Estimation Method'].mode()[0] if len(
                        extrapolated_rows['Estimation Method'].mode()) > 0 else ''
                if 'Data Quality Score' in extrapolated_rows.columns:
                    r2_score = extrapolated_rows['Data Quality Score'].mean()

        vq_change = None
        vq_change_pct = None
        if historic_vq_annual is not None and current_vq_annual is not None:
            vq_change = current_vq_annual - historic_vq_annual
            vq_change_pct = (vq_change / historic_vq_annual * 100) if historic_vq_annual != 0 else None

        turnover_change = None
        turnover_change_pct = None
        if historic_turnover_annual is not None and current_turnover_annual is not None:
            turnover_change = current_turnover_annual - historic_turnover_annual
            turnover_change_pct = (
                        turnover_change / historic_turnover_annual * 100) if historic_turnover_annual != 0 else None

        flag = None
        if vq_change_pct is not None:
            if abs(vq_change_pct) > 50:
                flag = '⚠️ Large change'
            elif abs(vq_change_pct) < 10:
                flag = '✓ Normal'
            else:
                flag = '→ Moderate'

        row_data = {
            'Site': site_id,
            'Month': 'ANNUAL (Both years have annual turnover)',
            f'VQ_{period_year}': historic_vq_annual,
            f'VQ_{current_year}': current_vq_annual,
            'VQ_Change': vq_change,
            'VQ_Change_%': round(vq_change_pct, 1) if vq_change_pct is not None else None,
            f'Turnover_{period_year}': historic_turnover_annual,
            f'Turnover_{current_year}': current_turnover_annual,
            f'Turnover_Change': turnover_change,
            f'Turnover_Change_%': round(turnover_change_pct, 1) if turnover_change_pct is not None else None,
            'Estimation Method': estimation_method,
            'Data Quality Score': round(r2_score, 3) if r2_score is not None else None,
            'Estimated': 'Yes' if has_extrapolated else 'No',
            'Flag': flag
        }

        analysis_data.append(row_data)

        if has_extrapolated:
            bold_cells.append((row_idx, f'VQ_{current_year}'))
            bold_cells.append((row_idx, 'VQ_Change'))
            bold_cells.append((row_idx, 'VQ_Change_%'))

    def _create_monthly_comparison_rows(self, site_id, period, period_year, current_year,
                                        site_data, month_order,
                                        historic_turnover_annual, historic_is_repeated,
                                        current_turnover_annual, current_is_repeated,
                                        analysis_data, bold_cells, row_idx):
        """Create 12 monthly rows + 1 TOTAL row for sites with monthly data"""

        historic_vq_total = 0
        current_vq_total = 0
        historic_turnover_list = []
        current_turnover_list = []

        for month in month_order:
            historic_vq = period.monthly_data.get(month)

            if historic_is_repeated:
                historic_turnover = historic_turnover_annual
            else:
                historic_turnover = period.get_monthly_feature(month, 'Turnover')

            month_data = site_data[site_data['Month'] == month]

            if len(month_data) > 0:
                current_vq = month_data['Volumetric Quantity'].iloc[0]
                current_turnover = month_data['Turnover'].iloc[0] if 'Turnover' in month_data.columns else None

                if 'Data integrity' in month_data.columns:
                    is_extrapolated = (month_data['Data integrity'].iloc[0] == 'Estimated')
                else:
                    is_extrapolated = False

                estimation_method = ''
                r2_score = None
                if is_extrapolated:
                    if 'Estimation Method' in month_data.columns:
                        estimation_method = month_data['Estimation Method'].iloc[0]
                    if 'Data Quality Score' in month_data.columns:
                        r2_score = month_data['Data Quality Score'].iloc[0]
            else:
                current_vq = None
                current_turnover = None
                is_extrapolated = False
                estimation_method = ''
                r2_score = None

            vq_change = None
            vq_change_pct = None
            if historic_vq is not None and current_vq is not None:
                vq_change = current_vq - historic_vq
                vq_change_pct = (vq_change / historic_vq * 100) if historic_vq != 0 else None

            turnover_change = None
            turnover_change_pct = None
            if historic_turnover is not None and current_turnover is not None:
                turnover_change = current_turnover - historic_turnover
                turnover_change_pct = (turnover_change / historic_turnover * 100) if historic_turnover != 0 else None

            if historic_vq is not None:
                historic_vq_total += historic_vq
            if current_vq is not None:
                current_vq_total += current_vq
            if historic_turnover is not None:
                historic_turnover_list.append(historic_turnover)
            if current_turnover is not None:
                current_turnover_list.append(current_turnover)

            flag = None
            if vq_change_pct is not None:
                if abs(vq_change_pct) > 50:
                    flag = '⚠️ Large change'
                elif abs(vq_change_pct) < 10:
                    flag = '✓ Normal'
                else:
                    flag = '→ Moderate'

            row_data = {
                'Site': site_id,
                'Month': month,
                f'VQ_{period_year}': historic_vq,
                f'VQ_{current_year}': current_vq,
                'VQ_Change': vq_change,
                'VQ_Change_%': round(vq_change_pct, 1) if vq_change_pct is not None else None,
                f'Turnover_{period_year}': historic_turnover,
                f'Turnover_{current_year}': current_turnover,
                f'Turnover_Change': turnover_change,
                f'Turnover_Change_%': round(turnover_change_pct, 1) if turnover_change_pct is not None else None,
                'Estimation Method': estimation_method,
                'Data Quality Score': r2_score,
                'Estimated': 'Yes' if is_extrapolated else 'No',
                'Flag': flag
            }

            analysis_data.append(row_data)

            if is_extrapolated and current_vq is not None:
                bold_cells.append((row_idx, f'VQ_{current_year}'))
                bold_cells.append((row_idx, 'VQ_Change'))
                bold_cells.append((row_idx, 'VQ_Change_%'))

            row_idx += 1

        # Create TOTAL row
        if historic_is_repeated:
            historic_turnover_display = historic_turnover_annual
            historic_note = f"{period_year}: Annual"
        else:
            historic_turnover_display = sum(historic_turnover_list) if historic_turnover_list else None
            historic_note = ""

        if current_is_repeated:
            current_turnover_display = current_turnover_annual
            current_note = f"{current_year}: Annual"
        else:
            current_turnover_display = sum(current_turnover_list) if current_turnover_list else None
            current_note = ""

        turnover_note = " | ".join(filter(None, [historic_note, current_note]))

        total_vq_change = None
        total_vq_change_pct = None
        if historic_vq_total > 0 and current_vq_total > 0:
            total_vq_change = current_vq_total - historic_vq_total
            total_vq_change_pct = (total_vq_change / historic_vq_total * 100)

        total_turnover_change = None
        total_turnover_change_pct = None
        if historic_turnover_display is not None and current_turnover_display is not None:
            total_turnover_change = current_turnover_display - historic_turnover_display
            total_turnover_change_pct = (
                        total_turnover_change / historic_turnover_display * 100) if historic_turnover_display != 0 else None

        total_row = {
            'Site': site_id,
            'Month': f'TOTAL {turnover_note}'.strip(),
            f'VQ_{period_year}': historic_vq_total if historic_vq_total > 0 else None,
            f'VQ_{current_year}': current_vq_total if current_vq_total > 0 else None,
            'VQ_Change': total_vq_change,
            'VQ_Change_%': round(total_vq_change_pct, 1) if total_vq_change_pct is not None else None,
            f'Turnover_{period_year}': historic_turnover_display,
            f'Turnover_{current_year}': current_turnover_display,
            f'Turnover_Change': total_turnover_change,
            f'Turnover_Change_%': round(total_turnover_change_pct,
                                        1) if total_turnover_change_pct is not None else None,
            'Estimation Method': '',
            'Data Quality Score': None,
            'Estimated': '',
            'Flag': ''
        }

        analysis_data.append(total_row)
        row_idx += 1

        return row_idx



    def _format_monthly_availability(self, months, month_count):
        """
        Format monthly data availability description.

        Returns:
        - "Full" if all 12 months present
        - "Annual" if only 1 data point (annual data)
        - "Jan-Nov" for consecutive months
        - "Jan, Mar, Dec" for scattered months
        """
        if month_count == 12:
            return "Full"
        elif month_count == 1:
            # Check if it's an annual record or a single month
            if not months or len(months) == 0:
                return "Annual"
            elif len(months) == 1:
                return months[0][:3]  # Abbreviate
        elif month_count == 0:
            return "None"

        if not months or len(months) == 0:
            return f"{month_count} months"

        # Define month order
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        month_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Convert to abbreviated form and get indices
        month_indices = []
        for m in months:
            if m in month_order:
                month_indices.append(month_order.index(m))

        month_indices = sorted(set(month_indices))

        if len(month_indices) == 0:
            return f"{month_count} months"

        # Check if consecutive
        is_consecutive = True
        for i in range(len(month_indices) - 1):
            if month_indices[i + 1] - month_indices[i] != 1:
                is_consecutive = False
                break

        if is_consecutive and len(month_indices) >= 2:
            # Show as range
            start_month = month_abbr[month_indices[0]]
            end_month = month_abbr[month_indices[-1]]
            return f"{start_month}-{end_month}"
        else:
            # Show as list
            month_names = [month_abbr[i] for i in month_indices]
            if len(month_names) <= 4:
                return ', '.join(month_names)
            else:
                return f"{', '.join(month_names[:3])}, ... ({len(month_names)} months)"

    def run_extrapolation(self):
        """Main method to run the complete extrapolation process"""
        print("=" * 70)
        print("VOLUMETRIC DATA EXTRAPOLATION TOOL WITH HISTORIC DATA")
        print("=" * 70)

        print("\n1. Loading data...")
        self.load_data()

        print("\n2. Determining timeframes...")
        self.add_timeframe_columns()

        print("\n3. Loading historic data...")
        self.load_historic_data()

        print("\n4. Engineering features from historic data...")
        self.engineer_historic_features()

        print("\n5. Identifying blank rows...")
        self.identify_blank_rows()

        print("\n6. Classifying features...")
        self.classify_features()

        print("\n7. Calculating seasonal patterns...")
        self.calculate_seasonal_patterns()

        print("\n8. Training models...")
        self.train_models()

        print("\n9. Extrapolating blank rows...")
        results = self.extrapolate_blank_rows()

        print("\n10. Creating output dataframe...")
        output_df = self.create_output_dataframe(results)

        print(f"\n{'=' * 70}")
        print(f"EXTRAPOLATION COMPLETE")
        print(f"{'=' * 70}")
        print(f"Total rows: {len(output_df)}")
        print(f"Extrapolated rows: {len(results)}")
        if results:
            print(f"Average quality score: {np.mean([r['r2'] for r in results]):.3f}")

        # Create output sheets
        model_testing_matrix = self.create_model_testing_matrix()
        model_performance = self.create_model_performance_sheet()
        data_availability = self.create_data_availability_sheet()
        historic_analysis = self.create_historic_analysis_sheet(output_df)  # Pass output_df!
        training_data = self.create_training_data_sheet()

        return output_df, results, model_testing_matrix, model_performance, data_availability, historic_analysis, training_data

    @staticmethod
    def apply_bold_formatting_to_sheet(writer, sheet_name, dataframe):
        """
        Apply bold formatting to cells marked as extrapolated in the historic analysis sheet.

        Args:
            writer: pd.ExcelWriter object
            sheet_name: Name of the sheet
            dataframe: DataFrame that may have _bold_cells attribute
        """
        try:
            from openpyxl.styles import Font

            # Check if DataFrame has bold cell markers
            if not hasattr(dataframe, '_bold_cells') or not dataframe._bold_cells:
                return

            # Get the worksheet
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]

            # Apply bold formatting to marked cells
            bold_font = Font(bold=True)

            for row_idx, col_name in dataframe._bold_cells:
                # Find column index
                if col_name in dataframe.columns:
                    col_idx = dataframe.columns.get_loc(col_name)
                    # Excel rows are 1-indexed, +2 for header row
                    excel_row = row_idx + 2
                    # Excel columns are 1-indexed
                    excel_col = col_idx + 1

                    # Get cell and apply bold
                    cell = worksheet.cell(row=excel_row, column=excel_col)
                    cell.font = bold_font
        except ImportError:
            # openpyxl not available - skip formatting
            pass
        except Exception as e:
            # Don't fail if formatting doesn't work
            print(f"⚠️  Could not apply bold formatting: {e}")