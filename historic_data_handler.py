"""
PART 1: Historic Data Loading & Alignment Module (PERIOD-BASED VERSION)
Handles loading historic records and features, organizing by PERIODS instead of years

✅ UPDATED: Monthly-specific feature support
- get_period_feature() now accepts optional 'month' parameter
- get_historic_feature() is a new method supporting month-specific lookups
- detect_feature_variability() reports which features are Annual vs Monthly
- load_historic_features() reports variability on load
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import defaultdict


class HistoricPeriod:
    """
    Represents a single historic period for a site.
    A period is a contiguous span of time (usually 12 months) with VQ data.
    """

    def __init__(self, site, start_date, end_date):
        self.site = site
        self.start_date = start_date
        self.end_date = end_date
        self.year = start_date.year

        # Data storage
        self.monthly_data = {}  # {month_name: vq_value}
        self.annual_total = None
        self.features = {}  # {feature_name: value} - period-level features
        self.monthly_features = {}  # {month_name: {feature_name: value}}

        # Metadata
        self.months_covered = 0
        self.is_complete = False  # True if has all 12 months or annual total

    def add_monthly_vq(self, month, vq):
        """Add a monthly VQ value"""
        self.monthly_data[month] = vq
        self.months_covered = len(self.monthly_data)
        if self.months_covered == 12:
            self.is_complete = True
            self.annual_total = sum(self.monthly_data.values())

    def set_annual_total(self, total):
        """Set annual total directly"""
        self.annual_total = total
        self.is_complete = True
        if not self.monthly_data:
            self.months_covered = 12  # Assume full year

    def add_feature(self, feature_name, value):
        """Add a period-level feature value"""
        self.features[feature_name] = value

    def add_monthly_feature(self, month, feature_name, value):
        """Add a feature value for a specific month"""
        if month not in self.monthly_features:
            self.monthly_features[month] = {}
        self.monthly_features[month][feature_name] = value

    def get_monthly_feature(self, month, feature_name):
        """Get a feature value for a specific month"""
        if month in self.monthly_features:
            return self.monthly_features[month].get(feature_name)
        return None

    def get_annual_feature(self, feature_name):
        """
        Get annual feature value, detecting if monthly values are repeated (annual) or different (monthly sum).
        Returns: (value, is_repeated)

        is_repeated meanings:
            True  - same value every month (e.g. annual SqFt repeated across rows)
            False - varies by month (e.g. monthly Turnover)
            None  - insufficient data to determine (fall back to period-level)

        FIXED: requires at least 6 months of data before classifying as 'repeated'.
        With fewer months, a single stored value would falsely look like a constant.
        """
        if not self.monthly_features:
            return self.features.get(feature_name), None

        monthly_values = []
        for month in self.monthly_features:
            val = self.monthly_features[month].get(feature_name)
            if val is not None:
                monthly_values.append(val)

        if not monthly_values:
            return self.features.get(feature_name), None

        unique_values = set(round(v, 2) if isinstance(v, (int, float)) else v for v in monthly_values)

        if len(unique_values) == 1:
            if len(monthly_values) >= 6:
                # Enough months with identical value - confidently annual/repeated
                return monthly_values[0], True
            else:
                # Too few months stored to be sure - treat as unknown
                # so per-month lookup is used in the YoY sheet
                return monthly_values[0], None
        else:
            # Values differ across months - monthly-varying
            return sum(monthly_values), False

    def get_monthly_average(self):
        """Get average monthly VQ"""
        if self.monthly_data:
            return sum(self.monthly_data.values()) / len(self.monthly_data)
        elif self.annual_total:
            return self.annual_total / 12
        return None

    def get_normalized_annual(self):
        """Get normalized annual total (extrapolate to 12 months if incomplete)"""
        if self.annual_total:
            return self.annual_total
        elif self.monthly_data:
            if len(self.monthly_data) == 12:
                return sum(self.monthly_data.values())
            else:
                # Extrapolate to 12 months
                avg_monthly = sum(self.monthly_data.values()) / len(self.monthly_data)
                return avg_monthly * 12
        return None

    def __repr__(self):
        return f"HistoricPeriod({self.site}, {self.start_date.date()} to {self.end_date.date()}, {self.months_covered} months)"


class HistoricDataManager:
    """
    Manages historic data using PERIODS instead of years.
    A period represents a contiguous span of time with VQ data.

    ✅ UPDATED: Monthly-specific feature support throughout.
    """

    def __init__(self):
        self.historic_records = None
        self.historic_features = None
        self.current_year = None
        self.historic_years = []

        # PERIOD-BASED STORAGE
        self.site_periods = defaultdict(list)  # {site: [HistoricPeriod, ...]}

        # Availability matrix
        self.availability_matrix = None

    def parse_dates(self, date_str):
        """Parse date string handling both standard format and Excel serial dates"""
        try:
            if isinstance(date_str, (int, float)):
                excel_epoch = datetime(1899, 12, 30)
                return excel_epoch + pd.Timedelta(days=date_str)
            return pd.to_datetime(date_str, format='%d/%m/%Y')
        except:
            return pd.to_datetime(date_str, dayfirst=True)

    def determine_timeframe(self, date_from, date_to):
        """Determine if data is Monthly, Annual, or custom days"""
        if date_from.day == 1:
            next_month = date_from + relativedelta(months=1)
            last_day = next_month - pd.Timedelta(days=1)

            if date_to.date() == last_day.date():
                return 'Monthly', date_from.strftime('%B')

        days_diff = (date_to - date_from).days + 1
        if days_diff >= 365 and days_diff <= 366:
            return 'Annual', None

        return f'{days_diff} days', None

    def detect_current_period(self, current_df):
        """
        Detect the current period from input data.
        PRIORITY: Use FY/Financial Year column if available
        FALLBACK: Use calendar year from dates
        """
        import re

        # Try to detect from FY column first
        fy_column = None
        for col in current_df.columns:
            if col in ['FY', 'fy', 'Financial Year', 'financial_year', 'Financial_Year']:
                fy_column = col
                break

        if fy_column and current_df[fy_column].notna().any():
            fy_values = current_df[fy_column].dropna()
            from collections import Counter
            most_common_fy = Counter(fy_values).most_common(1)[0][0]

            if isinstance(most_common_fy, str):
                match = re.search(r'(\d{4})', str(most_common_fy))
                if match:
                    self.current_year = int(match.group(1))
                    self.current_fy_label = most_common_fy
                else:
                    self.current_year = int(most_common_fy)
                    self.current_fy_label = f"FY{most_common_fy}"
            else:
                self.current_year = int(most_common_fy)
                self.current_fy_label = f"FY{most_common_fy}"

            print(f"\n🗓️  Detected current period: {self.current_fy_label} (FY column)")
            print(f"   Financial year identifier: {self.current_year}")

            self.use_fy_logic = True
            return self.current_year

        # Fallback: Use calendar year from dates
        years = []
        dates = []

        for date_str in current_df['Date from'].dropna():
            parsed = self.parse_dates(date_str)
            years.append(parsed.year)
            dates.append(parsed)

        if years:
            from collections import Counter
            year_counts = Counter(years)
            self.current_year = year_counts.most_common(1)[0][0]
            self.current_fy_label = f"{self.current_year}"

            min_date = min(dates)
            max_date = max([self.parse_dates(d) for d in current_df['Date to'].dropna()])

            print(f"\n🗓️  Detected current period: {self.current_year} (calendar year from dates)")
            print(f"   Date range: {min_date.date()} to {max_date.date()}")

            self.use_fy_logic = False
            return self.current_year
        else:
            self.current_year = datetime.now().year
            self.current_fy_label = f"{self.current_year}"
            self.use_fy_logic = False
            print(f"\n⚠️  Could not detect period from data, using system year: {self.current_year}")
            return self.current_year

    def load_historic_records(self, filepath, ghg_category_filter=None):
        """
        Load historic records and organize into PERIODS.
        """
        print("\n" + "=" * 70)
        print("LOADING HISTORIC RECORDS (PERIOD-BASED)")
        print("=" * 70)

        if filepath.endswith('.csv'):
            self.historic_records = pd.read_csv(filepath)
        else:
            self.historic_records = pd.read_excel(filepath)

        self.historic_records.columns = self.historic_records.columns.str.strip()

        print(f"✓ Loaded {len(self.historic_records)} historic records")

        if ghg_category_filter:
            if isinstance(ghg_category_filter, str):
                ghg_category_filter = [ghg_category_filter]

            self.historic_records = self.historic_records[
                self.historic_records['GHG Category'].isin(ghg_category_filter)
            ]
            print(f"✓ Filtered to {len(self.historic_records)} records matching GHG categories")

        self._organize_into_periods()

        self.historic_years = sorted(list(set([p.year for periods in self.site_periods.values() for p in periods])))
        print(f"✓ Detected {len(self.historic_years)} historic years: {self.historic_years}")

        return self.historic_records

    def _organize_into_periods(self):
        """
        Organize historic records into PERIODS.
        Groups data by site and contiguous time spans.
        """
        print("\n📅 Organizing data into periods...")

        year_column = None
        for col in self.historic_records.columns:
            if col in ['Year', 'year', 'FY', 'fy', 'Financial Year', 'financial_year']:
                year_column = col
                break

        if year_column:
            print(f"  ✓ Found year column: '{year_column}' - using for period organization")
            self._organize_by_year_column(year_column)
        else:
            print("  ✓ Using Date from/Date to columns for period organization")
            self._organize_by_dates()

    def _organize_by_year_column(self, year_column):
        """Organize historic records using explicit Year/FY column"""
        import re

        site_groups = self.historic_records.groupby('Site identifier')

        for site, site_data in site_groups:
            for year_val in site_data[year_column].unique():
                if pd.isna(year_val):
                    continue

                if isinstance(year_val, str):
                    match = re.search(r'(\d{4})', year_val)
                    if match:
                        year = int(match.group(1))
                    else:
                        continue
                else:
                    year = int(year_val)

                if year == self.current_year:
                    continue

                year_data = site_data[site_data[year_column] == year_val]

                first_row = year_data.iloc[0]
                date_from = self.parse_dates(first_row['Date from'])
                date_to = self.parse_dates(first_row['Date to'])

                period = HistoricPeriod(site, date_from, date_to)

                for idx, row in year_data.iterrows():
                    vq = row.get('Volumetric Quantity')
                    if pd.isna(vq):
                        continue

                    date_from_row = self.parse_dates(row['Date from'])
                    date_to_row = self.parse_dates(row['Date to'])
                    timeframe, month = self.determine_timeframe(date_from_row, date_to_row)

                    if timeframe == 'Monthly' and month:
                        period.add_monthly_vq(month, vq)
                        if date_to_row > period.end_date:
                            period.end_date = date_to_row
                    elif timeframe == 'Annual':
                        period.set_annual_total(vq)
                        period.end_date = date_to_row

                self.site_periods[site].append(period)

        total_periods = sum(len(periods) for periods in self.site_periods.values())
        print(f"✓ Organized into {total_periods} periods across {len(self.site_periods)} sites")

        for site in list(self.site_periods.keys())[:3]:
            periods = self.site_periods[site]
            print(f"  • {site}: {len(periods)} periods")
            for period in periods:
                status = "✓ Complete" if period.is_complete else f"⚠ {period.months_covered} months"
                print(f"    - {period.year}: {status}")

    def _organize_by_dates(self):
        """Organize historic records by extracting year from Date columns"""
        import re

        fy_column = None
        for col in self.historic_records.columns:
            if col in ['FY', 'fy', 'Financial Year', 'financial_year', 'Financial_Year']:
                fy_column = col
                break

        site_groups = self.historic_records.groupby('Site identifier')

        for site, site_data in site_groups:
            site_data = site_data.sort_values('Date from')

            current_period = None

            for idx, row in site_data.iterrows():
                date_from = self.parse_dates(row['Date from'])
                date_to = self.parse_dates(row['Date to'])
                vq = row.get('Volumetric Quantity')

                if pd.isna(vq):
                    continue

                should_skip = False
                if fy_column and pd.notna(row.get(fy_column)):
                    fy_val = row[fy_column]
                    if isinstance(fy_val, str):
                        match = re.search(r'(\d{4})', fy_val)
                        if match:
                            fy_year = int(match.group(1))
                            should_skip = (fy_year == self.current_year)
                    else:
                        should_skip = (int(fy_val) == self.current_year)
                else:
                    should_skip = (date_from.year == self.current_year)

                if should_skip:
                    continue

                timeframe, month = self.determine_timeframe(date_from, date_to)

                if current_period is None or date_from.year != current_period.year:
                    if current_period is not None:
                        self.site_periods[site].append(current_period)
                    current_period = HistoricPeriod(site, date_from, date_to)

                if timeframe == 'Monthly' and month:
                    current_period.add_monthly_vq(month, vq)
                    if date_to > current_period.end_date:
                        current_period.end_date = date_to
                elif timeframe == 'Annual':
                    current_period.set_annual_total(vq)
                    current_period.start_date = date_from
                    current_period.end_date = date_to

            if current_period is not None:
                self.site_periods[site].append(current_period)

        total_periods = sum(len(periods) for periods in self.site_periods.values())
        print(f"✓ Organized into {total_periods} periods across {len(self.site_periods)} sites")

        for site in list(self.site_periods.keys())[:3]:
            periods = self.site_periods[site]
            print(f"  • {site}: {len(periods)} periods")
            for period in periods:
                status = "✓ Complete" if period.is_complete else f"⚠ {period.months_covered} months"
                print(f"    - {period.year}: {status}")

    def load_historic_features(self, filepath):
        """Load historic features and attach to periods"""
        print("\n" + "=" * 70)
        print("LOADING HISTORIC FEATURES")
        print("=" * 70)

        if filepath.endswith('.csv'):
            self.historic_features = pd.read_csv(filepath)
        else:
            self.historic_features = pd.read_excel(filepath)

        self.historic_features.columns = self.historic_features.columns.str.strip()

        print(f"✓ Loaded historic features file with {len(self.historic_features)} rows")
        print(f"  Columns: {list(self.historic_features.columns)}")

        self._convert_numeric_features()

        year_column = None
        for col in self.historic_features.columns:
            if col in ['Year', 'year', 'FY', 'fy', 'Financial Year', 'financial_year']:
                year_column = col
                break

        if year_column:
            print(f"✓ Detected Format 2A: Year column present ('{year_column}')")
            self._attach_features_format2(year_column)
        elif 'Date from' in self.historic_features.columns and 'Date to' in self.historic_features.columns:
            print("✓ Detected Format 2B: Date from/Date to columns present")
            self._attach_features_format2_dates()
        else:
            print("✓ Detected Format 1: Year suffix in column names")
            self._attach_features_format1()

        # ✅ NEW: Detect and report feature variability (Annual vs Monthly)
        feature_types = self.detect_feature_variability()
        if feature_types:
            print("\n📊 Feature Variability Analysis:")
            for feature, ftype in sorted(feature_types.items()):
                icon = "📅" if "Monthly" in ftype else "📆" if "Annual" in ftype else "🔀"
                print(f"  {icon} {feature}: {ftype}")
        else:
            print("\n⚠️  No feature variability data available")

        return self.historic_features

    def _convert_numeric_features(self):
        """Convert numeric columns in historic features that might be stored as text"""
        numeric_keywords = ['turnover', 'transaction', 'revenue', 'sales',
                            'square', 'footage', 'floor', 'space', 'sqft']

        for col in self.historic_features.columns:
            if col == 'Site identifier' or col == 'Year':
                continue

            if pd.api.types.is_numeric_dtype(self.historic_features[col]):
                continue

            col_lower = col.lower()
            should_be_numeric = any(keyword in col_lower for keyword in numeric_keywords)

            if should_be_numeric:
                try:
                    converted = pd.to_numeric(self.historic_features[col], errors='coerce')
                    successful = converted.notna().sum()
                    total = self.historic_features[col].notna().sum()

                    if total > 0 and successful / total > 0.8:
                        self.historic_features[col] = converted
                        print(f"  ✓ Converted '{col}' to numeric")
                except:
                    pass

    def _attach_features_format1(self):
        """Attach features with year suffix (e.g., Turnover_2023)"""
        feature_years = set()
        base_features = set()

        for col in self.historic_features.columns:
            if col == 'Site identifier':
                continue

            parts = col.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
                base_features.add(parts[0])
                feature_years.add(int(parts[1]))

        print(f"✓ Detected {len(base_features)} features: {sorted(base_features)}")
        print(f"✓ Detected {len(feature_years)} years: {sorted(feature_years)}")

        for idx, row in self.historic_features.iterrows():
            site = row.get('Site identifier')
            if pd.isna(site) or site not in self.site_periods:
                continue

            for col in self.historic_features.columns:
                if col == 'Site identifier':
                    continue

                parts = col.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
                    feature_name = parts[0]
                    year = int(parts[1])
                    value = row[col]

                    if pd.notna(value):
                        for period in self.site_periods[site]:
                            if period.year == year:
                                period.add_feature(feature_name, value)

        print(f"✓ Attached features to periods")

    def _attach_features_format2(self, year_column='Year'):
        """
        Attach features with Year/FY column.

        ✅ FIXED: If the file also has Date from/Date to columns (i.e. one row per month),
        features are stored monthly using add_monthly_feature() so that month-varying
        values (e.g. Turnover) are preserved rather than overwritten by the last row.

        After all rows are processed, any feature whose monthly values are all identical
        is also promoted to period.features (annual level) for fast annual lookups.
        """
        import re

        has_dates = ('Date from' in self.historic_features.columns and
                     'Date to' in self.historic_features.columns)

        feature_columns = [col for col in self.historic_features.columns
                           if col not in ['Site identifier', year_column, 'Location',
                                          'Date from', 'Date to']]

        if not feature_columns:
            print("⚠️  No feature columns found (only metadata columns present)")
            return

        print(f"✓ Detected {len(feature_columns)} features: {feature_columns}")
        if has_dates:
            print("  ℹ️  Date from/Date to present — storing features monthly to preserve variation")

        years_detected = set()

        for idx, row in self.historic_features.iterrows():
            site = row.get('Site identifier')
            year_raw = row.get(year_column)

            if pd.isna(site) or pd.isna(year_raw) or site not in self.site_periods:
                continue

            # Parse year (handles "FY2024", 2024, etc.)
            if isinstance(year_raw, str):
                match = re.search(r'(\d{4})', year_raw)
                if match:
                    year = int(match.group(1))
                else:
                    continue
            else:
                year = int(year_raw)

            years_detected.add(year)

            # Find matching period
            target_period = None
            for period in self.site_periods[site]:
                if period.year == year:
                    target_period = period
                    break

            if target_period is None:
                continue

            if has_dates:
                # Parse month from Date from so we store features per-month
                date_from_raw = row.get('Date from')
                month_name = None
                if pd.notna(date_from_raw):
                    try:
                        if isinstance(date_from_raw, str):
                            for fmt in ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y']:
                                try:
                                    month_name = datetime.strptime(date_from_raw, fmt).strftime('%B')
                                    break
                                except ValueError:
                                    continue
                            if month_name is None:
                                month_name = pd.to_datetime(date_from_raw, dayfirst=True).strftime('%B')
                        elif hasattr(date_from_raw, 'strftime'):
                            month_name = date_from_raw.strftime('%B')
                    except Exception:
                        month_name = None

                for feature in feature_columns:
                    value = row[feature]
                    if pd.notna(value):
                        if month_name:
                            target_period.add_monthly_feature(month_name, feature, value)
                        else:
                            # Can't determine month — store at period level as fallback
                            target_period.add_feature(feature, value)
            else:
                # No date columns — single row per site/year, store at period level
                for feature in feature_columns:
                    value = row[feature]
                    if pd.notna(value):
                        target_period.add_feature(feature, value)

        # ✅ Post-process: for monthly-stored features that are actually constant
        # across all months, also write to period.features for fast annual lookups
        if has_dates:
            promoted = 0
            for site, periods in self.site_periods.items():
                for period in periods:
                    if not period.monthly_features:
                        continue
                    # Collect all feature names stored monthly
                    all_monthly_feat_names = set(
                        f for mf in period.monthly_features.values() for f in mf.keys()
                    )
                    for feat_name in all_monthly_feat_names:
                        vals = [
                            period.monthly_features[m][feat_name]
                            for m in period.monthly_features
                            if feat_name in period.monthly_features[m]
                        ]
                        if len(vals) > 0:
                            unique = set(round(v, 2) if isinstance(v, (int, float)) else v for v in vals)
                            if len(unique) == 1:
                                # Constant — promote to period level
                                if feat_name not in period.features:
                                    period.add_feature(feat_name, vals[0])
                                    promoted += 1
            if promoted:
                print(f"  ✓ Promoted {promoted} constant monthly features to period level")

        print(f"✓ Detected {len(years_detected)} years: {sorted(years_detected)}")
        print(f"✓ Attached features to periods")

    def _attach_features_format2_dates(self):
        """
        Attach features from format with 'Date from' and 'Date to' columns.
        Extracts year and month from dates and matches to periods.
        Stores features PER MONTH for proper month-by-month comparison.
        """
        from datetime import datetime

        exclude_cols = ['Site identifier', 'Location', 'Date from', 'Date to', 'Year', 'Financial Year']
        feature_columns = [col for col in self.historic_features.columns
                           if col not in exclude_cols]

        if not feature_columns:
            print("⚠️  No feature columns found (only metadata columns present)")
            return

        print(f"✓ Detected {len(feature_columns)} features: {feature_columns}")

        years_detected = set()

        for idx, row in self.historic_features.iterrows():
            site = row.get('Site identifier')
            if pd.isna(site) or site not in self.site_periods:
                continue

            date_from = row.get('Date from')
            date_to = row.get('Date to')
            if pd.isna(date_from):
                continue

            try:
                if isinstance(date_from, str):
                    for fmt in ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y']:
                        try:
                            date_from_obj = datetime.strptime(date_from, fmt)
                            year = date_from_obj.year
                            month_name = date_from_obj.strftime('%B')
                            break
                        except:
                            continue
                    else:
                        date_from_obj = pd.to_datetime(date_from, dayfirst=True)
                        year = date_from_obj.year
                        month_name = date_from_obj.strftime('%B')
                elif hasattr(date_from, 'year'):
                    year = date_from.year
                    month_name = date_from.strftime('%B')
                else:
                    continue

                years_detected.add(year)

                for period in self.site_periods[site]:
                    if period.year == year:
                        for feature in feature_columns:
                            value = row[feature]
                            if pd.notna(value):
                                period.add_monthly_feature(month_name, feature, value)
                        break
            except Exception as e:
                continue

        print(f"✓ Detected {len(years_detected)} years: {sorted(years_detected)}")
        print(f"✓ Attached features to periods (monthly tracking enabled)")

    # =========================================================================
    # ✅ NEW: Feature Variability Detection
    # =========================================================================

    def detect_feature_variability(self):
        """
        Detect which features are annual (repeated) vs monthly (varying).

        Checks each feature across all sites and periods:
        - If a site has the same Turnover every month → 'Annual (Repeated)'
        - If a site has different Turnover each month → 'Monthly (Varying)'
        - If it varies by site → 'Mixed'

        Returns:
        --------
        dict: {feature_name: 'Annual (Repeated)' | 'Monthly (Varying)' | 'Mixed'}
        """
        feature_analysis = {}  # {feature_name: {'annual': 0, 'monthly': 0}}

        for site, periods in self.site_periods.items():
            for period in periods:
                # Collect all feature names from both period-level and monthly features
                all_feature_names = set(period.features.keys())
                for month_features in period.monthly_features.values():
                    all_feature_names.update(month_features.keys())

                for feature_name in all_feature_names:
                    if feature_name not in feature_analysis:
                        feature_analysis[feature_name] = {'annual': 0, 'monthly': 0}

                    # Use get_annual_feature which already handles the
                    # repeated vs varying logic
                    annual_val, is_repeated = period.get_annual_feature(feature_name)

                    if annual_val is None:
                        continue

                    if is_repeated is True:
                        feature_analysis[feature_name]['annual'] += 1
                    elif is_repeated is False:
                        feature_analysis[feature_name]['monthly'] += 1
                    else:
                        # is_repeated is None → came from period-level only (no monthly data)
                        # Treat as annual
                        feature_analysis[feature_name]['annual'] += 1

        # Classify each feature
        result = {}
        for feature_name, counts in feature_analysis.items():
            total = counts['annual'] + counts['monthly']
            if total == 0:
                continue

            annual_pct = counts['annual'] / total

            if annual_pct > 0.8:
                result[feature_name] = 'Annual (Repeated)'
            elif annual_pct < 0.2:
                result[feature_name] = 'Monthly (Varying)'
            else:
                result[feature_name] = f'Mixed ({counts["monthly"]} monthly, {counts["annual"]} annual sites)'

        return result

    # =========================================================================
    # ✅ UPDATED: Feature Access Methods (now month-aware)
    # =========================================================================

    def get_period_feature(self, period, feature_name, month=None):
        """
        Get a feature value from a period, with optional month-specific lookup.

        Priority:
        1. If month provided → check monthly_features for that month first
        2. Fall back to period-level features (annual / repeated value)

        Parameters:
        -----------
        period : HistoricPeriod
        feature_name : str
        month : str, optional
            Month name (e.g., 'January'). If provided, returns month-specific value.

        Returns:
        --------
        float or None

        Examples:
        ---------
        # Turnover varies by month - get January's value
        jan_turnover = hdm.get_period_feature(period, 'Turnover', month='January')

        # SqFt is the same all year - month doesn't matter
        sqft = hdm.get_period_feature(period, 'SqFt')
        """
        if month:
            # Try to get month-specific feature first
            month_val = period.get_monthly_feature(month, feature_name)
            if month_val is not None:
                return month_val

        # Fall back to period-level feature (annual or repeated value)
        return period.features.get(feature_name)

    def get_historic_feature(self, site, year, feature_name, month=None):
        """
        Get a historic feature value by site, year, and optionally month.

        Convenience wrapper around get_period_feature() for backward compatibility
        and easy access without needing a HistoricPeriod object.

        Parameters:
        -----------
        site : str
            Site identifier
        year : int
            Year to retrieve feature from
        feature_name : str
            Name of the feature (e.g., 'Turnover', 'SqFt')
        month : str, optional
            Month name (e.g., 'January'). If provided, returns month-specific value.
            If not provided (or if month-specific value not available), returns
            period-level (annual / repeated) value.

        Returns:
        --------
        float or None

        Examples:
        ---------
        # Monthly-varying Turnover
        jan_2024 = hdm.get_historic_feature('Site001', 2024, 'Turnover', month='January')
        feb_2024 = hdm.get_historic_feature('Site001', 2024, 'Turnover', month='February')

        # Annual SqFt (same regardless of month)
        sqft_2024 = hdm.get_historic_feature('Site001', 2024, 'SqFt')
        sqft_2024_jan = hdm.get_historic_feature('Site001', 2024, 'SqFt', month='January')
        # Both return the same value
        """
        if site not in self.site_periods:
            return None

        for period in self.site_periods[site]:
            if period.year == year:
                return self.get_period_feature(period, feature_name, month=month)

        return None

    # =========================================================================
    # Existing accessor methods (unchanged)
    # =========================================================================

    def get_most_recent_period(self, site, before_date):
        """
        Get the most recent historic period for a site ending before a given date.

        Parameters:
        -----------
        site : str
        before_date : datetime

        Returns:
        --------
        HistoricPeriod or None
        """
        if site not in self.site_periods:
            return None

        eligible_periods = [p for p in self.site_periods[site] if p.end_date < before_date]

        if not eligible_periods:
            return None

        return max(eligible_periods, key=lambda p: p.end_date)

    def get_all_periods(self, site):
        """Get all periods for a site"""
        return self.site_periods.get(site, [])

    def get_period_vq_for_month(self, period, month):
        """
        Get VQ for a specific month from a period.

        Parameters:
        -----------
        period : HistoricPeriod
        month : str

        Returns:
        --------
        float or None
        """
        if month in period.monthly_data:
            return period.monthly_data[month]
        return None

    def get_period_annual_total(self, period):
        """Get annual total from a period"""
        return period.annual_total

    def create_availability_matrix(self, current_df):
        """Create availability matrix for analysis"""
        print("\n" + "=" * 70)
        print("CREATING AVAILABILITY MATRIX")
        print("=" * 70)

        matrix_data = []
        current_sites = current_df['Site identifier'].unique()

        months = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']

        for site in current_sites:
            site_current = current_df[current_df['Site identifier'] == site]
            ghg_cat = site_current['GHG Category'].iloc[0] if len(site_current) > 0 else None

            for month in months:
                month_current = site_current[site_current['Month'] == month]

                has_current_vq = False
                current_vq = None
                if len(month_current) > 0:
                    has_current_vq = month_current['Volumetric Quantity'].notna().any()
                    if has_current_vq:
                        current_vq = month_current['Volumetric Quantity'].iloc[0]

                historic_availability = {}

                if site in self.site_periods:
                    for period in self.site_periods[site]:
                        year = period.year

                        has_month_vq = month in period.monthly_data
                        has_features = len(period.features) > 0 or len(period.monthly_features) > 0

                        historic_availability[f'{year}_VQ'] = has_month_vq
                        historic_availability[f'{year}_Features'] = has_features

                        if has_month_vq:
                            historic_availability[f'{year}_VQ_Value'] = period.monthly_data[month]

                years_with_vq = sum(1 for k, v in historic_availability.items() if k.endswith('_VQ') and v)
                completeness_score = years_with_vq / len(self.historic_years) if self.historic_years else 0

                if has_current_vq and years_with_vq >= 2:
                    data_quality = 'Excellent'
                elif has_current_vq and years_with_vq == 1:
                    data_quality = 'Good'
                elif not has_current_vq and years_with_vq >= 2:
                    data_quality = 'Historic Only'
                elif not has_current_vq and years_with_vq == 1:
                    data_quality = 'Sparse Historic'
                elif has_current_vq and years_with_vq == 0:
                    data_quality = 'Current Only'
                else:
                    data_quality = 'No Data'

                matrix_row = {
                    'Site identifier': site,
                    'GHG Category': ghg_cat,
                    'Month': month,
                    f'{self.current_year}_Has_VQ': has_current_vq,
                    f'{self.current_year}_VQ_Value': current_vq,
                    'Historic_Years_With_VQ': years_with_vq,
                    'Completeness_Score': completeness_score,
                    'Data_Quality': data_quality,
                    **historic_availability
                }

                matrix_data.append(matrix_row)

        self.availability_matrix = pd.DataFrame(matrix_data)
        print(f"✓ Created availability matrix with {len(self.availability_matrix)} records")

        quality_counts = self.availability_matrix['Data_Quality'].value_counts()
        print("\nData Quality Distribution:")
        for quality, count in quality_counts.items():
            print(f"  {quality}: {count} ({count / len(self.availability_matrix) * 100:.1f}%)")

        return self.availability_matrix

    def identify_new_sites(self, current_sites):
        """Identify sites with no historic data"""
        new_sites = [s for s in current_sites if s not in self.site_periods or len(self.site_periods[s]) == 0]

        print(f"\n📍 Identified {len(new_sites)} new sites (no historic data)")
        if new_sites and len(new_sites) <= 20:
            print(f"  New sites: {new_sites}")

        return new_sites

    def get_summary_statistics(self):
        """Get summary statistics"""
        total_periods = sum(len(periods) for periods in self.site_periods.values())

        return {
            'current_year': self.current_year,
            'historic_years': self.historic_years,
            'total_periods': total_periods,
            'sites_with_periods': len(self.site_periods)
        }

    # =========================================================================
    # Backward compatibility helpers
    # =========================================================================

    def get_historic_vq(self, site, year, month):
        """Get historic VQ by year and month (backward compatibility)"""
        if site not in self.site_periods:
            return None

        for period in self.site_periods[site]:
            if period.year == year and month in period.monthly_data:
                return period.monthly_data[month]
        return None