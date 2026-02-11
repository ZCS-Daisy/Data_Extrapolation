"""
PART 1: Historic Data Loading & Alignment Module (PERIOD-BASED VERSION)
Handles loading historic records and features, organizing by PERIODS instead of years
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
        self.monthly_features = {}  # {month_name: {feature_name: value}} - NEW!

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
        """
        # Check if we have monthly feature data
        if not self.monthly_features:
            # Fall back to period-level feature
            return self.features.get(feature_name), None

        # Collect monthly values
        monthly_values = []
        for month in self.monthly_features:
            val = self.monthly_features[month].get(feature_name)
            if val is not None:
                monthly_values.append(val)

        if not monthly_values:
            return None, None

        # Check if all values are the same (within small tolerance for floating point)
        unique_values = set(round(v, 2) if isinstance(v, (int, float)) else v for v in monthly_values)

        if len(unique_values) == 1:
            # All same → This is an annual value repeated
            return monthly_values[0], True
        else:
            # Different → Sum to get annual total
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
            # Use FY column (most common value)
            fy_values = current_df[fy_column].dropna()
            from collections import Counter
            most_common_fy = Counter(fy_values).most_common(1)[0][0]

            # Extract numeric year from FY (e.g., "FY2025" → 2025)
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

            # Store that we're using FY-based logic
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

            # Find date range
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

        # Load file
        if filepath.endswith('.csv'):
            self.historic_records = pd.read_csv(filepath)
        else:
            self.historic_records = pd.read_excel(filepath)

        # CRITICAL FIX: Strip whitespace from column names
        self.historic_records.columns = self.historic_records.columns.str.strip()

        print(f"✓ Loaded {len(self.historic_records)} historic records")

        # Filter by GHG Category
        if ghg_category_filter:
            if isinstance(ghg_category_filter, str):
                ghg_category_filter = [ghg_category_filter]

            before_count = len(self.historic_records)
            self.historic_records = self.historic_records[
                self.historic_records['GHG Category'].isin(ghg_category_filter)
            ]
            print(f"✓ Filtered to {len(self.historic_records)} records matching GHG categories")

        # Organize into periods
        self._organize_into_periods()

        # Extract historic years
        self.historic_years = sorted(list(set([p.year for periods in self.site_periods.values() for p in periods])))
        print(f"✓ Detected {len(self.historic_years)} historic years: {self.historic_years}")

        return self.historic_records

    def _organize_into_periods(self):
        """
        Organize historic records into PERIODS.
        Groups data by site and contiguous time spans.
        """
        print("\n📅 Organizing data into periods...")

        # Check if Year/FY column exists - use that instead of dates if available
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

        # Group by site
        site_groups = self.historic_records.groupby('Site identifier')

        for site, site_data in site_groups:
            # Group by year within site
            for year_val in site_data[year_column].unique():
                if pd.isna(year_val):
                    continue

                # Extract numeric year from FY2024 format
                if isinstance(year_val, str):
                    match = re.search(r'(\d{4})', year_val)
                    if match:
                        year = int(match.group(1))
                    else:
                        continue
                else:
                    year = int(year_val)

                # Skip current year
                if year == self.current_year:
                    continue

                year_data = site_data[site_data[year_column] == year_val]

                # Create period for this year
                first_row = year_data.iloc[0]
                date_from = self.parse_dates(first_row['Date from'])
                date_to = self.parse_dates(first_row['Date to'])

                period = HistoricPeriod(site, date_from, date_to)

                # Add all VQ data for this year
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

        # Summary
        total_periods = sum(len(periods) for periods in self.site_periods.values())
        print(f"✓ Organized into {total_periods} periods across {len(self.site_periods)} sites")

        # Show examples
        for site in list(self.site_periods.keys())[:3]:
            periods = self.site_periods[site]
            print(f"  • {site}: {len(periods)} periods")
            for period in periods:
                status = "✓ Complete" if period.is_complete else f"⚠ {period.months_covered} months"
                print(f"    - {period.year}: {status}")

    def _organize_by_dates(self):
        """Organize historic records by extracting year from Date columns"""
        import re

        # Check if FY column exists in historic records
        fy_column = None
        for col in self.historic_records.columns:
            if col in ['FY', 'fy', 'Financial Year', 'financial_year', 'Financial_Year']:
                fy_column = col
                break

        # Group by site
        site_groups = self.historic_records.groupby('Site identifier')

        for site, site_data in site_groups:
            # Sort by date
            site_data = site_data.sort_values('Date from')

            # Detect periods (groups of contiguous monthly data or annual records)
            current_period = None

            for idx, row in site_data.iterrows():
                date_from = self.parse_dates(row['Date from'])
                date_to = self.parse_dates(row['Date to'])
                vq = row.get('Volumetric Quantity')

                # Skip if no VQ
                if pd.isna(vq):
                    continue

                # CRITICAL: Skip current period data
                # If FY column exists, use FY comparison (handles financial years correctly)
                # Otherwise, fall back to calendar year comparison
                should_skip = False
                if fy_column and pd.notna(row.get(fy_column)):
                    # Extract year from FY column
                    fy_val = row[fy_column]
                    if isinstance(fy_val, str):
                        match = re.search(r'(\d{4})', fy_val)
                        if match:
                            fy_year = int(match.group(1))
                            should_skip = (fy_year == self.current_year)
                    else:
                        should_skip = (int(fy_val) == self.current_year)
                else:
                    # Fallback: Use calendar year from date
                    should_skip = (date_from.year == self.current_year)

                if should_skip:
                    continue

                timeframe, month = self.determine_timeframe(date_from, date_to)

                # Start new period if needed
                if current_period is None or date_from.year != current_period.year:
                    if current_period is not None:
                        self.site_periods[site].append(current_period)

                    # Create new period
                    current_period = HistoricPeriod(site, date_from, date_to)

                # Add data to period
                if timeframe == 'Monthly' and month:
                    current_period.add_monthly_vq(month, vq)
                    # Update period end date
                    if date_to > current_period.end_date:
                        current_period.end_date = date_to

                elif timeframe == 'Annual':
                    current_period.set_annual_total(vq)
                    current_period.start_date = date_from
                    current_period.end_date = date_to

            # Add final period
            if current_period is not None:
                self.site_periods[site].append(current_period)

        # Summary
        total_periods = sum(len(periods) for periods in self.site_periods.values())
        print(f"✓ Organized into {total_periods} periods across {len(self.site_periods)} sites")

        # Show examples
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

        # CRITICAL FIX: Strip whitespace from column names
        self.historic_features.columns = self.historic_features.columns.str.strip()

        print(f"✓ Loaded historic features file with {len(self.historic_features)} rows")
        print(f"  Columns: {list(self.historic_features.columns)}")

        # Convert numeric columns that might be stored as text
        self._convert_numeric_features()

        # Detect format - check for year column (various names)
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

        return self.historic_features

    def _convert_numeric_features(self):
        """Convert numeric columns in historic features that might be stored as text"""
        # Known numeric feature names
        numeric_keywords = ['turnover', 'transaction', 'revenue', 'sales',
                            'square', 'footage', 'floor', 'space', 'sqft']

        for col in self.historic_features.columns:
            if col == 'Site identifier' or col == 'Year':
                continue

            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(self.historic_features[col]):
                continue

            # Check if column name suggests numeric
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

        # Attach to periods
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
                        # Find matching period
                        for period in self.site_periods[site]:
                            if period.year == year:
                                period.add_feature(feature_name, value)

        print(f"✓ Attached features to periods")

    def _attach_features_format2(self, year_column='Year'):
        """Attach features with Year/FY column"""
        feature_columns = [col for col in self.historic_features.columns
                           if col not in ['Site identifier', year_column, 'Location', 'Date from', 'Date to']]

        if not feature_columns:
            print("⚠️  No feature columns found (only metadata columns present)")
            return

        print(f"✓ Detected {len(feature_columns)} features: {feature_columns}")

        years_detected = set()

        for idx, row in self.historic_features.iterrows():
            site = row.get('Site identifier')
            year = row.get(year_column)

            if pd.isna(site) or pd.isna(year) or site not in self.site_periods:
                continue

            # Handle FY2024 format - extract numeric year
            if isinstance(year, str):
                # Try to extract year from strings like "FY2024", "2024", "fy2024"
                import re
                match = re.search(r'(\d{4})', year)
                if match:
                    year = int(match.group(1))
                else:
                    continue
            else:
                year = int(year)

            years_detected.add(year)

            # Find matching period
            for period in self.site_periods[site]:
                if period.year == year:
                    for feature in feature_columns:
                        value = row[feature]
                        if pd.notna(value):
                            period.add_feature(feature, value)

        print(f"✓ Detected {len(years_detected)} years: {sorted(years_detected)}")
        print(f"✓ Attached features to periods")

        print(f"✓ Attached features to periods")

    def _attach_features_format2_dates(self):
        """
        Attach features from format with 'Date from' and 'Date to' columns.
        Extracts year and month from dates and matches to periods.
        NOW stores features PER MONTH for proper month-by-month comparison.
        """
        from datetime import datetime

        # Get feature columns (exclude metadata)
        exclude_cols = ['Site identifier', 'Location', 'Date from', 'Date to', 'Year', 'Financial Year']
        feature_columns = [col for col in self.historic_features.columns
                           if col not in exclude_cols]

        if not feature_columns:
            print("⚠️  No feature columns found (only metadata columns present)")
            return

        print(f"✓ Detected {len(feature_columns)} features: {feature_columns}")

        # Parse dates and extract years
        years_detected = set()

        for idx, row in self.historic_features.iterrows():
            site = row.get('Site identifier')
            if pd.isna(site) or site not in self.site_periods:
                continue

            # Parse Date from to get year AND month
            date_from = row.get('Date from')
            date_to = row.get('Date to')
            if pd.isna(date_from):
                continue

            try:
                # Try to parse date
                if isinstance(date_from, str):
                    # Try common formats
                    for fmt in ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y']:
                        try:
                            date_from_obj = datetime.strptime(date_from, fmt)
                            year = date_from_obj.year
                            month_name = date_from_obj.strftime('%B')  # Full month name
                            break
                        except:
                            continue
                    else:
                        # Try pandas
                        date_from_obj = pd.to_datetime(date_from, dayfirst=True)
                        year = date_from_obj.year
                        month_name = date_from_obj.strftime('%B')
                elif hasattr(date_from, 'year'):
                    year = date_from.year
                    month_name = date_from.strftime('%B')
                else:
                    continue

                # Also parse date_to to check if monthly or annual
                try:
                    if isinstance(date_to, str):
                        for fmt in ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y']:
                            try:
                                date_to_obj = datetime.strptime(date_to, fmt)
                                break
                            except:
                                continue
                        else:
                            date_to_obj = pd.to_datetime(date_to, dayfirst=True)
                    elif hasattr(date_to, 'year'):
                        date_to_obj = date_to
                    else:
                        date_to_obj = None
                except:
                    date_to_obj = None

                years_detected.add(year)

                # Find matching period
                for period in self.site_periods[site]:
                    if period.year == year:
                        # Store features for this specific month
                        for feature in feature_columns:
                            value = row[feature]
                            if pd.notna(value):
                                period.add_monthly_feature(month_name, feature, value)
                        break
            except Exception as e:
                continue

        print(f"✓ Detected {len(years_detected)} years: {sorted(years_detected)}")
        print(f"✓ Attached features to periods (monthly tracking enabled)")

    def get_most_recent_period(self, site, before_date):
        """
        Get the most recent historic period for a site ending before a given date.

        Parameters:
        -----------
        site : str
            Site identifier
        before_date : datetime
            Find period ending before this date

        Returns:
        --------
        HistoricPeriod or None
        """
        if site not in self.site_periods:
            return None

        # Filter to periods ending before the date
        eligible_periods = [p for p in self.site_periods[site] if p.end_date < before_date]

        if not eligible_periods:
            return None

        # Return most recent
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
            Month name (e.g., 'January')

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

    def get_period_feature(self, period, feature_name):
        """Get a feature value from a period"""
        return period.features.get(feature_name)

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

                # Current data
                has_current_vq = False
                current_vq = None
                if len(month_current) > 0:
                    has_current_vq = month_current['Volumetric Quantity'].notna().any()
                    if has_current_vq:
                        current_vq = month_current['Volumetric Quantity'].iloc[0]

                # Historic data (by year)
                historic_availability = {}

                if site in self.site_periods:
                    for period in self.site_periods[site]:
                        year = period.year

                        # Check if this period has this month
                        has_month_vq = month in period.monthly_data
                        has_features = len(period.features) > 0

                        historic_availability[f'{year}_VQ'] = has_month_vq
                        historic_availability[f'{year}_Features'] = has_features

                        if has_month_vq:
                            historic_availability[f'{year}_VQ_Value'] = period.monthly_data[month]

                # Calculate quality
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

        # Summary
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

    # Helper methods for backward compatibility
    def get_historic_vq(self, site, year, month):
        """Get historic VQ by year and month (backward compatibility)"""
        if site not in self.site_periods:
            return None

        for period in self.site_periods[site]:
            if period.year == year and month in period.monthly_data:
                return period.monthly_data[month]
        return None

    def get_historic_feature(self, site, year, feature):
        """Get historic feature by year (backward compatibility)"""
        if site not in self.site_periods:
            return None

        for period in self.site_periods[site]:
            if period.year == year and feature in period.features:
                return period.features[feature]
        return None