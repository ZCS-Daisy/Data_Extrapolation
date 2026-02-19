"""
Volumetric Data Extrapolation Tool
====================================
Clean rewrite — simple, honest, effective.

Pipeline:
  1. Load & normalise data (flexible date formats, site ID / Location fallback)
  2. Detect features (numeric + categorical) and timeframe (Monthly / Annual)
  3. Statistical tests: Pearson, Spearman, Kendall, Point-biserial on every
     relevant data combination — features labelled exactly as in input file
  4. Rule-based methods: site average, site-seasonal, feature intensity,
     annual-feature x seasonal, historic-adjusted (if historic data present)
  5. ML models: Linear, Polynomial, Random Forest, Gradient Boosting
  6. Per-row selection: highest effective R2 wins
  7. Output: Complete Records, Data Availability, Statistical Analysis,
     Machine Learning Models (with train/test row detail), YoY Analysis

FIXES (latest):
  FIX 1 — VQ column is always 'Volumetric Quantity' (strict exact match, no fuzzy)
  FIX 2 — GHG Category is REQUIRED. All stat tests, ML models, seasonal indices,
           and predictions are scoped to matching GHG Category + Sub Category only.
  FIX 3 — Popup errors for: missing Date from/to columns, unparseable dates,
           missing VQ column, missing GHG Category column. Popup warnings for
           softer issues in historic files.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
from collections import defaultdict
import warnings
import tkinter as tk
from tkinter import messagebox
warnings.filterwarnings('ignore')

MONTHS_ORDER = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

_META = {
    'Row ID', 'Current Timestamp', 'Client Name', 'Client', 'Country',
    'Site identifier', 'Site_identifier', 'site identifier', 'site_id',
    'Location', 'location', 'Site Name', 'site_name',
    'Date from', 'Date to', 'Date_from', 'Date_to',
    'Data Timeframe', 'Timeframe', 'timeframe',
    'GHG Category', 'GHG Sub Category',
    'Volumetric Quantity',
    'Data integrity', 'Estimation Method', 'Data Quality Score',
    'Month', 'Year', 'Financial Year', 'FY',
    'Scope', 'Functional Unit', 'Calculation Method',
    'Emission source', 'Meter Number',
    '_Timeframe', '_Month',
}

# ─────────────────────────────────────────────────────────────────────────────
# FIX 3: Popup error / warning helpers
# ─────────────────────────────────────────────────────────────────────────────

def _popup_error(title, message):
    """Show a blocking error popup and raise ValueError. Falls back to console."""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        messagebox.showerror(title, message, parent=root)
        root.destroy()
    except Exception:
        print(f"\n{'='*60}\n  ERROR: {title}\n  {message}\n{'='*60}\n")
    raise ValueError(f"{title}: {message}")


def _popup_warning(title, message):
    """Show a non-blocking warning popup. Falls back to console."""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        messagebox.showwarning(title, message, parent=root)
        root.destroy()
    except Exception:
        print(f"\n{'='*60}\n  WARNING: {title}\n  {message}\n{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Date helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date(val):
    if val is None:
        return None
    if isinstance(val, (pd.Timestamp, datetime)):
        return pd.Timestamp(val)
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, (int, float)):
        try:
            return pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(val))
        except Exception:
            return None
    if isinstance(val, str):
        for fmt in ('%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y',
                    '%d/%m/%y', '%Y/%m/%d', '%d %b %Y', '%d %B %Y'):
            try:
                return pd.Timestamp(datetime.strptime(val.strip(), fmt))
            except Exception:
                pass
        try:
            return pd.to_datetime(val, dayfirst=True)
        except Exception:
            return None
    return None


def _timeframe(date_from, date_to):
    if date_from is None or date_to is None:
        return 'Unknown', None
    if date_from.day == 1:
        next_mo = date_from + relativedelta(months=1)
        last_day = next_mo - pd.Timedelta(days=1)
        if date_to.date() == last_day.date():
            return 'Monthly', date_from.strftime('%B')
    days = (date_to - date_from).days + 1
    if 365 <= days <= 366:
        return 'Annual', None
    return f'{days}d', None


# ─────────────────────────────────────────────────────────────────────────────
# Feature scanning — used by GUI to populate feature selector
# ─────────────────────────────────────────────────────────────────────────────

def scan_features(filepath):
    """
    Scan a file and return feature candidates with quality scores.
    Returns list of dicts:
      { name, type, fill_pct, unique_count, variance_score, recommended, reason }
    """
    if filepath.lower().endswith('.csv'):
        df = pd.read_csv(filepath, nrows=5000)
    else:
        df = pd.read_excel(filepath, nrows=5000)
    df.columns = df.columns.str.strip()

    skip = set(_META)
    results = []
    n = len(df)

    for col in df.columns:
        if col in skip or col.startswith('_'):
            continue

        non_null = df[col].dropna()
        fill_pct = round(len(non_null) / n * 100, 1) if n > 0 else 0
        if fill_pct < 10 or len(non_null) < 4:
            continue

        unique_count = non_null.nunique()

        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if not is_numeric:
            cleaned = non_null.astype(str).str.replace(r'[,£$€]', '', regex=True).str.strip()
            conv = pd.to_numeric(cleaned, errors='coerce')
            if conv.notna().sum() / len(non_null) > 0.8:
                is_numeric = True
                non_null = conv.dropna()
                unique_count = non_null.nunique()

        if is_numeric:
            if unique_count < 2:
                continue
            mean_val = non_null.mean()
            std_val  = non_null.std()
            cv = (std_val / mean_val) if mean_val != 0 else 0
            variance_score = round(min(float(cv), 1.0), 3)
            feat_type = 'Numeric'
            if fill_pct >= 70 and variance_score >= 0.3 and unique_count >= 10:
                recommended = True
                reason = f'Good fill rate ({fill_pct}%), strong variance (CV={variance_score:.2f}), {unique_count} unique values'
            elif fill_pct >= 50 and variance_score >= 0.15:
                recommended = True
                reason = f'Adequate fill ({fill_pct}%), moderate variance (CV={variance_score:.2f})'
            else:
                recommended = False
                reason = (f'Low fill rate ({fill_pct}%)' if fill_pct < 50
                          else f'Low variance (CV={variance_score:.2f}) — little predictive signal')
        else:
            if unique_count < 2 or unique_count > 30:
                continue
            counts = non_null.value_counts(normalize=True)
            entropy = float(-np.sum(counts * np.log2(counts + 1e-10)))
            max_entropy = np.log2(unique_count) if unique_count > 1 else 1
            variance_score = round(entropy / max_entropy, 3) if max_entropy > 0 else 0
            feat_type = 'Categorical'
            if fill_pct >= 70 and unique_count >= 2 and variance_score >= 0.5:
                recommended = True
                reason = f'Good fill ({fill_pct}%), {unique_count} categories, well-distributed (entropy={variance_score:.2f})'
            elif fill_pct >= 50 and unique_count >= 2:
                recommended = True
                reason = f'Adequate fill ({fill_pct}%), {unique_count} categories'
            else:
                recommended = False
                reason = (f'Low fill rate ({fill_pct}%)' if fill_pct < 50
                          else f'Skewed distribution — one category dominates (entropy={variance_score:.2f})')

        results.append({
            'name': col, 'type': feat_type, 'fill_pct': fill_pct,
            'unique_count': unique_count, 'variance_score': variance_score,
            'recommended': recommended, 'reason': reason,
        })

    results.sort(key=lambda x: (-int(x['recommended']), -x['variance_score']))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Stat helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pearson(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4: return None, None, None
    try:
        r, p = stats.pearsonr(x[m], y[m])
        return float(r), float(r**2), float(p)
    except Exception:
        return None, None, None

def _spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4: return None, None, None
    try:
        r, p = stats.spearmanr(x[m], y[m])
        return float(r), float(r**2), float(p)
    except Exception:
        return None, None, None

def _kendall(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4: return None, None, None
    try:
        tau, p = stats.kendalltau(x[m], y[m])
        return float(tau), float(tau**2), float(p)
    except Exception:
        return None, None, None

def _pointbiserial(codes, y):
    m = np.isfinite(codes) & np.isfinite(y)
    if m.sum() < 4 or len(np.unique(codes[m])) < 2: return None, None, None
    try:
        r, p = stats.pointbiserialr(codes[m], y[m])
        return float(r), float(r**2), float(p)
    except Exception:
        return None, None, None

def _eff_r2(pr2, sr2, kr2=None, pbr2=None):
    cands = [v for v in (pr2, sr2, kr2, pbr2) if v is not None]
    return max(cands) if cands else 0.0

def _cv_r2_with_meta(model, X, y, cv=5):
    n = len(X)
    if n < 4:
        return float('nan'), n, 0, 0, 'insufficient'
    if n < 10:
        scores = cross_val_score(model, X, y, cv=LeaveOneOut(), scoring='r2')
        mean_r2 = float(np.nanmean(scores)) if np.any(np.isfinite(scores)) else float('nan')
        return mean_r2, n, 1, n, 'LeaveOneOut'
    else:
        k = min(cv, n // 2)
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        return float(np.mean(scores)), n, n // k, k, f'{k}-Fold'


# ─────────────────────────────────────────────────────────────────────────────
# Main tool class
# ─────────────────────────────────────────────────────────────────────────────

class ExtrapolationTool:

    # FIX 1: VQ column name is a class constant — always exact, never fuzzy
    VQ_COL = 'Volumetric Quantity'

    def __init__(self, input_file,
                 historic_records_file=None,
                 historic_features_file=None,
                 forced_numeric_features=None,
                 forced_categorical_features=None):
        self.input_file = input_file
        self.historic_records_file = historic_records_file
        self.historic_features_file = historic_features_file
        self.forced_numeric_features = forced_numeric_features
        self.forced_categorical_features = forced_categorical_features

        self.df = None
        self.known = None
        self.blank = None

        self.site_col    = None
        self.vq_col      = self.VQ_COL   # always 'Volumetric Quantity'
        self.ghg_col     = None          # FIX 2: REQUIRED
        self.ghg_sub_col = None          # used when present

        self.numeric_features     = []
        self.categorical_features = []

        # FIX 2: seasonal patterns keyed by (ghg [, ghg_sub]) not just month
        self.seasonal_patterns      = {}   # {(ghg[, sub]): {month: factor}}
        self.site_seasonal_patterns = {}   # {(site, ghg[, sub]): {month: factor}}

        self._vq_lo = 0.0
        self._vq_hi = float('inf')

        self.stat_methods = {}
        self.ml_models    = {}
        self.cat_breakdown = {}

        self.historic_vq            = {}
        self.historic_growth_rates  = {}
        self.historic_feats         = {}
        self.historic_anchor_quality = {}
        self.historic_year  = None
        self.current_year   = None

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self):
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)

        if self.input_file.lower().endswith('.csv'):
            self.df = pd.read_csv(self.input_file)
        else:
            self.df = pd.read_excel(self.input_file)
        self.df.columns = self.df.columns.str.strip()
        print(f"  {len(self.df)} rows x {len(self.df.columns)} columns")

        # ── FIX 1: VQ column must be exactly 'Volumetric Quantity' ───────────
        if self.VQ_COL not in self.df.columns:
            _popup_error(
                "Missing Column: Volumetric Quantity",
                f"The input file must contain a column named exactly "
                f"'Volumetric Quantity'.\n\n"
                f"Columns found: {list(self.df.columns)}\n\n"
                f"Please rename the column and re-run."
            )

        # ── FIX 2: GHG Category is REQUIRED ──────────────────────────────────
        self.ghg_col = self._col(['GHG Category', 'GHG_Category', 'ghg_category'])
        if not self.ghg_col:
            _popup_error(
                "Missing Column: GHG Category",
                "The input file must contain a 'GHG Category' column.\n\n"
                "GHG Category is required so that each emission type "
                "(Electricity, Gas, Water, etc.) is predicted only from "
                "data within the same category.\n\n"
                "Please add this column and re-run."
            )

        self.ghg_sub_col = self._col(['GHG Sub Category', 'GHG_Sub_Category'])

        # ── Site column ───────────────────────────────────────────────────────
        self.site_col = self._col(['Site identifier', 'Site_identifier', 'Site ID', 'SiteID'])
        if not self.site_col:
            self.site_col = self._col(['Location', 'location', 'Site Name'])
            if self.site_col:
                print(f"  Using '{self.site_col}' as site key")
            else:
                _popup_error(
                    "Missing Column: Site Identifier",
                    "Cannot find a site identifier or location column.\n\n"
                    "Expected one of: 'Site identifier', 'Location', 'Site Name'."
                )

        # ── FIX 3: Date columns validated before anything else ────────────────
        self._validate_date_columns()
        self._add_timeframe_cols()
        self._coerce_numeric()

        cats = self.df[self.ghg_col].dropna().unique()
        print(f"  GHG categories found: {list(cats)}")
        print(f"  Predictions keyed on: Site + GHG Category"
              + (" + GHG Sub Category" if self.ghg_sub_col else ""))

        self._detect_features()

        self.known = self.df[self.df[self.vq_col].notna()].copy()
        self.blank = self.df[self.df[self.vq_col].isna()].copy()
        print(f"  Known VQ: {len(self.known)} | Missing VQ: {len(self.blank)}")

        from_col = self._col(['Date from', 'Date_from'])
        if from_col:
            for v in self.df[from_col].dropna():
                p = _parse_date(v)
                if p:
                    self.current_year = p.year
                    break

        if self.historic_records_file:
            self._load_historic_records()
        if self.historic_features_file:
            self._load_historic_features()
        return self

    # ── FIX 3: Date validation ────────────────────────────────────────────────

    def _validate_date_columns(self):
        """Popup errors for missing or unparseable date columns."""
        from_col = self._col(['Date from', 'Date_from'])
        to_col   = self._col(['Date to',   'Date_to'])

        if not from_col:
            _popup_error(
                "Missing Column: Date from",
                "The input file must contain a 'Date from' column.\n\n"
                "This is required to classify rows as Monthly or Annual "
                "and to align historic data with the correct year.\n\n"
                "Please add this column (format: DD/MM/YYYY) and re-run."
            )
        if not to_col:
            _popup_error(
                "Missing Column: Date to",
                "The input file must contain a 'Date to' column.\n\n"
                "Required alongside 'Date from' to determine timeframe.\n\n"
                "Please add this column (format: DD/MM/YYYY) and re-run."
            )

        for col_label, col in [('Date from', from_col), ('Date to', to_col)]:
            sample = self.df[col].dropna().head(50)
            if len(sample) == 0:
                _popup_error(
                    f"Empty Column: {col_label}",
                    f"'{col_label}' exists but contains no values.\n\n"
                    f"Please populate this column and re-run."
                )
            failed = [v for v in sample if _parse_date(v) is None]
            if failed:
                examples = ', '.join(str(v) for v in failed[:5])
                _popup_error(
                    f"Unparseable Dates: {col_label}",
                    f"Some values in '{col_label}' could not be parsed as dates.\n\n"
                    f"Examples: {examples}\n\n"
                    f"Supported formats: DD/MM/YYYY, YYYY-MM-DD, DD-MM-YYYY, "
                    f"DD/MM/YY, DD Mon YYYY.\n\n"
                    f"Please fix these and re-run."
                )

        print(f"  Date columns validated: '{from_col}', '{to_col}'")

    def _col(self, candidates):
        low = {c.lower().strip(): c for c in self.df.columns}
        for c in candidates:
            if c in self.df.columns:
                return c
            if c.lower() in low:
                return low[c.lower()]
        return None

    def _composite_key(self, row_or_site, ghg=None, ghg_sub=None):
        """
        Build composite lookup key: (site, ghg [, ghg_sub]).
        GHG Category is always present in the main file (enforced above).
        GHG Sub Category only included when populated on this row.
        """
        _BLANK = {'', 'nan', 'none', 'n/a', '-'}

        if isinstance(row_or_site, (dict, pd.Series)):
            row     = row_or_site
            site    = str(row.get(self.site_col, '') or '').strip()
            ghg     = str(row.get(self.ghg_col,     '') or '').strip()
            ghg_sub = str(row.get(self.ghg_sub_col, '') or '').strip() if self.ghg_sub_col else ''
        else:
            site    = str(row_or_site or '').strip()
            ghg     = str(ghg     or '').strip()
            ghg_sub = str(ghg_sub or '').strip()

        if ghg.lower()     in _BLANK: ghg     = ''
        if ghg_sub.lower() in _BLANK: ghg_sub = ''

        if ghg and ghg_sub:
            return (site, ghg, ghg_sub)
        elif ghg:
            return (site, ghg)
        else:
            return (site,)

    def _add_timeframe_cols(self):
        from_col = self._col(['Date from', 'Date_from'])
        to_col   = self._col(['Date to',   'Date_to'])
        fs = self.df[from_col] if from_col else pd.Series([None]*len(self.df))
        ts = self.df[to_col]   if to_col   else pd.Series([None]*len(self.df))
        tfs, mos = [], []
        for f, t in zip(fs, ts):
            tf, mo = _timeframe(_parse_date(f), _parse_date(t))
            tfs.append(tf)
            mos.append(mo or '')
        self.df['_Timeframe'] = tfs
        self.df['_Month']     = mos
        print(f"  Timeframes: {pd.Series(tfs).value_counts().to_dict()}")

    def _coerce_numeric(self):
        hints = ['turnover', 'transaction', 'revenue', 'sales', 'sqft', 'floor',
                 'space', 'electricity', 'area', 'count', 'quantity', 'volume',
                 'consumption', 'spend', 'units', 'kw', 'kwh', 'water', 'waste',
                 'headcount', 'staff', 'seats', 'covers', 'hours']
        for col in self.df.columns:
            if col in _META or col.startswith('_'):
                continue
            if pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            non_null = self.df[col].dropna()
            if not len(non_null):
                continue
            cleaned = non_null.astype(str).str.replace(r'[,£$€]', '', regex=True).str.strip()
            conv    = pd.to_numeric(cleaned, errors='coerce')
            rate    = conv.notna().sum() / len(non_null)
            hint    = any(h in col.lower() for h in hints)
            if (rate > 0.8 and hint) or rate > 0.95:
                self.df[col] = pd.to_numeric(
                    self.df[col].astype(str).str.replace(r'[,£$€]', '', regex=True).str.strip(),
                    errors='coerce'
                )
                if rate > 0.8 and hint:
                    print(f"  Coerced '{col}' to numeric ({rate:.0%} convertible)")

    def _detect_features(self):
        if self.forced_numeric_features is not None or self.forced_categorical_features is not None:
            self.numeric_features     = list(self.forced_numeric_features or [])
            self.categorical_features = list(self.forced_categorical_features or [])
            print(f"  Numeric features (user-selected):     {self.numeric_features}")
            print(f"  Categorical features (user-selected): {self.categorical_features}")
            return

        skip = _META | {self.site_col, self.vq_col}
        self.numeric_features, self.categorical_features = [], []
        for col in self.df.columns:
            if col in skip or col.startswith('_'):
                continue
            nn = self.df[col].dropna()
            if len(nn) < 4:
                continue
            if pd.api.types.is_numeric_dtype(self.df[col]) and nn.nunique() > 1:
                self.numeric_features.append(col)
            elif not pd.api.types.is_numeric_dtype(self.df[col]) and 2 <= nn.nunique() <= 30:
                self.categorical_features.append(col)

        print(f"  Numeric features (auto):       {self.numeric_features}")
        print(f"  Categorical features (auto):   {self.categorical_features}")

    # ── GHG filter helpers ────────────────────────────────────────────────────

    def _filter_to_ghg(self, df_in, ghg_val, ghg_sub_val=None):
        """
        FIX 2: Filter df to rows matching ghg_val (and ghg_sub_val if provided).
        All stat tests and ML training call this before using any data.
        """
        mask = df_in[self.ghg_col].astype(str).str.strip() == str(ghg_val).strip()
        result = df_in[mask]
        if (self.ghg_sub_col and ghg_sub_val is not None
                and str(ghg_sub_val).strip().lower() not in ('', 'nan', 'none', 'n/a', '-')):
            sub_mask = result[self.ghg_sub_col].astype(str).str.strip() == str(ghg_sub_val).strip()
            result = result[sub_mask]
        return result

    def _unique_ghg_combos(self, df_in):
        """List of (ghg, ghg_sub_or_None) tuples present in df_in."""
        cols = [self.ghg_col] + ([self.ghg_sub_col] if self.ghg_sub_col else [])
        combos = set()
        for _, row in df_in[cols].drop_duplicates().iterrows():
            ghg = str(row[self.ghg_col]).strip() if pd.notna(row[self.ghg_col]) else ''
            sub = None
            if self.ghg_sub_col:
                raw = row.get(self.ghg_sub_col)
                if pd.notna(raw) and str(raw).strip().lower() not in ('', 'nan', 'none', 'n/a', '-'):
                    sub = str(raw).strip()
            if ghg:
                combos.add((ghg, sub))
        return sorted(combos, key=lambda x: (x[0], x[1] or ''))

    # ── Historic loading ──────────────────────────────────────────────────────

    def _load_historic_records(self):
        print("\n  Loading historic records...")
        try:
            hrec = (pd.read_csv(self.historic_records_file)
                    if self.historic_records_file.lower().endswith('.csv')
                    else pd.read_excel(self.historic_records_file))
            hrec.columns = hrec.columns.str.strip()

            scol = next((c for c in ['Site identifier', 'Site_identifier', 'Location']
                         if c in hrec.columns), None)
            if not scol:
                _popup_error(
                    "Missing Column in Historic Records: Site Identifier",
                    "Historic records file must have a site identifier column "
                    "('Site identifier' or 'Location')."
                )

            # FIX 1: VQ column must be 'Volumetric Quantity' in historic file too
            if self.VQ_COL not in hrec.columns:
                _popup_error(
                    "Missing Column in Historic Records: Volumetric Quantity",
                    f"Historic records file must contain 'Volumetric Quantity'.\n\n"
                    f"Columns found: {list(hrec.columns)}"
                )
            vcol = self.VQ_COL

            from_col      = next((c for c in hrec.columns if 'date from' in c.lower()), None)
            to_col        = next((c for c in hrec.columns if 'date to'   in c.lower()), None)
            integrity_col = next((c for c in hrec.columns if 'data integrity' in c.lower()), None)
            h_ghg_col     = next((c for c in hrec.columns if c.lower() in ('ghg category', 'ghg_category')), None)
            h_ghgsub_col  = next((c for c in hrec.columns if c.lower() in ('ghg sub category', 'ghg_sub_category')), None)

            # FIX 3: Warn if historic file missing date or GHG columns
            if not from_col:
                _popup_warning(
                    "Missing Date Column in Historic Records",
                    "Historic records file has no 'Date from' column.\n\n"
                    "Rows cannot be assigned to a year — growth rate projection "
                    "and historic_year detection will not be available.\n\n"
                    "Historic VQ will still be loaded for direct matching."
                )
            if not h_ghg_col:
                _popup_warning(
                    "Missing GHG Category in Historic Records",
                    "Historic records file has no 'GHG Category' column.\n\n"
                    "Data will be matched on site identifier only, which may "
                    "cause cross-category contamination. It is strongly recommended "
                    "to add a GHG Category column."
                )

            raw_rows = defaultdict(list)
            years_seen = set()
            historic_integrity = {}

            for _, row in hrec.iterrows():
                site = row.get(scol)
                vq   = row.get(vcol)
                if pd.isna(site) or pd.isna(vq):
                    continue
                site = str(site).strip()

                pf = _parse_date(row.get(from_col)) if from_col else None
                pt = _parse_date(row.get(to_col))   if to_col   else pf
                tf, mo = _timeframe(pf, pt) if pf else ('Unknown', None)
                year = pf.year if pf else None
                if year:
                    years_seen.add(year)

                period = mo or 'ANNUAL'
                integ  = None
                if integrity_col:
                    iv = row.get(integrity_col)
                    integ = str(iv).strip() if pd.notna(iv) else None

                ghg     = str(row.get(h_ghg_col,    '')).strip() if h_ghg_col    else ''
                ghg_sub = str(row.get(h_ghgsub_col, '')).strip() if h_ghgsub_col else ''
                if ghg and ghg_sub and self.ghg_sub_col:
                    ckey = (site, ghg, ghg_sub)
                elif ghg and self.ghg_col:
                    ckey = (site, ghg)
                else:
                    ckey = (site,)

                raw_rows[(ckey, period)].append((year, float(vq), integ))

            curr_yrs = set()
            fc = self._col(['Date from', 'Date_from'])
            if fc:
                for v in self.df[fc].dropna():
                    p = _parse_date(v)
                    if p:
                        curr_yrs.add(p.year)
            past_yrs = years_seen - curr_yrs
            if past_yrs:
                self.historic_year = max(past_yrs)

            growth_accumulator = defaultdict(list)

            for (ckey, period), year_rows in raw_rows.items():
                year_rows_sorted = sorted(
                    [(y, vq, ig) for y, vq, ig in year_rows if y is not None],
                    key=lambda x: x[0])
                no_year = [(y, vq, ig) for y, vq, ig in year_rows if y is None]

                if year_rows_sorted:
                    _, best_vq, best_integ = year_rows_sorted[-1]
                    self.historic_vq.setdefault(ckey, {})[period] = best_vq
                    if best_integ:
                        historic_integrity.setdefault(ckey, {})[period] = best_integ
                    for i in range(1, len(year_rows_sorted)):
                        y0, vq0, _ = year_rows_sorted[i-1]
                        y1, vq1, _ = year_rows_sorted[i]
                        yr_gap = y1 - y0
                        if yr_gap > 0 and vq0 > 0 and vq1 > 0:
                            growth_accumulator[ckey].append((vq1 / vq0) ** (1.0 / yr_gap))
                elif no_year:
                    _, best_vq, best_integ = no_year[-1]
                    self.historic_vq.setdefault(ckey, {})[period] = best_vq
                    if best_integ:
                        historic_integrity.setdefault(ckey, {})[period] = best_integ

            for ckey, ratios in growth_accumulator.items():
                if ratios:
                    self.historic_growth_rates[ckey] = float(np.median(ratios))

            self.historic_anchor_quality = {}
            for ckey in self.historic_vq:
                if not integrity_col or ckey not in historic_integrity:
                    self.historic_anchor_quality[ckey] = 'unknown'
                    continue
                site_integ = historic_integrity[ckey]
                if 'ANNUAL' in site_integ:
                    val = site_integ['ANNUAL']
                    self.historic_anchor_quality[ckey] = 'good' if (val and val.lower() == 'actual') else 'poor'
                else:
                    n_actual = sum(1 for v in site_integ.values() if v and v.lower() == 'actual')
                    self.historic_anchor_quality[ckey] = 'good' if n_actual >= 3 else 'poor'

            n_growth = len(self.historic_growth_rates)
            print(f"  Historic records: {len(self.historic_vq)} keys, year={self.historic_year}")
            print(f"  Years in historic file: {sorted(years_seen)}")
            if n_growth:
                print(f"  Growth rates computed for {n_growth} keys")

        except ValueError:
            raise
        except Exception as e:
            import traceback
            print(f"  ⚠  Historic records error: {e}")
            traceback.print_exc()

    def _load_historic_features(self):
        print("  Loading historic features...")
        try:
            hf = (pd.read_csv(self.historic_features_file)
                  if self.historic_features_file.lower().endswith('.csv')
                  else pd.read_excel(self.historic_features_file))
            hf.columns = hf.columns.str.strip()
            scol = next((c for c in ['Site identifier', 'Site_identifier', 'Location']
                         if c in hf.columns), None)
            if not scol:
                _popup_error(
                    "Missing Column in Historic Features: Site Identifier",
                    "Historic features file must have a site identifier column."
                )

            _GHG_COLS = {'GHG Category', 'GHG_Category', 'ghg_category',
                         'GHG Sub Category', 'GHG_Sub_Category'}
            skip_cols = _META | _GHG_COLS | {scol}
            feat_cols = [c for c in hf.columns if c not in skip_cols]

            if not feat_cols:
                _popup_warning(
                    "No Feature Columns in Historic Features File",
                    "No usable feature columns found after excluding metadata.\n\n"
                    "Historic feature adjustment will not be available."
                )
                return

            for _, row in hf.iterrows():
                site = row.get(scol)
                if pd.isna(site):
                    continue
                site = str(site).strip()
                self.historic_feats.setdefault(site, {})
                for fc in feat_cols:
                    v = row.get(fc)
                    if pd.notna(v):
                        try:
                            self.historic_feats[site][fc] = float(v)
                        except Exception:
                            self.historic_feats[site][fc] = v

            print(f"  Historic features: {len(self.historic_feats)} sites, columns: {feat_cols}")
        except ValueError:
            raise
        except Exception as e:
            print(f"  ⚠  Historic features error: {e}")

    # ── Seasonal ──────────────────────────────────────────────────────────────

    def _calc_seasonal(self):
        """
        FIX 2: Seasonal indices computed per GHG category, not across all categories.
        self.seasonal_patterns keyed on (ghg [, ghg_sub]): {month: factor}
        self.site_seasonal_patterns keyed on (site, ghg [, ghg_sub]): {month: factor}
        """
        mo = self.known[self.known['_Timeframe'] == 'Monthly']
        if len(mo) < 4:
            return

        self.seasonal_patterns      = {}
        self.site_seasonal_patterns = {}

        for ghg, ghg_sub in self._unique_ghg_combos(mo):
            cat_mo  = self._filter_to_ghg(mo, ghg, ghg_sub)
            cat_key = (ghg, ghg_sub) if ghg_sub else (ghg,)
            if len(cat_mo) < 4:
                continue

            avgs    = cat_mo.groupby('_Month')[self.vq_col].mean()
            overall = avgs.mean()
            if overall > 0:
                self.seasonal_patterns[cat_key] = {m: float(v / overall) for m, v in avgs.items()}

            for site, grp in cat_mo.groupby(self.site_col):
                if grp['_Month'].nunique() < 3:
                    continue
                site_avgs    = grp.groupby('_Month')[self.vq_col].mean()
                site_overall = site_avgs.mean()
                if site_overall > 0:
                    sk = (str(site).strip(),) + cat_key
                    self.site_seasonal_patterns[sk] = {
                        m: float(v / site_overall) for m, v in site_avgs.items()}

        print(f"  Seasonal index: {len(self.seasonal_patterns)} GHG categories, "
              f"{len(self.site_seasonal_patterns)} site+category combos")

    def _seasonal_factor(self, site, month, ghg=None, ghg_sub=None):
        if not month:
            return 1.0
        cat_key      = (ghg, ghg_sub) if ghg_sub else (ghg,) if ghg else ()
        site_cat_key = (site,) + cat_key if cat_key else (site,)
        site_idx = self.site_seasonal_patterns.get(site_cat_key, {})
        if month in site_idx:
            return site_idx[month]
        return self.seasonal_patterns.get(cat_key, {}).get(month, 1.0)

    def _sanity_bounds(self):
        vq_vals = self.known[self.vq_col].dropna().values
        if len(vq_vals) == 0:
            return 0.0, float('inf')
        return 0.0, float(np.percentile(vq_vals, 99)) * 3.0

    def _clip(self, value, lo, hi):
        if value is None or not np.isfinite(value):
            return None
        if value < lo:
            return None
        return min(value, hi)

    # ── Statistical tests ─────────────────────────────────────────────────────

    def run_statistical_tests(self):
        """
        FIX 2: All methods computed per GHG category. Method names prefixed
        with '<GHG>[__<Sub>]__' to ensure predictions only use same-category stats.
        """
        print("\n" + "=" * 60)
        print("STATISTICAL ANALYSIS")
        print("=" * 60)
        self._calc_seasonal()

        for ghg, ghg_sub in self._unique_ghg_combos(self.known):
            pfx       = ghg + (f'__{ghg_sub}' if ghg_sub else '') + '__'
            cat_known = self._filter_to_ghg(self.known, ghg, ghg_sub)
            mn        = cat_known[cat_known['_Timeframe'] == 'Monthly']
            an        = cat_known[cat_known['_Timeframe'] == 'Annual']
            cat_key   = (ghg, ghg_sub) if ghg_sub else (ghg,)
            cat_sea   = self.seasonal_patterns.get(cat_key, {})

            if len(cat_known) < 4:
                continue

            label = ghg + (f' / {ghg_sub}' if ghg_sub else '')
            print(f"  [{label}] {len(cat_known)} rows ({len(mn)} monthly, {len(an)} annual)")

            # 1 – Site average
            if len(mn) >= 4:
                sa    = mn.groupby(self.site_col)[self.vq_col].mean()
                preds = mn[self.site_col].map(sa).fillna(sa.mean()).values.astype(float)
                acts  = mn[self.vq_col].values.astype(float)
                pr, pr2, pp = _pearson(preds, acts)
                sr, sr2, sp = _spearman(preds, acts)
                kr, kr2, kp = _kendall(preds, acts)
                self._reg(f'{pfx}Site_Average',
                          f'[{label}] Site average VQ (monthly)',
                          f'{len(mn)} monthly rows',
                          None, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None,
                          _eff_r2(pr2, sr2, kr2),
                          _site_avgs=sa.to_dict(), _ghg=ghg, _ghg_sub=ghg_sub)

            # 2 – Site average x seasonal
            if len(mn) >= 4 and cat_sea:
                sa2  = mn.groupby(self.site_col)[self.vq_col].mean().to_dict()
                fall = mn[self.vq_col].mean()
                p2   = np.array([sa2.get(r[self.site_col], fall) * cat_sea.get(r['_Month'], 1.0)
                                 for _, r in mn.iterrows()], float)
                a2   = mn[self.vq_col].values.astype(float)
                pr, pr2, pp = _pearson(p2, a2); sr, sr2, sp = _spearman(p2, a2); kr, kr2, kp = _kendall(p2, a2)
                self._reg(f'{pfx}Site_Average_Seasonal',
                          f'[{label}] Site average x seasonal factor',
                          f'{len(mn)} monthly rows',
                          None, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None,
                          _eff_r2(pr2, sr2, kr2),
                          _site_avgs=sa2, _seasonal=True, _ghg=ghg, _ghg_sub=ghg_sub)

            # 3 – Cross-site seasonal average
            if len(mn) >= 4 and cat_sea:
                ma     = mn.groupby('_Month')[self.vq_col].mean().to_dict()
                preds3 = [ma.get(r['_Month']) for _, r in mn.iterrows()]
                p3 = np.array([v for v in preds3 if v is not None], float)
                a3 = mn[self.vq_col].values[:len(p3)].astype(float)
                pr, pr2, pp = _pearson(p3, a3); sr, sr2, sp = _spearman(p3, a3); kr, kr2, kp = _kendall(p3, a3)
                self._reg(f'{pfx}Seasonal_Average',
                          f'[{label}] Cross-site mean VQ per month',
                          f'{len(mn)} monthly rows',
                          None, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None,
                          _eff_r2(pr2, sr2, kr2),
                          _month_avgs=ma, _ghg=ghg, _ghg_sub=ghg_sub)

            # 4 – Overall average fallback
            for sub, lbl in [(cat_known, 'all'), (an, 'annual')]:
                if len(sub) >= 4:
                    self._reg(f'{pfx}Overall_Average_{lbl}',
                              f'[{label}] Overall mean {lbl} VQ',
                              f'{len(sub)} rows',
                              None, None, None, None, None, None, None, None, None, None,
                              None, None, None, 0.0,
                              _avg=float(sub[self.vq_col].mean()), _ghg=ghg, _ghg_sub=ghg_sub)
                    break

            # 5 – Feature intensity
            for feat in self.numeric_features:
                pool = cat_known[cat_known[feat].notna()]
                if len(pool) < 4:
                    continue
                vq = pool[self.vq_col].values.astype(float)
                fv = pool[feat].values.astype(float)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratios = np.where(fv > 0, vq / fv, np.nan)
                slope = float(np.nanmedian(ratios)) if np.any(np.isfinite(ratios)) else None
                pr, pr2, pp = _pearson(fv, vq); sr, sr2, sp = _spearman(fv, vq); kr, kr2, kp = _kendall(fv, vq)
                eff = _eff_r2(pr2, sr2, kr2)
                self._reg(f'{pfx}Intensity_{feat}',
                          f'[{label}] VQ ~ {feat} intensity',
                          f'{len(pool)} rows',
                          slope, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None, eff,
                          _feat=feat, _mtype='Feature_Intensity', _ghg=ghg, _ghg_sub=ghg_sub)

                pm = pool[pool['_Timeframe'] == 'Monthly']
                if len(pm) >= 4 and cat_sea and slope:
                    ps  = np.array([float(r[feat]) * slope * cat_sea.get(r['_Month'], 1.0)
                                    for _, r in pm.iterrows()], float)
                    as_ = pm[self.vq_col].values.astype(float)
                    if len(ps) >= 4:
                        pr, pr2, pp = _pearson(ps, as_); sr, sr2, sp = _spearman(ps, as_); kr, kr2, kp = _kendall(ps, as_)
                        self._reg(f'{pfx}Intensity_{feat}_Seasonal',
                                  f'[{label}] VQ ~ {feat} intensity x seasonal',
                                  f'{len(ps)} monthly rows',
                                  slope, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None,
                                  _eff_r2(pr2, sr2, kr2),
                                  _feat=feat, _mtype='Feature_Intensity_Seasonal',
                                  _seasonal=True, _ghg=ghg, _ghg_sub=ghg_sub)

            # 6 – Categorical
            for feat in self.categorical_features:
                pool = cat_known[cat_known[feat].notna()]
                if len(pool) < 4:
                    continue
                vq    = pool[self.vq_col].values.astype(float)
                codes = pd.Categorical(pool[feat]).codes.astype(float)
                pbr, pbr2, pbp = _pointbiserial(codes, vq)
                self._reg(f'{pfx}Categorical_{feat}',
                          f'[{label}] Group mean VQ by {feat}',
                          f'{len(pool)} rows',
                          None, None, None, None, None, None, None, None, None, None,
                          pbr, pbr2, pbp, _eff_r2(None, None, pbr2=pbr2),
                          _feat=feat, _mtype='Categorical',
                          _cat_means=pool.groupby(feat)[self.vq_col].mean().to_dict(),
                          _ghg=ghg, _ghg_sub=ghg_sub)

            # 7 – Historic-adjusted
            if self.historic_vq:
                for feat in self.numeric_features:
                    preds_h, acts_h = [], []
                    for _, row in mn.iterrows():
                        site  = str(row.get(self.site_col, '')).strip()
                        month = row['_Month']
                        cf    = row.get(feat)
                        if pd.isna(cf) or not month:
                            continue
                        ckey = self._composite_key(row)
                        hvq  = self.historic_vq.get(ckey, {}).get(month)
                        if hvq is None:
                            ha = self.historic_vq.get(ckey, {}).get('ANNUAL')
                            if ha:
                                hvq = (ha / 12) * cat_sea.get(month, 1.0)
                        if hvq is None:
                            continue
                        hf = self.historic_feats.get(site, {}).get(feat)
                        if hf and float(hf) != 0:
                            preds_h.append(hvq * (float(cf) / float(hf)))
                            acts_h.append(row[self.vq_col])
                    if len(preds_h) >= 4:
                        ph, ah = np.array(preds_h, float), np.array(acts_h, float)
                        pr, pr2, pp = _pearson(ph, ah); sr, sr2, sp = _spearman(ph, ah); kr, kr2, kp = _kendall(ph, ah)
                        self._reg(f'{pfx}Historic_Adjusted_{feat}',
                                  f'[{label}] Historic VQ x ({feat} ratio)',
                                  f'{len(preds_h)} rows',
                                  None, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None,
                                  _eff_r2(pr2, sr2, kr2),
                                  _feat=feat, _mtype='Historic_Adjusted', _ghg=ghg, _ghg_sub=ghg_sub)

                pl, al = [], []
                for _, row in mn.iterrows():
                    ckey = self._composite_key(row)
                    hvq  = self.historic_vq.get(ckey, {}).get(row['_Month'])
                    if hvq is not None:
                        pl.append(hvq); al.append(row[self.vq_col])
                if len(pl) >= 4:
                    pla, ala = np.array(pl, float), np.array(al, float)
                    pr, pr2, pp = _pearson(pla, ala); sr, sr2, sp = _spearman(pla, ala); kr, kr2, kp = _kendall(pla, ala)
                    self._reg(f'{pfx}Historic_LastYear_Direct',
                              f'[{label}] Same month previous year',
                              f'{len(pl)} rows',
                              None, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None,
                              _eff_r2(pr2, sr2, kr2),
                              _mtype='Historic_Direct', _ghg=ghg, _ghg_sub=ghg_sub)

        print(f"  {len(self.stat_methods)} stat/rule methods registered")

    def _reg(self, name, desc, input_data, slope,
             pr, pr2, pp, sr, sr2, sp, kr, kr2, kp,
             pbr, pbr2, pbp, eff, **kw):
        self.stat_methods[name] = dict(
            description=desc, input_data=input_data, slope=slope,
            pearson_r=pr, pearson_r2=pr2, pearson_p=pp,
            spearman_r=sr, spearman_r2=sr2, spearman_p=sp,
            kendall_tau=kr, kendall_r2=kr2, kendall_p=kp,
            pointbiserial_r=pbr, pointbiserial_r2=pbr2, pointbiserial_p=pbp,
            effective_r2=eff, predictions_made=0, **kw
        )

    # ── ML models ─────────────────────────────────────────────────────────────

    def train_ml_models(self):
        """FIX 2: Models trained per GHG category; keys prefixed accordingly."""
        print("\n" + "=" * 60)
        print("MACHINE LEARNING MODELS")
        print("=" * 60)

        data_sources = [f'Input file ({len(self.known)} known rows)']
        if self.historic_vq:
            data_sources.append(f'Historic records ({len(self.historic_vq)} keys)')
        if self.historic_feats:
            data_sources.append(f'Historic features ({len(self.historic_feats)} sites)')
        self._data_sources_label = ' + '.join(data_sources)

        for ghg, ghg_sub in self._unique_ghg_combos(self.known):
            pfx       = ghg + (f'__{ghg_sub}' if ghg_sub else '') + '__'
            cat_known = self._filter_to_ghg(self.known, ghg, ghg_sub)
            mn        = cat_known[cat_known['_Timeframe'] == 'Monthly']

            if len(mn) >= 6:
                self._train_subset(mn,
                                   ['_Month'] + self.numeric_features,
                                   f'{pfx}Monthly', self.categorical_features)
            if len(cat_known) >= 6:
                self._train_subset(cat_known,
                                   self.numeric_features,
                                   f'{pfx}All', self.categorical_features)
            for cat in self.categorical_features:
                sub = cat_known[cat_known[cat].notna()]
                if len(sub) >= 6:
                    self._train_subset(sub,
                                       ['_Month'] + self.numeric_features,
                                       f'{pfx}By_{cat}', [cat] + self.categorical_features)

        print(f"  {len(self.ml_models)} ML models trained")

    def _train_subset(self, df_sub, feat_cols, prefix, cat_cols=None):
        cat_cols = cat_cols or []
        encoders, valid = {}, []

        for fc in feat_cols:
            if fc == '_Month' and '_Month' in df_sub.columns:
                enc = LabelEncoder()
                enc.fit(df_sub['_Month'].fillna('Unknown'))
                encoders['_Month'] = enc
                valid.append('_Month')
            elif fc in cat_cols and fc in df_sub.columns:
                enc = LabelEncoder()
                enc.fit(df_sub[fc].fillna('Unknown').astype(str))
                encoders[fc] = enc
                valid.append(fc)
            elif (fc in df_sub.columns
                  and pd.api.types.is_numeric_dtype(df_sub[fc])
                  and df_sub[fc].notna().mean() >= 0.2):
                valid.append(fc)

        if not valid:
            return

        rows_ok = df_sub.copy()
        for fc in valid:
            if fc not in encoders:
                rows_ok = rows_ok[rows_ok[fc].notna()]
        rows_ok = rows_ok[rows_ok[self.vq_col].notna()]
        if len(rows_ok) < 6:
            return

        X_parts = []
        for fc in valid:
            if fc in encoders:
                vals = rows_ok[fc].fillna('Unknown').astype(str)
                try:
                    X_parts.append(encoders[fc].transform(vals).reshape(-1, 1))
                except Exception:
                    enc2 = LabelEncoder()
                    enc2.fit(vals)
                    encoders[fc] = enc2
                    X_parts.append(enc2.transform(vals).reshape(-1, 1))
            else:
                med = rows_ok[fc].median()
                X_parts.append(rows_ok[fc].fillna(med).values.reshape(-1, 1))

        X = np.hstack(X_parts).astype(float)
        y = rows_ok[self.vq_col].values.astype(float)
        feat_label = ', '.join(fc.replace('_Month', 'Month').replace('_', ' ').strip() for fc in valid)

        for mname, mobj in [
            ('Linear_Regression',    LinearRegression()),
            ('Polynomial_Regression', make_pipeline(PolynomialFeatures(2), LinearRegression())),
            ('Random_Forest',         RandomForestRegressor(n_estimators=100, random_state=42,
                                                             max_depth=10, n_jobs=-1)),
            ('Gradient_Boosting',     GradientBoostingRegressor(n_estimators=100, random_state=42,
                                                                  max_depth=5)),
        ]:
            key = f'{prefix}_{mname}'
            r2, n_train, n_test, n_folds, cv_type = _cv_r2_with_meta(mobj, X, y)
            try:
                mobj.fit(X, y)
            except Exception:
                continue
            self.ml_models[key] = dict(
                model=mobj, features=valid, encoders=encoders,
                r2=float(r2), type=mname.replace('_', ' '),
                description=f'{mname.replace("_"," ")} using {feat_label}. CV R2={r2:.3f} (n={n_train})',
                n_train=n_train, n_test_per_fold=n_test, n_folds=n_folds, cv_type=cv_type,
                data_sources=getattr(self, '_data_sources_label', ''),
                predictions_made=0,
            )
            print(f"  {'✓' if r2>=0.3 else '~'} {key}: R2={r2:.4f} "
                  f"[train={n_train}, test/fold={n_test}, {cv_type}]")

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_row(self, row):
        """
        Two-stage waterfall. FIX 2: every tier uses GHG-prefixed methods so
        only same-category data informs a prediction.

        STAGE 1 — anchor:
          Tier 1: Site has own actuals (same category)
          Tier 2: Site has historic data (same category) ± growth projection
          Tier 3: categorical mean → GHG-specific ML → feature intensity → seasonal avg
        STAGE 2 — multiply by per-category seasonal index.
        """
        site    = str(row.get(self.site_col, '')).strip()
        month   = row.get('_Month', '')
        tf      = row.get('_Timeframe', '')
        ghg     = str(row.get(self.ghg_col, '') or '').strip()
        ghg_sub = str(row.get(self.ghg_sub_col, '') or '').strip() if self.ghg_sub_col else ''
        if ghg_sub.lower() in ('', 'nan', 'none', 'n/a', '-'):
            ghg_sub = ''

        pfx    = ghg + (f'__{ghg_sub}' if ghg_sub else '') + '__'
        lo, hi = self._vq_lo, self._vq_hi
        ckey   = self._composite_key(row)

        anchor, anchor_method, anchor_r2 = None, None, 0.0

        # ── Tier 1: site has own actuals for this GHG combo ───────────────────
        site_known = self._filter_to_ghg(
            self.known[self.known[self.site_col].astype(str).str.strip() == site],
            ghg, ghg_sub or None)

        if len(site_known) > 0:
            anchor        = float(site_known[self.vq_col].mean())
            anchor_method = f'{pfx}Site_Average_Seasonal'
            anchor_r2     = self.stat_methods.get(f'{pfx}Site_Average_Seasonal',
                            self.stat_methods.get(f'{pfx}Site_Average', {})
                            ).get('effective_r2', 0.6)

        # ── Tier 2: historic data for same GHG combo ──────────────────────────
        elif ckey in self.historic_vq:
            hist           = self.historic_vq[ckey]
            hval           = hist.get(month)
            specific_month = hval is not None
            if hval is None and 'ANNUAL' in hist:
                hval = hist['ANNUAL'] / 12.0

            if hval is not None:
                growth = self.historic_growth_rates.get(ckey)
                if growth and self.historic_year and self.current_year:
                    yf = self.current_year - self.historic_year
                    if yf > 0:
                        hval = hval * (growth ** yf)
                        anchor_method = f'{pfx}Historic_GrowthProjected'
                    else:
                        anchor_method = f'{pfx}Historic_Direct'
                else:
                    anchor_method = f'{pfx}Historic_Direct'

                anchor    = float(hval)
                anchor_r2 = self.stat_methods.get(f'{pfx}Historic_LastYear_Direct',
                            {}).get('effective_r2', 0.35)

                best_adj_r2 = 0.0
                for feat in self.numeric_features:
                    cf = row.get(feat)
                    hf = self.historic_feats.get(site, {}).get(feat)
                    if pd.isna(cf) or hf is None or float(hf) == 0:
                        continue
                    ratio = float(cf) / float(hf)
                    if not (0.1 <= ratio <= 10.0):
                        continue
                    adj_r2 = self.stat_methods.get(
                        f'{pfx}Historic_Adjusted_{feat}', {}).get('effective_r2', 0.0)
                    if adj_r2 > best_adj_r2:
                        anchor        = float(hval) * ratio
                        anchor_method = f'{pfx}Historic_Adjusted_{feat}'
                        anchor_r2     = adj_r2
                        best_adj_r2   = adj_r2

                if specific_month and anchor_method in (
                        f'{pfx}Historic_Direct', f'{pfx}Historic_GrowthProjected'):
                    sf   = self._seasonal_factor(site, month, ghg, ghg_sub or None)
                    pred = self._clip(anchor * sf, lo, hi)
                    return (pred if pred is not None
                            else float(self.known[self.vq_col].mean()),
                            anchor_method, anchor_r2)

        # ── Tier 3: derive from features (GHG-scoped only) ────────────────────
        if anchor is None:
            # 3a: categorical
            for feat in self.categorical_features:
                cv   = row.get(feat)
                info = self.stat_methods.get(f'{pfx}Categorical_{feat}', {})
                cm   = info.get('_cat_means', {})
                if pd.isna(cv) or cv not in cm:
                    continue
                r2 = info.get('effective_r2', 0.0)
                if r2 > anchor_r2:
                    anchor, anchor_method, anchor_r2 = float(cm[cv]), f'{pfx}Categorical_{feat}', r2

            # 3b: GHG-specific ML models only (FIX 2: prefix filter)
            for name, info in self.ml_models.items():
                if not name.startswith(pfx):
                    continue
                if tf == 'Annual' and '_Month' in info['features']:
                    continue
                pred = self._apply_ml_level(info, row)
                if pred is None:
                    continue
                r2 = info['r2'] or 0.0
                if isinstance(r2, float) and np.isnan(r2):
                    r2 = 0.0
                if r2 > anchor_r2:
                    anchor, anchor_method, anchor_r2 = pred, name, r2

            # 3c: feature intensity (GHG-scoped)
            for feat in self.numeric_features:
                info  = self.stat_methods.get(f'{pfx}Intensity_{feat}', {})
                slope = info.get('slope')
                fval  = row.get(feat)
                if slope is None or pd.isna(fval):
                    continue
                r2 = info.get('effective_r2', 0.0)
                if r2 > anchor_r2:
                    anchor, anchor_method, anchor_r2 = float(fval) * slope, f'{pfx}Intensity_{feat}', r2

            # 3d: GHG seasonal average
            if anchor is None:
                ma = self.stat_methods.get(f'{pfx}Seasonal_Average', {}).get('_month_avgs', {})
                v  = ma.get(month) if month else None
                if v is not None:
                    pred = self._clip(float(v), lo, hi)
                    r2   = self.stat_methods.get(f'{pfx}Seasonal_Average', {}).get('effective_r2', 0.0)
                    return (pred if pred is not None
                            else float(self.known[self.vq_col].mean()),
                            f'{pfx}Seasonal_Average', r2)

            if anchor is None:
                return float(self.known[self.vq_col].mean()), f'{pfx}Overall_Average_fallback', 0.0

        # ── Stage 2: seasonal index ───────────────────────────────────────────
        sf         = self._seasonal_factor(site, month, ghg, ghg_sub or None) if month else 1.0
        prediction = self._clip(anchor * sf, lo, hi)
        if prediction is None:
            return float(self.known[self.vq_col].mean()), f'{pfx}Overall_Average_fallback', 0.0

        return prediction, anchor_method, anchor_r2

    def _apply_ml_level(self, info, row):
        X_parts = []
        for fc in info['features']:
            if fc == '_Month':
                enc = info['encoders'].get('_Month')
                mid = float(len(enc.classes_) - 1) / 2.0 if enc else 6.0
                X_parts.append([[mid]])
            elif fc in info['encoders']:
                val = str(row.get(fc, 'Unknown'))
                enc = info['encoders'][fc]
                try:
                    X_parts.append([[enc.transform([val])[0]]])
                except Exception:
                    X_parts.append([[0]])
            else:
                val = row.get(fc)
                if pd.isna(val):
                    return None
                X_parts.append([[float(val)]])
        if not X_parts:
            return None
        X = np.hstack([np.array(p) for p in X_parts]).astype(float)
        try:
            return float(info['model'].predict(X)[0])
        except Exception:
            return None

    # ── Fill blanks ───────────────────────────────────────────────────────────

    def fill_blanks(self):
        print("\n" + "=" * 60)
        print("FILLING MISSING VALUES")
        print("=" * 60)
        self._vq_lo, self._vq_hi = self._sanity_bounds()
        print(f"  Sanity bounds: [{self._vq_lo:.2f}, {self._vq_hi:.2f}]")
        self._build_cat_breakdown()

        results = []
        for idx, row in self.blank.iterrows():
            pred, method, r2 = self.predict_row(row)
            results.append(dict(index=idx, prediction=pred, method=method, r2=r2))
            if method in self.stat_methods:
                self.stat_methods[method]['predictions_made'] += 1
            elif method in self.ml_models:
                self.ml_models[method]['predictions_made'] += 1

        print(f"  Filled {len(results)} rows")
        if results:
            print(f"  Avg R2: {np.mean([r['r2'] for r in results]):.4f}")
        return results

    def _build_cat_breakdown(self):
        mn = self.known[self.known['_Timeframe'] == 'Monthly']
        for feat in self.categorical_features:
            pool = self.known[self.known[feat].notna()]
            if len(pool) < 4:
                continue
            breakdown = {}
            for cat_val, grp in pool.groupby(feat):
                vq_vals = grp[self.vq_col].dropna()
                if len(vq_vals) == 0:
                    continue
                cat_mean    = float(vq_vals.mean())
                monthly_grp = mn[mn[feat] == cat_val]
                month_means = {}
                if len(monthly_grp) >= 2:
                    for mo, mg in monthly_grp.groupby('_Month'):
                        if len(mg) >= 1:
                            month_means[mo] = round(float(mg[self.vq_col].mean()), 2)
                breakdown[str(cat_val)] = {
                    'n_rows': int(len(vq_vals)),
                    'n_sites': int(grp[self.site_col].nunique()),
                    'mean_vq': round(cat_mean, 2),
                    'min_vq':  round(float(vq_vals.min()), 2),
                    'max_vq':  round(float(vq_vals.max()), 2),
                    'month_means': month_means,
                }
            self.cat_breakdown[feat] = breakdown

    # ── Output sheets ─────────────────────────────────────────────────────────

    def build_output(self, results):
        out = self.df.copy()

        if 'Data integrity' not in out.columns:
            out['Data integrity'] = 'Actual'
        else:
            out['Data integrity'] = out['Data integrity'].fillna('Actual')
        if 'Estimation Method'  not in out.columns: out['Estimation Method']  = ''
        if 'Data Quality Score' not in out.columns: out['Data Quality Score'] = np.nan

        for r in results:
            i = r['index']
            out.at[i, self.vq_col]          = r['prediction']
            out.at[i, 'Data integrity']     = 'Estimated'
            out.at[i, 'Estimation Method']  = r['method']
            out.at[i, 'Data Quality Score'] = round(r['r2'], 4)

        out.drop(columns=['_Timeframe', '_Month'], errors='ignore', inplace=True)

        cols        = list(out.columns)
        result_cols = ['Data integrity', 'Estimation Method', 'Data Quality Score']
        base_cols   = [c for c in cols if c not in result_cols]
        vq_idx      = base_cols.index(self.vq_col) if self.vq_col in base_cols else len(base_cols)
        ordered     = base_cols[:vq_idx+1] + result_cols + base_cols[vq_idx+1:]
        out         = out[[c for c in ordered if c in out.columns]]

        avail   = self._sheet_availability()
        stat    = self._sheet_stat()
        ml      = self._sheet_ml()
        cat_bkd = self._sheet_cat_breakdown()
        yoy     = self._sheet_yoy(out) if self.historic_vq else None
        return out, avail, stat, ml, cat_bkd, yoy

    def _sheet_availability(self):
        rows = []
        for idx, row in self.df.iterrows():
            month = row.get('_Month', '')
            site  = str(row.get(self.site_col, '')).strip()
            entry = {
                'Site Identifier': site,
                'Month': month,
                self.vq_col: row.get(self.vq_col),
                'Data integrity': 'Actual' if pd.notna(row.get(self.vq_col)) else 'Missing',
            }
            for f in self.numeric_features:
                v = row.get(f); entry[f] = v if pd.notna(v) else ''
            for f in self.categorical_features:
                v = row.get(f); entry[f] = v if pd.notna(v) else ''
            if self.historic_vq:
                ckey = self._composite_key(row)
                hvq  = (self.historic_vq.get(ckey, {}).get(month)
                        or self.historic_vq.get(ckey, {}).get('ANNUAL', ''))
                entry['Historic Volumetric Quantity'] = hvq
                for f in self.numeric_features:
                    entry[f'Historic {f}'] = self.historic_feats.get(site, {}).get(f, '')
            rows.append(entry)
        df = pd.DataFrame(rows)
        return df.loc[:, (df != '').any(axis=0)]

    def _sheet_stat(self):
        rows = []
        for name, info in self.stat_methods.items():
            rows.append({
                'Method':             name,
                'Description':        info['description'],
                'Input Data':         info['input_data'],
                'GHG_Category':       info.get('_ghg', ''),
                'GHG_Sub_Category':   info.get('_ghg_sub', ''),
                'Intensity_Slope':    info.get('slope'),
                'Pearson_r':          info.get('pearson_r'),
                'Pearson_R2':         info.get('pearson_r2'),
                'Pearson_p':          info.get('pearson_p'),
                'Spearman_r':         info.get('spearman_r'),
                'Spearman_R2':        info.get('spearman_r2'),
                'Spearman_p':         info.get('spearman_p'),
                'Kendall_tau':        info.get('kendall_tau'),
                'Kendall_R2':         info.get('kendall_r2'),
                'Kendall_p':          info.get('kendall_p'),
                'PointBiserial_r':    info.get('pointbiserial_r'),
                'PointBiserial_R2':   info.get('pointbiserial_r2'),
                'PointBiserial_p':    info.get('pointbiserial_p'),
                'Effective_R2':       round(info['effective_r2'], 4),
                'Used_In_Prediction': 'Yes' if info['predictions_made'] > 0 else 'No',
                'Predictions_Made':   info['predictions_made'],
            })
        df = pd.DataFrame(rows)
        if len(df):
            df = df.sort_values(['GHG_Category', 'Effective_R2'],
                                ascending=[True, False]).reset_index(drop=True)
        return df

    def _sheet_ml(self):
        rows = []
        for name, info in self.ml_models.items():
            feats  = ', '.join(fc.replace('_Month', 'Month').replace('_', ' ').strip()
                               for fc in info['features'])
            r2_raw = info['r2']
            if r2_raw is None or (isinstance(r2_raw, float) and np.isnan(r2_raw)):
                cv_r2, reliability = None, 'R² undefined — n too small for CV'
            elif r2_raw < 0:
                cv_r2, reliability = round(r2_raw, 4), 'Negative R² — worse than mean (overfitting)'
            elif r2_raw < 0.1:
                cv_r2, reliability = round(r2_raw, 4), 'Very weak'
            elif r2_raw < 0.3:
                cv_r2, reliability = round(r2_raw, 4), 'Weak'
            elif r2_raw < 0.6:
                cv_r2, reliability = round(r2_raw, 4), 'Moderate'
            else:
                cv_r2, reliability = round(r2_raw, 4), 'Good'
            rows.append({
                'Model': name, 'Type': info['type'], 'CV_R2': cv_r2,
                'Reliability': reliability, 'Features_Used': feats,
                'N_Features': len(info['features']),
                'N_Train_Rows': info.get('n_train', ''),
                'N_Test_Rows_Per_Fold': info.get('n_test_per_fold', ''),
                'N_Folds': info.get('n_folds', ''),
                'CV_Method': info.get('cv_type', ''),
                'Data_Sources': info.get('data_sources', ''),
                'Description': info['description'],
                'Predictions_Made': info['predictions_made'],
            })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df['_s'] = df['CV_R2'].fillna(-999)
        return df.sort_values('_s', ascending=False).drop(columns='_s').reset_index(drop=True)

    def _sheet_yoy(self, out):
        hy        = self.historic_year or 'Prev'
        cy        = self.current_year  or 'Current'
        month_map = self.df['_Month'].to_dict() if '_Month' in self.df.columns else {}
        rows      = []

        out_ckey_map = {}
        for idx, row in out.iterrows():
            out_ckey_map.setdefault(self._composite_key(row), []).append(idx)

        matched = [ck for ck in sorted(self.historic_vq.keys(), key=str)
                   if ck in out_ckey_map]

        for ckey in matched:
            site    = ckey[0]
            ghg     = ckey[1] if len(ckey) > 1 else ''
            ghg_sub = ckey[2] if len(ckey) > 2 else ''
            label   = site + (f' / {ghg}' if ghg else '') + (f' / {ghg_sub}' if ghg_sub else '')
            cat_key = (ghg, ghg_sub) if ghg_sub else (ghg,) if ghg else ()
            cat_sea = self.seasonal_patterns.get(cat_key, {})

            ckey_rows = out.loc[out_ckey_map[ckey]]
            hist      = self.historic_vq[ckey]
            h_tot, c_tot = 0.0, 0.0

            for month in MONTHS_ORDER:
                m_idx   = [i for i in ckey_rows.index if month_map.get(i, '') == month]
                curr_vq = float(ckey_rows.loc[m_idx[0], self.vq_col]) if m_idx else None
                hist_vq = hist.get(month)
                if hist_vq is None and 'ANNUAL' in hist:
                    hist_vq = (hist['ANNUAL'] / 12) * cat_sea.get(month, 1.0)

                chg = (curr_vq - hist_vq) if curr_vq is not None and hist_vq else None
                pct = round(chg / hist_vq * 100, 1) if (chg and hist_vq) else None

                is_est, em, dqs = False, '', None
                if m_idx and m_idx[0] in out.index:
                    r = out.loc[m_idx[0]]
                    is_est = r.get('Data integrity', '') == 'Estimated'
                    em     = r.get('Estimation Method', '') if is_est else ''
                    dqs    = r.get('Data Quality Score')    if is_est else None

                rows.append({
                    'Site': label, 'Month': month,
                    f'VQ_{hy}': round(hist_vq, 2) if hist_vq else None,
                    f'VQ_{cy}': round(curr_vq, 2) if curr_vq else None,
                    'VQ_Change': round(chg, 2) if chg else None,
                    'VQ_Change_%': pct,
                    'Estimation Method': em,
                    'Data Quality Score': round(dqs, 4) if dqs else None,
                    'Estimated': 'Yes' if is_est else 'No',
                    'Flag': ('Large change' if pct and abs(pct) > 50
                             else 'Normal' if pct and abs(pct) < 10
                             else 'Moderate') if pct else '',
                })
                if hist_vq: h_tot += hist_vq
                if curr_vq: c_tot += curr_vq

            chg_tot = c_tot - h_tot if h_tot and c_tot else None
            rows.append({
                'Site': label, 'Month': 'TOTAL',
                f'VQ_{hy}': round(h_tot, 2) if h_tot else None,
                f'VQ_{cy}': round(c_tot, 2) if c_tot else None,
                'VQ_Change': round(chg_tot, 2) if chg_tot else None,
                'VQ_Change_%': round(chg_tot / h_tot * 100, 1) if (chg_tot and h_tot) else None,
                'Estimated': '', 'Flag': '', 'Estimation Method': '', 'Data Quality Score': None,
            })

        return pd.DataFrame(rows) if rows else None

    def _sheet_cat_breakdown(self):
        rows = []
        for feat, cats in self.cat_breakdown.items():
            for cat_val, s in cats.items():
                row = {
                    'Feature': feat, 'Category': cat_val,
                    'N_Rows': s['n_rows'], 'N_Sites': s['n_sites'],
                    'Mean_VQ': s['mean_vq'], 'Min_VQ': s['min_vq'], 'Max_VQ': s['max_vq'],
                }
                for mo in MONTHS_ORDER:
                    row[mo] = s['month_means'].get(mo, '')
                rows.append(row)
        if not rows:
            return pd.DataFrame({'Message': ['No categorical features selected']})
        return pd.DataFrame(rows)

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        self.load()
        self.run_statistical_tests()
        self.train_ml_models()
        results = self.fill_blanks()
        return self.build_output(results)