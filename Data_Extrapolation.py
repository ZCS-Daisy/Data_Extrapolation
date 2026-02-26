"""
Volumetric Data Extrapolation Tool
====================================
Pipeline:
  1. Load & normalise data. If multiple GHG categories are present and more than
     one has blank VQ rows, the GUI presents a category picker — the user selects
     exactly one target category to extrapolate. All other categories are injected
     as named feature columns (e.g. "Fuels", "1. PG&S Water") using monthly values
     where available, annualised otherwise.
  2. Detect features (numeric + categorical) and timeframe (Monthly / Annual)
  3. Statistical tests: Pearson, Spearman, Kendall, Point-biserial
  4. Rule-based methods: site average, site-seasonal, feature intensity,
     historic-adjusted (if historic data present)
  5. ML models: Linear, Polynomial, Random Forest, Gradient Boosting
  6. Per-row selection: highest effective R2 wins
  7. Output: Complete Records, Data Availability, Statistical Analysis,
     Machine Learning Models, YoY Analysis
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


def scan_categories_with_blanks(filepath):
    """
    Scan the input file and return a list of dicts for every GHG category/
    subcategory combo that has at least one blank Volumetric Quantity row.

    Returns:
        list of dicts: [{label, ghg, ghg_sub, n_blank, n_total}, ...]
        Empty list if only one or zero categories have blanks.
    """
    if filepath.lower().endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()

    ghg_col = next((c for c in df.columns
                    if c.lower() in ('ghg category', 'ghg_category')), None)
    sub_col = next((c for c in df.columns
                    if c.lower() in ('ghg sub category', 'ghg_sub_category')), None)
    vq_col  = 'Volumetric Quantity' if 'Volumetric Quantity' in df.columns else None

    if not ghg_col or not vq_col:
        return []

    _BLANK = {'', 'nan', 'none', 'n/a', '-'}
    results = []
    group_cols = [ghg_col] + ([sub_col] if sub_col else [])

    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        ghg     = str(keys[0]).strip() if pd.notna(keys[0]) else ''
        ghg_sub = str(keys[1]).strip() if len(keys) > 1 and pd.notna(keys[1]) else ''
        if ghg.lower() in _BLANK:
            continue
        if ghg_sub.lower() in _BLANK:
            ghg_sub = ''

        n_blank = int(grp[vq_col].isna().sum())
        if n_blank == 0:
            continue

        label = (ghg + ' ' + ghg_sub).strip()
        results.append({
            'label':   label,
            'ghg':     ghg,
            'ghg_sub': ghg_sub,
            'n_blank': n_blank,
            'n_total': int(len(grp)),
        })

    results.sort(key=lambda x: x['label'])
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
                 forced_categorical_features=None,
                 target_ghg=None,
                 target_ghg_sub=None):
        """
        target_ghg / target_ghg_sub: the GHG category (and optional sub-category)
        to extrapolate. All other categories in the file are treated as features.
        Set by the GUI category picker when multiple categories have blank rows.
        If None and only one category has blanks, that category is used automatically.
        """
        self.input_file = input_file
        self.historic_records_file = historic_records_file
        self.historic_features_file = historic_features_file
        self.forced_numeric_features = forced_numeric_features
        self.forced_categorical_features = forced_categorical_features
        self.target_ghg     = target_ghg
        self.target_ghg_sub = target_ghg_sub

        self.df = None
        self.known = None
        self.blank = None

        self.site_col    = None
        self.vq_col      = self.VQ_COL
        self.ghg_col     = None
        self.ghg_sub_col = None

        self.numeric_features     = []
        self.categorical_features = []

        self.seasonal_patterns      = {}   # {month: factor}
        self.site_seasonal_patterns = {}   # {site: {month: factor}}

        self._vq_lo = 0.0
        self._vq_hi = float('inf')

        self.stat_methods = {}
        self.ml_models    = {}
        self.cat_breakdown = {}

        self.historic_vq              = {}  # {ckey: {period: vq}} — most recent year, used for prediction
        self.historic_vq_by_year      = {}  # {year: {ckey: {period: vq}}} — all years, for availability sheet
        self.historic_annual_trajectory = {}  # {ckey: [(year, annual_vq), ...]} — all years sorted, for trajectory stat test
        self.historic_growth_rates    = {}
        self.historic_feats           = {}  # {site: {feat: val}} — most recent year, used for prediction
        self.historic_feats_by_year   = {}  # {year: {site: {feat: val}}} — all years, for availability sheet
        self.historic_anchor_quality  = {}
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

        # ── Resolve target category ───────────────────────────────────────────
        # If caller didn't specify a target, auto-detect (only one with blanks).
        if not self.target_ghg:
            cats_with_blanks = scan_categories_with_blanks(self.input_file)
            if len(cats_with_blanks) == 1:
                self.target_ghg     = cats_with_blanks[0]['ghg']
                self.target_ghg_sub = cats_with_blanks[0]['ghg_sub']
            elif len(cats_with_blanks) == 0:
                _popup_error("No Blank Rows",
                             "No blank Volumetric Quantity rows found in the input file.")
            # If >1 and target still unset the GUI should have caught this already.
            # Fall through — target_ghg stays None and we use all rows (legacy behaviour).

        if self.target_ghg:
            tgt_label = self.target_ghg + (f' {self.target_ghg_sub}' if self.target_ghg_sub else '')
            print(f"  Target category: {tgt_label}")

        # ── Inject other-category VQ as features ──────────────────────────────
        self._inject_category_features()

        self._detect_features()

        # known/blank scoped to target category only
        if self.target_ghg:
            target_mask = (self.df[self.ghg_col].astype(str).str.strip()
                           == str(self.target_ghg).strip())
            if self.target_ghg_sub and self.ghg_sub_col:
                sub_mask = (self.df[self.ghg_sub_col].astype(str).str.strip()
                            == str(self.target_ghg_sub).strip())
                target_mask = target_mask & sub_mask
            target_df = self.df[target_mask]
        else:
            target_df = self.df

        self.known = target_df[target_df[self.vq_col].notna()].copy()
        self.blank = target_df[target_df[self.vq_col].isna()].copy()
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


    def _inject_category_features(self):
        """
        For each non-target GHG category present in the input file, compute
        an annualised VQ per site and inject it as a named feature column.

        Column name: '<GHG Category>' or '<GHG Category> <GHG Sub Category>'
        e.g. 'Fuels', '1. PG&S Water', 'Electricity (Market Based)'

        Timeframe logic (mirrors how other features are used):
          - If monthly rows exist for a site+category, the monthly value is
            used directly on matching rows; sites without monthly data for
            that period get the annualised value (annual ÷ 12 × seasonal factor).
          - If only annual rows exist, annualised ÷ 12 × seasonal factor.

        Only known (non-blank VQ) rows of other categories are used.
        The target category itself is never injected.
        """
        if not self.ghg_col:
            return

        _BLANK = {'', 'nan', 'none', 'n/a', '-'}

        def _cat_label(ghg, sub):
            parts = [str(ghg).strip()]
            if sub and str(sub).strip().lower() not in _BLANK:
                parts.append(str(sub).strip())
            return ' '.join(parts)

        target_label = _cat_label(self.target_ghg or '', self.target_ghg_sub or '')

        # Rows from all OTHER categories (non-blank VQ only)
        other_mask = self.df[self.vq_col].notna()
        if self.target_ghg:
            tgt_mask = self.df[self.ghg_col].astype(str).str.strip() == str(self.target_ghg).strip()
            if self.target_ghg_sub and self.ghg_sub_col:
                sub_mask = self.df[self.ghg_sub_col].astype(str).str.strip() == str(self.target_ghg_sub).strip()
                tgt_mask = tgt_mask & sub_mask
            other_mask = other_mask & ~tgt_mask

        other_df = self.df[other_mask].copy()
        if len(other_df) == 0:
            return

        group_cols = [self.site_col, self.ghg_col] + ([self.ghg_sub_col] if self.ghg_sub_col else [])
        injected = []

        for keys, grp in other_df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            site    = str(keys[0]).strip()
            ghg     = str(keys[1]).strip() if len(keys) > 1 and pd.notna(keys[1]) else ''
            ghg_sub = str(keys[2]).strip() if len(keys) > 2 and pd.notna(keys[2]) else ''
            if not ghg or ghg.lower() in _BLANK:
                continue

            col_name = _cat_label(ghg, ghg_sub)
            if col_name == target_label:
                continue

            # Build per-site monthly values and annual total
            monthly = grp[grp['_Timeframe'] == 'Monthly']
            annual  = grp[grp['_Timeframe'] == 'Annual']

            # Monthly lookup: {month_name: vq}
            monthly_vals = {}
            if len(monthly) > 0:
                for _, mrow in monthly.iterrows():
                    mo = mrow.get('_Month', '')
                    if mo:
                        monthly_vals[mo] = float(mrow[self.vq_col])

            # Annual equivalent for fallback
            annual_vq = None
            if len(annual) > 0:
                annual_vq = float(annual[self.vq_col].mean())
            elif monthly_vals:
                n = len(monthly_vals)
                annual_vq = sum(monthly_vals.values()) * (12.0 / n)

            if not monthly_vals and annual_vq is None:
                continue

            # Ensure column exists
            if col_name not in self.df.columns:
                self.df[col_name] = np.nan
                injected.append(col_name)

            # Fill rows for this site
            site_mask = self.df[self.site_col].astype(str).str.strip() == site
            for idx in self.df[site_mask].index:
                row_month = self.df.at[idx, '_Month'] if '_Month' in self.df.columns else ''
                if row_month and row_month in monthly_vals:
                    self.df.at[idx, col_name] = monthly_vals[row_month]
                elif annual_vq is not None:
                    # Annualised ÷ 12; seasonal factor applied when predictions use it
                    self.df.at[idx, col_name] = annual_vq / 12.0

        if injected:
            # Deduplicate (multiple sub-categories may share same col_name)
            injected = list(dict.fromkeys(injected))
            print(f"  Injected {len(injected)} other-category feature columns: {injected}")
        else:
            print("  No other-category features injected (single category or no other data)")

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
                    # Store every year individually for the availability sheet
                    for yr, vq, integ in year_rows_sorted:
                        self.historic_vq_by_year.setdefault(yr, {}).setdefault(ckey, {})[period] = vq

                    # Most recent year → prediction anchor (models + prediction waterfall)
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

            # Build annual trajectory for each ckey: sum monthly periods per year,
            # scale to 12 months if incomplete, prefer ANNUAL row if present.
            # Current-year known data is folded in later inside _run_trajectory_tests
            # once self.known is available.
            # Used only by the trajectory stat test — not predictions.
            traj_accumulator = defaultdict(dict)  # {ckey: {year: annual_vq or [monthly_vqs]}}
            for (ckey, period), year_rows in raw_rows.items():
                for yr, vq, _ in year_rows:
                    if yr is None:
                        continue
                    if period == 'ANNUAL':
                        traj_accumulator[ckey][yr] = vq
                    else:
                        if yr not in traj_accumulator[ckey]:
                            traj_accumulator[ckey][yr] = []
                        existing = traj_accumulator[ckey][yr]
                        if isinstance(existing, list):
                            existing.append(vq)
                        # If an ANNUAL row already set a float, monthly rows are ignored

            for ckey, yr_data in traj_accumulator.items():
                pts = []
                for yr, val in sorted(yr_data.items()):
                    if isinstance(val, list):
                        n_months = len(val)
                        annual_vq = sum(val) * (12.0 / n_months) if n_months else None
                    else:
                        annual_vq = val
                    if annual_vq and annual_vq > 0:
                        pts.append((yr, annual_vq))
                if len(pts) >= 2:
                    self.historic_annual_trajectory[ckey] = pts

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

            # Try to find a date column to extract year for labelling
            date_col = next((c for c in hf.columns
                             if 'date' in c.lower() and 'from' in c.lower()), None)
            if not date_col:
                date_col = next((c for c in hf.columns if 'date' in c.lower()), None)

            _GHG_COLS = {'GHG Category', 'GHG_Category', 'ghg_category',
                         'GHG Sub Category', 'GHG_Sub_Category'}
            skip_cols = _META | _GHG_COLS | {scol}
            if date_col:
                skip_cols = skip_cols | {date_col}
            feat_cols = [c for c in hf.columns if c not in skip_cols]

            if not feat_cols:
                _popup_warning(
                    "No Feature Columns in Historic Features File",
                    "No usable feature columns found after excluding metadata.\n\n"
                    "Historic feature adjustment will not be available."
                )
                return

            # Track which years appear so we can warn if none found
            years_found = set()

            for _, row in hf.iterrows():
                site = row.get(scol)
                if pd.isna(site):
                    continue
                site = str(site).strip()

                # Determine year for this row
                year = None
                if date_col:
                    p = _parse_date(row.get(date_col))
                    if p:
                        year = p.year
                        years_found.add(year)

                self.historic_feats.setdefault(site, {})
                if year:
                    self.historic_feats_by_year.setdefault(year, {}).setdefault(site, {})

                for fc in feat_cols:
                    v = row.get(fc)
                    if pd.notna(v):
                        try:
                            val = float(v)
                        except Exception:
                            val = v
                        # Always overwrite — later rows (higher year when sorted) win
                        self.historic_feats[site][fc] = val
                        if year:
                            self.historic_feats_by_year[year][site][fc] = val

            if not years_found and date_col:
                _popup_warning(
                    "No Parseable Dates in Historic Features File",
                    f"Found a date column ('{date_col}') but no dates could be parsed.\n\n"
                    "Historic features will be loaded without year labels."
                )
            elif not date_col:
                _popup_warning(
                    "No Date Column in Historic Features File",
                    "No date column found in the historic features file.\n\n"
                    "Features will be loaded but cannot be labelled by year "
                    "in the availability sheet."
                )

            print(f"  Historic features: {len(self.historic_feats)} sites, "
                  f"years={sorted(years_found) if years_found else 'unknown'}, "
                  f"columns: {feat_cols}")
        except ValueError:
            raise
        except Exception as e:
            print(f"  ⚠  Historic features error: {e}")

    # ── Seasonal ──────────────────────────────────────────────────────────────

    def _calc_seasonal(self):
        """
        Seasonal indices from target-category known rows only.
        self.seasonal_patterns:      {month: factor}  — cross-site
        self.site_seasonal_patterns: {site:  {month: factor}}  — per site
        """
        mo = self.known[self.known['_Timeframe'] == 'Monthly']
        if len(mo) < 4:
            return

        avgs    = mo.groupby('_Month')[self.vq_col].mean()
        overall = avgs.mean()
        if overall > 0:
            self.seasonal_patterns = {m: float(v / overall) for m, v in avgs.items()}

        self.site_seasonal_patterns = {}
        for site, grp in mo.groupby(self.site_col):
            if grp['_Month'].nunique() < 3:
                continue
            site_avgs    = grp.groupby('_Month')[self.vq_col].mean()
            site_overall = site_avgs.mean()
            if site_overall > 0:
                self.site_seasonal_patterns[str(site).strip()] = {
                    m: float(v / site_overall) for m, v in site_avgs.items()}

        print(f"  Seasonal index: cross-site ({len(self.seasonal_patterns)} months), "
              f"per-site ({len(self.site_seasonal_patterns)} sites)")

    def _seasonal_factor(self, site, month):
        if not month:
            return 1.0
        site_idx = self.site_seasonal_patterns.get(site, {})
        if month in site_idx:
            return site_idx[month]
        return self.seasonal_patterns.get(month, 1.0)

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
        All stat tests run on target-category known rows only.
        Other-category VQ columns have already been injected as named features.
        """
        print("\n" + "=" * 60)
        print("STATISTICAL ANALYSIS")
        print("=" * 60)
        self._calc_seasonal()

        mn  = self.known[self.known['_Timeframe'] == 'Monthly']
        an  = self.known[self.known['_Timeframe'] == 'Annual']
        sea = self.seasonal_patterns   # {month: factor}

        if len(self.known) < 4:
            print("  Not enough known rows for stat tests")
            return

        print(f"  {len(self.known)} known rows ({len(mn)} monthly, {len(an)} annual)")

        # 1 – Site average (monthly)
        if len(mn) >= 4:
            sa    = mn.groupby(self.site_col)[self.vq_col].mean()
            preds = mn[self.site_col].map(sa).fillna(sa.mean()).values.astype(float)
            acts  = mn[self.vq_col].values.astype(float)
            pr, pr2, pp = _pearson(preds, acts)
            sr, sr2, sp = _spearman(preds, acts)
            kr, kr2, kp = _kendall(preds, acts)
            self._reg('Site Average', 'Site average VQ (monthly)', f'{len(mn)} monthly rows',
                      None, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None,
                      _eff_r2(pr2, sr2, kr2), _site_avgs=sa.to_dict())

        # 2 – Site average x seasonal
        if len(mn) >= 4 and sea:
            sa2  = mn.groupby(self.site_col)[self.vq_col].mean().to_dict()
            fall = mn[self.vq_col].mean()
            p2   = np.array([sa2.get(r[self.site_col], fall) * sea.get(r['_Month'], 1.0)
                             for _, r in mn.iterrows()], float)
            a2   = mn[self.vq_col].values.astype(float)
            pr, pr2, pp = _pearson(p2, a2); sr, sr2, sp = _spearman(p2, a2); kr, kr2, kp = _kendall(p2, a2)
            self._reg('Site Average × Seasonal', 'Site average VQ × seasonal factor',
                      f'{len(mn)} monthly rows',
                      None, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None,
                      _eff_r2(pr2, sr2, kr2), _site_avgs=sa2, _seasonal=True)

        # 3 – Cross-site seasonal average
        if len(mn) >= 4 and sea:
            ma     = mn.groupby('_Month')[self.vq_col].mean().to_dict()
            preds3 = [ma.get(r['_Month']) for _, r in mn.iterrows()]
            p3 = np.array([v for v in preds3 if v is not None], float)
            a3 = mn[self.vq_col].values[:len(p3)].astype(float)
            pr, pr2, pp = _pearson(p3, a3); sr, sr2, sp = _spearman(p3, a3); kr, kr2, kp = _kendall(p3, a3)
            self._reg('Fleet Seasonal Average', 'Cross-site mean VQ per calendar month',
                      f'{len(mn)} monthly rows',
                      None, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None,
                      _eff_r2(pr2, sr2, kr2), _month_avgs=ma)

        # 4 – Overall average fallback
        for sub, lbl in [(self.known, 'all'), (an, 'annual')]:
            if len(sub) >= 4:
                self._reg(f'Fleet Overall Average ({lbl})', f'Overall mean {lbl} VQ', f'{len(sub)} rows',
                          None, None, None, None, None, None, None, None, None, None,
                          None, None, None, 0.0, _avg=float(sub[self.vq_col].mean()))
                break

        # 5 – Feature intensity (all numeric features — includes injected category cols)
        for feat in self.numeric_features:
            pool = self.known[self.known[feat].notna()]
            if len(pool) < 4:
                continue
            vq = pool[self.vq_col].values.astype(float)
            fv = pool[feat].values.astype(float)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = np.where(fv > 0, vq / fv, np.nan)
            slope = float(np.nanmedian(ratios)) if np.any(np.isfinite(ratios)) else None
            pr, pr2, pp = _pearson(fv, vq); sr, sr2, sp = _spearman(fv, vq); kr, kr2, kp = _kendall(fv, vq)
            eff = _eff_r2(pr2, sr2, kr2)
            self._reg(f'Intensity Ratio (VQ/{feat})', f'Intensity ratio VQ/{feat}',
                      f'{len(pool)} rows',
                      slope, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None, eff,
                      _feat=feat, _mtype='Feature_Intensity')

            pm = pool[pool['_Timeframe'] == 'Monthly']
            if len(pm) >= 4 and sea and slope:
                ps  = np.array([float(r[feat]) * slope * sea.get(r['_Month'], 1.0)
                                for _, r in pm.iterrows()], float)
                as_ = pm[self.vq_col].values.astype(float)
                if len(ps) >= 4:
                    pr, pr2, pp = _pearson(ps, as_); sr, sr2, sp = _spearman(ps, as_); kr, kr2, kp = _kendall(ps, as_)
                    self._reg(f'Intensity Ratio (VQ/{feat}) × Seasonal', f'Intensity ratio VQ/{feat} × seasonal',
                              f'{len(ps)} monthly rows',
                              slope, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None,
                              _eff_r2(pr2, sr2, kr2),
                              _feat=feat, _mtype='Feature_Intensity_Seasonal', _seasonal=True)

        # 6 – Categorical features
        for feat in self.categorical_features:
            pool = self.known[self.known[feat].notna()]
            if len(pool) < 4:
                continue
            vq    = pool[self.vq_col].values.astype(float)
            codes = pd.Categorical(pool[feat]).codes.astype(float)
            pbr, pbr2, pbp = _pointbiserial(codes, vq)
            self._reg(f'Group Mean VQ ({feat})', f'Group mean VQ by {feat}',
                      f'{len(pool)} rows',
                      None, None, None, None, None, None, None, None, None, None,
                      pbr, pbr2, pbp, _eff_r2(None, None, pbr2=pbr2),
                      _feat=feat, _mtype='Categorical',
                      _cat_means=pool.groupby(feat)[self.vq_col].mean().to_dict())

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
                            hvq = (ha / 12) * sea.get(month, 1.0)
                    if hvq is None:
                        continue
                    hf = self.historic_feats.get(site, {}).get(feat)
                    if hf and float(hf) != 0:
                        preds_h.append(hvq * (float(cf) / float(hf)))
                        acts_h.append(row[self.vq_col])
                if len(preds_h) >= 4:
                    ph, ah = np.array(preds_h, float), np.array(acts_h, float)
                    pr, pr2, pp = _pearson(ph, ah); sr, sr2, sp = _spearman(ph, ah); kr, kr2, kp = _kendall(ph, ah)
                    self._reg(f'Historic Direct × {feat} Ratio',
                              f'Historic VQ × {feat} ratio', f'{len(preds_h)} rows',
                              None, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None,
                              _eff_r2(pr2, sr2, kr2), _feat=feat, _mtype='Historic_Adjusted')

            pl, al = [], []
            for _, row in mn.iterrows():
                ckey = self._composite_key(row)
                hvq  = self.historic_vq.get(ckey, {}).get(row['_Month'])
                if hvq is not None:
                    pl.append(hvq); al.append(row[self.vq_col])
            if len(pl) >= 4:
                pla, ala = np.array(pl, float), np.array(al, float)
                pr, pr2, pp = _pearson(pla, ala); sr, sr2, sp = _spearman(pla, ala); kr, kr2, kp = _kendall(pla, ala)
                self._reg('Historic Direct (Same Month)', 'Same calendar month previous year',
                          f'{len(pl)} monthly rows',
                          None, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None,
                          _eff_r2(pr2, sr2, kr2), _mtype='Historic Direct')

            # 8 – Historic YoY site delta
            # For each site with ≥3 known current-year monthly rows AND matching
            # historic monthly rows, compute the mean YoY % change across those
            # known months. Predict each known month by leave-one-out: compute
            # delta from the OTHER known months, apply to that month's historic VQ.
            # Fleet R² is accumulated across all sites that qualify.
            # _site_yoy_deltas stored for use in predict_row: {site: delta_ratio}
            site_yoy_deltas = {}   # {site: mean ratio current/historic across known months}
            pd_all, ad_all  = [], []

            for site, site_grp in mn.groupby(self.site_col):
                site = str(site).strip()
                ckey_example = self._composite_key(site_grp.iloc[0])

                # Collect matched month pairs: (current_vq, historic_vq)
                pairs = []
                for _, row in site_grp.iterrows():
                    month = row['_Month']
                    if not month:
                        continue
                    ckey = self._composite_key(row)
                    hvq  = self.historic_vq.get(ckey, {}).get(month)
                    if hvq is not None and hvq > 0:
                        pairs.append((float(row[self.vq_col]), float(hvq), month))

                if len(pairs) < 3:
                    continue

                # Overall site delta ratio (current / historic mean)
                ratios = [p[0] / p[1] for p in pairs]
                site_yoy_deltas[site] = float(np.mean(ratios))

                # Leave-one-out cross-validation for fleet R²
                for i, (curr_vq, hist_vq, _) in enumerate(pairs):
                    other_ratios = [pairs[j][0] / pairs[j][1]
                                    for j in range(len(pairs)) if j != i]
                    loo_delta = float(np.mean(other_ratios))
                    pd_all.append(hist_vq * loo_delta)
                    ad_all.append(curr_vq)

            if len(pd_all) >= 4:
                pd_arr, ad_arr = np.array(pd_all, float), np.array(ad_all, float)
                pr, pr2, pp = _pearson(pd_arr, ad_arr)
                sr, sr2, sp = _spearman(pd_arr, ad_arr)
                kr, kr2, kp = _kendall(pd_arr, ad_arr)
                n_sites = len(site_yoy_deltas)
                self._reg('Historic Direct × Site YoY Trend',
                          f'Historic VQ × site mean YoY ratio (LOO CV, {n_sites} sites)',
                          f'{len(pd_all)} monthly rows across {n_sites} qualifying sites',
                          None, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None,
                          _eff_r2(pr2, sr2, kr2),
                          _mtype='Historic Direct × Site YoY Trend',
                          _site_yoy_deltas=site_yoy_deltas)

            # 9 – Fleet YoY delta
            # Fleet-wide mean YoY ratio across ALL sites with ≥1 matched pair.
            # Used for sites with no current-year data at all — applies the
            # observed fleet trend to their historic figure.
            # Cross-validated by leave-one-SITE-out: for each site's pairs,
            # compute the delta from all OTHER sites and test against this site.
            # Also computed per categorical feature group for finer matching.

            # Collect all site ratios from ANY site with ≥1 matched pair
            all_site_ratios = {}   # {site: mean_ratio}
            for site_s, site_grp in mn.groupby(self.site_col):
                site_s = str(site_s).strip()
                pairs_s = []
                for _, row in site_grp.iterrows():
                    mo = row['_Month']
                    if not mo:
                        continue
                    ck = self._composite_key(row)
                    hv = self.historic_vq.get(ck, {}).get(mo)
                    if hv is not None and hv > 0:
                        pairs_s.append((float(row[self.vq_col]), float(hv)))
                if pairs_s:
                    all_site_ratios[site_s] = float(np.mean(
                        [p[0] / p[1] for p in pairs_s]))

            # Fleet delta = mean across all sites
            # Median — robust to outlier sites (new openings, expansions, meter changes)
            fleet_delta = float(np.median(list(all_site_ratios.values()))) if all_site_ratios else None

            if fleet_delta is not None:
                # Leave-one-SITE-out CV for R²
                pf_all, af_all = [], []
                sites_list = list(all_site_ratios.keys())
                for site_s, site_grp in mn.groupby(self.site_col):
                    site_s = str(site_s).strip()
                    other_ratios = [v for k, v in all_site_ratios.items()
                                    if k != site_s]
                    if not other_ratios:
                        continue
                    loso_delta = float(np.median(other_ratios))
                    for _, row in site_grp.iterrows():
                        mo = row['_Month']
                        if not mo:
                            continue
                        ck = self._composite_key(row)
                        hv = self.historic_vq.get(ck, {}).get(mo)
                        if hv is not None and hv > 0:
                            pf_all.append(float(hv) * loso_delta)
                            af_all.append(float(row[self.vq_col]))

                if len(pf_all) >= 4:
                    pf_arr, af_arr = np.array(pf_all, float), np.array(af_all, float)
                    pr, pr2, pp = _pearson(pf_arr, af_arr)
                    sr, sr2, sp = _spearman(pf_arr, af_arr)
                    kr, kr2, kp = _kendall(pf_arr, af_arr)
                    n_fleet = len(all_site_ratios)
                    self._reg('Historic Direct × Fleet YoY Trend',
                              f'Historic VQ × fleet mean YoY ratio (LOSO CV, {n_fleet} sites)',
                              f'{len(pf_all)} monthly rows across {n_fleet} sites',
                              None, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None,
                              _eff_r2(pr2, sr2, kr2),
                              _mtype='Historic Direct × Fleet YoY Trend',
                              _fleet_yoy_delta=fleet_delta,
                              _all_site_ratios=all_site_ratios)

                # Category-band versions — one per categorical feature
                for cat_feat in self.categorical_features:
                    if cat_feat not in mn.columns:
                        continue
                    for cat_val, cat_grp in mn.groupby(cat_feat):
                        cat_val = str(cat_val).strip()
                        # Sites in this category that have ratios
                        cat_site_ratios = {
                            str(s).strip(): r
                            for s, r in all_site_ratios.items()
                            if s in cat_grp[self.site_col].astype(str).str.strip().values
                        }
                        if len(cat_site_ratios) < 2:
                            continue
                        cat_delta = float(np.median(list(cat_site_ratios.values())))
                        # LOSO CV within category
                        pc_all, ac_all = [], []
                        for site_s, site_grp2 in cat_grp.groupby(self.site_col):
                            site_s = str(site_s).strip()
                            other = [v for k, v in cat_site_ratios.items()
                                     if k != site_s]
                            if not other:
                                continue
                            loso_c = float(np.median(other))
                            for _, row in site_grp2.iterrows():
                                mo = row['_Month']
                                if not mo:
                                    continue
                                ck = self._composite_key(row)
                                hv = self.historic_vq.get(ck, {}).get(mo)
                                if hv is not None and hv > 0:
                                    pc_all.append(float(hv) * loso_c)
                                    ac_all.append(float(row[self.vq_col]))
                        if len(pc_all) >= 4:
                            pca, aca = np.array(pc_all, float), np.array(ac_all, float)
                            pr, pr2, pp = _pearson(pca, aca)
                            sr, sr2, sp = _spearman(pca, aca)
                            kr, kr2, kp = _kendall(pca, aca)
                            method_key = f'Historic Direct × Fleet YoY Trend ({cat_feat}={cat_val})'
                            self._reg(method_key,
                                      f'Historic VQ × {cat_feat} group YoY ratio (LOSO CV)',
                                      f'{len(pc_all)} rows, {cat_feat}={cat_val}',
                                      None, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None,
                                      _eff_r2(pr2, sr2, kr2),
                                      _mtype='Historic Direct × Fleet YoY Trend',
                                      _fleet_yoy_delta=cat_delta,
                                      _cat_feat=cat_feat, _cat_val=cat_val,
                                      _all_site_ratios=cat_site_ratios)

        print(f"  {len(self.stat_methods)} stat/rule methods registered")
        self._run_trajectory_tests()

    def _run_trajectory_tests(self):
        """
        Trajectory stat test — only runs when ≥2 historic years exist for a ckey.

        For each site+GHG combo with a multi-year annual VQ trajectory, fits a
        linear trend across years and records:
          - Slope (VQ change per year)
          - R² of the linear fit (how consistent the trend is)
          - Direction: Growth / Decline / Flat
          - Pearson r and p-value across the year points

        Data source: historic records only (monthly rows annualised by 12/n scaling,
        annual rows used as-is, mixed years handled per-year with ANNUAL taking priority).

        Informational only — does NOT feed predictions.
        Results stored in self.stat_methods under key 'Trajectory__<ckey>'.
        """
        if not self.historic_annual_trajectory:
            return

        # Fold in current-year known rows from the main file so the trajectory
        # extends to the present. Monthly rows are annualised by 12/n scaling.
        # This is a copy of the trajectory dict — we don't mutate the stored version.
        trajectory = {ckey: list(pts) for ckey, pts in self.historic_annual_trajectory.items()}

        if self.current_year and self.known is not None:
            fc = self._col(['Date from', 'Date_from'])
            curr_accumulator = defaultdict(list)  # {ckey: [vq, ...]} for monthly
            curr_annual      = {}                  # {ckey: vq} for annual

            for _, row in self.known.iterrows():
                pf = _parse_date(row.get(fc)) if fc else None
                if pf is None or pf.year != self.current_year:
                    continue
                ckey = self._composite_key(row)
                tf   = row.get('_Timeframe', '')
                vq   = row.get(self.vq_col)
                if pd.isna(vq):
                    continue
                if tf == 'Annual':
                    curr_annual[ckey] = float(vq)
                else:
                    curr_accumulator[ckey].append(float(vq))

            # Merge current-year point into trajectory for matching ckeys only
            all_curr_ckeys = set(curr_annual) | set(curr_accumulator)
            for ckey in all_curr_ckeys:
                if ckey not in trajectory:
                    continue  # only extend existing multi-year trajectories
                if ckey in curr_annual:
                    annual_vq = curr_annual[ckey]
                else:
                    monthly_vqs = curr_accumulator[ckey]
                    n = len(monthly_vqs)
                    annual_vq = sum(monthly_vqs) * (12.0 / n) if n else None
                if annual_vq and annual_vq > 0:
                    # Only add if current year not already in trajectory
                    existing_years = {p[0] for p in trajectory[ckey]}
                    if self.current_year not in existing_years:
                        trajectory[ckey].append((self.current_year, annual_vq))
                        trajectory[ckey].sort(key=lambda x: x[0])

        n_added = 0
        for ckey, pts in trajectory.items():
            if len(pts) < 2:
                continue

            site    = ckey[0]
            ghg     = ckey[1] if len(ckey) > 1 else ''
            ghg_sub = ckey[2] if len(ckey) > 2 else ''

            years = np.array([p[0] for p in pts], float)
            vqs   = np.array([p[1] for p in pts], float)

            # Linear fit: VQ = slope * year + intercept
            slope_val, intercept = np.polyfit(years, vqs, 1)

            # R² of the linear fit
            vq_pred = slope_val * years + intercept
            ss_res  = np.sum((vqs - vq_pred) ** 2)
            ss_tot  = np.sum((vqs - vqs.mean()) ** 2)
            r2_fit  = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Pearson across year points
            pr, pr2, pp = _pearson(years, vqs)

            pct_change_per_yr = float(slope_val / vqs[0] * 100) if vqs[0] != 0 else 0.0

            if abs(pct_change_per_yr) < 2:
                direction = 'Flat'
            elif slope_val > 0:
                direction = 'Growth'
            else:
                direction = 'Decline'

            yr_range = f'{int(years[0])}–{int(years[-1])}'
            pts_str  = ', '.join(f'{int(y)}: {v:.0f}' for y, v in pts)

            # Internal key uses site only — GHG context stored in _ghg fields
            key = f'Trajectory__{site}'
            self.stat_methods[key] = dict(
                description=(f'Annual VQ trajectory {yr_range}: {pts_str}. '
                             f'{direction} of {abs(pct_change_per_yr):.1f}% per year.'),
                input_data=f'{len(pts)} annual data points ({yr_range})',
                slope=round(float(slope_val), 4),
                pearson_r=pr, pearson_r2=pr2, pearson_p=pp,
                spearman_r=None, spearman_r2=None, spearman_p=None,
                kendall_tau=None, kendall_r2=None, kendall_p=None,
                pointbiserial_r=None, pointbiserial_r2=None, pointbiserial_p=None,
                effective_r2=round(r2_fit, 4),
                predictions_made=0,
                _mtype='Trajectory',
                _ghg=ghg, _ghg_sub=ghg_sub,
                _trajectory_direction=direction,
                _trajectory_pct_per_yr=round(pct_change_per_yr, 2),
                _trajectory_pts=pts,
                _trajectory_r2=round(r2_fit, 4),
            )
            n_added += 1

        if n_added:
            print(f"  Trajectory tests: {n_added} site+category combos "
                  f"({sum(1 for v in self.stat_methods.values() if v.get('_mtype') == 'Trajectory' and v.get('_trajectory_direction') == 'Growth')} growth, "
                  f"{sum(1 for v in self.stat_methods.values() if v.get('_mtype') == 'Trajectory' and v.get('_trajectory_direction') == 'Decline')} decline, "
                  f"{sum(1 for v in self.stat_methods.values() if v.get('_mtype') == 'Trajectory' and v.get('_trajectory_direction') == 'Flat')} flat)")

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

    def _build_ml_features(self, df):
        """
        Enrich a dataframe with two additional ML feature columns:

          '_Historic_VQ' — last year VQ for this site, timeframe-adjusted:
            Monthly row + historic monthly  → use exact month value
            Monthly row + historic annual   → annual ÷ 12
            Annual row  + historic annual   → use annual directly
            Annual row  + historic monthly  → sum all months
            No match                        → NaN (row excluded from that feature)

          '_Site' — site identifier as a string category for label encoding,
                    lets ML models learn site-level fixed effects.
        """
        out = df.copy()
        out['_Site'] = out[self.site_col].astype(str).str.strip()

        if not self.historic_vq:
            out['_Historic_VQ'] = np.nan
            return out

        hvq_vals = []
        for _, row in out.iterrows():
            ckey = self._composite_key(row)
            hist = self.historic_vq.get(ckey, {})
            tf   = row.get('_Timeframe', '')
            mo   = row.get('_Month', '')

            if not hist:
                hvq_vals.append(np.nan)
                continue

            if tf == 'Monthly' and mo:
                if mo in hist:
                    hvq_vals.append(float(hist[mo]))
                elif 'ANNUAL' in hist:
                    hvq_vals.append(float(hist['ANNUAL']) / 12.0)
                else:
                    month_vals = [v for k, v in hist.items() if k != 'ANNUAL']
                    hvq_vals.append(float(np.mean(month_vals)) if month_vals else np.nan)
            else:
                # Annual or unknown — prefer ANNUAL key, else sum months
                if 'ANNUAL' in hist:
                    hvq_vals.append(float(hist['ANNUAL']))
                else:
                    month_vals = [v for k, v in hist.items() if k != 'ANNUAL']
                    hvq_vals.append(float(sum(month_vals)) if month_vals else np.nan)

        out['_Historic_VQ'] = hvq_vals
        n_matched = sum(1 for v in hvq_vals if not np.isnan(v))
        print(f"  ML feature _Historic_VQ: {n_matched}/{len(out)} rows matched")
        return out

    def train_ml_models(self):
        print("\n" + "=" * 60)
        print("MACHINE LEARNING MODELS")
        print("=" * 60)

        data_sources = [f'Input file ({len(self.known)} known rows)']
        if self.historic_vq:
            data_sources.append(f'Historic records ({len(self.historic_vq)} keys)')
        if self.historic_feats:
            data_sources.append(f'Historic features ({len(self.historic_feats)} sites)')
        self._data_sources_label = ' + '.join(data_sources)

        # Enrich known rows with Historic VQ and Site identifier features
        known_enriched = self._build_ml_features(self.known)
        # _Site must appear in BOTH feat_cols and cat_cols so _train_subset
        # picks it up in the loop and knows to label-encode it
        num_feats = self.numeric_features + ['_Historic_VQ']
        cat_feats = self.categorical_features + ['_Site']
        all_feats = num_feats + ['_Site']   # _Site in feat_cols so it gets processed

        mn = known_enriched[known_enriched['_Timeframe'] == 'Monthly']

        if len(mn) >= 6:
            self._train_subset(mn, ['_Month'] + all_feats,
                               'Monthly', cat_feats)
        if len(known_enriched) >= 6:
            self._train_subset(known_enriched, all_feats,
                               'All', cat_feats)
        for cat in self.categorical_features:
            sub = known_enriched[known_enriched[cat].notna()]
            if len(sub) >= 6:
                self._train_subset(sub, ['_Month'] + all_feats + [cat],
                                   f'By_{cat}', [cat] + cat_feats)

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
            # Build human-readable model name
            _mtype = mname.replace('_', ' ').title()
            _mtype = _mtype.replace('Linear Regression', 'Multiple Linear Regression')\
                           .replace('Polynomial Regression', 'Polynomial Regression')\
                           .replace('Random Forest', 'Random Forest')\
                           .replace('Gradient Boosting', 'Gradient Boosting')
            # Extract categorical qualifier from prefix if present
            _cat_qualifier = ''
            if prefix.startswith('By_'):
                _cat_name = prefix[3:]  # strip 'By_'
                _cat_qualifier = f' ({_cat_name})'
            key = f'{prefix}_{mname}'   # keep internal key stable for deduplication
            _display_key = f'{_mtype}{_cat_qualifier}'
            r2, n_train, n_test, n_folds, cv_type = _cv_r2_with_meta(mobj, X, y)
            try:
                mobj.fit(X, y)
            except Exception:
                continue
            self.ml_models[_display_key] = dict(
                model=mobj, features=valid, encoders=encoders,
                r2=float(r2), type=_mtype,
                description=f'{mname.replace("_"," ")} using {feat_label}. CV R2={r2:.3f} (n={n_train})',
                n_train=n_train, n_test_per_fold=n_test, n_folds=n_folds, cv_type=cv_type,
                data_sources=getattr(self, '_data_sources_label', ''),
                predictions_made=0,
            )
            print(f"  {'✓' if r2>=0.3 else '~'} {_display_key}: R2={r2:.4f} "
                  f"[train={n_train}, test/fold={n_test}, {cv_type}]")

    # ── Prediction ────────────────────────────────────────────────────────────

    def _apply_ml_level(self, info, row):
        """
        Apply a trained ML model to a single row dict and return the prediction,
        or None if the row is missing required features.

        info dict contains: model, features (list), encoders (dict).
        Mirrors the feature-building logic in _train_subset.
        """
        model    = info.get('model')
        features = info.get('features', [])
        encoders = info.get('encoders', {})

        if model is None or not features:
            return None

        x_parts = []
        for fc in features:
            val = row.get(fc)
            if fc in encoders:
                # Categorical / month — label-encode
                str_val = str(val) if pd.notna(val) else 'Unknown'
                enc = encoders[fc]
                try:
                    code = enc.transform([str_val])[0]
                except ValueError:
                    # Unseen label — use most frequent class (index 0)
                    code = 0
                x_parts.append(float(code))
            else:
                # Numeric — must be present
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return None
                try:
                    x_parts.append(float(val))
                except (TypeError, ValueError):
                    return None

        if not x_parts:
            return None

        try:
            X = np.array(x_parts, dtype=float).reshape(1, -1)
            pred = float(model.predict(X)[0])
            return pred if np.isfinite(pred) else None
        except Exception:
            return None

    def predict_row(self, row):
        """
        Two-stage waterfall — target category rows only.

        STAGE 1 — anchor:
          Tier 1: Site has own actuals  → site mean VQ
          Tier 2: Site has historic data → historic monthly ± growth projection
                                           + optional feature-ratio adjustment
          Tier 3: categorical mean → ML model → feature intensity → seasonal avg
        STAGE 2 — multiply anchor by per-site (or cross-site) seasonal index.
        """
        site  = str(row.get(self.site_col, '')).strip()
        month = row.get('_Month', '')
        tf    = row.get('_Timeframe', '')
        lo, hi = self._vq_lo, self._vq_hi
        ckey   = self._composite_key(row)

        anchor, anchor_method, anchor_r2 = None, None, 0.0

        # ── Tier 1: site has actuals in current target-category data ──────────
        site_known = self.known[self.known[self.site_col].astype(str).str.strip() == site]
        if len(site_known) > 0:
            anchor        = float(site_known[self.vq_col].mean())
            anchor_method = 'Site Average × Seasonal'
            anchor_r2     = self.stat_methods.get('Site Average × Seasonal',
                            self.stat_methods.get('Site Average', {})
                            ).get('effective_r2', 0.6)

        # ── Tier 2: historic target-category data for this site ───────────────
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
                        anchor_method = 'Historic Direct (Growth Projected)'
                    else:
                        anchor_method = 'Historic Direct'
                else:
                    anchor_method = 'Historic Direct'

                anchor    = float(hval)
                anchor_r2 = self.stat_methods.get('Historic Direct (Same Month)',
                            {}).get('effective_r2', 0.35)

                best_adj_r2 = anchor_r2  # only upgrade if adjusted method is actually better

                # Check site-specific YoY delta first — competes on R²
                yoy_info   = self.stat_methods.get('Historic Direct × Site YoY Trend', {})
                yoy_r2     = yoy_info.get('effective_r2', 0.0)
                site_delta = yoy_info.get('_site_yoy_deltas', {}).get(site)
                if site_delta is not None and yoy_r2 > best_adj_r2 and specific_month:
                    anchor        = float(hval) * site_delta
                    anchor_method = 'Historic Direct × Site YoY Trend'
                    anchor_r2     = yoy_r2
                    best_adj_r2   = yoy_r2

                # Fleet YoY delta — fires when site has no own trend
                # Try category-band version first, then fleet-wide
                if site_delta is None and specific_month:
                    # Category-band: find best matching category group for this site
                    best_cat_r2    = best_adj_r2
                    best_cat_delta = None
                    best_cat_key   = None
                    for cat_feat in self.categorical_features:
                        cat_val = str(row.get(cat_feat, '')).strip()
                        if not cat_val or cat_val == 'nan':
                            continue
                        mkey  = f'Historic Direct × Fleet YoY Trend ({cat_feat}={cat_val})'
                        minfo = self.stat_methods.get(mkey, {})
                        mr2   = minfo.get('effective_r2', 0.0)
                        mdelta = minfo.get('_fleet_yoy_delta')
                        if mdelta is not None and mr2 > best_cat_r2:
                            best_cat_r2    = mr2
                            best_cat_delta = mdelta
                            best_cat_key   = mkey
                    if best_cat_delta is not None:
                        anchor        = float(hval) * best_cat_delta
                        anchor_method = best_cat_key
                        anchor_r2     = best_cat_r2
                        best_adj_r2   = best_cat_r2
                    else:
                        # Fall back to fleet-wide delta
                        fleet_info  = self.stat_methods.get('Historic Direct × Fleet YoY Trend', {})
                        fleet_r2    = fleet_info.get('effective_r2', 0.0)
                        fleet_delta = fleet_info.get('_fleet_yoy_delta')
                        if fleet_delta is not None and fleet_r2 > best_adj_r2:
                            anchor        = float(hval) * fleet_delta
                            anchor_method = 'Historic Direct × Fleet YoY Trend'
                            anchor_r2     = fleet_r2
                            best_adj_r2   = fleet_r2

                for feat in self.numeric_features:
                    cf = row.get(feat)
                    hf = self.historic_feats.get(site, {}).get(feat)
                    if pd.isna(cf) or hf is None or float(hf) == 0:
                        continue
                    ratio = float(cf) / float(hf)
                    if not (0.1 <= ratio <= 10.0):
                        continue
                    adj_r2 = self.stat_methods.get(
                        f'Historic Direct × {feat} Ratio', {}).get('effective_r2', 0.0)
                    if adj_r2 > best_adj_r2:
                        anchor        = float(hval) * ratio
                        anchor_method = f'Historic Direct × {feat} Ratio'
                        anchor_r2     = adj_r2
                        best_adj_r2   = adj_r2

                if specific_month and anchor_method in ('Historic Direct', 'Historic Direct (Growth Projected)'):
                    sf   = self._seasonal_factor(site, month)
                    pred = self._clip(anchor * sf, lo, hi)
                    return (pred if pred is not None
                            else float(self.known[self.vq_col].mean()),
                            anchor_method, anchor_r2)

        # ── Tier 3: derive from features ──────────────────────────────────────
        if anchor is None:
            # 3a: categorical group mean
            for feat in self.categorical_features:
                cv   = row.get(feat)
                info = self.stat_methods.get(f'Group Mean VQ ({feat})', {})
                cm   = info.get('_cat_means', {})
                if pd.isna(cv) or cv not in cm:
                    continue
                r2 = info.get('effective_r2', 0.0)
                if r2 > anchor_r2:
                    anchor, anchor_method, anchor_r2 = float(cm[cv]), f'Group Mean VQ ({feat})', r2

            # 3b: ML models
            # Enrich the row with _Historic_VQ and _Site so models that were
            # trained with these features can use them at prediction time.
            ml_row = dict(row)
            ml_row['_Site'] = site

            # Compute _Historic_VQ for this blank row using same timeframe logic
            # as _build_ml_features
            ckey_hist = self._composite_key(row)
            hist_data = self.historic_vq.get(ckey_hist, {})
            if hist_data:
                if tf == 'Monthly' and month:
                    if month in hist_data:
                        ml_row['_Historic_VQ'] = float(hist_data[month])
                    elif 'ANNUAL' in hist_data:
                        ml_row['_Historic_VQ'] = float(hist_data['ANNUAL']) / 12.0
                    else:
                        month_vals = [v for k, v in hist_data.items() if k != 'ANNUAL']
                        ml_row['_Historic_VQ'] = float(np.mean(month_vals)) if month_vals else np.nan
                else:
                    if 'ANNUAL' in hist_data:
                        ml_row['_Historic_VQ'] = float(hist_data['ANNUAL'])
                    else:
                        month_vals = [v for k, v in hist_data.items() if k != 'ANNUAL']
                        ml_row['_Historic_VQ'] = float(sum(month_vals)) if month_vals else np.nan
            else:
                ml_row['_Historic_VQ'] = np.nan

            for name, info in self.ml_models.items():
                if tf == 'Annual' and '_Month' in info['features']:
                    continue
                pred = self._apply_ml_level(info, ml_row)
                if pred is None:
                    continue
                r2 = info['r2'] or 0.0
                if isinstance(r2, float) and np.isnan(r2):
                    r2 = 0.0
                if r2 > anchor_r2:
                    anchor, anchor_method, anchor_r2 = pred, name, r2

            # 3c: feature intensity slope
            for feat in self.numeric_features:
                info  = self.stat_methods.get(f'Intensity Ratio (VQ/{feat})', {})
                slope = info.get('slope')
                fval  = row.get(feat)
                if slope is None or pd.isna(fval):
                    continue
                r2 = info.get('effective_r2', 0.0)
                if r2 > anchor_r2:
                    anchor, anchor_method, anchor_r2 = float(fval) * slope, f'Intensity Ratio (VQ/{feat})', r2

            # 3d: cross-site seasonal average
            if anchor is None:
                ma = self.stat_methods.get('Fleet Seasonal Average', {}).get('_month_avgs', {})
                v  = ma.get(month) if month else None
                if v is not None:
                    pred = self._clip(float(v), lo, hi)
                    r2   = self.stat_methods.get('Fleet Seasonal Average', {}).get('effective_r2', 0.0)
                    return (pred if pred is not None
                            else float(self.known[self.vq_col].mean()),
                            'Fleet Seasonal Average', r2)

            if anchor is None:
                return float(self.known[self.vq_col].mean()), 'Fleet Overall Average', 0.0

        # ── Stage 2: seasonal index ───────────────────────────────────────────
        sf         = self._seasonal_factor(site, month) if month else 1.0
        prediction = self._clip(anchor * sf, lo, hi)
        if prediction is None:
            return float(self.known[self.vq_col].mean()), 'Fleet Overall Average', 0.0

        # Update method name to reflect whether seasonal factor was actually applied
        if anchor_method:
            seasonal_was_applied = month and sf != 1.0
            if seasonal_was_applied and '× Seasonal' not in anchor_method:
                # Seasonal factor changed the number — add × Seasonal to name
                anchor_method = anchor_method + ' × Seasonal'
            elif not month and '× Seasonal' in anchor_method:
                # Annual row — seasonal factor was 1.0 — strip from name
                anchor_method = anchor_method.replace(' × Seasonal', '')

        return prediction, anchor_method, anchor_r2

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
        # Determine sorted historic years for VQ and features
        vq_years   = sorted(self.historic_vq_by_year.keys())
        feat_years = sorted(self.historic_feats_by_year.keys())
        # Collect all numeric feature names that appear in any historic year
        hist_feat_names = []
        seen = set()
        for yr_data in self.historic_feats_by_year.values():
            for site_data in yr_data.values():
                for fn in site_data:
                    if fn not in seen:
                        hist_feat_names.append(fn)
                        seen.add(fn)

        rows = []
        for idx, row in self.df.iterrows():
            month = row.get('_Month', '')
            site  = str(row.get(self.site_col, '')).strip()
            ckey  = self._composite_key(row)

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

            # One 'FY20XX VQ' column per historic year
            for yr in vq_years:
                yr_vq_map = self.historic_vq_by_year.get(yr, {})
                periods   = yr_vq_map.get(ckey, {})
                val = periods.get(month) or periods.get('ANNUAL', '')
                entry[f'FY{yr} VQ'] = val

            # One 'FY20XX <Feature>' column per historic year × feature
            for yr in feat_years:
                yr_feat_map = self.historic_feats_by_year.get(yr, {})
                site_feats  = yr_feat_map.get(site, {})
                for fn in hist_feat_names:
                    entry[f'FY{yr} {fn}'] = site_feats.get(fn, '')

            rows.append(entry)

        df = pd.DataFrame(rows)
        return df.loc[:, (df != '').any(axis=0)]

    def _sheet_stat(self):
        rows = []
        for name, info in self.stat_methods.items():
            rows.append({
                'Method':               name,
                'Description':          info['description'],
                'Input Data':           info['input_data'],
                'Method_Type':          info.get('_mtype', ''),
                'Trajectory_Direction': info.get('_trajectory_direction', ''),
                'Trajectory_%_Per_Yr':  info.get('_trajectory_pct_per_yr', ''),
                'Trajectory_R2':        info.get('_trajectory_r2', ''),
                'Intensity_Slope':      info.get('slope'),
                'Pearson_r':            info.get('pearson_r'),
                'Pearson_R2':           info.get('pearson_r2'),
                'Pearson_p':            info.get('pearson_p'),
                'Spearman_r':           info.get('spearman_r'),
                'Spearman_R2':          info.get('spearman_r2'),
                'Spearman_p':           info.get('spearman_p'),
                'Kendall_tau':          info.get('kendall_tau'),
                'Kendall_R2':           info.get('kendall_r2'),
                'Kendall_p':            info.get('kendall_p'),
                'PointBiserial_r':      info.get('pointbiserial_r'),
                'PointBiserial_R2':     info.get('pointbiserial_r2'),
                'PointBiserial_p':      info.get('pointbiserial_p'),
                'Effective_R2':         round(info['effective_r2'], 4),
                'Used_In_Prediction':   'Yes' if info['predictions_made'] > 0 else 'No',
                'Predictions_Made':     info['predictions_made'],
            })
        df = pd.DataFrame(rows)
        if len(df):
            # Trajectory rows at the top, then by GHG category + R2 descending
            df['_is_traj'] = (df['Method_Type'] == 'Trajectory').astype(int)
            df = df.sort_values(['_is_traj', 'Effective_R2'],
                                ascending=[False, False]).drop(columns='_is_traj').reset_index(drop=True)
        return df

    def _sheet_ml(self):
        rows = []
        for name, info in self.ml_models.items():
            feats  = ', '.join(fc.replace('_Month', 'Month').replace('_', ' ').strip()
                               for fc in info['features'])
            r2_raw = info['r2']
            if r2_raw is None or (isinstance(r2_raw, float) and np.isnan(r2_raw)):
                cv_r2 = None
            else:
                cv_r2 = round(r2_raw, 4)
            rows.append({
                'Model': name, 'Type': info['type'], 'CV_R2': cv_r2,
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
            cat_sea = self.seasonal_patterns   # flat {month: factor}

            ckey_rows = out.loc[out_ckey_map[ckey]]
            hist      = self.historic_vq[ckey]
            h_tot, c_tot = 0.0, 0.0

            for month in MONTHS_ORDER:
                m_idx   = [i for i in ckey_rows.index if month_map.get(i, '') == month]
                curr_vq = float(ckey_rows.loc[m_idx[0], self.vq_col]) if m_idx else None
                hist_vq = hist.get(month)
                if hist_vq is None and 'ANNUAL' in hist:
                    hist_vq = (hist['ANNUAL'] / 12) * cat_sea.get(month, 1.0)  # noqa

                chg = (curr_vq - hist_vq) if curr_vq is not None and hist_vq else None
                pct = round(chg / hist_vq * 100, 1) if (chg and hist_vq) else None

                is_est, em, dqs = False, '', None
                if m_idx and m_idx[0] in out.index:
                    r = out.loc[m_idx[0]]
                    is_est = r.get('Data integrity', '') == 'Estimated'
                    em     = r.get('Estimation Method', '') if is_est else ''
                    dqs    = r.get('Data Quality Score')    if is_est else None

                rows.append({
                    'Site': site, 'GHG Category': ghg, 'GHG Sub Category': ghg_sub,
                    'Month': month,
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
                'Site': site, 'GHG Category': ghg, 'GHG Sub Category': ghg_sub,
                'Month': 'TOTAL',
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