"""
Volumetric Data Extrapolation Tool
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
     Machine Learning Models, YoY Analysis (if historic data detected)
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
warnings.filterwarnings('ignore')

MONTHS_ORDER = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

# Columns that are always metadata, never predictive features
_META = {
    'Row ID', 'Current Timestamp', 'Client Name', 'Client', 'Country',
    'Site identifier', 'Site_identifier', 'site identifier', 'site_id',
    'Location', 'location', 'Site Name', 'site_name',
    'Date from', 'Date to', 'Date_from', 'Date_to',
    'Data Timeframe', 'Timeframe', 'timeframe',
    'GHG Category', 'GHG Sub Category',
    'Volumetric Quantity', 'Volumetric_Quantity',
    'Data integrity', 'Estimation Method', 'Data Quality Score',
    'Month', 'Year', 'Financial Year', 'FY',
    'Scope', 'Functional Unit', 'Calculation Method',
    'Emission source', 'Meter Number',
    '_Timeframe', '_Month',
}

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
    """Return ('Monthly', 'January') or ('Annual', None) or ('Unknown', None)."""
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
# Stat helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pearson(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return None, None, None
    try:
        r, p = stats.pearsonr(x[m], y[m])
        return float(r), float(r**2), float(p)
    except Exception:
        return None, None, None


def _spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return None, None, None
    try:
        r, p = stats.spearmanr(x[m], y[m])
        return float(r), float(r**2), float(p)
    except Exception:
        return None, None, None


def _kendall(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return None, None, None
    try:
        tau, p = stats.kendalltau(x[m], y[m])
        return float(tau), float(tau**2), float(p)
    except Exception:
        return None, None, None


def _pointbiserial(codes, y):
    m = np.isfinite(codes) & np.isfinite(y)
    if m.sum() < 4 or len(np.unique(codes[m])) < 2:
        return None, None, None
    try:
        r, p = stats.pointbiserialr(codes[m], y[m])
        return float(r), float(r**2), float(p)
    except Exception:
        return None, None, None


def _eff_r2(pr2, sr2, kr2=None, pbr2=None):
    """Best R2 across all stat tests — no significance gate, just best signal."""
    cands = [v for v in (pr2, sr2, kr2, pbr2) if v is not None]
    return max(cands) if cands else 0.0


def _cv_r2(model, X, y, cv=5):
    n = len(X)
    if n < 4:
        return 0.0
    cv_obj = (LeaveOneOut() if n < 10
               else KFold(n_splits=min(cv, n // 2), shuffle=True, random_state=42))
    scores = cross_val_score(model, X, y, cv=cv_obj, scoring='r2')
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────────────────
# Main tool class
# ─────────────────────────────────────────────────────────────────────────────

class ExtrapolationTool:

    def __init__(self, input_file,
                 historic_records_file=None,
                 historic_features_file=None):
        self.input_file = input_file
        self.historic_records_file = historic_records_file
        self.historic_features_file = historic_features_file

        self.df = None
        self.known = None
        self.blank = None

        self.site_col = None
        self.vq_col = None
        self.numeric_features = []
        self.categorical_features = []
        self.seasonal_patterns = {}     # {month_name: factor}

        self.stat_methods = {}          # {name: info}
        self.ml_models = {}             # {name: info}

        # Historic
        self.historic_vq = {}           # {site: {month_or_ANNUAL: float}}
        self.historic_feats = {}        # {site: {feat: float}}
        self.historic_year = None
        self.current_year = None

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

        self.vq_col = self._col(['Volumetric Quantity', 'Volumetric_Quantity', 'VQ'])
        if not self.vq_col:
            raise ValueError("Cannot find 'Volumetric Quantity' column.")

        self.site_col = self._col(
            ['Site identifier', 'Site_identifier', 'Site ID', 'SiteID'])
        if not self.site_col:
            self.site_col = self._col(['Location', 'location', 'Site Name'])
            if self.site_col:
                print(f"  Using '{self.site_col}' as site key (no Site identifier found)")
            else:
                raise ValueError("Cannot find site identifier or location column.")

        self._add_timeframe_cols()
        self._coerce_numeric()
        self._detect_features()

        self.known = self.df[self.df[self.vq_col].notna()].copy()
        self.blank = self.df[self.df[self.vq_col].isna()].copy()
        print(f"  Known VQ: {len(self.known)} | Missing VQ: {len(self.blank)}")

        # Detect current year
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

    def _col(self, candidates):
        """Find first matching column (case-insensitive)."""
        low = {c.lower().strip(): c for c in self.df.columns}
        for c in candidates:
            if c in self.df.columns:
                return c
            if c.lower() in low:
                return low[c.lower()]
        return None

    def _add_timeframe_cols(self):
        from_col = self._col(['Date from', 'Date_from'])
        to_col   = self._col(['Date to', 'Date_to'])
        fs = self.df[from_col] if from_col else pd.Series([None]*len(self.df))
        ts = self.df[to_col]   if to_col   else pd.Series([None]*len(self.df))
        tfs, mos = [], []
        for f, t in zip(fs, ts):
            tf, mo = _timeframe(_parse_date(f), _parse_date(t))
            tfs.append(tf)
            mos.append(mo or '')
        self.df['_Timeframe'] = tfs
        self.df['_Month'] = mos
        print(f"  Timeframes: {pd.Series(tfs).value_counts().to_dict()}")

    def _coerce_numeric(self):
        hints = ['turnover', 'transaction', 'revenue', 'sales', 'sqft', 'floor',
                 'space', 'electricity', 'area', 'count', 'quantity', 'volume',
                 'consumption', 'spend', 'units', 'kw', 'kwh', 'water', 'waste']
        for col in self.df.columns:
            if col in _META or col.startswith('_'):
                continue
            if pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            non_null = self.df[col].dropna()
            if not len(non_null):
                continue
            conv = pd.to_numeric(non_null, errors='coerce')
            rate = conv.notna().sum() / len(non_null)
            hint = any(h in col.lower() for h in hints)
            if rate > 0.8 and hint:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

    def _detect_features(self):
        skip = _META | {self.site_col, self.vq_col}
        self.numeric_features = []
        self.categorical_features = []
        for col in self.df.columns:
            if col in skip or col.startswith('_'):
                continue
            nn = self.df[col].dropna()
            if len(nn) < 4:
                continue
            if pd.api.types.is_numeric_dtype(self.df[col]) and nn.nunique() > 1:
                self.numeric_features.append(col)
            elif not pd.api.types.is_numeric_dtype(self.df[col]):
                u = nn.nunique()
                if 2 <= u <= 30:
                    self.categorical_features.append(col)
        print(f"  Numeric features:     {self.numeric_features}")
        print(f"  Categorical features: {self.categorical_features}")

    # ── Historic ──────────────────────────────────────────────────────────────

    def _load_historic_records(self):
        print("\n  Loading historic records...")
        try:
            hrec = (pd.read_csv(self.historic_records_file)
                    if self.historic_records_file.lower().endswith('.csv')
                    else pd.read_excel(self.historic_records_file))
            hrec.columns = hrec.columns.str.strip()
            scol = next((c for c in ['Site identifier', 'Site_identifier', 'Location']
                         if c in hrec.columns), None)
            vcol = next((c for c in hrec.columns if 'volumetric' in c.lower()), None)
            if not scol or not vcol:
                print("  ⚠  Missing site or VQ column in historic records")
                return
            from_col = next((c for c in hrec.columns
                             if 'date from' in c.lower() or c.lower() == 'date_from'), None)
            to_col   = next((c for c in hrec.columns
                             if 'date to' in c.lower()   or c.lower() == 'date_to'), None)

            years_seen = set()
            for _, row in hrec.iterrows():
                site = row.get(scol)
                vq   = row.get(vcol)
                if pd.isna(site) or pd.isna(vq):
                    continue
                site = str(site).strip()
                pf = _parse_date(row.get(from_col)) if from_col else None
                pt = _parse_date(row.get(to_col))   if to_col   else pf
                tf, mo = _timeframe(pf, pt) if pf else ('Unknown', None)
                if pf:
                    years_seen.add(pf.year)
                self.historic_vq.setdefault(site, {})[mo or 'ANNUAL'] = float(vq)

            curr_yrs = set()
            fc = self._col(['Date from', 'Date_from'])
            if fc:
                for v in self.df[fc].dropna():
                    p = _parse_date(v)
                    if p:
                        curr_yrs.add(p.year)
            past = years_seen - curr_yrs
            if past:
                self.historic_year = max(past)
            print(f"  Historic records: {len(self.historic_vq)} sites, year={self.historic_year}")
        except Exception as e:
            print(f"  ⚠  Historic records error: {e}")

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
                return
            feat_cols = [c for c in hf.columns if c not in _META and c != scol]
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
            print(f"  Historic features: {len(self.historic_feats)} sites")
        except Exception as e:
            print(f"  ⚠  Historic features error: {e}")

    # ── Seasonal ──────────────────────────────────────────────────────────────

    def _calc_seasonal(self):
        mo = self.known[self.known['_Timeframe'] == 'Monthly']
        if len(mo) < 4:
            return
        avgs = mo.groupby('_Month')[self.vq_col].mean()
        overall = avgs.mean()
        if overall > 0:
            self.seasonal_patterns = {m: float(v/overall) for m, v in avgs.items()}

    # ── Statistical tests ─────────────────────────────────────────────────────

    def run_statistical_tests(self):
        print("\n" + "=" * 60)
        print("STATISTICAL ANALYSIS")
        print("=" * 60)
        self._calc_seasonal()
        mn = self.known[self.known['_Timeframe'] == 'Monthly']
        an = self.known[self.known['_Timeframe'] == 'Annual']

        # 1 – Site average (monthly)
        if len(mn) >= 4:
            sa = mn.groupby(self.site_col)[self.vq_col].mean()
            preds = mn[self.site_col].map(sa).fillna(sa.mean()).values.astype(float)
            acts  = mn[self.vq_col].values.astype(float)
            pr, pr2, pp = _pearson(preds, acts)
            sr, sr2, sp = _spearman(preds, acts)
            kr, kr2, kp = _kendall(preds, acts)
            eff = _eff_r2(pr2, sr2, kr2)
            self._reg('Site_Average',
                      'Site average VQ across all known monthly rows',
                      f'{len(mn)} monthly rows',
                      None, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None, eff,
                      _site_avgs=sa.to_dict())

        # 2 – Site average x seasonal
        if len(mn) >= 4 and self.seasonal_patterns:
            sa2 = mn.groupby(self.site_col)[self.vq_col].mean().to_dict()
            fall = mn[self.vq_col].mean()
            preds2 = []
            for _, row in mn.iterrows():
                avg = sa2.get(row[self.site_col], fall)
                sf  = self.seasonal_patterns.get(row['_Month'], 1.0)
                preds2.append(avg * sf)
            p2 = np.array(preds2, float)
            a2 = mn[self.vq_col].values.astype(float)
            pr, pr2, pp = _pearson(p2, a2)
            sr, sr2, sp = _spearman(p2, a2)
            kr, kr2, kp = _kendall(p2, a2)
            eff = _eff_r2(pr2, sr2, kr2)
            self._reg('Site_Average_Seasonal',
                      'Site average VQ scaled by monthly seasonal factor',
                      f'{len(mn)} monthly rows',
                      None, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None, eff,
                      _site_avgs=sa2, _seasonal=True)

        # 3 – Cross-site seasonal average
        if len(mn) >= 4 and self.seasonal_patterns:
            ma = mn.groupby('_Month')[self.vq_col].mean().to_dict()
            fall = mn[self.vq_col].mean()
            preds3 = [ma.get(row['_Month'], fall) for _, row in mn.iterrows()]
            p3 = np.array(preds3, float)
            a3 = mn[self.vq_col].values.astype(float)
            pr, pr2, pp = _pearson(p3, a3)
            sr, sr2, sp = _spearman(p3, a3)
            kr, kr2, kp = _kendall(p3, a3)
            eff = _eff_r2(pr2, sr2, kr2)
            self._reg('Seasonal_Average',
                      'Cross-site mean VQ per calendar month',
                      f'{len(mn)} monthly rows',
                      None, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None, eff,
                      _month_avgs=ma)

        # 4 – Overall average fallback
        for sub, lbl in [(self.known, 'all'), (an, 'annual')]:
            if len(sub) >= 4:
                avg = float(sub[self.vq_col].mean())
                self._reg(f'Overall_Average_{lbl}',
                          f'Overall mean of {lbl} VQ values',
                          f'{len(sub)} rows',
                          None, None, None, None, None, None, None, None, None, None,
                          None, None, None, 0.0,
                          _avg=avg)
                break

        # 5 – Numeric feature intensity
        for feat in self.numeric_features:
            pool = self.known[self.known[feat].notna()]
            if len(pool) < 4:
                continue
            vq = pool[self.vq_col].values.astype(float)
            fv = pool[feat].values.astype(float)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = np.where(fv > 0, vq / fv, np.nan)
            slope = float(np.nanmedian(ratios)) if np.any(np.isfinite(ratios)) else None
            pr, pr2, pp = _pearson(fv, vq)
            sr, sr2, sp = _spearman(fv, vq)
            kr, kr2, kp = _kendall(fv, vq)
            eff = _eff_r2(pr2, sr2, kr2)
            self._reg(f'Intensity_{feat}',
                      f'VQ proportional to {feat} (median intensity slope)',
                      f'{len(pool)} rows with {feat} + VQ',
                      slope, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None, eff,
                      _feat=feat, _mtype='Feature_Intensity')

            # Monthly + seasonal variant
            pm = pool[pool['_Timeframe'] == 'Monthly']
            if len(pm) >= 4 and self.seasonal_patterns and slope:
                preds_s, acts_s = [], []
                for _, row in pm.iterrows():
                    sf = self.seasonal_patterns.get(row['_Month'], 1.0)
                    preds_s.append(float(row[feat]) * slope * sf)
                    acts_s.append(row[self.vq_col])
                if len(preds_s) >= 4:
                    ps = np.array(preds_s, float)
                    as_ = np.array(acts_s, float)
                    pr, pr2, pp = _pearson(ps, as_)
                    sr, sr2, sp = _spearman(ps, as_)
                    kr, kr2, kp = _kendall(ps, as_)
                    eff = _eff_r2(pr2, sr2, kr2)
                    self._reg(f'Intensity_{feat}_Seasonal',
                              f'VQ ~ {feat} intensity x seasonal factor',
                              f'{len(preds_s)} monthly rows with {feat}',
                              slope, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None, eff,
                              _feat=feat, _mtype='Feature_Intensity_Seasonal', _seasonal=True)

        # 6 – Categorical features (point-biserial)
        for feat in self.categorical_features:
            pool = self.known[self.known[feat].notna()]
            if len(pool) < 4:
                continue
            vq    = pool[self.vq_col].values.astype(float)
            codes = pd.Categorical(pool[feat]).codes.astype(float)
            pbr, pbr2, pbp = _pointbiserial(codes, vq)
            eff = _eff_r2(None, None, pbr2=pbr2)
            cat_means = pool.groupby(feat)[self.vq_col].mean().to_dict()
            self._reg(f'Categorical_{feat}',
                      f'Group mean VQ by {feat}',
                      f'{len(pool)} rows grouped by {feat}',
                      None, None, None, None, None, None, None, None, None, None,
                      pbr, pbr2, pbp, eff,
                      _feat=feat, _mtype='Categorical', _cat_means=cat_means)

        # 7 – Historic-adjusted methods
        if self.historic_vq:
            for feat in self.numeric_features:
                preds_h, acts_h = [], []
                for _, row in mn.iterrows():
                    site  = str(row.get(self.site_col, '')).strip()
                    month = row['_Month']
                    cf    = row.get(feat)
                    if pd.isna(cf) or not month:
                        continue
                    hvq = self.historic_vq.get(site, {}).get(month)
                    if hvq is None:
                        ha = self.historic_vq.get(site, {}).get('ANNUAL')
                        if ha:
                            hvq = (ha/12) * self.seasonal_patterns.get(month, 1.0)
                    if hvq is None:
                        continue
                    hf = self.historic_feats.get(site, {}).get(feat)
                    if hf and float(hf) != 0:
                        preds_h.append(hvq * (float(cf)/float(hf)))
                        acts_h.append(row[self.vq_col])
                if len(preds_h) >= 4:
                    ph = np.array(preds_h, float)
                    ah = np.array(acts_h, float)
                    pr, pr2, pp = _pearson(ph, ah)
                    sr, sr2, sp = _spearman(ph, ah)
                    kr, kr2, kp = _kendall(ph, ah)
                    eff = _eff_r2(pr2, sr2, kr2)
                    self._reg(f'Historic_Adjusted_{feat}',
                              f'Historic VQ x (current {feat} / historic {feat})',
                              f'{len(preds_h)} monthly rows with historic VQ + {feat}',
                              None, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None, eff,
                              _feat=feat, _mtype='Historic_Adjusted')

            # Last-year direct
            pl, al = [], []
            for _, row in mn.iterrows():
                site  = str(row.get(self.site_col, '')).strip()
                month = row['_Month']
                hvq   = self.historic_vq.get(site, {}).get(month)
                if hvq is not None:
                    pl.append(hvq)
                    al.append(row[self.vq_col])
            if len(pl) >= 4:
                pla = np.array(pl, float)
                ala = np.array(al, float)
                pr, pr2, pp = _pearson(pla, ala)
                sr, sr2, sp = _spearman(pla, ala)
                kr, kr2, kp = _kendall(pla, ala)
                eff = _eff_r2(pr2, sr2, kr2)
                self._reg('Historic_LastYear_Direct',
                          'Same calendar month value from previous year',
                          f'{len(pl)} monthly rows with historic monthly VQ',
                          None, pr, pr2, pp, sr, sr2, sp, kr, kr2, kp, None, None, None, eff,
                          _mtype='Historic_Direct')

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
        print("\n" + "=" * 60)
        print("MACHINE LEARNING MODELS")
        print("=" * 60)
        mn  = self.known[self.known['_Timeframe'] == 'Monthly']
        all_k = self.known

        if len(mn) >= 6:
            self._train_subset(mn, ['_Month'] + self.numeric_features,
                               'Monthly', self.categorical_features)
        if len(all_k) >= 6:
            self._train_subset(all_k, self.numeric_features,
                               'All', self.categorical_features)
        for cat in self.categorical_features:
            sub = self.known[self.known[cat].notna()]
            if len(sub) >= 6:
                self._train_subset(sub, ['_Month'] + self.numeric_features,
                                   f'By_{cat}', [cat] + self.categorical_features)
        print(f"  {len(self.ml_models)} ML models trained")

    def _train_subset(self, df_sub, feat_cols, prefix, cat_cols=None):
        cat_cols = cat_cols or []
        encoders = {}
        valid = []
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
                    enc = encoders[fc]
                    X_parts.append(enc.transform(vals).reshape(-1, 1))
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
        feat_label = ', '.join(
            fc.replace('_Month', 'Month').replace('_', ' ').strip() for fc in valid)

        for mname, mobj in [
            ('Linear_Regression', LinearRegression()),
            ('Polynomial_Regression', make_pipeline(PolynomialFeatures(2), LinearRegression())),
            ('Random_Forest',
             RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)),
            ('Gradient_Boosting',
             GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)),
        ]:
            key = f'{prefix}_{mname}'
            r2  = _cv_r2(mobj, X, y)
            try:
                mobj.fit(X, y)
            except Exception:
                continue
            self.ml_models[key] = dict(
                model=mobj, features=valid, encoders=encoders,
                r2=float(r2), type=mname.replace('_', ' '),
                description=f'{mname.replace("_"," ")} using {feat_label}. CV R2={r2:.3f} (n={len(rows_ok)})',
                n_train=len(rows_ok), predictions_made=0
            )
            print(f"  {'✓' if r2>=0.3 else '~'} {key}: R2={r2:.4f}")

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_row(self, row):
        site     = str(row.get(self.site_col, '')).strip()
        month    = row.get('_Month', '')
        tf       = row.get('_Timeframe', '')
        cands    = []

        for name, info in self.stat_methods.items():
            pred = self._apply_stat(info, row, site, month)
            if pred is not None and np.isfinite(pred) and pred >= 0:
                cands.append((pred, info['effective_r2'], name))

        for name, info in self.ml_models.items():
            if tf == 'Annual' and '_Month' in info['features']:
                continue
            pred = self._apply_ml(info, row)
            if pred is not None and np.isfinite(pred) and pred >= 0:
                cands.append((pred, info['r2'], name))

        if not cands:
            return float(self.known[self.vq_col].mean()), 'Overall_Average_fallback', 0.0
        best = max(cands, key=lambda x: x[1])
        return best[0], best[2], best[1]

    def _apply_stat(self, info, row, site, month):
        mtype = info.get('_mtype', '')
        feat  = info.get('_feat')

        if '_site_avgs' in info:
            sa = info['_site_avgs']
            # Only use this method if the site actually has known data.
            # Do NOT fall back to a global mean — that would misrepresent the method
            # and steal credit from better-suited methods for unknown sites.
            avg = sa.get(site)
            if avg is None:
                return None
            return float(avg) * self.seasonal_patterns.get(month, 1.0) if info.get('_seasonal') and month else float(avg)

        if '_month_avgs' in info:
            ma = info['_month_avgs']
            # Only use if we actually have data for this specific month
            if not ma or not month:
                return None
            v = ma.get(month)
            return float(v) if v is not None else None

        if '_avg' in info:
            return float(info['_avg'])

        if mtype in ('Feature_Intensity', 'Feature_Intensity_Seasonal'):
            slope = info.get('slope')
            fval  = row.get(feat)
            if slope is None or feat is None or pd.isna(fval):
                return None
            pred = float(fval) * slope
            if mtype == 'Feature_Intensity_Seasonal' and month:
                pred *= self.seasonal_patterns.get(month, 1.0)
            return pred

        if mtype == 'Categorical':
            cm  = info.get('_cat_means', {})
            cv  = row.get(feat)
            if not cm or pd.isna(cv):
                return None
            # Only use if this specific category value was seen in training
            v = cm.get(cv)
            return float(v) if v is not None else None

        if mtype == 'Historic_Adjusted':
            fval = row.get(feat)
            if feat is None or pd.isna(fval):
                return None
            hvq = self.historic_vq.get(site, {}).get(month)
            if hvq is None:
                ha = self.historic_vq.get(site, {}).get('ANNUAL')
                if ha:
                    hvq = (ha/12) * self.seasonal_patterns.get(month, 1.0)
            if hvq is None:
                return None
            hf = self.historic_feats.get(site, {}).get(feat)
            if hf and float(hf) != 0:
                return hvq * (float(fval)/float(hf))

        if mtype == 'Historic_Direct':
            h = self.historic_vq.get(site, {}).get(month)
            if h:
                return float(h)
            a = self.historic_vq.get(site, {}).get('ANNUAL')
            if a:
                return (a/12) * self.seasonal_patterns.get(month, 1.0)

        return None

    def _apply_ml(self, info, row):
        X_parts = []
        for fc in info['features']:
            if fc in info['encoders']:
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

    # ── Output sheets ─────────────────────────────────────────────────────────

    def build_output(self, results):
        out = self.df.copy()
        out['Data integrity']     = 'Actual'
        out['Estimation Method']  = ''
        out['Data Quality Score'] = np.nan

        for r in results:
            i = r['index']
            out.at[i, self.vq_col]          = r['prediction']
            out.at[i, 'Data integrity']     = 'Estimated'
            out.at[i, 'Estimation Method']  = r['method']
            out.at[i, 'Data Quality Score'] = round(r['r2'], 4)

        out.drop(columns=['_Timeframe', '_Month'], errors='ignore', inplace=True)

        avail  = self._sheet_availability()
        stat   = self._sheet_stat()
        ml     = self._sheet_ml()
        yoy    = self._sheet_yoy(out) if self.historic_vq else None
        return out, avail, stat, ml, yoy

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
                v = row.get(f)
                entry[f] = v if pd.notna(v) else ''
            for f in self.categorical_features:
                v = row.get(f)
                entry[f] = v if pd.notna(v) else ''
            if self.historic_vq:
                hvq = (self.historic_vq.get(site, {}).get(month)
                       or self.historic_vq.get(site, {}).get('ANNUAL', ''))
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
                'Method': name,
                'Description': info['description'],
                'Input Data': info['input_data'],
                'Intensity_Slope': info.get('slope'),
                'Pearson_r': info.get('pearson_r'),
                'Pearson_R2': info.get('pearson_r2'),
                'Pearson_p': info.get('pearson_p'),
                'Spearman_r': info.get('spearman_r'),
                'Spearman_R2': info.get('spearman_r2'),
                'Spearman_p': info.get('spearman_p'),
                'Kendall_tau': info.get('kendall_tau'),
                'Kendall_R2': info.get('kendall_r2'),
                'Kendall_p': info.get('kendall_p'),
                'PointBiserial_r': info.get('pointbiserial_r'),
                'PointBiserial_R2': info.get('pointbiserial_r2'),
                'PointBiserial_p': info.get('pointbiserial_p'),
                'Effective_R2': round(info['effective_r2'], 4),
                'Used_In_Prediction': 'Yes' if info['predictions_made'] > 0 else 'No',
                'Predictions_Made': info['predictions_made'],
            })
        df = pd.DataFrame(rows)
        if len(df):
            df = df.sort_values('Effective_R2', ascending=False).reset_index(drop=True)
        return df

    def _sheet_ml(self):
        rows = []
        for name, info in self.ml_models.items():
            feats = ', '.join(
                fc.replace('_Month', 'Month').replace('_', ' ').strip()
                for fc in info['features'])
            rows.append({
                'Model': name,
                'Type': info['type'],
                'R2': round(info['r2'], 4),
                'Features_Used': feats,
                'Description': info['description'],
                'N_Training': info.get('n_train', ''),
                'Predictions_Made': info['predictions_made'],
            })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values('R2', ascending=False).reset_index(drop=True)

    def _sheet_yoy(self, out):
        hy = self.historic_year or 'Prev'
        cy = self.current_year  or 'Current'
        # Re-attach _Month for YoY
        month_map = self.df['_Month'].to_dict() if '_Month' in self.df.columns else {}
        rows = []
        sites = [str(s).strip() for s in out[self.site_col].unique()
                 if str(s).strip() in self.historic_vq]

        for site in sorted(sites):
            site_rows = out[out[self.site_col] == site]
            hist = self.historic_vq.get(site, {})
            h_tot, c_tot = 0.0, 0.0

            for month in MONTHS_ORDER:
                # Find matching output row for this site+month
                m_idx = [i for i in site_rows.index
                         if month_map.get(i, '') == month]
                curr_vq  = float(site_rows.loc[m_idx[0], self.vq_col]) if m_idx else None
                hist_vq  = hist.get(month)
                if hist_vq is None and 'ANNUAL' in hist:
                    sf = self.seasonal_patterns.get(month, 1.0)
                    hist_vq = (hist['ANNUAL']/12) * sf

                chg = (curr_vq - hist_vq) if curr_vq is not None and hist_vq else None
                pct = round(chg/hist_vq*100, 1) if (chg and hist_vq) else None

                is_est = False
                em, dqs = '', None
                if m_idx and m_idx[0] in out.index:
                    r = out.loc[m_idx[0]]
                    is_est = r.get('Data integrity', '') == 'Estimated'
                    em  = r.get('Estimation Method', '') if is_est else ''
                    dqs = r.get('Data Quality Score')    if is_est else None

                row = {
                    'Site': site, 'Month': month,
                    f'VQ_{hy}': round(hist_vq, 2) if hist_vq else None,
                    f'VQ_{cy}': round(curr_vq, 2) if curr_vq else None,
                    'VQ_Change': round(chg, 2)  if chg else None,
                    'VQ_Change_%': pct,
                    'Estimation Method': em,
                    'Data Quality Score': round(dqs, 4) if dqs else None,
                    'Estimated': 'Yes' if is_est else 'No',
                    'Flag': ('Large change' if pct and abs(pct) > 50
                             else 'Normal' if pct and abs(pct) < 10
                             else 'Moderate') if pct else '',
                }
                # Feature YoY
                for f in self.numeric_features:
                    cf_val = float(site_rows.loc[m_idx[0], f]) if m_idx and pd.notna(site_rows.loc[m_idx[0], f] if m_idx else np.nan) else None
                    hf_val = self.historic_feats.get(site, {}).get(f)
                    row[f'{f}_{hy}'] = hf_val
                    row[f'{f}_{cy}'] = cf_val
                    if hf_val and cf_val and hf_val != 0:
                        row[f'{f}_Change_%'] = round((cf_val-hf_val)/hf_val*100, 1)
                rows.append(row)
                if hist_vq: h_tot += hist_vq
                if curr_vq: c_tot += curr_vq

            # TOTAL row
            chg_tot = c_tot - h_tot if h_tot and c_tot else None
            rows.append({
                'Site': site, 'Month': 'TOTAL',
                f'VQ_{hy}': round(h_tot, 2) if h_tot else None,
                f'VQ_{cy}': round(c_tot, 2) if c_tot else None,
                'VQ_Change': round(chg_tot, 2) if chg_tot else None,
                'VQ_Change_%': round(chg_tot/h_tot*100,1) if (chg_tot and h_tot) else None,
                'Estimated': '', 'Flag': '', 'Estimation Method': '', 'Data Quality Score': None,
            })
        return pd.DataFrame(rows) if rows else None

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        """Full pipeline. Returns (complete, availability, stat, ml, yoy)."""
        self.load()
        self.run_statistical_tests()
        self.train_ml_models()
        results = self.fill_blanks()
        return self.build_output(results)