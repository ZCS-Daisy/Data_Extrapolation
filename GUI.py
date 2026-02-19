"""
GUI for Volumetric Extrapolation Tool
======================================
Matches original interface. New: Features button opens a selector popup
that scans the input file, scores columns, and lets the user choose which
to include as numeric/categorical features.
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import sys
import os
import pandas as pd
import json
from pathlib import Path
import numpy as np

from Data_Extrapolation import ExtrapolationTool, scan_features


# ─────────────────────────────────────────────────────────────────────────────
# Text redirector
# ─────────────────────────────────────────────────────────────────────────────

class TextRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, s):
        self.widget.insert(tk.END, s)
        self.widget.see(tk.END)
        self.widget.update_idletasks()

    def flush(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Feature selector popup
# ─────────────────────────────────────────────────────────────────────────────

class FeatureSelectorDialog(tk.Toplevel):
    """
    Modal popup showing detected feature columns from the input file.
    Recommended features are pre-ticked. User can toggle any on/off.
    Returns selected numeric and categorical features on confirm.
    """

    BG      = "#F4F6F7"
    CARD    = "#FFFFFF"
    PRIMARY = "#2C3E50"
    ACCENT  = "#3498DB"
    SUCCESS = "#27AE60"
    WARNING = "#E67E22"
    TEXT    = "#2C3E50"
    LITE    = "#7F8C8D"
    REC_BG  = "#EAF6FF"
    OPT_BG  = "#FDFEFE"

    def __init__(self, parent, filepath):
        super().__init__(parent)
        self.title("Select Features")
        self.geometry("780x620")
        self.resizable(True, True)
        self.configure(bg=self.BG)
        self.transient(parent)
        self.grab_set()

        self.filepath = filepath
        self.result_numeric     = None
        self.result_categorical = None
        self.cancelled = True

        self._scan_and_build()

    def _scan_and_build(self):
        # Loading label while scanning
        loading = ttk.Label(self, text="Scanning file for feature columns...",
                            font=('Segoe UI', 11), background=self.BG,
                            foreground=self.LITE)
        loading.pack(pady=40)
        self.update()

        try:
            self.features = scan_features(self.filepath)
        except Exception as e:
            loading.destroy()
            ttk.Label(self, text=f"Error scanning file:\n{e}",
                      foreground='red', background=self.BG).pack(pady=20)
            return

        loading.destroy()
        self._build_ui()

    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=self.PRIMARY, pady=16, padx=20)
        hdr.pack(fill='x')
        tk.Label(hdr, text="Feature Selection",
                 font=('Segoe UI', 16, 'bold'),
                 fg='white', bg=self.PRIMARY).pack(anchor='w')
        tk.Label(hdr,
                 text="Choose which columns to use as predictive features. "
                      "Recommended columns are pre-selected based on fill rate and variance.",
                 font=('Segoe UI', 9), fg='#BDC3C7', bg=self.PRIMARY,
                 wraplength=720, justify='left').pack(anchor='w', pady=(4, 0))

        # ── Legend ────────────────────────────────────────────────────────────
        leg = tk.Frame(self, bg=self.BG, pady=8, padx=20)
        leg.pack(fill='x')
        for colour, label in [(self.SUCCESS, '● Recommended'), (self.WARNING, '● Optional')]:
            tk.Label(leg, text=label, font=('Segoe UI', 9, 'bold'),
                     fg=colour, bg=self.BG).pack(side='left', padx=(0, 20))
        tk.Label(leg,
                 text="Recommendations based on fill rate (≥50%) and data variance",
                 font=('Segoe UI', 8), fg=self.LITE, bg=self.BG).pack(side='left')

        # ── Scrollable feature list ───────────────────────────────────────────
        list_outer = tk.Frame(self, bg=self.BG, padx=15, pady=5)
        list_outer.pack(fill='both', expand=True)

        canvas = tk.Canvas(list_outer, bg=self.BG, highlightthickness=0, bd=0)
        vsb    = ttk.Scrollbar(list_outer, orient='vertical', command=canvas.yview)
        inner  = tk.Frame(canvas, bg=self.BG)
        inner.bind('<Configure>',
                   lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        cw = canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(cw, width=e.width))
        canvas.configure(yscrollcommand=vsb.set)

        # Bind mousewheel to canvas only (not bind_all) so it doesn't
        # outlive this dialog and try to scroll a destroyed widget
        def _on_scroll(e):
            try:
                canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units')
            except Exception:
                pass

        canvas.bind('<MouseWheel>', _on_scroll)
        # Also bind to inner frame and children so scroll works anywhere in dialog
        self.bind('<MouseWheel>', _on_scroll)

        vsb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        self.check_vars = {}  # {col_name: BooleanVar}

        if not self.features:
            tk.Label(inner, text="No suitable feature columns found in this file.",
                     font=('Segoe UI', 10), fg=self.LITE, bg=self.BG).pack(pady=20)
        else:
            # Section headers
            sections = [
                ('Numeric Features',     [f for f in self.features if f['type'] == 'Numeric']),
                ('Categorical Features', [f for f in self.features if f['type'] == 'Categorical']),
            ]
            for section_title, feats in sections:
                if not feats:
                    continue
                # Section label
                sec_lbl = tk.Frame(inner, bg=self.BG)
                sec_lbl.pack(fill='x', pady=(12, 4), padx=4)
                tk.Label(sec_lbl, text=section_title,
                         font=('Segoe UI', 10, 'bold'),
                         fg=self.PRIMARY, bg=self.BG).pack(anchor='w')
                tk.Frame(sec_lbl, bg='#D5D8DC', height=1).pack(fill='x', pady=(2, 0))

                for feat in feats:
                    self._add_feature_row(inner, feat)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_bar = tk.Frame(self, bg=self.BG, pady=12, padx=20)
        btn_bar.pack(fill='x', side='bottom')

        # Select all / none shortcuts
        tk.Button(btn_bar, text='Select All',
                  font=('Segoe UI', 9), fg=self.ACCENT, bg=self.BG,
                  relief='flat', cursor='hand2',
                  command=lambda: self._toggle_all(True)).pack(side='left')
        tk.Button(btn_bar, text='Select None',
                  font=('Segoe UI', 9), fg=self.LITE, bg=self.BG,
                  relief='flat', cursor='hand2',
                  command=lambda: self._toggle_all(False)).pack(side='left', padx=(8, 0))
        tk.Button(btn_bar, text='Recommended Only',
                  font=('Segoe UI', 9), fg=self.SUCCESS, bg=self.BG,
                  relief='flat', cursor='hand2',
                  command=self._select_recommended).pack(side='left', padx=(8, 0))

        tk.Button(btn_bar, text='Cancel',
                  font=('Segoe UI', 10), fg='white',
                  bg='#BDC3C7', activebackground='#95A5A6',
                  relief='flat', padx=18, pady=8, cursor='hand2',
                  command=self.destroy).pack(side='right', padx=(8, 0))
        tk.Button(btn_bar, text='Confirm Selection',
                  font=('Segoe UI', 10, 'bold'), fg='white',
                  bg=self.ACCENT, activebackground='#2980B9',
                  relief='flat', padx=18, pady=8, cursor='hand2',
                  command=self._confirm).pack(side='right')

    def _add_feature_row(self, parent, feat):
        recommended = feat['recommended']
        row_bg = self.REC_BG if recommended else self.OPT_BG

        row = tk.Frame(parent, bg=row_bg, pady=8, padx=12,
                       highlightbackground='#D5D8DC', highlightthickness=1)
        row.pack(fill='x', pady=2, padx=4)

        # Checkbox
        var = tk.BooleanVar(value=recommended)
        self.check_vars[feat['name']] = var
        cb = tk.Checkbutton(row, variable=var, bg=row_bg,
                            activebackground=row_bg, cursor='hand2')
        cb.pack(side='left', padx=(0, 8))

        # Status dot + name
        dot_colour = self.SUCCESS if recommended else self.WARNING
        status     = 'Recommended' if recommended else 'Optional'
        tk.Label(row, text='●', fg=dot_colour, bg=row_bg,
                 font=('Segoe UI', 10)).pack(side='left')

        name_f = tk.Frame(row, bg=row_bg)
        name_f.pack(side='left', fill='x', expand=True, padx=(6, 0))

        tk.Label(name_f, text=feat['name'],
                 font=('Segoe UI', 10, 'bold'),
                 fg=self.PRIMARY, bg=row_bg).pack(anchor='w')
        tk.Label(name_f,
                 text=f"{feat['type']}  ·  {status}  ·  {feat['reason']}",
                 font=('Segoe UI', 8), fg=self.LITE, bg=row_bg).pack(anchor='w')

        # Stats badges
        stats_f = tk.Frame(row, bg=row_bg)
        stats_f.pack(side='right', padx=(10, 0))
        for label, value in [
            ('Fill', f"{feat['fill_pct']}%"),
            ('Unique', str(feat['unique_count'])),
            ('Variance', f"{feat['variance_score']:.2f}"),
        ]:
            badge = tk.Frame(stats_f, bg='#D5D8DC', padx=6, pady=2)
            badge.pack(side='left', padx=2)
            tk.Label(badge, text=f"{label}: {value}",
                     font=('Segoe UI', 8), fg=self.TEXT,
                     bg='#D5D8DC').pack()

    def _toggle_all(self, state):
        for var in self.check_vars.values():
            var.set(state)

    def _select_recommended(self):
        for feat in self.features:
            self.check_vars[feat['name']].set(feat['recommended'])

    def _confirm(self):
        selected = {name for name, var in self.check_vars.items() if var.get()}
        feat_map = {f['name']: f for f in self.features}

        self.result_numeric = [
            name for name in selected
            if name in feat_map and feat_map[name]['type'] == 'Numeric'
        ]
        self.result_categorical = [
            name for name in selected
            if name in feat_map and feat_map[name]['type'] == 'Categorical'
        ]
        self.cancelled = False
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────

class ExtrapolationGUI:

    BG_MAIN   = "#E8E8E8"
    BG_CARD   = "#FFFFFF"
    PRIMARY   = "#2C3E50"
    ACCENT    = "#3498DB"
    SUCCESS   = "#27AE60"
    TEXT_DARK = "#2C3E50"
    TEXT_LITE = "#7F8C8D"
    DANGER    = "#E74C3C"
    WARNING   = "#E67E22"

    def __init__(self, root):
        self.root = root
        self.root.title("Extrapolation")
        self.root.geometry("1200x880")
        self.root.resizable(True, True)
        self.root.configure(bg=self.BG_MAIN)

        self.input_file_path        = tk.StringVar()
        self.historic_records_path  = tk.StringVar()
        self.historic_features_path = tk.StringVar()
        # Factor database persists across sessions — saved to config file
        self.factor_database_path   = tk.StringVar()
        self.is_processing = False

        # Feature selection state — None means "auto-detect"
        self.selected_numeric_features     = None
        self.selected_categorical_features = None

        self.config_file = Path.home() / '.extrapolation_tool_config.json'

        self._build_styles()
        self._build_ui()
        self._load_config()   # restore persisted paths after UI is built

        sys.stdout = TextRedirector(self.log_text)

    def _save_config(self):
        """Save persistent state to JSON so it survives app restarts."""
        try:
            data = {
                'factor_database_path': self.factor_database_path.get(),
            }
            self.config_file.write_text(json.dumps(data, indent=2))
        except Exception:
            pass  # never crash the app over config saving

    def _load_config(self):
        """Restore persistent state from JSON on startup."""
        try:
            if self.config_file.exists():
                data = json.loads(self.config_file.read_text())
                ef = data.get('factor_database_path', '')
                if ef and Path(ef).exists():
                    self.factor_database_path.set(ef)
        except Exception:
            pass

    # ── Styles ────────────────────────────────────────────────────────────────

    def _build_styles(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('Modern.TFrame', background=self.BG_MAIN)
        s.configure('Modern.TButton',
                    background=self.ACCENT, foreground='white',
                    borderwidth=0, focuscolor='none', padding=10,
                    font=('Segoe UI', 10))
        s.map('Modern.TButton', background=[('active', '#2980B9')])
        s.configure('Template.TButton',
                    background='#BDC3C7', foreground='white',
                    borderwidth=0, focuscolor='none', padding=10,
                    font=('Segoe UI', 9))
        s.map('Template.TButton', background=[('active', '#95A5A6')])
        s.configure('Card.TLabelframe',
                    background=self.BG_CARD, borderwidth=0, relief='flat')
        s.configure('Card.TLabelframe.Label',
                    background=self.BG_CARD, foreground=self.TEXT_DARK,
                    font=('Segoe UI', 10, 'bold'))
        s.configure('TLabel',
                    background=self.BG_CARD, foreground=self.TEXT_DARK,
                    font=('Segoe UI', 9))
        s.configure('Modern.Horizontal.TProgressbar',
                    background=self.ACCENT, troughcolor=self.BG_MAIN,
                    borderwidth=0, thickness=8)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        main = ttk.Frame(self.root, style='Modern.TFrame')
        main.pack(fill='both', expand=True)

        canvas = tk.Canvas(main, bg=self.BG_MAIN, highlightthickness=0, bd=0)
        vsb    = ttk.Scrollbar(main, orient='vertical', command=canvas.yview)
        scroll_frame = ttk.Frame(canvas, style='Modern.TFrame')

        scroll_frame.bind('<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        cw = canvas.create_window((0, 0), window=scroll_frame, anchor='nw')
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(cw, width=e.width))
        canvas.configure(yscrollcommand=vsb.set)

        def _main_scroll(e):
            try:
                canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units')
            except Exception:
                pass

        canvas.bind('<MouseWheel>', _main_scroll)
        scroll_frame.bind('<MouseWheel>', _main_scroll)

        vsb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        scroll_frame.columnconfigure(0, weight=1)

        # ── Title
        title_f = ttk.Frame(scroll_frame, style='Modern.TFrame', padding='20 20 20 10')
        title_f.grid(row=0, column=0, sticky='ew')
        ttk.Label(title_f, text='Extrapolation',
                  font=('Segoe UI', 28, 'bold'),
                  foreground=self.PRIMARY, background=self.BG_MAIN).pack(anchor='w')
        ttk.Label(title_f,
                  text='Extrapolate blank volumetric quantities using Machine Learning and statistical data patterns',
                  font=('Segoe UI', 11), foreground=self.TEXT_LITE,
                  background=self.BG_MAIN).pack(anchor='w', pady=(5, 0))

        # ── File selection card
        file_f = ttk.LabelFrame(scroll_frame, style='Card.TLabelframe',
                                text='File Selection', padding='20')
        file_f.grid(row=1, column=0, sticky='ew', padx=0, pady=(0, 1))

        def _file_row(parent, label, textvar, r, bold=False,
                      browse_cmd=None, template_cmd=None):
            font = ('Segoe UI', 9, 'bold') if bold else ('Segoe UI', 9)
            pady = (10, 5) if r == 0 else (5, 5)
            ttk.Label(parent, text=label, font=font).grid(
                row=r, column=0, sticky='w', pady=pady, padx=(0, 10))
            ttk.Entry(parent, textvariable=textvar, width=55,
                      font=('Segoe UI', 9)).grid(row=r, column=1, padx=5, pady=pady)
            if browse_cmd:
                ttk.Button(parent, text='Browse', command=browse_cmd,
                           style='Modern.TButton', width=12).grid(
                    row=r, column=2, pady=pady, padx=(5, 5))
            if template_cmd:
                ttk.Button(parent, text='Export Template', command=template_cmd,
                           style='Template.TButton', width=15).grid(
                    row=r, column=3, pady=pady, padx=(0, 0))

        _file_row(file_f, 'Input File:', self.input_file_path, 0, bold=True,
                  browse_cmd=self._browse_input,
                  template_cmd=self._export_input_template)
        _file_row(file_f, 'Historic Records (Optional):', self.historic_records_path, 1,
                  browse_cmd=self._browse_historic_records,
                  template_cmd=self._export_historic_records_template)
        _file_row(file_f, 'Historic Features (Optional):', self.historic_features_path, 2,
                  browse_cmd=self._browse_historic_features,
                  template_cmd=self._export_historic_features_template)

        # Factor Database row — persists across uploads until user changes it
        _file_row(file_f, 'Factor Database (Optional):', self.factor_database_path, 3,
                  browse_cmd=self._browse_factor_database)

        # Clear factor database button
        clear_ef_row = tk.Frame(file_f, bg=self.BG_CARD)
        clear_ef_row.grid(row=3, column=3, padx=(4, 0), pady=2, sticky='w')
        ttk.Button(clear_ef_row, text='✕ Clear',
                   command=lambda: [self.factor_database_path.set(''), self._save_config()],
                   style='Template.TButton', width=8).pack(side='left')

        # ── Features row (below file rows, spanning full width)
        feat_row = tk.Frame(file_f, bg=self.BG_CARD)
        feat_row.grid(row=4, column=0, columnspan=4, sticky='ew', pady=(8, 4))

        ttk.Button(feat_row, text='⚙  Select Features',
                   command=self._open_feature_selector,
                   style='Modern.TButton', width=20).pack(side='left')

        self.feature_status_label = tk.Label(
            feat_row,
            text='Features: auto-detect from input file',
            font=('Segoe UI', 9, 'italic'),
            fg=self.TEXT_LITE, bg=self.BG_CARD)
        self.feature_status_label.pack(side='left', padx=(12, 0))

        self.feature_reset_btn = tk.Button(
            feat_row, text='✕ Reset to auto',
            font=('Segoe UI', 8), fg=self.DANGER, bg=self.BG_CARD,
            relief='flat', cursor='hand2',
            command=self._reset_features)
        # Only shown when features have been manually selected

        # ── About card
        about_f = ttk.LabelFrame(scroll_frame, style='Card.TLabelframe',
                                 text='About This Tool', padding='20')
        about_f.grid(row=2, column=0, sticky='ew', padx=0, pady=1)
        about_text = (
            "Volumetric Quantity Extrapolation\n\n"
            "This tool compares data-dependent prediction models, filling in blank Volumetric "
            "Quantities using the most suited Statistical or Machine Learning method for each row.\n\n"
            "Required Input Columns:\n"
            "    \u2022 Site identifier (or Location if no site ID)\n"
            "    \u2022 Date from and Date to (DD/MM/YYYY or YYYY-MM-DD)\n"
            "    \u2022 Volumetric Quantity (with blank values to extrapolate)\n\n"
            "Optional \u2014 To Improve Predictions:\n"
            "    \u2022 Any numeric feature columns (e.g. Turnover, Floor Space, kWh)\n"
            "    \u2022 Categorical feature columns (e.g. Brand)\n"
            "    \u2022 Historic Records: Previous years' volumetric quantities\n"
            "    \u2022 Historic Features: Previous years' feature values by site\n\n"
            "Use 'Select Features' to choose which columns act as predictive features, "
            "or leave on auto-detect.\n\n"
            "Output Sheets:\n"
            "    1. Complete Carbon Records \u2014 all rows with filled VQ + integrity + method + quality score\n"
            "    2. Data Availability \u2014 per-row summary of site, month, features, historic data\n"
            "    3. Statistical Analysis \u2014 all stat tests with R\u00b2 and p-values\n"
            "    4. Machine Learning Models \u2014 CV R\u00b2, train/test row counts, data sources\n"
            "    5. YoY Analysis \u2014 site-by-site comparison with historic data (if provided)"
        )
        ttk.Label(about_f, text=about_text, justify='left',
                  font=('Segoe UI', 9), foreground=self.TEXT_DARK).pack(anchor='w')

        # ── Progress
        prog_f = ttk.Frame(scroll_frame, style='Modern.TFrame', padding='20 15')
        prog_f.grid(row=3, column=0, sticky='ew')
        self.progress_bar = ttk.Progressbar(
            prog_f, mode='indeterminate', length=1140,
            style='Modern.Horizontal.TProgressbar')
        self.progress_bar.pack(fill='x')
        self.status_label = ttk.Label(
            prog_f, text='Ready to process',
            foreground=self.SUCCESS, background=self.BG_MAIN,
            font=('Segoe UI', 11, 'bold'))
        self.status_label.pack(pady=8)

        # ── Action buttons
        btn_f = ttk.Frame(scroll_frame, style='Modern.TFrame', padding='0 10 20 15')
        btn_f.grid(row=4, column=0, sticky='ew')
        self.process_btn = ttk.Button(
            btn_f, text='Process Data', command=self._start_processing,
            style='Modern.TButton', width=18)
        self.process_btn.pack(side='left', padx=(20, 5))
        ttk.Button(btn_f, text='Clear Log', command=self._clear_log,
                   style='Modern.TButton', width=18).pack(side='left', padx=5)

        # ── Log
        log_f = ttk.LabelFrame(scroll_frame, style='Card.TLabelframe',
                               text='Processing Log', padding='20')
        log_f.grid(row=5, column=0, sticky='ewns', padx=0, pady=(1, 0))
        self.log_text = scrolledtext.ScrolledText(
            log_f, width=140, height=18, wrap='word',
            font=('Consolas', 9), bg='white', relief='flat', borderwidth=0)
        self.log_text.pack(fill='both', expand=True)
        scroll_frame.rowconfigure(5, weight=1)

        self.root.protocol('WM_DELETE_WINDOW', self.root.destroy)

    # ── Feature selector ──────────────────────────────────────────────────────

    def _open_feature_selector(self):
        path = self.input_file_path.get().strip()
        if not path:
            messagebox.showerror('No File', 'Please select an input file first.')
            return
        if not os.path.exists(path):
            messagebox.showerror('File Not Found', f'Cannot find:\n{path}')
            return

        dlg = FeatureSelectorDialog(self.root, path)
        self.root.wait_window(dlg)

        if not dlg.cancelled:
            self.selected_numeric_features     = dlg.result_numeric
            self.selected_categorical_features = dlg.result_categorical

            n_num = len(self.selected_numeric_features)
            n_cat = len(self.selected_categorical_features)
            total = n_num + n_cat

            if total == 0:
                self._reset_features()
                messagebox.showinfo('No Features',
                                    'No features selected — the tool will auto-detect features from the file.')
                return

            parts = []
            if n_num:
                parts.append(f'{n_num} numeric: {", ".join(self.selected_numeric_features)}')
            if n_cat:
                parts.append(f'{n_cat} categorical: {", ".join(self.selected_categorical_features)}')
            summary = '  |  '.join(parts)

            self.feature_status_label.config(
                text=f'Features ({total} selected): {summary}',
                fg=self.PRIMARY)
            self.feature_reset_btn.pack(side='left', padx=(8, 0))

            print(f"\n✓ Features selected manually:")
            if self.selected_numeric_features:
                print(f"  Numeric:     {self.selected_numeric_features}")
            if self.selected_categorical_features:
                print(f"  Categorical: {self.selected_categorical_features}\n")

    def _reset_features(self):
        self.selected_numeric_features     = None
        self.selected_categorical_features = None
        self.feature_status_label.config(
            text='Features: auto-detect from input file',
            fg=self.TEXT_LITE)
        self.feature_reset_btn.pack_forget()
        print("\n  Features reset to auto-detect\n")

    # ── Browse helpers ────────────────────────────────────────────────────────

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title='Select Input Data File',
            filetypes=[('Excel/CSV', '*.xlsx *.xls *.csv'), ('All files', '*.*')])
        if path:
            self.input_file_path.set(path)
            # Clear any manually selected features when a new file is loaded
            self._reset_features()
            print(f"\n✓ Input file selected: {os.path.basename(path)}\n")

    def _browse_historic_records(self):
        path = filedialog.askopenfilename(
            title='Select Historic Records File',
            filetypes=[('Excel/CSV', '*.xlsx *.xls *.csv'), ('All files', '*.*')])
        if path:
            self.historic_records_path.set(path)
            print(f"\n✓ Historic records selected: {os.path.basename(path)}\n")

    def _browse_historic_features(self):
        path = filedialog.askopenfilename(
            title='Select Historic Features File',
            filetypes=[('Excel/CSV', '*.xlsx *.xls *.csv'), ('All files', '*.*')])
        if path:
            self.historic_features_path.set(path)
            print(f"\n✓ Historic features selected: {os.path.basename(path)}\n")

    def _browse_factor_database(self):
        path = filedialog.askopenfilename(
            title='Select Factor Database',
            filetypes=[('Excel/CSV', '*.xlsx *.xls *.csv'), ('All files', '*.*')])
        if path:
            self.factor_database_path.set(path)
            self._save_config()
            print(f"\n✓ Factor database loaded: {os.path.basename(path)}")
            print("  (Saved — will persist when app is closed and reopened)\n")

    # ── Template exports ──────────────────────────────────────────────────────

    def _export_input_template(self):
        path = filedialog.asksaveasfilename(
            title='Save Input Template', defaultextension='.xlsx',
            filetypes=[('Excel', '*.xlsx'), ('CSV', '*.csv')],
            initialfile='Input_Template.xlsx')
        if not path:
            return
        try:
            df = pd.DataFrame({
                'Site identifier': ['SITE001', 'SITE001', 'SITE002'],
                'Location':        ['London, UK', 'London, UK', 'Manchester, UK'],
                'Date from':       ['01/01/2025', '01/02/2025', '01/01/2025'],
                'Date to':         ['31/01/2025', '28/02/2025', '31/01/2025'],
                'GHG Category':    ['Electricity', 'Electricity', 'Electricity'],
                'Volumetric Quantity': [12000, '', 9500],
                'Turnover':        [500000, 520000, 430000],
                'Brand':           ['Brand A', 'Brand A', 'Brand B'],
            })
            if path.endswith('.csv'):
                df.to_csv(path, index=False)
            else:
                df.to_excel(path, index=False)
            self.input_file_path.set(path)
            self._reset_features()
            print(f"\n✓ Input template exported: {path}")
            if messagebox.askyesno('Template Exported',
                                   f'Template saved to:\n{path}\n\nOpen it now?'):
                os.startfile(path)
        except Exception as e:
            messagebox.showerror('Error', f'Could not export template: {e}')

    def _export_historic_records_template(self):
        path = filedialog.asksaveasfilename(
            title='Save Historic Records Template', defaultextension='.xlsx',
            filetypes=[('Excel', '*.xlsx'), ('CSV', '*.csv')],
            initialfile='Historic_Records_Template.xlsx')
        if not path:
            return
        try:
            df = pd.DataFrame({
                'Site identifier': ['SITE001', 'SITE001', 'SITE002'],
                'Date from':       ['01/01/2024', '01/02/2024', '01/01/2024'],
                'Date to':         ['31/01/2024', '28/02/2024', '31/01/2024'],
                'GHG Category':    ['Electricity', 'Electricity', 'Electricity'],
                'Volumetric Quantity': [11000, 10500, 8800],
            })
            if path.endswith('.csv'):
                df.to_csv(path, index=False)
            else:
                df.to_excel(path, index=False)
            self.historic_records_path.set(path)
            print(f"\n✓ Historic records template exported: {path}")
            if messagebox.askyesno('Template Exported',
                                   f'Template saved to:\n{path}\n\nOpen it now?'):
                os.startfile(path)
        except Exception as e:
            messagebox.showerror('Error', f'Could not export template: {e}')

    def _export_historic_features_template(self):
        path = filedialog.asksaveasfilename(
            title='Save Historic Features Template', defaultextension='.xlsx',
            filetypes=[('Excel', '*.xlsx'), ('CSV', '*.csv')],
            initialfile='Historic_Features_Template.xlsx')
        if not path:
            return
        try:
            df = pd.DataFrame({
                'Site identifier': ['SITE001', 'SITE002'],
                'Date from':       ['01/01/2024', '01/01/2024'],
                'Date to':         ['31/12/2024', '31/12/2024'],
                'Turnover':        [6000000, 5200000],
                'Brand':           ['Brand A', 'Brand B'],
            })
            if path.endswith('.csv'):
                df.to_csv(path, index=False)
            else:
                df.to_excel(path, index=False)
            self.historic_features_path.set(path)
            print(f"\n✓ Historic features template exported: {path}")
            if messagebox.askyesno('Template Exported',
                                   f'Template saved to:\n{path}\n\nOpen it now?'):
                os.startfile(path)
        except Exception as e:
            messagebox.showerror('Error', f'Could not export template: {e}')

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _clear_log(self):
        self.log_text.delete('1.0', tk.END)

    def _set_status(self, msg, colour='black'):
        colours = {'green': self.SUCCESS, 'blue': self.ACCENT, 'red': self.DANGER}
        self.status_label.config(text=msg,
                                 foreground=colours.get(colour, colour))

    # ── Processing ────────────────────────────────────────────────────────────

    def _start_processing(self):
        if self.is_processing:
            messagebox.showwarning('Processing', 'Already processing. Please wait.')
            return
        if not self.input_file_path.get():
            messagebox.showerror('Error', 'Please select an input data file.')
            return

        self.is_processing = True
        self.process_btn.config(state='disabled')
        self.progress_bar.start()
        self._set_status('Processing...', 'blue')

        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def _apply_factor_database(self, complete_df, factor_db_path):
        """
        Three jobs:
        1. Strip internal Cross-cat feature columns — these are computation
           artefacts, not carbon records fields, and must not appear in output.
        2. Use the factor database as a COLUMN TEMPLATE — the output columns
           follow the factor DB column order exactly. Any input column not in
           the factor DB is appended at the end.
        3. Left-join factor rows onto the output where keys match, filling in
           emission factor values (EF kgCO2e FU, tCO2e, etc.).
        """
        try:
            print("\n  Applying factor database as column template + lookup...")

            # ── Step 1: strip internal cross-category feature columns ─────────
            cross_cat_cols = [c for c in complete_df.columns if c.startswith('Cross-cat:')]
            if cross_cat_cols:
                complete_df = complete_df.drop(columns=cross_cat_cols)
                print(f"  Removed {len(cross_cat_cols)} internal Cross-cat columns from output")

            if factor_db_path.lower().endswith('.csv'):
                ef_df = pd.read_csv(factor_db_path)
            else:
                ef_df = pd.read_excel(factor_db_path)
            ef_df.columns = ef_df.columns.str.strip()

            # Columns that are factor values — never join on these
            value_cols = {
                'Volumetric Quantity', 'Data integrity', 'Estimation Method',
                'Data Quality Score', 'kgCO2e', 'tCO2e',
                'EF kgCO2e FU', 'EF Source', 'Factor name',
                'Generic kgCO2e', 'Generic tCO2e', 'Generic EF kgCO2e FU',
                'Generic EF source', 'Generic factor name',
                'CO2 tCO2e', 'CH4 tCO2e', 'N2O tCO2e',
                'kg CO2e of CO2 per unit', 'kg CO2e of CH4 per unit',
                'kg CO2e of N2O per unit',
            }

            # ── Step 2: establish join keys ───────────────────────────────────
            join_keys = [
                c for c in ef_df.columns
                if c in complete_df.columns and c not in value_cols
                and ef_df[c].notna().mean() > 0.5
            ]

            # ── Step 3: join factor values where keys match ────────────────────
            new_ef_cols = [c for c in ef_df.columns if c not in complete_df.columns]
            print(f"  Join keys: {join_keys if join_keys else 'none found'}")
            print(f"  New columns from factor DB: {len(new_ef_cols)}")

            if join_keys and new_ef_cols:
                ef_dedup = ef_df[join_keys + new_ef_cols].drop_duplicates(subset=join_keys)
                merged = complete_df.merge(ef_dedup, on=join_keys, how='left')
                matched = merged[new_ef_cols[0]].notna().sum()
                print(f"  ✓ {matched}/{len(merged)} rows matched emission factors")
            elif new_ef_cols:
                print("  ⚠  No join keys — adding empty factor columns for template shape")
                merged = complete_df.copy()
                for col in new_ef_cols:
                    merged[col] = np.nan
            else:
                print("  ⚠  Factor database has no new columns — stripping cross-cat only")
                merged = complete_df.copy()

            # ── Step 4: enforce EXACT factor DB column order ──────────────────
            # Factor DB columns come first in their exact order.
            # Any remaining input columns not in the factor DB are appended after.
            ef_col_order    = list(ef_df.columns)
            remaining_input = [c for c in complete_df.columns if c not in ef_col_order]
            final_order     = ef_col_order + remaining_input
            merged = merged[[c for c in final_order if c in merged.columns]]

            return merged

        except Exception as e:
            print(f"  ⚠  Factor database error: {e} — output unchanged")
            # Still strip cross-cat columns even if factor DB fails
            cross_cat_cols = [c for c in complete_df.columns if c.startswith('Cross-cat:')]
            return complete_df.drop(columns=cross_cat_cols, errors='ignore')

    def _build_xcat_sheet(self, complete_df, tool):
        """
        Build the Cross-Category Features explanation sheet.
        Shows each injected feature, how it was calculated, its correlation
        with VQ in the training data, and whether it was actually selected
        as the best prediction method for any rows.
        """
        xcat_cols = [c for c in complete_df.columns if c.startswith('Cross-cat:')]
        if not xcat_cols:
            return None

        rows = []
        # Which methods were actually used in predictions?
        used_methods = set()
        if 'Estimation Method' in complete_df.columns:
            used_methods = set(complete_df['Estimation Method'].dropna().unique())

        for col in xcat_cols:
            # Parse the label: "Cross-cat: Electricity (Location Based) | fleet index"
            label = col.replace('Cross-cat: ', '')
            if ' | ' in label:
                category_part, metric_part = label.split(' | ', 1)
            else:
                category_part, metric_part = label, ''

            is_fleet     = 'fleet index' in metric_part
            is_intensity = 'intensity'   in metric_part

            if is_fleet:
                how_calculated = (
                    f"Site's annual {category_part} VQ ÷ fleet mean annual {category_part} VQ. "
                    f"Value of 1.0 = average; >1.0 = above average consumer; <1.0 = below average. "
                    f"Annualised: monthly data scaled to 12 months before computing."
                )
            elif is_intensity:
                how_calculated = (
                    f"Site's annual {category_part} VQ ÷ site's annual Turnover. "
                    f"Dimensionless ratio — comparable across sites regardless of size. "
                    f"Annualised: monthly data scaled to 12 months before computing."
                )
            else:
                how_calculated = f"Derived from {category_part} VQ."

            # Correlation with VQ in known rows
            pearson_r, spearman_r, fill_pct = '', '', ''
            if col in tool.df.columns and tool.vq_col in tool.df.columns:
                known = tool.df[tool.df[tool.vq_col].notna() & tool.df[col].notna()]
                n = len(known)
                fill_pct = f"{n}/{len(tool.df)} rows ({n/len(tool.df)*100:.0f}%)"
                if n >= 4:
                    try:
                        pr = known[col].corr(known[tool.vq_col], method='pearson')
                        sr = known[col].corr(known[tool.vq_col], method='spearman')
                        pearson_r  = f"{pr:.3f}" if pd.notna(pr) else ''
                        spearman_r = f"{sr:.3f}" if pd.notna(sr) else ''
                    except Exception:
                        pass

            # Check if used directly or as part of an Intensity_ or ML method
            intensity_method = f'Intensity_{col}'
            used_directly = (
                intensity_method in used_methods
                or any(col in m for m in used_methods if 'ML' in m or 'GBT' in m or 'RF' in m)
            )
            # Also check stat_methods for the effective R2
            stat_info = tool.stat_methods.get(intensity_method, {})
            eff_r2 = stat_info.get('effective_r2', '')
            if eff_r2 != '':
                eff_r2 = f"{eff_r2:.4f}"

            rows.append({
                'Feature Name':        col,
                'GHG Category':        category_part,
                'Metric Type':         'Fleet Index' if is_fleet else 'Turnover Intensity',
                'How Calculated':      how_calculated,
                'Data Fill Rate':      fill_pct,
                'Pearson r vs VQ':     pearson_r,
                'Spearman r vs VQ':    spearman_r,
                'Effective R² (stat)': eff_r2,
                'Used in Predictions': 'Yes' if used_directly else 'No',
                'Note': (
                    "Fleet index is dimensionless — a ratio relative to portfolio average. "
                    "Intensity is VQ per unit of Turnover. Neither uses raw VQ directly, "
                    "so cross-unit contamination (kWh vs litres) is avoided."
                ),
            })

        return pd.DataFrame(rows)

    def _run(self):
        try:
            input_file  = self.input_file_path.get()
            hist_rec    = self.historic_records_path.get()  or None
            hist_feat   = self.historic_features_path.get() or None
            factor_db   = self.factor_database_path.get()   or None

            tool = ExtrapolationTool(
                input_file=input_file,
                historic_records_file=hist_rec,
                historic_features_file=hist_feat,
                forced_numeric_features=self.selected_numeric_features,
                forced_categorical_features=self.selected_categorical_features,
            )

            complete, avail, stat, ml, cat_bkd, yoy = tool.run()

            # ── Build cross-category features sheet (before stripping cols) ───
            xcat_sheet = self._build_xcat_sheet(complete, tool)

            # ── Apply factor database (also strips cross-cat cols + reorders) ─
            if factor_db:
                complete = self._apply_factor_database(complete, factor_db)
            else:
                # No factor DB — just strip the internal cross-cat columns
                cross_cat_cols = [c for c in complete.columns if c.startswith('Cross-cat:')]
                if cross_cat_cols:
                    complete = complete.drop(columns=cross_cat_cols)

            base = input_file.rsplit('.', 1)[0]
            output_file = base + ' - Extrapolated.xlsx'

            print(f"\n{'='*60}")
            print("SAVING OUTPUT")
            print(f"{'='*60}\n")

            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                sheet_num = 1

                complete.to_excel(writer, sheet_name='Complete Carbon Records', index=False)
                print(f"✓ Sheet {sheet_num}: Complete Carbon Records ({len(complete)} rows)")
                sheet_num += 1

                avail.to_excel(writer, sheet_name='Data Availability', index=False)
                print(f"✓ Sheet {sheet_num}: Data Availability ({len(avail)} rows)")
                sheet_num += 1

                if stat is not None and len(stat):
                    stat.to_excel(writer, sheet_name='Statistical Analysis', index=False)
                    print(f"✓ Sheet {sheet_num}: Statistical Analysis ({len(stat)} methods)")
                    sheet_num += 1

                if ml is not None and len(ml):
                    ml.to_excel(writer, sheet_name='Machine Learning Models', index=False)
                    print(f"✓ Sheet {sheet_num}: Machine Learning Models ({len(ml)} models)")
                    sheet_num += 1
                else:
                    print("  (ML Models sheet skipped — no models trained)")

                if cat_bkd is not None and len(cat_bkd) and 'Message' not in cat_bkd.columns:
                    cat_bkd.to_excel(writer, sheet_name='Categorical Breakdown', index=False)
                    print(f"✓ Sheet {sheet_num}: Categorical Breakdown ({len(cat_bkd)} rows)")
                    sheet_num += 1
                else:
                    print("  (Categorical Breakdown sheet skipped — no categorical features)")

                if yoy is not None and len(yoy):
                    yoy.to_excel(writer, sheet_name='YoY Analysis', index=False)
                    print(f"✓ Sheet {sheet_num}: YoY Analysis ({len(yoy)} rows)")
                    sheet_num += 1
                else:
                    print("  (YoY Analysis sheet skipped — no historic records provided)")

                if xcat_sheet is not None and len(xcat_sheet):
                    xcat_sheet.to_excel(writer, sheet_name='Cross-Category Features', index=False)
                    print(f"✓ Sheet {sheet_num}: Cross-Category Features ({len(xcat_sheet)} features)")
                    sheet_num += 1

            print(f"\n✓ Output saved to:\n  {output_file}\n")
            print('=' * 60)
            print('PROCESSING COMPLETE!')
            print('=' * 60)

            self.root.after(0, lambda: self._set_status('Processing complete!', 'green'))
            self.root.after(0, lambda: messagebox.showinfo(
                'Success', f'Processing complete!\n\nOutput saved to:\n{output_file}'))
            self.root.after(0, lambda: self._ask_open(output_file))

        except Exception as e:
            import traceback
            msg = str(e)
            print(f"\n❌ Error: {msg}\n")
            traceback.print_exc()
            self.root.after(0, lambda: self._set_status(f'Error: {msg}', 'red'))
            self.root.after(0, lambda: messagebox.showerror('Error', msg))

        finally:
            self.root.after(0, self.progress_bar.stop)
            self.root.after(0, lambda: self.process_btn.config(state='normal'))
            self.is_processing = False

    def _ask_open(self, filepath):
        if messagebox.askyesno('Open File', 'Would you like to open the output file?'):
            try:
                os.startfile(filepath)
            except Exception as e:
                messagebox.showerror('Error', f'Could not open file: {e}')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    ExtrapolationGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()