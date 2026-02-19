"""
GUI for Volumetric Extrapolation Tool with Historic Data Support
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox, simpledialog
import threading
import sys
import os
import pandas as pd
import json
from pathlib import Path
import numpy as np

# Import the main extrapolation tool
from Data_Extrapolation import VolumetricExtrapolationTool, CORE_COLUMNS


def analyze_feature_column(dataframe, column_name):
    """Analyze a feature column to provide recommendations"""
    col_data = dataframe[column_name]
    non_null = col_data.dropna()

    if len(non_null) == 0:
        return None

    completeness = len(non_null) / len(dataframe)

    analysis = {
        'name': column_name,
        'completeness': completeness,
        'details': {}
    }

    is_numeric = pd.api.types.is_numeric_dtype(col_data)

    if not is_numeric:
        try:
            numeric_conversion = pd.to_numeric(non_null, errors='coerce')
            successful_conversions = numeric_conversion.notna().sum()
            conversion_rate = successful_conversions / len(non_null) if len(non_null) > 0 else 0

            if conversion_rate > 0.8:
                is_numeric = True
                col_data = pd.to_numeric(col_data, errors='coerce')
                non_null = col_data.dropna()
                print(f"  ✓ Converted '{column_name}' from text to numeric ({conversion_rate * 100:.0f}% successful)")
        except:
            pass

    if is_numeric:
        analysis['type'] = 'Numerical'
        analysis['details'] = {
            'min': float(non_null.min()),
            'max': float(non_null.max()),
            'mean': float(non_null.mean()),
            'std': float(non_null.std())
        }

        if completeness < 0.2:
            analysis['recommendation'] = 'Skip'
            analysis['reason'] = f'Only {completeness * 100:.0f}% complete - too sparse for reliable predictions'
        elif non_null.nunique() == 1:
            analysis['recommendation'] = 'Skip'
            analysis['reason'] = 'All values are the same - no predictive value'
        else:
            analysis['recommendation'] = 'Use'
            analysis['reason'] = f'Good coverage ({completeness * 100:.0f}%) with variance - helpful for predictions'

    else:
        analysis['type'] = 'Categorical'
        unique_count = non_null.nunique()
        value_counts = non_null.value_counts()

        analysis['details'] = {
            'unique_values': unique_count,
            'most_common': value_counts.head(5).to_dict()
        }

        if completeness < 0.2:
            analysis['recommendation'] = 'Skip'
            analysis['reason'] = f'Only {completeness * 100:.0f}% complete - too sparse'
        elif unique_count > 20:
            analysis['recommendation'] = 'Skip'
            analysis['reason'] = f'Too many categories ({unique_count}) - would create too many features'
        elif unique_count == 1:
            analysis['recommendation'] = 'Skip'
            analysis['reason'] = 'Only one category - no predictive value'
        else:
            analysis['recommendation'] = 'Use'
            analysis['reason'] = f'Good coverage ({completeness * 100:.0f}%) with {unique_count} categories'

    return analysis


class ColumnSelectionDialog:
    """Dialog for selecting which extra feature columns to use in ML models with detailed analysis"""

    def __init__(self, parent, columns_analysis):
        self.result = None
        self.columns_analysis = columns_analysis

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("New Columns Detected")
        self.dialog.geometry("800x600")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.dialog.protocol("WM_DELETE_WINDOW", self.skip_all)

        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (800 // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (600 // 2)
        self.dialog.geometry(f"+{x}+{y}")

        title_frame = tk.Frame(self.dialog, bg='white', pady=15, padx=20)
        title_frame.pack(fill="x", side="top")

        title = tk.Label(title_frame, text="🔍 New Columns Detected",
                         font=('Segoe UI', 14, 'bold'), foreground="#2C3E50", bg='white')
        title.pack()

        subtitle = tk.Label(title_frame,
                            text="Select which columns to use in ML predictions:",
                            font=('Segoe UI', 9), foreground="#7F8C8D", bg='white')
        subtitle.pack(pady=(3, 0))

        container = ttk.Frame(self.dialog)
        container.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        canvas = tk.Canvas(container, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind("<Configure>",
                                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=750)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.checkboxes = {}
        for i, analysis in enumerate(columns_analysis):
            self.create_column_widget(self.scrollable_frame, analysis, i)

        button_frame = tk.Frame(self.dialog, bg='#f0f0f0', pady=12, padx=15)
        button_frame.pack(fill="x", side="bottom")

        btn_style = {'font': ('Segoe UI', 9), 'width': 16, 'pady': 8}

        self.use_selected_btn = tk.Button(button_frame,
                                          text=f"Use Selected ({sum(1 for a in columns_analysis if a['recommendation'] == 'Use')})",
                                          command=self.use_selected,
                                          bg='#3498DB', fg='white',
                                          relief=tk.FLAT, **btn_style)
        self.use_selected_btn.pack(side="left", padx=5)

        tk.Button(button_frame, text="Select All Recommended",
                  command=self.select_recommended,
                  bg='#3498DB', fg='white',
                  relief=tk.FLAT, **btn_style).pack(side="left", padx=3)

        tk.Button(button_frame, text="Skip All",
                  command=self.skip_all,
                  bg='#95A5A6', fg='white',
                  relief=tk.FLAT, **btn_style).pack(side="left", padx=3)

        for var in self.checkboxes.values():
            var.trace_add('write', lambda *args: self.update_button_text())

        self.dialog.wait_window()

    def update_button_text(self):
        count = sum(1 for var in self.checkboxes.values() if var.get())
        self.use_selected_btn.config(text=f"Use Selected ({count})")

    def create_column_widget(self, parent, analysis, index):
        outer_frame = ttk.Frame(parent)
        outer_frame.pack(fill="x", padx=5, pady=6)

        frame = tk.Frame(outer_frame, bg='white', relief=tk.SOLID, borderwidth=1)
        frame.pack(fill="x")

        inner_frame = tk.Frame(frame, bg='white', padx=12, pady=10)
        inner_frame.pack(fill="x")

        header_frame = tk.Frame(inner_frame, bg='white')
        header_frame.pack(fill="x", pady=(0, 6))

        var = tk.BooleanVar(value=(analysis['recommendation'] == 'Use'))
        cb = tk.Checkbutton(header_frame, variable=var, bg='white',
                            text=f"  {analysis['name']}",
                            font=('Segoe UI', 10, 'bold'),
                            foreground='#2C3E50')
        cb.pack(side="left")
        self.checkboxes[analysis['name']] = var

        if analysis['recommendation'] == 'Use':
            badge_text = "✓ Recommended"
            badge_bg = "#d4edda"
            badge_fg = "#155724"
        else:
            badge_text = "⚠ Not Recommended"
            badge_bg = "#fff3cd"
            badge_fg = "#856404"

        badge = tk.Label(header_frame, text=badge_text, bg=badge_bg, fg=badge_fg,
                         font=('Segoe UI', 8, 'bold'), padx=8, pady=3)
        badge.pack(side="right")

        type_text = f"{analysis['type']} • {analysis['completeness'] * 100:.0f}% complete"
        type_label = tk.Label(inner_frame, text=type_text, bg='white',
                              font=('Segoe UI', 8), foreground='#7F8C8D')
        type_label.pack(anchor="w", pady=(0, 4))

        reason_color = "#155724" if analysis['recommendation'] == 'Use' else "#856404"
        reason_label = tk.Label(inner_frame, text=f"→ {analysis['reason']}", bg='white',
                                font=('Segoe UI', 8), foreground=reason_color,
                                wraplength=700, justify=tk.LEFT)
        reason_label.pack(anchor="w")

    def use_selected(self):
        selected = {name: var.get() for name, var in self.checkboxes.items()}
        self.result = {'selected_columns': selected}
        self.dialog.destroy()

    def select_recommended(self):
        for analysis in self.columns_analysis:
            if analysis['name'] in self.checkboxes:
                if analysis['recommendation'] == 'Use':
                    self.checkboxes[analysis['name']].set(True)
                else:
                    self.checkboxes[analysis['name']].set(False)

    def skip_all(self):
        self.result = {'selected_columns': {name: False for name in self.checkboxes.keys()}}
        self.dialog.destroy()


class ExtrapolationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Extrapolation")
        self.root.geometry("1200x850")
        self.root.resizable(True, True)

        self.bg_main = "#E8E8E8"
        self.bg_card = "#FFFFFF"
        self.primary = "#2C3E50"
        self.accent = "#3498DB"
        self.success = "#27AE60"
        self.text_dark = "#2C3E50"
        self.text_light = "#7F8C8D"
        self.root.configure(bg=self.bg_main)

        self.input_file_path = tk.StringVar()
        self.factor_db_path = tk.StringVar()
        self.config_file = Path.home() / '.extrapolation_config.json'
        self.load_config()
        self.historic_records_path = tk.StringVar()
        self.historic_features_path = tk.StringVar()
        self.is_processing = False
        self.selected_features = []
        self.loaded_data = None

        self.create_widgets()

        sys.stdout = TextRedirector(self.log_text)

    def load_config(self):
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)

                if 'factor_db_path' in config:
                    saved_path = config['factor_db_path']
                    if saved_path and Path(saved_path).exists():
                        self.factor_db_path.set(saved_path)
                        print(f"✓ Loaded saved Factor Database: {Path(saved_path).name}")
                    else:
                        print("⚠️  Saved Factor Database path no longer valid")
        except Exception as e:
            print(f"⚠️  Could not load config: {e}")

    def save_config(self):
        try:
            config = {
                'factor_db_path': self.factor_db_path.get() if self.factor_db_path.get() else None
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save config: {e}")

    def create_widgets(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('Modern.TButton',
                        background=self.accent, foreground='white',
                        borderwidth=0, focuscolor='none', padding=10,
                        font=('Segoe UI', 10))
        style.map('Modern.TButton', background=[('active', '#2980B9')])

        style.configure('Template.TButton',
                        background='#BDC3C7', foreground='white',
                        borderwidth=0, focuscolor='none', padding=10,
                        font=('Segoe UI', 9))
        style.map('Template.TButton', background=[('active', '#95A5A6')])

        style.configure('Card.TLabelframe', background=self.bg_card, borderwidth=0, relief='flat')
        style.configure('Card.TLabelframe.Label',
                        background=self.bg_card, foreground=self.text_dark,
                        font=('Segoe UI', 10, 'bold'))
        style.configure('TLabel', background=self.bg_card, foreground=self.text_dark, font=('Segoe UI', 9))
        style.configure('Modern.Horizontal.TProgressbar',
                        background=self.accent, troughcolor=self.bg_main, borderwidth=0, thickness=8)

        main_container = ttk.Frame(self.root, style='Modern.TFrame')
        main_container.pack(fill="both", expand=True)
        style.configure('Modern.TFrame', background=self.bg_main)

        canvas = tk.Canvas(main_container, bg=self.bg_main, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style='Modern.TFrame')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def configure_canvas_width(event):
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind('<Configure>', configure_canvas_width)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Title
        title_frame = ttk.Frame(scrollable_frame, style='Modern.TFrame', padding="20 20 20 10")
        title_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        ttk.Label(title_frame, text="Extrapolation",
                  font=('Segoe UI', 28, 'bold'), foreground=self.primary,
                  background=self.bg_main).pack(anchor="w")
        ttk.Label(title_frame,
                  text="Extrapolate blank volumetric quantities using Machine Learning and historic data patterns",
                  font=('Segoe UI', 11), foreground=self.text_light,
                  background=self.bg_main).pack(anchor="w", pady=(5, 0))

        # File Selection
        file_frame = ttk.LabelFrame(scrollable_frame, style='Card.TLabelframe', text="File Selection", padding="20")
        file_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=0, pady=(0, 1))

        ttk.Label(file_frame, text="Input File:", font=('Segoe UI', 9, 'bold')).grid(
            row=0, column=0, sticky=tk.W, pady=(10, 5), padx=(0, 10))
        ttk.Entry(file_frame, textvariable=self.input_file_path, width=55, font=('Segoe UI', 9)).grid(
            row=0, column=1, padx=5, pady=(10, 5))
        ttk.Button(file_frame, text="Browse", command=self.browse_input_file,
                   style='Modern.TButton', width=12).grid(row=0, column=2, pady=(10, 5), padx=(5, 5))
        ttk.Button(file_frame, text="Export Template", command=self.export_input_template,
                   style='Template.TButton', width=15).grid(row=0, column=3, pady=(10, 5), padx=(0, 0))

        ttk.Label(file_frame, text="Historic Records (Optional):", font=('Segoe UI', 9)).grid(
            row=1, column=0, sticky=tk.W, pady=5, padx=(0, 10))
        ttk.Entry(file_frame, textvariable=self.historic_records_path, width=55, font=('Segoe UI', 9)).grid(
            row=1, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_historic_records,
                   style='Modern.TButton', width=12).grid(row=1, column=2, pady=5, padx=(5, 5))
        ttk.Button(file_frame, text="Export Template", command=self.export_historic_records_template,
                   style='Template.TButton', width=15).grid(row=1, column=3, pady=5, padx=(0, 0))

        ttk.Label(file_frame, text="Historic Features (Optional):", font=('Segoe UI', 9)).grid(
            row=2, column=0, sticky=tk.W, pady=5, padx=(0, 10))
        ttk.Entry(file_frame, textvariable=self.historic_features_path, width=55, font=('Segoe UI', 9)).grid(
            row=2, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_historic_features,
                   style='Modern.TButton', width=12).grid(row=2, column=2, pady=5, padx=(5, 5))
        ttk.Button(file_frame, text="Export Template", command=self.export_historic_features_template,
                   style='Template.TButton', width=15).grid(row=2, column=3, pady=5, padx=(0, 0))

        ttk.Label(file_frame, text="Factor Database:", font=('Segoe UI', 9, 'bold')).grid(
            row=3, column=0, sticky=tk.W, pady=(5, 10), padx=(0, 10))
        ttk.Entry(file_frame, textvariable=self.factor_db_path, width=55, font=('Segoe UI', 9)).grid(
            row=3, column=1, padx=5, pady=(5, 10))
        ttk.Button(file_frame, text="Browse", command=self.browse_factor_db,
                   style='Modern.TButton', width=12).grid(row=3, column=2, pady=(5, 10), padx=(5, 5))

        # About
        about_frame = ttk.LabelFrame(scrollable_frame, style='Card.TLabelframe', text="About This Tool", padding="20")
        about_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=0, pady=1)

        about_text = """Volumetric Quantity Extrapolation

This tool compares data-dependent prediction models, filling in blank Volumetric Quantities using the most suited Rule-Based or Machine Learning Method for each row-specific scenario.

Required Input Columns:
    • Site identifier (or Location if no site ID)
    • Date from and Date to (DD/MM/YYYY format)
    • GHG Category
    • Volumetric Quantity (with blank values to extrapolate)
    • Data Timeframe (Annual or Monthly)

Optional - To Improve Predictions:
    • Turnover, Brand, or any other numerical/categorical features
    • Historic Records: Previous years' volumetric quantities (same format as output)
    • Historic Features: Previous years' turnover, sq ft, or other features by site"""

        ttk.Label(about_frame, text=about_text, justify=tk.LEFT,
                  font=('Segoe UI', 9), foreground=self.text_dark).pack(anchor="w")

        # Progress
        progress_frame = ttk.Frame(scrollable_frame, style='Modern.TFrame', padding="20 15")
        progress_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))

        self.progress_bar = ttk.Progressbar(
            progress_frame, mode='indeterminate', length=1140,
            style='Modern.Horizontal.TProgressbar'
        )
        self.progress_bar.pack(fill="x")

        self.status_label = ttk.Label(
            progress_frame, text="Ready to process", foreground=self.success,
            background=self.bg_main, font=('Segoe UI', 11, 'bold')
        )
        self.status_label.pack(pady=8)

        # Buttons
        button_frame = ttk.Frame(scrollable_frame, style='Modern.TFrame', padding="0 10 20 15")
        button_frame.grid(row=4, column=0, sticky=(tk.W, tk.E))

        self.process_button = ttk.Button(
            button_frame, text="Process Data", command=self.process_data,
            style='Modern.TButton', width=18
        )
        self.process_button.pack(side="left", padx=(20, 5))

        ttk.Button(button_frame, text="Clear Log", command=self.clear_log,
                   style='Modern.TButton', width=18).pack(side="left", padx=5)

        self.add_features_button = ttk.Button(
            button_frame, text="Add Features", command=self.add_features,
            style='Modern.TButton', state='normal', width=18
        )
        self.add_features_button.pack(side="left", padx=5)

        # Log
        log_frame = ttk.LabelFrame(scrollable_frame, style='Card.TLabelframe', text="Processing Log", padding="20")
        log_frame.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=0, pady=(1, 0))

        self.log_text = scrolledtext.ScrolledText(
            log_frame, width=140, height=18, wrap=tk.WORD,
            font=('Consolas', 9), bg='white', relief=tk.FLAT, borderwidth=0
        )
        self.log_text.pack(fill="both", expand=True)

        scrollable_frame.columnconfigure(0, weight=1)
        scrollable_frame.rowconfigure(5, weight=1)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        self.save_config()
        self.root.destroy()

    def export_input_template(self):
        filename = filedialog.asksaveasfilename(
            title="Save Input File Template", defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="Input_Template.xlsx"
        )
        if filename:
            try:
                template_df = pd.DataFrame(columns=[
                    'Brand', 'Site identifier', 'Location', 'Data timeframe',
                    'Date from', 'Date to', 'GHG Category', 'Volumetric Quantity', 'Turnover'
                ])
                template_df.loc[0] = [
                    'Example Brand', 'SITE001', 'London, UK', 'Monthly',
                    '01/01/2025', '31/01/2025', 'Category 3', '', '50000'
                ]
                if filename.endswith('.csv'):
                    template_df.to_csv(filename, index=False)
                else:
                    template_df.to_excel(filename, index=False)

                self.input_file_path.set(filename)
                print(f"\n✓ Input template exported: {filename}")
                print("  Fill in the template and click 'Process Data' when ready!")

                if messagebox.askyesno("Template Exported", f"Template saved to:\n{filename}\n\nOpen it now?"):
                    os.startfile(filename)
            except Exception as e:
                messagebox.showerror("Error", f"Could not export template: {e}")

    def export_historic_records_template(self):
        if self.factor_db_path.get() and Path(self.factor_db_path.get()).exists():
            try:
                factor_df = pd.read_csv(self.factor_db_path.get())
                template_df = pd.DataFrame(columns=factor_df.columns)
                example_row = {}
                for col in factor_df.columns:
                    if col == 'Site identifier':
                        example_row[col] = 'SITE001'
                    elif col == 'Date from':
                        example_row[col] = '01/01/2024'
                    elif col == 'Date to':
                        example_row[col] = '31/12/2024'
                    elif col == 'Volumetric Quantity':
                        example_row[col] = '120000'
                    else:
                        example_row[col] = ''
                template_df = pd.concat([template_df, pd.DataFrame([example_row])], ignore_index=True)
            except Exception as e:
                print(f"⚠️  Could not read Factor Database: {e}")
                template_df = self._get_default_historic_records_template()
        else:
            template_df = self._get_default_historic_records_template()

        filename = filedialog.asksaveasfilename(
            title="Save Historic Records Template", defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="Historic_Records_Template.xlsx"
        )
        if filename:
            try:
                if filename.endswith('.csv'):
                    template_df.to_csv(filename, index=False)
                else:
                    template_df.to_excel(filename, index=False)
                self.historic_records_path.set(filename)
                print(f"\n✓ Historic Records template exported: {filename}")
                if messagebox.askyesno("Template Exported", f"Template saved to:\n{filename}\n\nOpen it now?"):
                    os.startfile(filename)
            except Exception as e:
                messagebox.showerror("Error", f"Could not export template: {e}")

    def export_historic_features_template(self):
        if self.factor_db_path.get() and Path(self.factor_db_path.get()).exists():
            try:
                factor_df = pd.read_csv(self.factor_db_path.get())
                columns = [col for col in factor_df.columns if col != 'Volumetric Quantity']
                template_df = pd.DataFrame(columns=columns)
                example_row = {}
                for col in columns:
                    if col == 'Site identifier':
                        example_row[col] = 'SITE001'
                    elif col == 'Date from':
                        example_row[col] = '01/01/2024'
                    elif col == 'Date to':
                        example_row[col] = '31/12/2024'
                    elif col == 'Turnover':
                        example_row[col] = '500000'
                    else:
                        example_row[col] = ''
                template_df = pd.concat([template_df, pd.DataFrame([example_row])], ignore_index=True)
            except Exception as e:
                print(f"⚠️  Could not read Factor Database: {e}")
                template_df = self._get_default_historic_features_template()
        else:
            template_df = self._get_default_historic_features_template()

        filename = filedialog.asksaveasfilename(
            title="Save Historic Features Template", defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="Historic_Features_Template.xlsx"
        )
        if filename:
            try:
                if filename.endswith('.csv'):
                    template_df.to_csv(filename, index=False)
                else:
                    template_df.to_excel(filename, index=False)
                self.historic_features_path.set(filename)
                print(f"\n✓ Historic Features template exported: {filename}")
                if messagebox.askyesno("Template Exported", f"Template saved to:\n{filename}\n\nOpen it now?"):
                    os.startfile(filename)
            except Exception as e:
                messagebox.showerror("Error", f"Could not export template: {e}")

    def _get_default_historic_records_template(self):
        template_df = pd.DataFrame(columns=[
            'Client', 'Site identifier', 'Location', 'Date from', 'Date to',
            'GHG Category', 'Volumetric Quantity', 'Timeframe', 'Year'
        ])
        template_df.loc[0] = [
            'Example Client', 'SITE001', 'London, UK', '01/01/2024', '31/12/2024',
            'Category 3', '120000', 'Annual', '2024'
        ]
        return template_df

    def _get_default_historic_features_template(self):
        template_df = pd.DataFrame(columns=[
            'Client', 'Site identifier', 'Location', 'Date from', 'Date to',
            'Turnover', 'SqFt', 'Year'
        ])
        template_df.loc[0] = [
            'Example Client', 'SITE001', 'London, UK', '01/01/2024', '31/12/2024',
            '500000', '5000', '2024'
        ]
        return template_df

    def browse_input_file(self):
        filename = filedialog.askopenfilename(
            title="Select Input Data File",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if filename:
            self.input_file_path.set(filename)
            try:
                if filename.endswith('.csv'):
                    self.loaded_data = pd.read_csv(filename)
                else:
                    self.loaded_data = pd.read_excel(filename)
                print(f"\n✓ Loaded {len(self.loaded_data)} rows with {len(self.loaded_data.columns)} columns")
                print("  Click 'Add Features' to select additional features for ML models")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load file: {e}")
                self.loaded_data = None

    def browse_historic_records(self):
        filename = filedialog.askopenfilename(
            title="Select Historic Records File (Previous Years VQ Data)",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if filename:
            self.historic_records_path.set(filename)
            print(f"\n✓ Historic records file selected: {os.path.basename(filename)}")

    def browse_historic_features(self):
        filename = filedialog.askopenfilename(
            title="Select Historic Features File (Previous Years Turnover, SqFt, etc.)",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if filename:
            self.historic_features_path.set(filename)
            print(f"\n✓ Historic features file selected: {os.path.basename(filename)}")

    def browse_factor_db(self):
        filename = filedialog.askopenfilename(
            title="Select Factor Database",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.factor_db_path.set(filename)
            self.save_config()
            print(f"\n✓ Factor Database selected and saved: {Path(filename).name}")

    def add_features(self):
        if self.loaded_data is None:
            if self.input_file_path.get():
                try:
                    input_file = self.input_file_path.get()
                    if input_file.endswith('.csv'):
                        self.loaded_data = pd.read_csv(input_file)
                    else:
                        self.loaded_data = pd.read_excel(input_file)
                    print(f"\n✓ Loaded {len(self.loaded_data)} rows with {len(self.loaded_data.columns)} columns")
                except Exception as e:
                    messagebox.showerror("Error", f"Could not load file: {e}")
                    return
            else:
                messagebox.showwarning("No Data Loaded", "Please load an input file first using the Browse button")
                return

        available_features = []
        for col in self.loaded_data.columns:
            if col not in CORE_COLUMNS:
                non_null_count = self.loaded_data[col].notna().sum()
                if non_null_count > 0:
                    available_features.append(col)

        print(f"\n📊 Detected {len(available_features)} additional columns:")

        if not available_features:
            messagebox.showinfo("No Additional Features", "No additional features detected in the data.")
            return

        columns_analysis = []
        for col in available_features:
            try:
                analysis = analyze_feature_column(self.loaded_data, col)
                if analysis:
                    columns_analysis.append(analysis)
                    rec_symbol = "✓" if analysis['recommendation'] == 'Use' else "⚠"
                    print(f"  {rec_symbol} {col} ({analysis['type']}, {analysis['completeness'] * 100:.0f}% complete)")
            except Exception as e:
                print(f"  ❌ Error analyzing '{col}': {e}")

        if not columns_analysis:
            messagebox.showinfo("No Features to Add", "No valid features found to add.")
            return

        dialog = ColumnSelectionDialog(self.root, columns_analysis)

        if dialog.result:
            selected_columns = [col for col, is_selected in dialog.result['selected_columns'].items() if is_selected]
            self.selected_features = selected_columns

            if self.selected_features:
                print(f"\n✅ Selected {len(self.selected_features)} features for ML models:")
                for feat in self.selected_features:
                    print(f"   • {feat}")
                print("\n💡 These features will be used when you click 'Process Data'\n")
            else:
                self.selected_features = []
                print("\n⚠️ No features selected\n")
        else:
            print("\n❌ Feature selection cancelled\n")

    def clear_log(self):
        self.log_text.delete(1.0, tk.END)

    def update_status(self, message, color="black"):
        if color == "green":
            color = self.success
        elif color == "blue":
            color = self.accent
        elif color == "red":
            color = "#E74C3C"
        self.status_label.config(text=message, foreground=color)

    def process_data(self):
        if self.is_processing:
            messagebox.showwarning("Processing", "Already processing data. Please wait.")
            return

        if not self.input_file_path.get():
            messagebox.showerror("Error", "Please select an input data file")
            return

        self.is_processing = True
        self.process_button.config(state='disabled')
        self.progress_bar.start()
        self.update_status("Processing...", "blue")

        thread = threading.Thread(target=self.run_extrapolation, daemon=True)
        thread.start()

    def run_extrapolation(self):
        try:
            input_file = self.input_file_path.get()
            factor_db = self.factor_db_path.get() if self.factor_db_path.get() else None
            historic_records = self.historic_records_path.get() if self.historic_records_path.get() else None
            historic_features = self.historic_features_path.get() if self.historic_features_path.get() else None

            print("\n" + "=" * 70)
            print("STARTING DATA EXTRAPOLATION WITH HISTORIC DATA")
            print("=" * 70 + "\n")

            final_features = self.selected_features.copy()
            if 'Brand' in self.selected_features and self.loaded_data is not None:
                if 'Brand' in self.loaded_data.columns:
                    brand_counts = self.loaded_data.groupby('Brand').size()
                    insufficient_brands = brand_counts[brand_counts < 20]

                    if len(insufficient_brands) > 0:
                        message = f"⚠️ Brand Data Sufficiency Warning\n\n"
                        message += f"You selected 'Brand' as a feature, but some brands have insufficient data:\n\n"
                        for brand, count in insufficient_brands.items():
                            message += f"  • {brand}: {count} rows (need 20+ for reliable learning)\n"
                        message += f"\nWith insufficient data per brand, the model may:\n"
                        message += f"  ❌ Learn poor brand-specific patterns\n"
                        message += f"  ❌ Overfit to limited examples\n\n"
                        message += f"Recommendation: Remove Brand as a feature for this dataset.\n\n"
                        message += f"Would you like to REMOVE Brand and continue?\n"
                        message += f"(Models will learn general patterns instead)"

                        user_choice = {'response': None}

                        def ask_user():
                            user_choice['response'] = messagebox.askyesno(
                                "Brand Data Warning", message, icon='warning'
                            )

                        self.root.after(0, ask_user)

                        import time
                        while user_choice['response'] is None:
                            time.sleep(0.1)

                        if user_choice['response']:
                            final_features = [f for f in final_features if f != 'Brand']
                            print("\n⚠️  User chose to REMOVE Brand feature due to insufficient data")
                            print(f"    Continuing with {len(final_features)} features\n")
                        else:
                            print("\n⚠️  User chose to KEEP Brand feature despite insufficient data")
                            print("    Proceeding with caution - results may vary by brand\n")

            tool = VolumetricExtrapolationTool(
                extrapolate_file=input_file,
                factor_database_file=factor_db,
                selected_features=final_features,
                historic_records_file=historic_records,
                historic_features_file=historic_features
            )

            # FIX: Unpack 8 values (rule_based_sheet added)
            output_df, results, model_testing_matrix, model_performance, data_availability, historic_analysis, training_data, rule_based_sheet = tool.run_extrapolation()

            if input_file.endswith('.csv'):
                output_file = input_file.replace(".csv", " - With extrapolated.xlsx")
            else:
                output_file = input_file.rsplit('.', 1)[0] + " - With extrapolated.xlsx"

            print(f"\n{'=' * 70}")
            print("SAVING RESULTS")
            print(f"{'=' * 70}\n")

            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Sheet 1: Complete Records
                if factor_db:
                    try:
                        factor_df = pd.read_csv(factor_db)
                        factor_columns = factor_df.columns.tolist()

                        complete_df = pd.DataFrame()
                        for col in factor_columns:
                            if col in output_df.columns:
                                complete_df[col] = output_df[col]
                            else:
                                complete_df[col] = None

                        if 'Data Quality Score' not in factor_columns and 'Data Quality Score' in output_df.columns:
                            complete_df['Data Quality Score'] = output_df['Data Quality Score']

                        complete_df.to_excel(writer, sheet_name='Complete Records', index=False)
                        print(f"✓ Sheet 1: Complete Records ({len(complete_df.columns)} columns)")
                    except:
                        output_df.to_excel(writer, sheet_name='Complete Records', index=False)
                        print(f"✓ Sheet 1: Complete Records ({len(output_df.columns)} columns)")
                else:
                    output_df.to_excel(writer, sheet_name='Complete Records', index=False)
                    print(f"✓ Sheet 1: Complete Records ({len(output_df.columns)} columns)")

                # Sheet 2: Estimation Summary
                estimated_rows = output_df[output_df['Data integrity'] == 'Estimated'].copy()
                if not estimated_rows.empty:
                    summary_cols = ['Site identifier', 'GHG Category', 'Timeframe',
                                    'Date from', 'Date to', 'Volumetric Quantity',
                                    'Estimation Method', 'Data Quality', 'Data Quality Score']
                    summary_cols = [col for col in summary_cols if col in estimated_rows.columns]

                    if 'Brand' in estimated_rows.columns:
                        summary_cols.insert(1, 'Brand')
                    if 'Turnover' in estimated_rows.columns:
                        summary_cols.insert(-4, 'Turnover')
                    if 'Data_Months' in estimated_rows.columns:
                        summary_cols.append('Data_Months')
                    if 'Data_Has_Turnover' in estimated_rows.columns:
                        summary_cols.append('Data_Has_Turnover')
                    if 'Data_Has_Historic' in estimated_rows.columns:
                        summary_cols.append('Data_Has_Historic')
                    if 'Data_Availability' in estimated_rows.columns:
                        summary_cols.append('Data_Availability')

                    summary_df = estimated_rows[summary_cols].copy()
                    summary_df.to_excel(writer, sheet_name='Estimation Summary', index=False)
                    print(f"✓ Sheet 2: Estimation Summary ({len(summary_df)} predictions)")
                else:
                    pd.DataFrame({'Message': ['No predictions were made']}).to_excel(
                        writer, sheet_name='Estimation Summary', index=False)
                    print("✓ Sheet 2: Estimation Summary (no predictions)")

                # Sheet 3: Model Testing Matrix
                if model_testing_matrix is not None and len(model_testing_matrix) > 0:
                    model_testing_matrix.to_excel(writer, sheet_name='Model Testing Matrix', index=False)
                    print(f"✓ Sheet 3: Model Testing Matrix ({len(model_testing_matrix)} contexts)")
                else:
                    pd.DataFrame({'Message': ['No model testing data available']}).to_excel(
                        writer, sheet_name='Model Testing Matrix', index=False)
                    print("✓ Sheet 3: Model Testing Matrix (no data)")

                # Sheet 4: Model Performance
                if model_performance is not None and len(model_performance) > 0:
                    model_performance.to_excel(writer, sheet_name='Model Performance', index=False)
                    print(f"✓ Sheet 4: Model Performance ({len(model_performance)} models)")
                else:
                    pd.DataFrame({'Message': ['No model performance data available']}).to_excel(
                        writer, sheet_name='Model Performance', index=False)
                    print("✓ Sheet 4: Model Performance (no data)")

                # Sheet 5: Data Availability
                if data_availability is not None and len(data_availability) > 0:
                    data_availability.to_excel(writer, sheet_name='Data Availability', index=False)
                    print(f"✓ Sheet 5: Data Availability ({len(data_availability)} contexts)")
                else:
                    pd.DataFrame({'Message': ['No data availability summary available']}).to_excel(
                        writer, sheet_name='Data Availability', index=False)
                    print("✓ Sheet 5: Data Availability (no data)")

                # Sheet 6: Historic Data Analysis
                if historic_analysis is not None and len(historic_analysis) > 0:
                    historic_analysis.to_excel(writer, sheet_name='Historic Data Analysis', index=False)
                    print(f"✓ Sheet 6: Historic Data Analysis ({len(historic_analysis)} records)")
                else:
                    pd.DataFrame({'Message': ['No historic data was provided']}).to_excel(
                        writer, sheet_name='Historic Data Analysis', index=False)
                    print("✓ Sheet 6: Historic Data Analysis (no historic data)")

                # Sheet 7: Training Data
                if training_data is not None and len(training_data) > 0:
                    training_data.to_excel(writer, sheet_name='Training Data', index=False)
                    print(f"✓ Sheet 7: Training Data ({len(training_data)} records)")
                else:
                    pd.DataFrame({'Message': ['No training data available']}).to_excel(
                        writer, sheet_name='Training Data', index=False)
                    print("✓ Sheet 7: Training Data (no data)")

                # Sheet 8: Rule Based Methods
                if rule_based_sheet is not None and len(rule_based_sheet) > 0:
                    rule_based_sheet.to_excel(writer, sheet_name='Rule_Based_Methods', index=False)
                    print(f"✓ Sheet 8: Rule Based Methods ({len(rule_based_sheet)} methods)")
                else:
                    pd.DataFrame({'Message': ['No rule-based method data available']}).to_excel(
                        writer, sheet_name='Rule_Based_Methods', index=False)
                    print("✓ Sheet 8: Rule Based Methods (no data)")

            print(f"\n✓ Multi-sheet Excel saved to:\n  {output_file}\n")
            print("=" * 70)
            print("PROCESSING COMPLETE!")
            print("=" * 70)

            self.root.after(0, lambda: self.update_status("Processing complete!", "green"))
            self.root.after(0, lambda: messagebox.showinfo(
                "Success", f"Processing complete!\n\nOutput saved to:\n{output_file}"
            ))
            self.root.after(0, lambda: self.ask_open_file(output_file))

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n❌ {error_msg}\n")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.update_status(error_msg, "red"))
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

        finally:
            self.root.after(0, self.progress_bar.stop)
            self.root.after(0, lambda: self.process_button.config(state='normal'))
            self.is_processing = False

    def ask_open_file(self, filepath):
        if messagebox.askyesno("Open File", "Would you like to open the output file?"):
            try:
                os.startfile(filepath)
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file: {e}")


class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()

    def flush(self):
        pass


def main():
    root = tk.Tk()
    app = ExtrapolationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()