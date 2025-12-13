"""
Localization Results Viewer and Analyzer
=========================================

Comprehensive spreadsheet-based viewer for thunderSTORM localization results
with advanced filtering, plotting, and analysis tools.

Author: George K (with Claude)
Date: 2025-12-13
"""

import numpy as np
import pandas as pd
from qtpy import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from pathlib import Path
import json

try:
    from flika.window import Window
    from flika import global_vars as g
    FLIKA_AVAILABLE = True
except ImportError:
    FLIKA_AVAILABLE = False


class LocalizationTableModel(QtCore.QAbstractTableModel):
    """Custom table model for localization data with pandas backend."""
    
    def __init__(self, data=None):
        super().__init__()
        if data is None:
            self._data = pd.DataFrame()
        elif isinstance(data, dict):
            self._data = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            self._data = data
        else:
            raise ValueError("Data must be dict or DataFrame")
    
    def rowCount(self, parent=None):
        return len(self._data)
    
    def columnCount(self, parent=None):
        return len(self._data.columns)
    
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        
        if role == QtCore.Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            if isinstance(value, (int, np.integer)):
                return str(value)
            elif isinstance(value, (float, np.floating)):
                return f"{value:.4f}"
            return str(value)
        
        elif role == QtCore.Qt.TextAlignmentRole:
            return QtCore.Qt.AlignCenter
        
        return None
    
    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return str(self._data.columns[section])
            else:
                return str(section + 1)
        return None
    
    def get_dataframe(self):
        """Get the underlying DataFrame."""
        return self._data
    
    def update_data(self, new_data):
        """Update the model with new data."""
        self.beginResetModel()
        if isinstance(new_data, dict):
            self._data = pd.DataFrame(new_data)
        elif isinstance(new_data, pd.DataFrame):
            self._data = new_data
        self.endResetModel()
    
    def sort(self, column, order):
        """Sort table by column."""
        self.layoutAboutToBeChanged.emit()
        col_name = self._data.columns[column]
        ascending = (order == QtCore.Qt.AscendingOrder)
        self._data = self._data.sort_values(by=col_name, ascending=ascending)
        self._data = self._data.reset_index(drop=True)
        self.layoutChanged.emit()


class FilterWidget(QtWidgets.QWidget):
    """Widget for creating and managing filters."""
    
    filterChanged = QtCore.Signal()
    
    def __init__(self, columns=None, parent=None):
        super().__init__(parent)
        self.columns = columns or []
        self.filters = []  # List of filter dictionaries
        self.init_ui()
    
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # Filter list
        self.filter_list = QtWidgets.QListWidget()
        self.filter_list.setMaximumHeight(150)
        layout.addWidget(QtWidgets.QLabel("Active Filters:"))
        layout.addWidget(self.filter_list)
        
        # Add filter section
        filter_group = QtWidgets.QGroupBox("Add Filter")
        filter_layout = QtWidgets.QGridLayout()
        
        self.column_combo = QtWidgets.QComboBox()
        self.operator_combo = QtWidgets.QComboBox()
        self.operator_combo.addItems(['<', '<=', '==', '>=', '>', '!='])
        self.value_edit = QtWidgets.QLineEdit()
        
        filter_layout.addWidget(QtWidgets.QLabel("Column:"), 0, 0)
        filter_layout.addWidget(self.column_combo, 0, 1)
        filter_layout.addWidget(QtWidgets.QLabel("Operator:"), 1, 0)
        filter_layout.addWidget(self.operator_combo, 1, 1)
        filter_layout.addWidget(QtWidgets.QLabel("Value:"), 2, 0)
        filter_layout.addWidget(self.value_edit, 2, 1)
        
        add_btn = QtWidgets.QPushButton("Add Filter")
        add_btn.clicked.connect(self.add_filter)
        filter_layout.addWidget(add_btn, 3, 0, 1, 2)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # Expression filter
        expr_group = QtWidgets.QGroupBox("Expression Filter")
        expr_layout = QtWidgets.QVBoxLayout()
        
        self.expr_edit = QtWidgets.QLineEdit()
        self.expr_edit.setPlaceholderText("e.g., (intensity > 500) & (uncertainty < 50)")
        expr_layout.addWidget(self.expr_edit)
        
        expr_btn = QtWidgets.QPushButton("Add Expression Filter")
        expr_btn.clicked.connect(self.add_expression_filter)
        expr_layout.addWidget(expr_btn)
        
        expr_group.setLayout(expr_layout)
        layout.addWidget(expr_group)
        
        # Control buttons
        btn_layout = QtWidgets.QHBoxLayout()
        
        remove_btn = QtWidgets.QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_filter)
        btn_layout.addWidget(remove_btn)
        
        clear_btn = QtWidgets.QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_filters)
        btn_layout.addWidget(clear_btn)
        
        layout.addLayout(btn_layout)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def update_columns(self, columns):
        """Update available columns for filtering."""
        self.columns = columns
        self.column_combo.clear()
        self.column_combo.addItems(columns)
    
    def add_filter(self):
        """Add a simple comparison filter."""
        column = self.column_combo.currentText()
        operator = self.operator_combo.currentText()
        value_text = self.value_edit.text()
        
        if not column or not value_text:
            return
        
        try:
            # Try to parse as number
            value = float(value_text)
        except ValueError:
            value = value_text
        
        filter_dict = {
            'type': 'comparison',
            'column': column,
            'operator': operator,
            'value': value
        }
        
        self.filters.append(filter_dict)
        self.filter_list.addItem(f"{column} {operator} {value}")
        self.value_edit.clear()
        self.filterChanged.emit()
    
    def add_expression_filter(self):
        """Add an expression-based filter."""
        expression = self.expr_edit.text()
        
        if not expression:
            return
        
        filter_dict = {
            'type': 'expression',
            'expression': expression
        }
        
        self.filters.append(filter_dict)
        self.filter_list.addItem(f"Expression: {expression}")
        self.expr_edit.clear()
        self.filterChanged.emit()
    
    def remove_filter(self):
        """Remove selected filter."""
        current_row = self.filter_list.currentRow()
        if current_row >= 0:
            self.filters.pop(current_row)
            self.filter_list.takeItem(current_row)
            self.filterChanged.emit()
    
    def clear_filters(self):
        """Clear all filters."""
        self.filters.clear()
        self.filter_list.clear()
        self.filterChanged.emit()
    
    def apply_filters(self, df):
        """Apply all filters to a DataFrame."""
        if df.empty or not self.filters:
            return df
        
        filtered_df = df.copy()
        
        for filter_dict in self.filters:
            if filter_dict['type'] == 'comparison':
                column = filter_dict['column']
                operator = filter_dict['operator']
                value = filter_dict['value']
                
                if column not in filtered_df.columns:
                    continue
                
                if operator == '<':
                    filtered_df = filtered_df[filtered_df[column] < value]
                elif operator == '<=':
                    filtered_df = filtered_df[filtered_df[column] <= value]
                elif operator == '==':
                    filtered_df = filtered_df[filtered_df[column] == value]
                elif operator == '>=':
                    filtered_df = filtered_df[filtered_df[column] >= value]
                elif operator == '>':
                    filtered_df = filtered_df[filtered_df[column] > value]
                elif operator == '!=':
                    filtered_df = filtered_df[filtered_df[column] != value]
            
            elif filter_dict['type'] == 'expression':
                try:
                    filtered_df = filtered_df.query(filter_dict['expression'])
                except Exception as e:
                    print(f"Error applying expression filter: {e}")
        
        return filtered_df


class PlotWidget(QtWidgets.QWidget):
    """Widget for creating various plots from localization data."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.init_ui()
    
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # Plot type selection
        type_layout = QtWidgets.QHBoxLayout()
        type_layout.addWidget(QtWidgets.QLabel("Plot Type:"))
        
        self.plot_type = QtWidgets.QComboBox()
        self.plot_type.addItems([
            'Histogram',
            'Scatter Plot',
            '2D Histogram',
            'Localizations per Frame',
            'Cumulative Localizations',
            'Intensity vs Uncertainty'
        ])
        self.plot_type.currentTextChanged.connect(self.update_plot_options)
        type_layout.addWidget(self.plot_type)
        type_layout.addStretch()
        
        layout.addLayout(type_layout)
        
        # Plot options (dynamic based on plot type)
        self.options_group = QtWidgets.QGroupBox("Plot Options")
        self.options_layout = QtWidgets.QFormLayout()
        self.options_group.setLayout(self.options_layout)
        layout.addWidget(self.options_group)
        
        # Column selectors (will be populated dynamically)
        self.x_column = QtWidgets.QComboBox()
        self.y_column = QtWidgets.QComboBox()
        self.bins_spin = QtWidgets.QSpinBox()
        self.bins_spin.setRange(10, 200)
        self.bins_spin.setValue(50)
        
        # Create plot button
        self.plot_btn = QtWidgets.QPushButton("Generate Plot")
        self.plot_btn.clicked.connect(self.create_plot)
        layout.addWidget(self.plot_btn)
        
        layout.addStretch()
        self.setLayout(layout)
        
        self.update_plot_options()
    
    def update_plot_options(self):
        """Update plot options based on selected plot type."""
        # Clear existing options
        while self.options_layout.rowCount() > 0:
            self.options_layout.removeRow(0)
        
        plot_type = self.plot_type.currentText()
        
        if plot_type == 'Histogram':
            self.options_layout.addRow("Column:", self.x_column)
            self.options_layout.addRow("Bins:", self.bins_spin)
        
        elif plot_type == 'Scatter Plot':
            self.options_layout.addRow("X Column:", self.x_column)
            self.options_layout.addRow("Y Column:", self.y_column)
        
        elif plot_type == '2D Histogram':
            self.options_layout.addRow("X Column:", self.x_column)
            self.options_layout.addRow("Y Column:", self.y_column)
            self.options_layout.addRow("Bins:", self.bins_spin)
        
        elif plot_type in ['Localizations per Frame', 'Cumulative Localizations']:
            # No additional options needed
            pass
        
        elif plot_type == 'Intensity vs Uncertainty':
            # Fixed columns, no options
            pass
    
    def set_data(self, data):
        """Set data and update column lists."""
        if isinstance(data, dict):
            self.data = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            self.data = data
        else:
            return
        
        # Update column selectors
        columns = list(self.data.columns)
        self.x_column.clear()
        self.y_column.clear()
        self.x_column.addItems(columns)
        self.y_column.addItems(columns)
        
        # Set default selections
        if 'x' in columns:
            self.x_column.setCurrentText('x')
        if 'y' in columns:
            self.y_column.setCurrentText('y')
    
    def create_plot(self):
        """Create and display the selected plot."""
        if self.data is None or self.data.empty:
            QtWidgets.QMessageBox.warning(self, "No Data", "No data available for plotting")
            return
        
        plot_type = self.plot_type.currentText()
        
        try:
            if plot_type == 'Histogram':
                self.plot_histogram()
            elif plot_type == 'Scatter Plot':
                self.plot_scatter()
            elif plot_type == '2D Histogram':
                self.plot_2d_histogram()
            elif plot_type == 'Localizations per Frame':
                self.plot_localizations_per_frame()
            elif plot_type == 'Cumulative Localizations':
                self.plot_cumulative_localizations()
            elif plot_type == 'Intensity vs Uncertainty':
                self.plot_intensity_vs_uncertainty()
        
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Plot Error", f"Error creating plot:\n{str(e)}")
    
    def plot_histogram(self):
        """Create histogram plot."""
        column = self.x_column.currentText()
        bins = self.bins_spin.value()
        
        if column not in self.data.columns:
            return
        
        data = self.data[column].dropna()
        
        win = pg.plot(title=f"Histogram: {column}")
        y, x = np.histogram(data, bins=bins)
        win.plot(x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))
        win.setLabel('bottom', column)
        win.setLabel('left', 'Count')
    
    def plot_scatter(self):
        """Create scatter plot."""
        x_col = self.x_column.currentText()
        y_col = self.y_column.currentText()
        
        if x_col not in self.data.columns or y_col not in self.data.columns:
            return
        
        x_data = self.data[x_col].values
        y_data = self.data[y_col].values
        
        win = pg.plot(title=f"{y_col} vs {x_col}")
        win.plot(x_data, y_data, pen=None, symbol='o', symbolSize=3, symbolBrush=(100, 100, 255, 150))
        win.setLabel('bottom', x_col)
        win.setLabel('left', y_col)
    
    def plot_2d_histogram(self):
        """Create 2D histogram plot."""
        x_col = self.x_column.currentText()
        y_col = self.y_column.currentText()
        bins = self.bins_spin.value()
        
        if x_col not in self.data.columns or y_col not in self.data.columns:
            return
        
        x_data = self.data[x_col].values
        y_data = self.data[y_col].values
        
        # Create 2D histogram
        hist, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=bins)
        
        # Display with ImageView
        win = pg.image(hist.T, title=f"2D Histogram: {y_col} vs {x_col}")
        win.setWindowTitle(f"2D Histogram: {y_col} vs {x_col}")
    
    def plot_localizations_per_frame(self):
        """Plot number of localizations per frame."""
        if 'frame' not in self.data.columns:
            QtWidgets.QMessageBox.warning(self, "Missing Data", "'frame' column not found")
            return
        
        frame_counts = self.data['frame'].value_counts().sort_index()
        
        win = pg.plot(title="Localizations per Frame")
        win.plot(frame_counts.index.values, frame_counts.values, pen='b')
        win.setLabel('bottom', 'Frame Number')
        win.setLabel('left', 'Number of Localizations')
    
    def plot_cumulative_localizations(self):
        """Plot cumulative number of localizations over frames."""
        if 'frame' not in self.data.columns:
            QtWidgets.QMessageBox.warning(self, "Missing Data", "'frame' column not found")
            return
        
        frame_counts = self.data['frame'].value_counts().sort_index()
        cumulative = np.cumsum(frame_counts.values)
        
        win = pg.plot(title="Cumulative Localizations")
        win.plot(frame_counts.index.values, cumulative, pen='g')
        win.setLabel('bottom', 'Frame Number')
        win.setLabel('left', 'Cumulative Localizations')
    
    def plot_intensity_vs_uncertainty(self):
        """Plot intensity vs localization uncertainty."""
        if 'intensity' not in self.data.columns or 'uncertainty' not in self.data.columns:
            QtWidgets.QMessageBox.warning(
                self, "Missing Data",
                "'intensity' and 'uncertainty' columns required"
            )
            return
        
        intensity = self.data['intensity'].values
        uncertainty = self.data['uncertainty'].values
        
        win = pg.plot(title="Intensity vs Uncertainty")
        win.plot(intensity, uncertainty, pen=None, symbol='o', symbolSize=2, 
                symbolBrush=(100, 100, 255, 100))
        win.setLabel('bottom', 'Intensity (photons)')
        win.setLabel('left', 'Uncertainty (nm)')


class StatisticsWidget(QtWidgets.QWidget):
    """Widget for displaying summary statistics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # Statistics text display
        self.stats_text = QtWidgets.QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(300)
        layout.addWidget(QtWidgets.QLabel("Summary Statistics:"))
        layout.addWidget(self.stats_text)
        
        # Refresh button
        refresh_btn = QtWidgets.QPushButton("Refresh Statistics")
        refresh_btn.clicked.connect(self.calculate_statistics)
        layout.addWidget(refresh_btn)
        
        layout.addStretch()
        self.setLayout(layout)
        
        self.data = None
    
    def set_data(self, data):
        """Set data for statistics calculation."""
        if isinstance(data, dict):
            self.data = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            self.data = data
        self.calculate_statistics()
    
    def calculate_statistics(self):
        """Calculate and display summary statistics."""
        if self.data is None or self.data.empty:
            self.stats_text.setText("No data available")
            return
        
        stats_lines = []
        stats_lines.append("=== SUMMARY STATISTICS ===\n")
        stats_lines.append(f"Total Localizations: {len(self.data)}\n")
        
        # Frame statistics
        if 'frame' in self.data.columns:
            n_frames = self.data['frame'].nunique()
            stats_lines.append(f"Number of Frames: {n_frames}")
            stats_lines.append(f"Mean Localizations/Frame: {len(self.data)/n_frames:.2f}\n")
        
        # Numeric column statistics
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == 'frame':
                continue
            
            stats_lines.append(f"\n{col}:")
            stats_lines.append(f"  Mean: {self.data[col].mean():.4f}")
            stats_lines.append(f"  Median: {self.data[col].median():.4f}")
            stats_lines.append(f"  Std: {self.data[col].std():.4f}")
            stats_lines.append(f"  Min: {self.data[col].min():.4f}")
            stats_lines.append(f"  Max: {self.data[col].max():.4f}")
        
        # Spatial extent
        if 'x' in self.data.columns and 'y' in self.data.columns:
            stats_lines.append("\nSpatial Extent:")
            stats_lines.append(f"  X Range: {self.data['x'].min():.2f} - {self.data['x'].max():.2f} nm")
            stats_lines.append(f"  Y Range: {self.data['y'].min():.2f} - {self.data['y'].max():.2f} nm")
            stats_lines.append(f"  Width: {self.data['x'].max() - self.data['x'].min():.2f} nm")
            stats_lines.append(f"  Height: {self.data['y'].max() - self.data['y'].min():.2f} nm")
        
        self.stats_text.setText('\n'.join(stats_lines))


class LocalizationResultsViewer(QtWidgets.QMainWindow):
    """Main window for viewing and analyzing localization results."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_data = None  # Unfiltered data
        self.filtered_data = None  # Currently displayed data
        self.current_file = None
        
        self.setWindowTitle("ThunderSTORM Localization Results Viewer")
        self.setGeometry(100, 100, 1400, 800)
        
        self.init_ui()
    
    def init_ui(self):
        # Create central widget and main layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        
        # Left panel: Table view
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        
        # Info label
        self.info_label = QtWidgets.QLabel("No data loaded")
        left_layout.addWidget(self.info_label)
        
        # Table view
        self.table_view = QtWidgets.QTableView()
        self.table_view.setSortingEnabled(True)
        self.table_model = LocalizationTableModel()
        self.table_view.setModel(self.table_model)
        left_layout.addWidget(self.table_view)
        
        main_layout.addWidget(left_panel, stretch=3)
        
        # Right panel: Tabs for different tools
        self.tool_tabs = QtWidgets.QTabWidget()
        
        # Filter tab
        self.filter_widget = FilterWidget()
        self.filter_widget.filterChanged.connect(self.apply_filters)
        self.tool_tabs.addTab(self.filter_widget, "Filters")
        
        # Plot tab
        self.plot_widget = PlotWidget()
        self.tool_tabs.addTab(self.plot_widget, "Plots")
        
        # Statistics tab
        self.stats_widget = StatisticsWidget()
        self.tool_tabs.addTab(self.stats_widget, "Statistics")
        
        main_layout.addWidget(self.tool_tabs, stretch=1)
        
        # Create menu bar
        self.create_menus()
    
    def create_menus(self):
        """Create application menus."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        load_action = QtWidgets.QAction('Load CSV...', self)
        load_action.triggered.connect(self.load_csv)
        file_menu.addAction(load_action)
        
        save_action = QtWidgets.QAction('Save Filtered CSV...', self)
        save_action.triggered.connect(self.save_filtered_csv)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        export_stats_action = QtWidgets.QAction('Export Statistics...', self)
        export_stats_action.triggered.connect(self.export_statistics)
        file_menu.addAction(export_stats_action)
        
        file_menu.addSeparator()
        
        close_action = QtWidgets.QAction('Close', self)
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)
        
        # Edit menu
        edit_menu = menubar.addMenu('Edit')
        
        reset_filters_action = QtWidgets.QAction('Reset Filters', self)
        reset_filters_action.triggered.connect(self.reset_filters)
        edit_menu.addAction(reset_filters_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        show_all_action = QtWidgets.QAction('Show All Columns', self)
        show_all_action.triggered.connect(self.show_all_columns)
        view_menu.addAction(show_all_action)
        
        auto_resize_action = QtWidgets.QAction('Auto-Resize Columns', self)
        auto_resize_action.triggered.connect(self.auto_resize_columns)
        view_menu.addAction(auto_resize_action)
    
    def load_csv(self):
        """Load localizations from CSV file."""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Localization CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not filename:
            return
        
        try:
            # Load CSV
            df = pd.read_csv(filename)
            
            # Convert column names to lowercase and remove units
            df.columns = df.columns.str.lower()
            df.columns = df.columns.str.replace(r'\s*\[.*?\]', '', regex=True)
            df.columns = df.columns.str.strip()
            
            self.set_data(df)
            self.current_file = filename
            self.info_label.setText(f"Loaded: {Path(filename).name} ({len(df)} localizations)")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                f"Error loading CSV file:\n{str(e)}"
            )
    
    def set_data(self, data):
        """Set data from dict or DataFrame."""
        if isinstance(data, dict):
            self.original_data = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            self.original_data = data.copy()
        else:
            raise ValueError("Data must be dict or DataFrame")
        
        self.filtered_data = self.original_data.copy()
        
        # Update all components
        self.table_model.update_data(self.filtered_data)
        self.filter_widget.update_columns(list(self.original_data.columns))
        self.plot_widget.set_data(self.filtered_data)
        self.stats_widget.set_data(self.filtered_data)
        
        self.auto_resize_columns()
        self.update_info_label()
    
    def apply_filters(self):
        """Apply current filters to data."""
        if self.original_data is None:
            return
        
        self.filtered_data = self.filter_widget.apply_filters(self.original_data)
        
        # Update display
        self.table_model.update_data(self.filtered_data)
        self.plot_widget.set_data(self.filtered_data)
        self.stats_widget.set_data(self.filtered_data)
        
        self.update_info_label()
    
    def reset_filters(self):
        """Reset all filters and show original data."""
        self.filter_widget.clear_filters()
        if self.original_data is not None:
            self.filtered_data = self.original_data.copy()
            self.table_model.update_data(self.filtered_data)
            self.plot_widget.set_data(self.filtered_data)
            self.stats_widget.set_data(self.filtered_data)
            self.update_info_label()
    
    def save_filtered_csv(self):
        """Save currently filtered data to CSV."""
        if self.filtered_data is None or self.filtered_data.empty:
            QtWidgets.QMessageBox.warning(self, "No Data", "No data to save")
            return
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Filtered Localizations",
            "",
            "CSV Files (*.csv)"
        )
        
        if not filename:
            return
        
        try:
            self.filtered_data.to_csv(filename, index=False)
            QtWidgets.QMessageBox.information(
                self,
                "Success",
                f"Saved {len(self.filtered_data)} localizations to:\n{filename}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Save Error",
                f"Error saving CSV:\n{str(e)}"
            )
    
    def export_statistics(self):
        """Export statistics to text file."""
        if self.filtered_data is None or self.filtered_data.empty:
            QtWidgets.QMessageBox.warning(self, "No Data", "No data available")
            return
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Statistics",
            "",
            "Text Files (*.txt)"
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'w') as f:
                f.write(self.stats_widget.stats_text.toPlainText())
            
            QtWidgets.QMessageBox.information(
                self,
                "Success",
                f"Statistics exported to:\n{filename}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Export Error",
                f"Error exporting statistics:\n{str(e)}"
            )
    
    def show_all_columns(self):
        """Show all columns in table."""
        for i in range(self.table_model.columnCount()):
            self.table_view.showColumn(i)
    
    def auto_resize_columns(self):
        """Auto-resize table columns to contents."""
        self.table_view.resizeColumnsToContents()
    
    def update_info_label(self):
        """Update information label."""
        if self.filtered_data is None:
            self.info_label.setText("No data loaded")
            return
        
        total = len(self.original_data) if self.original_data is not None else 0
        filtered = len(self.filtered_data)
        
        if total == filtered:
            self.info_label.setText(f"{total} localizations")
        else:
            self.info_label.setText(
                f"{filtered} / {total} localizations "
                f"({100*filtered/total:.1f}% after filtering)"
            )


def show_results_viewer(localizations=None):
    """Convenience function to show the results viewer."""
    viewer = LocalizationResultsViewer()
    
    if localizations is not None:
        viewer.set_data(localizations)
    
    viewer.show()
    return viewer


# For testing
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    
    # Create some test data
    test_data = {
        'frame': np.repeat(np.arange(100), 10),
        'x': np.random.randn(1000) * 1000 + 5000,
        'y': np.random.randn(1000) * 1000 + 5000,
        'intensity': np.random.exponential(1000, 1000),
        'background': np.random.poisson(50, 1000),
        'sigma_x': np.abs(np.random.randn(1000) * 0.5 + 1.6),
        'sigma_y': np.abs(np.random.randn(1000) * 0.5 + 1.6),
        'uncertainty': np.abs(np.random.randn(1000) * 10 + 30),
    }
    
    viewer = show_results_viewer(test_data)
    sys.exit(app.exec_())
