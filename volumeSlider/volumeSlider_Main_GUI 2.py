from typing import Optional, Tuple, List, Dict
import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
import flika
from flika import global_vars as g
import os
from distutils.version import StrictVersion
from pyqtgraph.dockarea import *
from OpenGL.GL import *
import pyqtgraph as pg
import time

import logging
flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector

from .helperFunctions import windowGeometry, setSliderUp, setMenuUp
from .pyqtGraph_classOverwrites import *


from .texturePlot import *
from .volumeSlider_3DViewer import *

from .volume_processor import VolumeProcessor

dataType = np.float16
import gc
import glob

#########################################################################################
#############                  volumeViewer GUI setup            ########################
#########################################################################################

class Form2(QtWidgets.QDialog):
    def __init__(self, viewerInstance):
        super().__init__()
        self.viewer = viewerInstance
        self.batch = False
        self.s = g.settings['volumeSlider']
        self.setup_ui()
        self.initialize_variables()
        self.connect_signals()

    def setup_ui(self):
        self.create_widgets()
        self.create_layout()
        self.set_window_properties()

    def create_widgets(self):
        # Labels
        self.labels = {
            'slice': QtWidgets.QLabel("Slice #"),
            'slices_per_volume': QtWidgets.QLabel("# of slices per volume: "),
            'slices_removed': QtWidgets.QLabel("# of slices removed per volume: "),
            'baseline': QtWidgets.QLabel("baseline value: "),
            'f0_start': QtWidgets.QLabel("F0 start volume: "),
            'f0_end': QtWidgets.QLabel("F0 end volume: "),
            'multiplication': QtWidgets.QLabel("factor to multiply by: "),
            'theta': QtWidgets.QLabel("theta: "),
            'shift': QtWidgets.QLabel("shift factor: "),
            'volume': QtWidgets.QLabel("# of volumes: "),
            'current_volume': QtWidgets.QLabel("current volume: "),
            'shape': QtWidgets.QLabel("array shape: "),
            'data_type': QtWidgets.QLabel("current data type: "),
            'new_data_type': QtWidgets.QLabel("new data type: "),
            'input_array': QtWidgets.QLabel("input array order: "),
            'display_array': QtWidgets.QLabel("display array order: "),
            'trim_last_frame': QtWidgets.QLabel("Trim Last Frame: "),
            'file_name': QtWidgets.QLabel("file name: ")
        }

        # Text labels
        self.text_labels = {
            'volume': QtWidgets.QLabel("  "),
            'current_volume': QtWidgets.QLabel("0"),
            'shape': QtWidgets.QLabel(str(self.viewer.getArrayShape())),
            'data_type': QtWidgets.QLabel(str(self.viewer.getDataType())),
            'array_save_path': QtWidgets.QLabel(str(self.viewer.savePath)),
            'file_name': QtWidgets.QLabel(str(self.viewer.getFileName()))
        }

        # Spinboxes
        self.spinboxes = {name: QtWidgets.QSpinBox() for name in
            ['slice', 'slices_per_volume', 'slices_removed', 'baseline', 'f0_start', 'f0_end',
             'multiplication', 'theta', 'shift', 'f0_vol_start', 'f0_vol_end']}

        # Slider
        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)

        # Buttons
        self.buttons = {
            'autolevel': QtWidgets.QPushButton("Autolevel"),
            'set_slices': QtWidgets.QPushButton("Set Slices"),
            'subtract_baseline': QtWidgets.QPushButton("subtract baseline"),
            'run_df_f0': QtWidgets.QPushButton("run DF/F0"),
            'export_window': QtWidgets.QPushButton("export to Window"),
            'set_data_type': QtWidgets.QPushButton("set data Type"),
            'multiply': QtWidgets.QPushButton("multiply"),
            'export_array': QtWidgets.QPushButton("export to array"),
            'open_3d_viewer': QtWidgets.QPushButton("open 3D viewer"),
            'close_3d_viewer': QtWidgets.QPushButton("close 3D viewer"),
            'load_new_file': QtWidgets.QPushButton("load new file"),
            'set_overlay': QtWidgets.QPushButton("Set overlay to current volume")
        }

        # Comboboxes
        self.dTypeSelectorBox = QtWidgets.QComboBox()
        self.dTypeSelectorBox.addItems(["float16", "float32", "float64", "int8", "int16", "int32", "int64"])
        self.inputArraySelectorBox = QtWidgets.QComboBox()
        self.inputArraySelectorBox.addItems(self.viewer.getArrayKeys())
        self.displayArraySelectorBox = QtWidgets.QComboBox()
        self.displayArraySelectorBox.addItems(self.viewer.getArrayKeys())

        # Checkbox
        self.trim_last_frame_checkbox = QtWidgets.QCheckBox()

    def create_layout(self):
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(10)

        widgets = [
            (self.labels['slice'], 1, 0), (self.spinboxes['slice'], 1, 1),
            (self.slider1, 2, 0, 2, 5),
            (self.labels['slices_per_volume'], 4, 0), (self.spinboxes['slices_per_volume'], 4, 1),
            (self.buttons['set_slices'], 4, 4),
            (self.labels['slices_removed'], 4, 2), (self.spinboxes['slices_removed'], 4, 3),
            (self.labels['baseline'], 6, 0), (self.spinboxes['baseline'], 6, 1),
            (self.buttons['subtract_baseline'], 6, 2),
            (self.labels['f0_start'], 7, 0), (self.spinboxes['f0_start'], 7, 1),
            (self.labels['f0_end'], 7, 2), (self.spinboxes['f0_end'], 7, 3),
            (self.buttons['run_df_f0'], 7, 4),
            (QtWidgets.QLabel("ratio start volume: "), 8, 0), (self.spinboxes['f0_vol_start'], 8, 1),
            (QtWidgets.QLabel("ratio End volume: "), 8, 2), (self.spinboxes['f0_vol_end'], 8, 3),
            (self.labels['volume'], 9, 0), (self.text_labels['volume'], 9, 1),
            (self.labels['current_volume'], 9, 2), (self.text_labels['current_volume'], 9, 3),
            (self.labels['shape'], 10, 0), (self.text_labels['shape'], 10, 1),
            (self.labels['multiplication'], 11, 0), (self.spinboxes['multiplication'], 11, 1),
            (self.buttons['multiply'], 11, 2),
            (self.buttons['set_overlay'], 12, 0),
            (self.labels['data_type'], 13, 0), (self.text_labels['data_type'], 13, 1),
            (self.labels['new_data_type'], 13, 2), (self.dTypeSelectorBox, 13, 3),
            (self.buttons['set_data_type'], 13, 4),
            (self.buttons['export_window'], 15, 0), (self.buttons['autolevel'], 15, 4),
            (self.buttons['export_array'], 16, 0), (self.text_labels['array_save_path'], 16, 1, 1, 4),
            (self.labels['theta'], 18, 0), (self.spinboxes['theta'], 18, 1),
            (self.labels['shift'], 19, 0), (self.spinboxes['shift'], 19, 1),
            (self.labels['trim_last_frame'], 20, 0), (self.trim_last_frame_checkbox, 20, 1),
            (self.labels['input_array'], 21, 0), (self.inputArraySelectorBox, 21, 1),
            (self.labels['display_array'], 21, 2), (self.displayArraySelectorBox, 21, 3),
            (self.buttons['open_3d_viewer'], 22, 0), (self.buttons['close_3d_viewer'], 22, 1),
            (self.buttons['load_new_file'], 22, 2),
            (self.labels['file_name'], 23, 0), (self.text_labels['file_name'], 23, 1, 1, 4)
        ]

        for item in widgets:
            if len(item) == 3:
                layout.addWidget(item[0], item[1], item[2])
            elif len(item) == 5:
                layout.addWidget(item[0], item[1], item[2], item[3], item[4])

        self.setLayout(layout)

    def set_window_properties(self):
        windowGeometry(self, left=300, top=300, height=600, width=400)
        self.setWindowTitle("Volume Slider GUI")

    def initialize_variables(self):
        self.slicesPerVolume = self.s['slicesPerVolume']
        self.slicesDeletedPerVolume = self.s['slicesDeletedPerVolume']
        self.baselineValue = self.s['baselineValue']
        self.f0Start = self.s['f0Start']
        self.f0End = self.s['f0End']
        self.f0VolStart = self.s['f0VolStart']
        self.f0VolEnd = self.s['f0VolEnd']
        self.multiplicationFactor = self.s['multiplicationFactor']
        self.currentDataType = self.s['currentDataType']
        self.newDataType = self.s['newDataType']
        self.inputArrayOrder = self.s['inputArrayOrder']
        self.displayArrayOrder = self.s['displayArrayOrder'] = 16
        self.theta = self.s['theta']
        self.shiftFactor = self.s['shiftFactor']
        self.trim_last_frame = self.s['trimLastFrame']

        self.setup_spinboxes()
        self.setup_slider()

    def setup_spinboxes(self):
        spinbox_configs = {
            'slice': (0, self.viewer.getNFrames(), 0),
            'slices_per_volume': (1, self.viewer.getNFrames(), self.slicesPerVolume),
            'slices_removed': (0, self.viewer.getNFrames() - 1, self.slicesDeletedPerVolume),
            'baseline': (0, int(self.viewer.getMaxPixel()), int(self.baselineValue)),
            'f0_start': (0, self.viewer.getNVols(), self.f0Start),
            'f0_end': (0, self.viewer.getNVols(), self.f0End),
            'multiplication': (0, 10000, self.multiplicationFactor),
            'theta': (0, 360, self.theta),
            'shift': (0, 100, self.shiftFactor),
            'f0_vol_start': (0, self.viewer.getNVols(), self.f0VolStart),
            'f0_vol_end': (0, self.viewer.getNVols(), self.f0VolEnd)
        }

        for name, (min_val, max_val, value) in spinbox_configs.items():
            spinbox = self.spinboxes[name]
            spinbox.setRange(min_val, max_val)
            spinbox.setValue(value)

        self.trim_last_frame_checkbox.setChecked(self.trim_last_frame)
        self.inputArraySelectorBox.setCurrentIndex(4)
        self.displayArraySelectorBox.setCurrentIndex(18)

    def setup_slider(self):
        setSliderUp(self.slider1, minimum=0, maximum=self.viewer.getNFrames(), tickInterval=1, singleStep=1, value=0)

    def connect_signals(self):
        self.slider1.valueChanged.connect(self.slider1ValueChange)
        self.spinboxes['slice'].valueChanged.connect(self.spinBox1ValueChange)
        self.spinboxes['theta'].valueChanged.connect(self.setTheta)
        self.spinboxes['shift'].valueChanged.connect(self.setShiftFactor)

        button_connections = {
            'autolevel': self.autoLevel,
            'set_slices': self.updateVolumeValue,
            'subtract_baseline': self.subtractBaseline,
            'run_df_f0': self.ratioDFF0,
            'export_window': self.exportToWindow,
            'set_data_type': self.dTypeSelectionChange,
            'multiply': self.multiplyByFactor,
            'export_array': self.exportArray,
            'open_3d_viewer': self.startViewer,
            'close_3d_viewer': self.closeViewer,
            'load_new_file': lambda: self.loadNewFile(''),
            'set_overlay': self.setOverlay
        }

        for name, func in button_connections.items():
            self.buttons[name].clicked.connect(func)

        self.trim_last_frame_checkbox.stateChanged.connect(self.trim_last_frameClicked)
        self.inputArraySelectorBox.currentIndexChanged.connect(self.inputArraySelectionChange)
        self.displayArraySelectorBox.currentIndexChanged.connect(self.displayArraySelectionChange)

    def slider1ValueChange(self, value):
        self.spinboxes['slice'].setValue(value)

    def spinBox1ValueChange(self, value):
        if 0 <= value <= self.slider1.maximum():
            self.slider1.setValue(value)
            self.viewer.updateDisplay_sliceNumberChange(value)
        else:
            logging.warning(f"Invalid spinbox value: {value}")

    def autoLevel(self):
        self.viewer.displayWindow.imageview.autoLevels()

    def updateVolumeValue(self):
        try:
            self.slicesPerVolume = self.spinboxes['slices_per_volume'].value()
            noVols = int(self.viewer.getNFrames() / self.slicesPerVolume)
            self.framesToDelete = self.spinboxes['slices_removed'].value()
            self.viewer.updateVolsandFramesPerVol(noVols, self.slicesPerVolume, framesToDelete=self.framesToDelete)
            self.text_labels['volume'].setText(str(noVols))

            self.viewer.updateDisplay_volumeSizeChange()
            self.text_labels['shape'].setText(str(self.viewer.getArrayShape()))

            max_value = self.slicesPerVolume - 1 - self.framesToDelete if self.slicesPerVolume % 2 == 0 else self.slicesPerVolume - 2 - self.framesToDelete

            self.spinboxes['slice'].setRange(0, max_value)
            self.slider1.setMaximum(max_value)

            self.updateVolSpinBoxes()
        except ValueError as e:
            logging.error(f"Failed to update volume: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to update volume: {e}")

    def updateVolSpinBoxes(self):
        range_value = self.viewer.getNVols() - 1
        for name in ['f0_start', 'f0_end', 'f0_vol_start', 'f0_vol_end']:
            self.spinboxes[name].setRange(0, range_value)
        self.spinboxes['f0_vol_end'].setValue(range_value)

    def getBaseline(self):
        return self.spinboxes['baseline'].value()

    def getF0(self):
        return (self.spinboxes['f0_start'].value(),
                self.spinboxes['f0_end'].value(),
                self.spinboxes['f0_vol_start'].value(),
                self.spinboxes['f0_vol_end'].value())

    def subtractBaseline(self):
        self.viewer.subtractBaseline()

    def ratioDFF0(self):
        self.viewer.ratioDFF0()

    def exportToWindow(self):
        try:
            data = self.viewer.getExportData()
            if data is not None:
                Window(data)
            else:
                raise ValueError("No valid data to export")
        except Exception as e:
            logging.error(f"Failed to export to window: {e}")
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export to window: {e}")

    def exportArray(self, vol='None'):
        try:
            if vol == 'None':
                np.save(self.viewer.savePath, self.viewer.B)
            else:
                f, v, x, y = self.viewer.B.shape
                np.save(self.viewer.savePath, self.viewer.B[:, vol, :, :].reshape((f, 1, x, y)))
            logging.info(f"Array exported successfully to {self.viewer.savePath}")
        except Exception as e:
            logging.error(f"Failed to export array: {e}")
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export array: {e}")

    def dTypeSelectionChange(self):
        try:
            new_dtype = self.dTypeSelectorBox.currentText()
            self.viewer.setDType(new_dtype)
            self.text_labels['data_type'].setText(str(self.viewer.getDataType()))
        except Exception as e:
            logging.error(f"Failed to change data type: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to change data type: {e}")

    def multiplyByFactor(self):
        self.viewer.multiplyByFactor(self.spinboxes['multiplication'].value())

    def startViewer(self):
        self.saveSettings()
        self.viewer.startViewer()

    def setOverlay(self):
        self.viewer.setOverlay()

    def closeViewer(self):
        self.viewer.closeViewer()

    def setTheta(self):
        self.theta = self.spinboxes['theta'].value()

    def setShiftFactor(self):
        self.shiftFactor = self.spinboxes['shift'].value()

    def trim_last_frameClicked(self):
        self.trim_last_frame = self.trim_last_frame_checkbox.isChecked()

    def inputArraySelectionChange(self, value):
        self.viewer.setInputArrayOrder(self.inputArraySelectorBox.currentText())

    def displayArraySelectionChange(self, value):
        self.viewer.setDisplayArrayOrder(self.displayArraySelectorBox.currentText())

    def saveSettings(self):
        self.s['theta'] = self.theta
        self.s['slicesPerVolume'] = self.slicesPerVolume
        self.s['slicesDeletedPerVolume'] = self.slicesDeletedPerVolume
        self.s['baselineValue'] = self.baselineValue
        self.s['f0Start'] = self.f0Start
        self.s['f0End'] = self.f0End
        self.s['multiplicationFactor'] = self.multiplicationFactor
        self.s['currentDataType'] = self.currentDataType
        self.s['newDataType'] = self.newDataType
        self.s['shiftFactor'] = self.shiftFactor
        self.s['trimLastFrame'] = self.trim_last_frame
        self.s['f0VolStart'] = self.f0VolStart
        self.s['f0VolEnd'] = self.f0VolEnd

        g.settings['volumeSlider'] = self.s

    def setFileName(self, fileName):
        self.fileName = fileName
        self.text_labels['file_name'].setText(self.fileName)

    def getFileName(self):
        return self.fileName

    def close(self):
        self.saveSettings()
        self.viewer.closeViewer()
        self.viewer.displayWindow.close()
        self.viewer.dialogbox.destroy()
        self.viewer.end()
        self.closeAllWindows()
        gc.collect()

    def clearData(self):
        self.viewer.A = []
        self.viewer.B = []

    def loadNewFile(self, fileName):
        if not self.batch:
            fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File',
                                                                os.path.expanduser("~/Desktop"),
                                                                'tiff files (*.tif *.tiff)')
        if fileName:
            try:
                A, _, _ = self.viewer.openTiff(fileName)
                self.viewer.updateVolumeSlider(A)
                self.viewer.displayWindow.imageview.setImage(A)
                self.setFileName(fileName)
                self.viewer.setFileName(fileName)
            except Exception as e:
                logging.error(f"Failed to load new file: {e}")
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load new file: {e}")

    def update_ui(self):
        self.text_labels['volume'].setText(str(self.viewer.getNVols()))
        self.text_labels['shape'].setText(str(self.viewer.getArrayShape()))
        self.text_labels['data_type'].setText(str(self.viewer.getDataType()))
        self.updateVolSpinBoxes()

    def batchProcess(self, paramDict: dict):
        logging.info(f"Starting batch process with parameters: {paramDict}")
        tiffFiles = glob.glob(os.path.join(paramDict['inputDirectory'], "*.tif*"))

        with QtWidgets.QProgressDialog("Processing files...", "Abort", 0, len(tiffFiles), self) as progress:
            progress.setWindowModality(QtCore.Qt.WindowModal)

            for i, tiff_file in enumerate(tiffFiles):
                if progress.wasCanceled():
                    break

                progress.setValue(i)
                progress.setLabelText(f"Processing file {i+1}/{len(tiffFiles)}: {tiff_file}")

                try:
                    self.process_single_file(tiff_file, paramDict, i == 0)
                except Exception as e:
                    logging.error(f"Error processing file {tiff_file}: {e}")
                    QtWidgets.QMessageBox.warning(self, "Processing Error", f"Error processing file {tiff_file}: {e}")

            progress.setValue(len(tiffFiles))

        logging.info("Batch processing completed")
        g.m.statusBar().showMessage('Batch processing finished')

    def process_single_file(self, tiff_file: str, paramDict: dict, is_first: bool):
        self.loadNewFile(tiff_file)
        self.spinboxes['slices_per_volume'].setValue(paramDict['slicesPerVolume'])
        self.updateVolumeValue()

        if paramDict['subtractBaseline']:
            self.spinboxes['baseline'].setValue(paramDict['baselineValue'])
            self.subtractBaseline()

        if paramDict['runDFF0']:
            self.spinboxes['f0_start'].setValue(paramDict['f0Start'])
            self.spinboxes['f0_end'].setValue(paramDict['f0End'])
            self.spinboxes['f0_vol_start'].setValue(paramDict['f0VolStart'])
            self.spinboxes['f0_vol_end'].setValue(paramDict['f0VolEnd'])
            self.ratioDFF0()

        if paramDict['runMultiplication']:
            self.spinboxes['multiplication'].setValue(paramDict['multiplicationFactor'])
            self.multiplyByFactor()

        self.spinboxes['theta'].setValue(paramDict['theta'])
        self.spinboxes['shift'].setValue(paramDict['shiftFactor'])
        self.trim_last_frame_checkbox.setChecked(paramDict['trim_last_frame'])

        if is_first:
            self.startViewer()
        else:
            self.viewer.viewer.changeMainImage(self.viewer.B)
            self.viewer.viewer.runBatchStep(tiff_file)

    def closeEvent(self, event):
        self.close()
        event.accept()
