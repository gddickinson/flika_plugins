import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
import flika
from flika import global_vars as g
from flika.window import Window
import os
from distutils.version import StrictVersion
from pyqtgraph.dockarea import *
import matplotlib.pyplot as plt
from OpenGL.GL import *
import glob
import gc
import logging
from typing import Optional, List, Tuple, Dict, Any

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector

from .helperFunctions import *
from .pyqtGraph_classOverwrites import *

from .volumeSlider_3DViewer import *
from .volumeSlider_Main_GUI import *
from .tiffLoader import openTiff
from .volume_processor import VolumeProcessor
from .volumeSlider_Main_GUI import Form2
from .volumeSlider_3DViewer import SliceViewer
from .helperFunctions import perform_shear_transform

dataType = np.float32


### disable messages from PyQt ################
def handler(msg_type, msg_log_context, msg_string):
    pass

QtCore.qInstallMessageHandler(handler)
############################################################################
##########                Create a logger      #############################
############################################################################

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create console handler and set level to info
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create file handler and set level to info
file_handler = logging.FileHandler('volumeslider.log')
file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Add formatter to handlers
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

#########################################################################################
#############                  CamVolumeSlider class                #####################
#########################################################################################
class CamVolumeSlider:
    def __init__(self):
        self.processor: Optional[VolumeProcessor] = None
        self.fileName: str = ''
        self.nVols: int = 1
        self.dataType = np.float32
        self.batch: bool = False
        self.overlayEmbeded: bool = False  # Add this line
        self.setup_data_structures()
        self.setup_array_parameters()
        self.setup_logger()

    def setup_data_structures(self):
        self.dTypeDict: Dict[str, np.dtype] = {
            'float16': np.float16, 'float32': np.float32, 'float64': np.float64,
            'int8': np.uint8, 'int16': np.uint16, 'int32': np.uint32, 'int64': np.uint64
        }
        self.arrayDict: Dict[str, List[int]] = {
                '[0, 1, 2, 3]': [0, 1, 2, 3],
                '[0, 1, 3, 2]': [0, 1, 3, 2],
                '[0, 2, 1, 3]': [0, 2, 1, 3],
                '[0, 2, 3, 1]': [0, 2, 3, 1],
                '[0, 3, 1, 2]': [0, 3, 1, 2],
                '[0, 3, 2, 1]': [0, 3, 2, 1],
                '[1, 0, 2, 3]': [1, 0, 2, 3],
                '[1, 0, 3, 2]': [1, 0, 3, 2],
                '[1, 2, 0, 3]': [1, 2, 0, 3],
                '[1, 2, 3, 0]': [1, 2, 3, 0],
                '[1, 3, 0, 2]': [1, 3, 0, 2],
                '[1, 3, 2, 0]': [1, 3, 2, 0],
                '[2, 0, 1, 3]': [2, 0, 1, 3],
                '[2, 0, 3, 1]': [2, 0, 3, 1],
                '[2, 1, 0, 3]': [2, 1, 0, 3],
                '[2, 1, 3, 0]': [2, 1, 3, 0],
                '[2, 3, 0, 1]': [2, 3, 0, 1],
                '[2, 3, 1, 0]': [2, 3, 1, 0],
                '[3, 0, 1, 2]': [3, 0, 1, 2],
                '[3, 0, 2, 1]': [3, 0, 2, 1],
                '[3, 1, 0, 2]': [3, 1, 0, 2],
                '[3, 1, 2, 0]': [3, 1, 2, 0],
                '[3, 2, 0, 1]': [3, 2, 0, 1],
                '[3, 2, 1, 0]': [3, 2, 1, 0]
        }

    def setup_array_parameters(self):
        self.inputArrayOrder: List[int] = [0, 3, 1, 2]
        self.displayArrayOrder: List[int] = [3, 0, 1, 2]
        self.B: Optional[np.ndarray] = None
        self.savePath: str = ''
        self._A: Optional[np.ndarray] = None
        self.nFrames: int = 0
        self.x: int = 0
        self.y: int = 0
        self.framesPerVol: int = 0

    def setup_logger(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    @property
    def A(self) -> np.ndarray:
        if self._A is None:
            raise ValueError("Data not loaded")
        return self._A

    @A.setter
    def A(self, value: np.ndarray):
        self._A = value
        self.nFrames, self.x, self.y = value.shape
        self.framesPerVol = int(self.nFrames / self.nVols)

    def startVolumeSlider(self, A: Optional[np.ndarray] = None, keepWindow: bool = False,
                          batch: bool = False, preProcess: bool = False,
                          nVols: Optional[int] = None, framesPerVol: Optional[int] = None,
                          framesToDelete: int = 0, overlayEmbeded: bool = False,
                          A_overlay: Optional[np.ndarray] = None):
        try:
            if batch:
                self.batch = True
                self.batchOptions = BatchOptions()
                self.batchOptions.start()
                return

            self.A = A if A is not None else np.array(g.win.image, dtype=self.dataType)
            if not keepWindow:
                g.win.close()

            self.processor = VolumeProcessor(self.A)
            self.B = None
            self.setup_display_window()
            self.setup_dialog_box()

            if preProcess:
                self.preProcess_stack(framesPerVol, framesToDelete)

            if overlayEmbeded:
                self.setup_overlay(A_overlay, framesPerVol, framesToDelete)

        except Exception as e:
            logging.error(f"Error in startVolumeSlider: {str(e)}")
            QtWidgets.QMessageBox.critical(None, "Error", f"Failed to start Volume Slider: {str(e)}")

    def setup_display_window(self):
        self.displayWindow = Window(self.A, 'Volume Slider Window')

    def setup_dialog_box(self):
        self.dialogbox = Form2(self)
        self.dialogbox.show()

    def setup_overlay(self, A_overlay: np.ndarray, framesPerVol: int, framesToDelete: int):
        logging.info('Generating overlay window...')
        self.A_overlay = A_overlay
        self.overlayWindow = Window(self.A_overlay, 'Overlay Window')
        self.processOverlay(framesPerVol, framesToDelete)
        self.overlayEmbeded = True  # Set this to True when overlay is set up
        self.startViewer()

    def setOverlay(self):
        if self.B is not None and len(self.B) > 0:
            self.overlayVolume = self.displayWindow.imageview.currentIndex
            self.A_overlay = self.B[:, self.overlayVolume, :, :]
            self.overlayWindow = Window(self.A_overlay[0], 'Overlay Window')
            self.overlayEmbeded = True  # Set this to True when overlay is set
            self.B = self.B[:, :self.overlayVolume, :, :]
            self.updateDisplay()
        else:
            logging.error("No valid data to set overlay")
            QtWidgets.QMessageBox.warning(None, "Error", "No valid data to set overlay")


    def preProcess_stack(self, framesPerVol: int, framesToDelete: int = 0):
        logging.info('Preprocessing Image Stack')
        logging.info(f"Input shape: {self.A.shape}, Frames per volume: {framesPerVol}")

        try:
            reshaped_data = self.processor.reshape_to_4d(self.A, framesPerVol)
            if reshaped_data is None:
                raise ValueError("Failed to reshape data")

            self.B = reshaped_data
            self.nVols = self.B.shape[1]

            logging.info(f"Reshaped data shape: {self.B.shape}")

            if framesToDelete > 0:
                self.B = self.processor.delete_frames(self.B, framesToDelete)
                logging.info(f"Shape after deleting frames: {self.B.shape}")

            self.updateDisplay()
        except Exception as e:
            logging.error(f"Error in preProcess_stack: {str(e)}")
            raise

    def processOverlay(self, framesPerVol: int, framesToDelete: int = 0):
        logging.info("Processing overlay")
        # TODO: Implement overlay processing

    def updateVolumeSlider(self, A: np.ndarray):
        self.A = A
        self.B = None
        self.framesPerVol = int(self.nFrames / self.nVols)

    def updateDisplay_volumeSizeChange(self):
        try:
            reshaped_data = self.processor.reshape_to_4d(self.A, self.getFramesPerVol(), self.getNVols())
            if reshaped_data is None:
                raise ValueError("Failed to reshape data")

            self.B = reshaped_data

            if self.framesToDelete != 0:
                self.B = self.processor.delete_frames(self.B, self.framesToDelete)

            first_slice = self.B[0] if self.B is not None else None
            if first_slice is not None:
                self.safe_update_display(first_slice)
            else:
                logging.warning("No valid first slice to display")

            self.connect_timeline()
            self.update_shape_text()

        except Exception as e:
            logging.error(f"Error in updateDisplay_volumeSizeChange: {str(e)}")
            QtWidgets.QMessageBox.critical(None, "Error", f"Failed to update display: {str(e)}")

    def connect_timeline(self):
        timeline = getattr(self.displayWindow.imageview, 'timeLine', None)
        if timeline is not None:
            timeline.sigPositionChanged.connect(self.displayCurrentVolume)
        else:
            logging.warning("Display window does not have a timeline")

    def update_shape_text(self):
        if hasattr(self, 'dialogbox') and hasattr(self.dialogbox, 'shapeText'):
            self.dialogbox.shapeText.setText(str(self.B.shape if self.B is not None else "N/A"))

    def safe_update_display(self, image: np.ndarray):
        if hasattr(self, 'displayWindow') and self.displayWindow is not None:
            try:
                self.displayWindow.imageview.setImage(image, autoLevels=False)
            except Exception as e:
                logging.error(f"Error updating display: {str(e)}")

    def updateDisplay_sliceNumberChange(self, index: int):
        displayIndex = self.displayWindow.imageview.currentIndex
        slice_data = self.safe_get_array_value(self.B, index)
        if slice_data is not None:
            self.displayWindow.imageview.setImage(slice_data, autoLevels=False)
            self.displayWindow.imageview.setCurrentIndex(displayIndex)
        else:
            logging.warning(f"No valid slice data at index {index}")

        if self.overlayEmbeded:
            self.update_overlay(index, displayIndex)

    def update_overlay(self, index: int, displayIndex: int):
        overlay_data = self.safe_get_array_value(self.A_overlay, index)
        if overlay_data is not None:
            self.overlayWindow.imageview.setImage(overlay_data, autoLevels=False)
            self.overlayWindow.imageview.setCurrentIndex(displayIndex)
        else:
            logging.warning(f"No valid overlay data at index {index}")

    def safe_get_array_value(self, arr: Optional[np.ndarray], index: int) -> Optional[np.ndarray]:
        try:
            return arr[index] if arr is not None else None
        except IndexError:
            logging.warning(f"Index {index} out of bounds for array of shape {arr.shape if arr is not None else 'None'}")
            return None

    def getNFrames(self) -> int:
        return self.nFrames

    def getNVols(self) -> int:
        return self.nVols

    def getFramesPerVol(self) -> int:
        return self.framesPerVol

    def updateVolsandFramesPerVol(self, nVols: int, framesPerVol: int, framesToDelete: int = 0):
        try:
            self.validate_parameters(nVols, framesPerVol, framesToDelete)
            self.nVols = nVols
            self.framesPerVol = framesPerVol
            self.framesToDelete = framesToDelete
            logging.info(f"Updated to {nVols} volumes with {framesPerVol} frames per volume, deleting {framesToDelete} frames")
        except ValueError as e:
            logging.error(f"Invalid parameters: {str(e)}")
            raise

    def validate_parameters(self, nVols: int, framesPerVol: int, framesToDelete: int):
        if nVols <= 0 or framesPerVol <= 0:
            raise ValueError("nVols and framesPerVol must be positive integers")
        if framesToDelete < 0 or framesToDelete >= framesPerVol:
            raise ValueError("framesToDelete must be non-negative and less than framesPerVol")

    def getArrayShape(self) -> Tuple[int, ...]:
        if self.B is None:
            logging.info(f'Array shape: {self.A.shape}')
            return self.A.shape
        return self.B.shape

    def subtractBaseline(self):
        if self.B is None:
            raise ValueError("No data to subtract baseline from")
        baseline = self.dialogbox.getBaseline()
        self.B = self.processor.subtract_baseline(self.B, baseline)
        self.updateDisplay()

    def ratioDFF0(self):
        if self.B is None:
            raise ValueError("No data to calculate DF/F0")
        f0_start, f0_end, vol_start, vol_end = self.dialogbox.getF0()
        self.B = self.processor.calculate_df_f0(self.B, f0_start, f0_end, vol_start, vol_end)
        self.updateDisplay()

    def exportToWindow(self):
        if self.B is not None:
            Window(self.B.reshape(self.nFrames, self.x, self.y))
        else:
            logging.error("No valid data to export")
            QtWidgets.QMessageBox.warning(None, "Export Error", "No valid data to export")

    def exportArray(self, vol: str = 'None'):
        try:
            with open(self.savePath, 'wb') as f:
                if vol == 'None':
                    np.save(f, self.B)
                else:
                    f, v, x, y = self.B.shape
                    np.save(f, self.B[:, int(vol), :, :].reshape((f, 1, x, y)))
            logging.info(f"Array exported successfully to {self.savePath}")
        except Exception as e:
            logging.error(f"Failed to export array: {str(e)}")
            QtWidgets.QMessageBox.critical(None, "Export Error", f"Failed to export array: {str(e)}")

    def getVolumeArray(self, vol: int) -> np.ndarray:
        f, v, x, y = self.B.shape
        return self.B[:, vol, :, :].reshape((f, 1, x, y))

    def getMaxPixel(self) -> float:
        return np.max(self.B) if self.B is not None else np.max(self.A)

    def setDType(self, newDataType: str):
        if self.B is not None:
            self.B = self.B.astype(self.dTypeDict[newDataType])
            self.dataType = self.dTypeDict[newDataType]
            self.dialogbox.dataTypeText.setText(self.getDataType())
            self.updateDisplay()
        else:
            logging.error("No valid data to change data type")
            QtWidgets.QMessageBox.warning(None, "Error", "No valid data to change data type")

    def getDataType(self) -> str:
        return str(self.dataType).split(".")[-1].split("'")[0]

    def getArrayKeys(self) -> List[str]:
        return list(self.arrayDict.keys())

    def setInputArrayOrder(self, value: str):
        self.inputArrayOrder = self.arrayDict[value]

    def setDisplayArrayOrder(self, value: str):
        self.displayArrayOrder = self.arrayDict[value]

    def multiplyByFactor(self, factor: float):
        if self.B is not None:
            self.B = self.processor.multiply_by_factor(self.B, float(factor))
            self.updateDisplay()
        else:
            logging.error("No valid data to multiply")
            QtWidgets.QMessageBox.warning(None, "Error", "No valid data to multiply")

    def startViewer(self):
        if self.B is None:
            logging.warning("First set number of frames per volume")
            g.m.statusBar().showMessage("First set number of frames per volume")
        else:
            if self.batch:
                self.viewer = SliceViewer(self, self.B, batch=True, imsExportPath=self.imsPath)
            else:
                logging.info('3D viewer starting...')
                self.viewer = SliceViewer(self, self.B)
                if self.overlayEmbeded:
                    logging.info('Overlay added from stack')
                    A_overlay_trim = self.A_overlay[0:self.getFramesPerVol(), :, :]
                    A_overlay_4D = A_overlay_trim.reshape((self.getFramesPerVol(), 1, self.x, self.y))
                    self.viewer.overlayArray_start(overlayFromStack=True, overlayArray=A_overlay_4D)

    def closeViewer(self):
        if hasattr(self, 'viewer'):
            self.viewer.close()
        else:
            logging.warning("No viewer to close")

    def displayCurrentVolume(self):
        if hasattr(self, 'dialogbox'):
            self.dialogbox.currentVolumeText.setText(str(self.displayWindow.imageview.currentIndex))

    def setFileName(self, fileName: str):
        self.fileName = fileName

    def getFileName(self) -> str:
        return self.fileName

    def updateDisplay(self):
        if self.B is not None:
            self.safe_update_display(self.B[0])
        else:
            logging.warning("No data to display")

    def batchProcess(self, paramDict: Dict[str, Any]):
        logging.info(f"Starting batch process with parameters: {paramDict}")
        tiffFiles = glob.glob(os.path.join(paramDict['inputDirectory'], "*.tif*"))

        progress = QtWidgets.QProgressDialog("Processing files...", "Abort", 0, len(tiffFiles))
        progress.setWindowModality(QtCore.Qt.WindowModal)

        for i, tiff_file in enumerate(tiffFiles):
            if progress.wasCanceled():
                break

            progress.setValue(i)
            progress.setLabelText(f"Processing file {i+1}/{len(tiffFiles)}: {tiff_file}")

            try:
                self.process_single_file(tiff_file, paramDict, i == 0)
            except Exception as e:
                logging.error(f"Error processing file {tiff_file}: {str(e)}")
                QtWidgets.QMessageBox.warning(None, "Processing Error", f"Error processing file {tiff_file}: {str(e)}")

        progress.setValue(len(tiffFiles))
        logging.info("Batch processing completed")
        g.m.statusBar().showMessage('Batch processing finished')

    def process_single_file(self, tiff_file: str, paramDict: Dict[str, Any], is_first: bool):
        self.imsPath = tiff_file
        A, _, _ = openTiff(tiff_file)
        self.updateVolumeSlider(A)
        self.dialogbox.SpinBox2.setValue(paramDict['slicesPerVolume'])
        self.dialogbox.updateVolumeValue()

        if paramDict['subtractBaseline']:
            self.dialogbox.SpinBox4.setValue(paramDict['baselineValue'])
            self.subtractBaseline()

        if paramDict['runDFF0']:
            self.dialogbox.SpinBox6.setValue(paramDict['f0Start'])
            self.dialogbox.SpinBox7.setValue(paramDict['f0End'])
            self.dialogbox.SpinBox11.setValue(paramDict['f0VolStart'])
            self.dialogbox.SpinBox12.setValue(paramDict['f0VolEnd'])
            self.ratioDFF0()

        if paramDict['runMultiplication']:
            self.dialogbox.SpinBox8.setValue(paramDict['multiplicationFactor'])
            self.multiplyByFactor(paramDict['multiplicationFactor'])

        self.dialogbox.SpinBox9.setValue(paramDict['theta'])
        self.dialogbox.SpinBox10.setValue(paramDict['shiftFactor'])
        self.dialogbox.trim_last_frame_checkbox.setChecked(paramDict['trim_last_frame'])

        if is_first:
            self.startViewer()
        else:
            self.viewer.changeMainImage(self.B)
            self.viewer.runBatchStep(tiff_file)

    def setOverlay(self):
        if self.B is not None and len(self.B) > 0:
            self.overlayVolume = self.displayWindow.imageview.currentIndex
            self.A_overlay = self.B[:, self.overlayVolume, :, :]
            self.overlayWindow = Window(self.A_overlay[0], 'Overlay Window')
            self.overlayEmbeded = True
            self.B = self.B[:, :self.overlayVolume, :, :]
            self.updateDisplay()
        else:
            logging.error("No valid data to set overlay")
            QtWidgets.QMessageBox.warning(None, "Error", "No valid data to set overlay")

    def changeMainImage(self, A: np.ndarray):
        if self.dialogbox.trim_last_frame:
            self.A = A[:, :-1, :, :]
        else:
            self.A = A

        self.A = perform_shear_transform(self.A, self.dialogbox.shiftFactor, False, self.A.dtype,
                                         self.dialogbox.theta, inputArrayOrder=self.inputArrayOrder,
                                         displayArrayOrder=self.displayArrayOrder)
        self.data = self.A[:, 0, :, :]
        self.updateAllMainWins()

    def updateAllMainWins(self):
        for win in [1, 2, 3, 6]:
            self.update(win)
        self.update_center()

    def update(self, win: int):
        # Implement update logic for different windows
        pass

    def update_center(self):
        # Implement update center logic
        pass

    def closeEvent(self, event):
        self.close()
        event.accept()

    def end(self):
        #TODO!
        ...

    def closeAllWindows(self):
        #TODO!
        ...

    def close(self):
        self.saveSettings()
        self.closeViewer()
        if hasattr(self, 'displayWindow'):
            self.displayWindow.close()
        if hasattr(self, 'dialogbox'):
            self.dialogbox.close()
        gc.collect()

    def getInputArrayOrder(self) -> List[int]:
        return self.inputArrayOrder

    def getDisplayArrayOrder(self) -> List[int]:
        return self.displayArrayOrder

camVolumeSlider = CamVolumeSlider()


class BatchOptions(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(BatchOptions, self).__init__(parent)
        self.s = g.settings['volumeSlider']
        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        self.setWindowTitle("Batch Options")
        self.setGeometry(300, 300, 400, 600)

        layout = QtWidgets.QGridLayout(self)
        layout.setSpacing(10)

        self.create_spinboxes()
        self.create_checkboxes()
        self.create_buttons()
        self.create_labels()

        self.add_widgets_to_layout(layout)

    def create_spinboxes(self):
        self.spinboxes = {
            'slicesPerVolume': self.create_spinbox(0, 10000, self.s['slicesPerVolume']),
            'theta': self.create_spinbox(0, 360, self.s['theta']),
            'baselineValue': self.create_spinbox(0, 100000, self.s['baselineValue']),
            'f0Start': self.create_spinbox(0, 100000, self.s['f0Start']),
            'f0End': self.create_spinbox(0, 100000, self.s['f0End']),
            'f0VolStart': self.create_spinbox(0, 100000, self.s['f0VolStart']),
            'f0VolEnd': self.create_spinbox(0, 100000, self.s['f0VolEnd']),
            'multiplicationFactor': self.create_spinbox(0, 100000, self.s['multiplicationFactor']),
            'shiftFactor': self.create_spinbox(0, 100000, self.s['shiftFactor']),
        }

    def create_spinbox(self, min_val: int, max_val: int, default_val: int) -> QtWidgets.QSpinBox:
        spinbox = QtWidgets.QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default_val)
        return spinbox

    def create_checkboxes(self):
        self.checkboxes = {
            'trim_last_frame': QtWidgets.QCheckBox(),
            'subtractBaseline': QtWidgets.QCheckBox(),
            'runDFF0': QtWidgets.QCheckBox(),
            'runMultiplication': QtWidgets.QCheckBox(),
        }
        self.checkboxes['trim_last_frame'].setChecked(self.s['trimLastFrame'])

    def create_buttons(self):
        self.button_setInputDirectory = QtWidgets.QPushButton("Set Folder")
        self.button_startBatch = QtWidgets.QPushButton("Go")

    def create_labels(self):
        self.labels = {
            'slicesPerVolume': QtWidgets.QLabel("slices per volume:"),
            'theta': QtWidgets.QLabel("theta:"),
            'baselineValue': QtWidgets.QLabel('baseline Value:'),
            'f0Start': QtWidgets.QLabel('f0 Start:'),
            'f0End': QtWidgets.QLabel('f0 End:'),
            'f0VolStart': QtWidgets.QLabel('f0Vol Start:'),
            'f0VolEnd': QtWidgets.QLabel('f0Vol End:'),
            'multiplicationFactor': QtWidgets.QLabel('multiplication Factor:'),
            'shiftFactor': QtWidgets.QLabel('shift Factor:'),
            'trim_last_frame': QtWidgets.QLabel('trim Last Frame:'),
            'subtractBaseline': QtWidgets.QLabel('subtract baseline:'),
            'runDFF0': QtWidgets.QLabel('run DF/F0:'),
            'runMultiplication': QtWidgets.QLabel('scale by multiplication factor:'),
            'inputDirectory': QtWidgets.QLabel('input directory:'),
        }
        self.inputDirectory_display = QtWidgets.QLabel("No directory selected")

    def add_widgets_to_layout(self, layout: QtWidgets.QGridLayout):
        for i, (key, label) in enumerate(self.labels.items()):
            layout.addWidget(label, i, 0)
            if key in self.spinboxes:
                layout.addWidget(self.spinboxes[key], i, 1)
            elif key in self.checkboxes:
                layout.addWidget(self.checkboxes[key], i, 1)

        layout.addWidget(self.inputDirectory_display, len(self.labels), 1)
        layout.addWidget(self.button_setInputDirectory, len(self.labels) + 1, 0)
        layout.addWidget(self.button_startBatch, len(self.labels) + 1, 1)

    def connect_signals(self):
        for key, spinbox in self.spinboxes.items():
            spinbox.valueChanged.connect(lambda value, k=key: self.update_setting(k, value))

        for key, checkbox in self.checkboxes.items():
            checkbox.stateChanged.connect(lambda state, k=key: self.update_checkbox(k, state))

        self.button_setInputDirectory.clicked.connect(self.setInput_button)
        self.button_startBatch.clicked.connect(self.start_button)

    def update_setting(self, key: str, value: int):
        self.s[key] = value

    def update_checkbox(self, key: str, state: int):
        self.s[key] = bool(state)

    def setInput_button(self):
        self.inputDirectory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        if self.inputDirectory:
            self.inputDirectory_display.setText(f".../{os.path.basename(self.inputDirectory)}")

    def start_button(self):
        if not hasattr(self, 'inputDirectory') or not self.inputDirectory:
            QtWidgets.QMessageBox.warning(self, "Error", "Please select an input directory first.")
            return

        paramDict = {
            'slicesPerVolume': self.spinboxes['slicesPerVolume'].value(),
            'theta': self.spinboxes['theta'].value(),
            'baselineValue': self.spinboxes['baselineValue'].value(),
            'f0Start': self.spinboxes['f0Start'].value(),
            'f0End': self.spinboxes['f0End'].value(),
            'f0VolStart': self.spinboxes['f0VolStart'].value(),
            'f0VolEnd': self.spinboxes['f0VolEnd'].value(),
            'multiplicationFactor': self.spinboxes['multiplicationFactor'].value(),
            'shiftFactor': self.spinboxes['shiftFactor'].value(),
            'trim_last_frame': self.checkboxes['trim_last_frame'].isChecked(),
            'inputDirectory': self.inputDirectory,
            'subtractBaseline': self.checkboxes['subtractBaseline'].isChecked(),
            'runDFF0': self.checkboxes['runDFF0'].isChecked(),
            'runMultiplication': self.checkboxes['runMultiplication'].isChecked()
        }

        self.hide()
        try:
            g.win.camVolumeSlider.batchProcess(paramDict)
        except Exception as e:
            logging.error(f"Error during batch processing: {str(e)}")
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred during batch processing: {str(e)}")
        finally:
            self.close()

    def closeEvent(self, event: QtCore.QEvent):
        g.settings['volumeSlider'] = self.s
        super().closeEvent(event)

    def start(self):
        """
        Show the BatchOptions dialog.
        """
        self.show()



