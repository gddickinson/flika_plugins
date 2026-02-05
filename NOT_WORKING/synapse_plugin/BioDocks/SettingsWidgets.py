"""
@author: Brett Settle
@Department: UCI Neurobiology and Behavioral Science
@Lab: Parker Lab
@Date: August 6, 2015
"""
from .Tools import *
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
from types import FunctionType

import sys
if sys.version.startswith('3'):
	QtCore.Signal = QtCore.pyqtSignal

class MultiSpinBox(QtGui.QWidget):
	valueChanged = QtCore.Signal(object)
	def __init__(self, values):
		super(MultiSpinBox, self).__init__()
		self.layout = QtGui.QGridLayout(self)
		self.spins = []
		for i,v in enumerate(values):
			to_add = pg.SpinBox(value=v)
			to_add.valueChanged.connect(lambda : self.valueChanged.emit(self.value()))
			self.spins.append(to_add)
			self.layout.addWidget(to_add, 0, i)
		self.setLayout(self.layout)

	def value(self):
		return [spin.value() for spin in self.spins]

	def setValue(self, val):
		for i in range(len(self.spins)):
			self.spins[i].setValue(val[i])

class DropDownBox(QtGui.QComboBox):
	valueChanged = QtCore.Signal(object)
	def __init__(self, parent, options):
		super(DropDownBox, self).__init__(parent)
		self.addItems(options)
		self.opts = options
		self.setEditable(False)
		self.setCurrentIndex(0)
		self.currentIndexChanged.connect(lambda f: self.valueChanged.emit(self.opts[f]))

	def setValue(self, val):
		assert val in self.opts, 'Value %s is not an option in the dropdown of %s' % (val, self.opts)
		self.setCurrentIndex(self.opts.index(val))

	def value(self):
		return str(self.currentText())

class SliderSpinBox(QtGui.QWidget):
	valueChanged = QtCore.Signal(float)
	updating=False
	def __init__(self, spin_args={}):
		super(SliderSpinBox, self).__init__()
		self.layout = QtGui.QGridLayout(self)
		self.spin = pg.SpinBox(value=0, **spin_args)
		self.spin.valueChanged.connect(lambda : self.setValue(self.spin.value()))
		self.slide = QtGui.QSlider(QtGui.Qt.Horizontal)
		self.slide.setTickPosition(2)
		self.slide.setTickInterval(1)
		self.slide.valueChanged.connect(lambda : self.setValue(self.slide.value()))
		self.layout.addWidget(self.slide, 0, 0, 1, 4)
		self.layout.addWidget(self.spin, 0, 4)
		self.setLayout(self.layout)

	def value(self):
		return self.spin.value()

	def setValue(self, value):
		if not self.updating:
			self.updating = True
			self.spin.setValue(value)
			self.slide.setValue(value)
			self.valueChanged.emit(value)
			self.updating = False

class OptionsWidget(QtGui.QGroupBox):
	'''
	Widget to be kept open and change values on the fly. Different than ParameterWidget, has no end button
	'''
	valueChanged = QtCore.Signal(dict)
	def __init__(self, title, options, shape=None):
		'''
		accepts as options a list of dictinaries, with arguments (key, name, value). name is optionally set
		to key if omitted. If action is given, a button is created
		'''
		super(OptionsWidget, self).__init__(title=title)
		self.pos = (0, 0)
		count = len(options * 2)
		self.shape = shape if shape != None else (np.ceil(count / 6.), 6)
		self.layout = QtGui.QGridLayout(self)
		self.widgets = {}
		self.updating_all = False
		for var_dict in options:
			self.build_widget(var_dict)


	def update(self, d):
		self.updating_all = True
		for key, val in d.items():
			if key in self.widgets:
				self.update_value(key, val)
		self.updating_all = False
		self.valueChanged.emit(self.getOptions())

	def increment_pos(self):
		self.pos = [self.pos[0], self.pos[1] + 1]
		if self.pos[1] >= self.shape[1]:
			self.pos[1] = 0
			self.pos[0] += 1
			self.layout.setRowStretch(self.pos[0], 1)
		self.layout.setColumnStretch(self.pos[1], 1)

	def addWidget(self, widget, where):
		self.layout.addWidget(widget, *where)

	def get(self, key):
		if key in self.widgets:
			if isinstance(self.widgets[key], QtGui.QCheckBox):
				return self.widgets[key].isChecked()
			else:
				return self.widgets[key].value()
		return None

	def update_value(self, key, value):
		if isinstance(self.widgets[key], QtGui.QCheckBox):
			self.widgets[key].setChecked(value)
		else:
			self.widgets[key].setValue(value)
		if not self.updating_all:
			self.valueChanged.emit(self.getOptions())

	def getOptions(self):
		ret = {}
		for name, widget in self.widgets.items():
			ret[name] = self.get(name)
		return ret

	def build_widget(self, var_dict):
		key = var_dict['key']
		del var_dict['key']
		if 'name' in var_dict:
			name = var_dict['name']
			del var_dict['name']
		else:
			name = key
		skip = False
		if 'value' in var_dict:
			value = var_dict['value']
			del var_dict['value']
			if type(value) == bool:
				new_widget = QtGui.QCheckBox(name)
				new_widget.setChecked(value)
				new_widget.toggled.connect(lambda : self.valueChanged.emit(self.getOptions()))
				skip = True
			else:
				self.layout.addWidget(QtGui.QLabel(name + ": "), self.pos[0], self.pos[1])
				self.increment_pos()
				if isinstance(value, (int, float, np.generic)):
					new_widget = pg.SpinBox(value=value, **var_dict)
				elif isinstance(value, (tuple, list)):
					if all([isinstance(i, (float, int, np.generic)) for i in value]):
						new_widget = MultiSpinBox(values=value)
					else:
						new_widget = DropDownBox(parent=self, options=value)
				elif value == 'range':
					new_widget = SliderSpinBox(**var_dict)
				else:
					return
				new_widget.valueChanged.connect(lambda : self.valueChanged.emit(self.getOptions()))
			self.widgets[key] = new_widget
			self.layout.addWidget(new_widget, self.pos[0], self.pos[1])
			self.increment_pos()
		elif 'action' in var_dict:
			action = var_dict['action']
			button = QtGui.QPushButton(name)
			button.clicked.connect(action)
			self.layout.addWidget(button, self.pos[0], self.pos[1], 1, 2)
			self.increment_pos()
			skip = True
		if skip:
			self.increment_pos()


class ScalableGroup(pTypes.GroupParameter):
	'''Group used for entering and adding parameters to ParameterWidget'''
	def __init__(self, **opts):
		opts['type'] = 'group'
		opts['addText'] = "Add"
		opts['addList'] = ['str', 'float', 'int', 'color']
		pTypes.GroupParameter.__init__(self, **opts)

	def addNew(self, typ):
		val = {
			'str': '',
			'float': 0.0,
			'int': 0,
			'color': (255, 255, 255)
		}[typ]
		self.addChild(dict(name="New Parameter %d" % (len(self.childs)+1), type=typ, value=val, removable=True, renamable=True))

	def getItems(self):
		items = {}
		for k, v in self.names.items():
			if isinstance(v, ScalableGroup):
				items[k] = v.getItems()
			else:
				items[k] = v.value()
		return items

class ParameterWidget(QtGui.QWidget):
	'''Settings Widget takes a list of dictionaries and provides a widget to edit the values (and key names)

		Each dict object must include these keys:
			key: The string used to get/set the value, shown to left of value if 'name' not given
			value: The value to show next to name, matches 'type' if given else use type of this value
				accepts QColor, lambda, list, generic types

		Optional keys are:
			name: the name which is shown to the left of the value for user readability
			type: Specify the type if it is not obvious by the value provided, as a string
			suffix: (for float/int values only) added for user readability
			children: a list of dicts under a 'group' parameter
			removable: bool specifying if this can be removed (set to False)
			renamable: bool specifying if this can be renamed (set to False)
			appendable: bool specifying if user can add to this group

		To pass in a parent parameter with children, pass 'group' to the 'value' key and the list of dicts to a new 'children' parameter

	'''
	done = QtCore.Signal(dict)
	valueChanged = QtCore.Signal(str, object)
	def __init__(self, title, paramlist, about="", doneButton=False, appendable=False):
		super(ParameterWidget, self).__init__()
		self.ParamGroup = ScalableGroup if appendable else pTypes.GroupParameter

		self.hasDone = doneButton

		self.parameters = self.ParamGroup(name="Parameters", children=ParameterWidget.build_parameter_list(paramlist))
		self.parameters.sigTreeStateChanged.connect(self.paramsChanged)
		self.info = about
		self.tree = ParameterTree()
		self.tree.setParameters(self.parameters, showTop=False)
		self.tree.setWindowTitle(title)
		self.makeLayout()
		self.resize(800,600)

	@staticmethod
	def type_as_str(var):
		if type(var) == tuple and len(var) != 3:
			var = list(var)
		elif isinstance(var, np.string_):
			return 'str'
		elif isinstance(var, np.generic):
			var = float(var)
		elif isinstance(var, QtGui.QColor) or (type(var) == tuple and len(var) == 3):
			return 'color'
		elif isinstance(var, dict) or (isinstance(var, list) and all([type(i) == dict for i in var])):
			return 'group'
		elif isinstance(var, (int, float, bool, list, str)):
			return type(var).__name__
		elif isinstance(var, FunctionType):
			return 'action'
		return 'text'

	def paramsChanged(self, params, change):
		obj, change, val = change[0]
		if change == 'value':
			self.valueChanged.emit(obj.opts['key'], val)
		else:
			pass

	def makeLayout(self):
		layout = QtGui.QGridLayout()
		self.setLayout(layout)

		if len(self.info) > 0:
			self.scrollArea = QtGui.QScrollArea(self)
			self.scrollArea.setWidgetResizable(True)
			self.scrollArea.setWidget(QtGui.QLabel(self.info))
			layout.addWidget(self.scrollArea, 0,  0, 1, 2)

		layout.addWidget(self.tree, 1, 0, 1, 2)

		if self.hasDone:
			cancelButton = QtGui.QPushButton("Cancel")
			cancelButton.clicked.connect(self.close)
			okButton = QtGui.QPushButton("Ok")
			okButton.clicked.connect(lambda : self.close(emit=True))
			layout.addWidget(cancelButton, 2, 0)
			layout.addWidget(okButton, 2, 1)

		layout.setRowStretch(1, 4)

	@staticmethod
	def get_group_dict(groupParam):
		d = {}
		for c in groupParam.childs:
			if isinstance(c, pTypes.GroupParameter):
				d[c.opts['name']] = ParameterWidget.get_group_dict(c)
			else:
				d[c.opts['key']] = c.opts['value']
		return d

	@staticmethod
	def build_parameter_list(params):
		return_params = []
		for param_dict in params:

			assert 'key' in param_dict, 'Must provide a key for each item'
			assert 'value' in param_dict, 'Must provide a value for each item; %s does not have a value' % param_dict['key']
			if param_dict['value'] == None:
				continue
			if 'name' not in param_dict:
				param_dict['name'] = param_dict['key']

			if param_dict['value'] == 'group':
				return_params.append(pTypes.GroupParameter(name=param_dict['name'], children=ParameterWidget.build_parameter_list(param_dict['children'])))
				continue

			if 'type' not in param_dict:
				param_dict['type'] = ParameterWidget.type_as_str(param_dict['value'])

			if param_dict['type'] == 'list':
				param_dict['values'] = param_dict.pop('value')

			return_params.append(param_dict)
		return return_params

	def close(self, emit=False):
		super(ParameterWidget, self).close()
		if emit == True:
			self.done.emit(ParameterWidget.get_group_dict(self.parameters))


def edit_plotitem(plotitem = None, **base_data):
    if plotitem:
        opts = {k: v for k, v in plotitem.opts.items() if v != None}

    p = [{'key': 'name', 'name': 'Name', 'value': plotitem.__name__}]
    if isinstance(plotitem, pg.ScatterPlotItem):
        p.extend([{'key': 'symbol', 'name': 'Symbol', 'value': ['s', 't', 'd', 'o', '+']}, {'key': 'brush', 'name': 'Symbol Color', 'value': opts['brush'].color()}])
    elif 'stepMode' in plotitem.opts:
        p.extend([{'key': 'binSize', 'name': 'Bin Size', 'value': opts['binSize'], 'type': 'int'}, {'key': 'fillBrush', 'name': 'Fill Color', 'value': opts['fillBrush'].color()}, {'key':'stepMode', 'name':'Step Mode (Don\'t Change)', 'value': True}])
    else:
        p.extend([{'key': 'pen', 'name': 'Line Color', 'value': opts['pen'].color()}, {'key': 'width', 'name': 'Line Width', 'value': opts['width']}])
    par = ParameterWidget('Title', p, doneButton=True)
    par.show()
    par.done.connect(lambda d: updateItem(d, olditem=plotitem))

def updateItem(d, olditem=None):
    opts = d
    x, y = olditem.getData()
    if 'stepMode' in d:
        opts['fillLevel'] = 0
        data = y
        mi, ma = np.min(y), np.max(y)
        mi = opts['binSize'] * (mi // opts['binSize'])
        ma = opts['binSize'] * (1 + (ma // opts['binSize']))
        count = ((ma - mi) // opts['binSize']) + 1
        y, x = np.histogram(data, np.linspace(mi, ma, count))
        olditem.setData(x=x, y=y, **opts)
        olditem.__name__ = opts['name']
        return
    elif 'symbol' in d:
        opts['pen'] = opts['brush']
    #item = pg.PlotDataItem(data, **opts)
    olditem.__name__ = opts['name']
    olditem.setData(x, y, **opts)
