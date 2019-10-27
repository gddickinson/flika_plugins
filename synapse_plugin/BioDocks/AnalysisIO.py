"""
@author: Brett Settle
@Department: UCI Neurobiology and Behavioral Science
@Lab: Parker Lab
@Date: August 6, 2015
"""
import tifffile
from .bin2mat import *
#from PyQt4.QtCore import *
#from PyQt4.QtGui import *
from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy import QtWidgets
import xlrd, re, os
import numpy as np
try:
	from scipy.ndimage import imread
except:
	print("Imread could not be loaded. Some image files will not be openable")
import sys
#from PyQt4.QtCore import pyqtSignal as Signal
from qtpy.QtCore import Signal
last_dir = ''

def getHeaders(fname, delimiter='\t'):
	return [s.strip() for s in open(fname, 'r').readline().split(delimiter)]

def export3DArray(fname, arr, header = '', fmt='%-7.2f'):
	with open(fname, 'w') as outfile:
		outfile.write('%s\n' % header)
		for sub in arr:
			for row in sub:
				outfile.write(['\t'.join([fmt % i for i in row])])

def exportDictionaryList(filename, dict_list, delimiter="\t", order=None):
	'''export a list of dictionaries with similar keys '''
	if order == None:
		order = set(dict_list[0].keys())
		for d in dict_list:
			order &= set(d.keys())
		order = list(order)
	with open(filename, 'w') as outf:
		outf.write("%s\n" % delimiter.join(order))
		for d in dict_list:
			outf.write("%s\n" % delimiter.join([str(d[name]) for name in order]))

def getFilename(title='Select a file to open', **args):
	global last_dir
	to_ret = str(QtWidgets.QFileDialog.getOpenFileName(caption=title, directory=last_dir, **args)[0])
	last_dir = os.path.dirname(to_ret)
	return to_ret

def getFilenames(title='Select a file to open', **args):
	global last_dir
	to_ret = [str(i) for i in QtWidgets.QFileDialog.getOpenFileNames(caption=title, directory=last_dir, **args)]
	if len(to_ret) == 0:
		return
	last_dir = os.path.dirname(to_ret[0])
	return to_ret

def getSaveFilename(title='Save to file...', initial='', **args):
	global last_dir
	if initial != '':
		last_dir = os.path.join(last_dir, initial) if "\\" not in initial else initial
	savename = str(QtWidgets.QFileDialog.getSaveFileName(caption=title, directory = last_dir, **args))
	last_dir = os.path.dirname(savename)
	return savename

def getDirectory(caption, initial=''):
	return QtWidgets.QFileDialog.getExistingDirectory(None, caption, directory=initial)

def readTif(fname=''):
	if fname == '':
		fname = getFilename(filter = "All files (*.*);;TIF Files (*.tif)")
	if not os.path.isfile(fname):
		raise Exception("No File Selected")
	tif = tifffile.TIFFfile(fname).asarray().astype(np.float32)
	img = np.squeeze(tif)
	return img

def fileToArray(f=''):
	arr = []
	def fix_str_type(s):
		if re.match(r'^[\d\.]+$', s):
			return float(s)
		else:
			return str(s)
	if f == '':
		f = getFilename(filter = "All files (*.*);;Image Files (*.jpg, *.png);;TIF Files (*.tif);;Other Files (*.stk, *.lsm, *.nih)")
	if f.endswith(('.tif', '.nd2', '.nih', '.stk')):
		return read_tif(f)
	elif f.endswith(('.jpg', '.png')):
		arr = imread(f)
	elif f.endswith('.txt'):
		try:
			arr = np.loadtxt(f)
			breaks = [i for i, line in enumerate(open(f, 'r')) if line.strip() == '']
			breaks = np.subtract(breaks, range(len(breaks)))
			if len(breaks) > 0:
				arr = [a for a in np.split(arr, breaks) if a.size > 0]
		except Exception as e:
			l = 0
			for i, line in enumerate(open(f)):
				arr.append([fix_str_type(l.strip()) for l in line.split('\t')])
	elif f.endswith(('.xls', 'xlsx')):
		workbook = xlrd.open_workbook(f)
		arr = []
		for sheet in workbook.sheets():
			num_rows = sheet.nrows - 1
			curr_row = -1
			while curr_row < num_rows:
				curr_row += 1
				row = sheet.row(curr_row)
				arr.append([float(fix_str_type(str(s.value))) for s in row])
	elif f.endswith(('.npy', '.npz')):
		arr = np.load(f)
		if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
			arr = arr.transpose()
	else:
		print('Failed to open')
		return

	if type(arr) == list and any([type(j) == str for j in arr[0]]):
		names = arr[0]
		arr_dict = {n : [[]] for n in arr[0]}
		count = 0
		for row in arr[1:]:
			if all([i == '' for i in row]):
				count += 1
				for n in names:
					arr_dict[n].append([])
			else:
				for i, j in enumerate(row):
					arr_dict[names[i]][count].append(j)

		return {k: np.squeeze(v) for k,v in arr_dict.items()}
	else:
		return np.array(arr)

class ImageImporter(QThread):
	done = Signal(object)
	def __init__(self, filename=''):
		super(ImageImporter, self).__init__()
		if filename == '':
			filename = getFilename(filter = "All files (*.*);;Image Files (*.jpg, *.png);;TIF Files (*.tif);;Other Files (*.stk, *.lsm, *.nih)")
		self.filename = filename
		if self.filename == '':
			raise Exception('No file selected')

	def run(self):
		if self.filename.endswith(('.tif', '.nd2', '.nih', '.stk')):
			image = tifffile.TIFFfile(self.filename).asarray().astype(np.float32)
			image = np.squeeze(np.swapaxes(image, 2, 1))
		elif self.filename.endswith(('.jpg', '.png')):
			image = imread(self.filename).swapaxes(0, 1)
		elif self.filename.endswith(('.npy', '.npz')):
			image = np.load(self.filename)
			if simage.ndim == 2 and image.shape[0] > image.shape[1]:
				image = self.image.transpose()
		self.done.emit(image)
		del image
		self.terminate()

class TiffExporter(QThread):
	done = Signal()
	def __init__(self, fname, tiff):
		super(TiffExporterd, self).__init__()
		self.filename = fname
		self.tiff = tiff

	def run(self):
		tifffile.imsave(self.filename, self.tiff)
		self.done.emit()
		self.terminate()

def importDictionary(filename='', mode='columns'):
	if filename == '':
		filename = getFilename(filter = "Text files (*.txt)")
	keys = []
	data = []
	for i, line in enumerate(open(filename, 'r')):
		if i == 0:
			keys = line.split()
			if k[0].startswith('# '):
				k[0] = k[0][2:]
		else:
			data.append([float(j) for j in line.split()])
	data = np.array(data)
	if mode == 'columns':
		data = data.transpose()
		return OrderedDict([(keys[i], data[i]) for i in range(len(keys))])
	elif mode == 'rows':
		return [dict(zip(keys, i)) for i in data]

def importFile(filename, delimiter="\t", columns=[], evaluateLines=True):
	'''read info from a file, into a list of columns (specified by args) or dictionaries (specified by kargs)'''
	data = []
	if evaluateLines: 
		lines = [[evaluate(i) for i in line.split(delimiter)] for line in open(filename, 'r')]
	else:
		lines = [[i for i in line.split(delimiter)] for line in open(filename, 'r')]        

	if len(columns) == 0: # no columns given, return data as it is read from file
		return lines
	else:
		names = lines[0]    # read data to dictionary given columns
		lines = lines[1:]
		if all([type(i) == str for i in columns]):
			data = {}
			for n in columns:
				data[n] = [lines[i][names.index(n)] for i in range(len(lines))]
		elif all([type(i) == int for i in columns]):
			data = {}
			for i in columns:
				n = names[i]
				data[n] = [lines[j][i] for j in range(len(lines))]
		return data

def evaluate(i):
	try:
		return eval(i)
	except:
		return i
