# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import re
import h5py
#from .helperFunctions import perform_shear_transform
from numpy import moveaxis
from skimage.transform import rescale
import logging
logger = logging.getLogger(__name__)

A_path = r"C:\Users\George\Desktop\overlayImages\puffsArray.npy"
A = np.load(str(A_path))
print(A.shape)

def get_transformation_matrix(theta=45):
    """
    theta is the angle of the light sheet
    """

    theta = theta/360 * 2 * np.pi # in radians
    hx = np.cos(theta)
    sy = np.sin(theta)

    S = np.array([[1, hx, 0],
                  [0, sy, 0],
                  [0, 0, 1]])

    return S


def get_transformation_coordinates(I, theta):
    negative_new_max = False
    S = get_transformation_matrix(theta)
    S_inv = np.linalg.inv(S)
    mx, my = I.shape

    four_corners = np.matmul(S, np.array([[0, 0, mx, mx],
                                          [0, my, 0, my],
                                          [1, 1, 1, 1]]))[:-1,:]
    range_x = np.round(np.array([np.min(four_corners[0]), np.max(four_corners[0])])).astype(np.int)
    range_y = np.round(np.array([np.min(four_corners[1]), np.max(four_corners[1])])).astype(np.int)
    all_new_coords = np.meshgrid(np.arange(range_x[0], range_x[1]), np.arange(range_y[0], range_y[1]))
    new_coords = [all_new_coords[0].flatten(), all_new_coords[1].flatten()]
    new_homog_coords = np.stack([new_coords[0], new_coords[1], np.ones(len(new_coords[0]))])
    old_coords = np.matmul(S_inv, new_homog_coords)
    old_coords = old_coords[:-1, :]
    old_coords = old_coords
    old_coords[0, old_coords[0, :] >= mx-1] = -1
    old_coords[1, old_coords[1, :] >= my-1] = -1
    old_coords[0, old_coords[0, :] < 1] = -1
    old_coords[1, old_coords[1, :] < 1] = -1
    new_coords[0] -= np.min(new_coords[0])
    keep_coords = np.logical_not(np.logical_or(old_coords[0] == -1, old_coords[1] == -1))
    new_coords = [new_coords[0][keep_coords], new_coords[1][keep_coords]]
    old_coords = [old_coords[0][keep_coords], old_coords[1][keep_coords]]
    return old_coords, new_coords


def perform_shear_transform(A, shift_factor, interpolate, datatype, theta, inputArrayOrder = [0, 3, 1, 2], displayArrayOrder = [3, 0, 1, 2]):
    A = moveaxis(A, inputArrayOrder, [0, 1, 2, 3]) # INPUT
    m1, m2, m3, m4 = A.shape
    if interpolate:
        A_rescaled = np.zeros((m1*int(shift_factor), m2, m3, m4))
        for v in np.arange(m4):
            print('Upsampling Volume #{}/{}'.format(v+1, m4))
            #g.m.statusBar().showMessage('Upsampling Volume #{}/{}'.format(v+1, m4))
            A_rescaled[:, :, :, v] = rescale(A[:, :, :, v], (shift_factor, 1.), mode='constant', preserve_range=True)
    else:
        A_rescaled = np.repeat(A, shift_factor, axis=1)
    mx, my, mz, mt = A_rescaled.shape
    I = A_rescaled[:, :, 0, 0]
    old_coords, new_coords = get_transformation_coordinates(I, theta)
    old_coords = np.round(old_coords).astype(np.int)
    new_mx, new_my = np.max(new_coords[0]) + 1, np.max(new_coords[1]) + 1

    D = np.zeros((new_mx, new_my, mz, mt))
    D[new_coords[0], new_coords[1], :, :] = A_rescaled[old_coords[0], old_coords[1], :, :]
    E = moveaxis(D, [0, 1, 2, 3], displayArrayOrder) # AXIS INDEX CHANGED FROM INPUT TO MATCH KYLE'S CODE
    #E = np.flip(E, 1)

    return E

shift_factor=1
interpolate=False
theta=45
inputArrayOrder=[0, 3, 1, 2]
displayArrayOrder=[3, 0, 1, 2]

B = perform_shear_transform(A, shift_factor, interpolate, A.dtype, theta, inputArrayOrder=inputArrayOrder,displayArrayOrder=displayArrayOrder)

print(B.shape)

B = moveaxis(B,0,1)
print(B.shape)

B = moveaxis(B,2,3)

B = np.expand_dims(B,axis=1)
print(B.shape)

B = B.astype(A.dtype)

#def make_thumbnail(array, size=256):
#    """ array should be 4D array """
#    # TODO: don't just crop to the upper left corner
#    mip = np.squeeze(array).max(1)[:3, :size, :size].astype(np.float)
#    for i in range(mip.shape[0]):
#        mip[i] -= np.min(mip[i])
#        mip[i] *= 255 / np.max(mip[i])
#    mip = np.pad(mip, ((0, 3 - mip.shape[0]),
#                       (0, size - mip.shape[1]),
#                       (0, size - mip.shape[2])
#                       ), 'constant', constant_values=0)
#    mip = np.pad(mip, ((0, 1), (0, 0), (0, 0)), 'constant',
#                 constant_values=255).astype('|u1')
#    return np.squeeze(mip.T.reshape(1, size, size * 4)).astype('|u1')

def make_thumbnail(array, size=256):
    empty = np.zeros((size,size))
    clip = array[0][0].astype(np.float)[0:size,0:size]   
    #TODO
    mip = empty
    return mip.astype('|u1')


def h5str(s, coding='ASCII', dtype='S1'):
    return np.frombuffer(str(s).encode(coding), dtype=dtype)


def subsample_data(data, subsamp):
    return data[0::int(subsamp[0]), 0::int(subsamp[1]), 0::int(subsamp[2])]

C = np.squeeze(B,axis=1)
B_thumb = make_thumbnail(C)
plt.imshow(B_thumb)


########################


def np_to_ims(array, fname='myfile.ims',
              subsamp=((1, 1, 1), (1, 2, 2)),
              chunks=((16, 128, 128), (64, 64, 64)),
              compression='gzip',
              thumbsize=256,
              dx=0.1, dz=0.25):

    assert len(subsamp) == len(chunks)
    assert all([len(i) == 3 for i in subsamp]), 'Only deal with 3D chunks'
    assert all([len(i) == len(x) for i, x in zip(subsamp, chunks)])
    assert compression in (None, 'gzip', 'lzf', 'szip'), 'Unknown compression type'
    if not fname.endswith('.ims'):
        fname = fname + '.ims'

    # force 5D
    if not array.ndim == 5:
        array = array.reshape(tuple([1] * (5 - array.ndim)) + array.shape)


    nt, nc, nz, ny, nx = array.shape
    print('array: ',nt,nc,nz,ny,nx)
    nr = len(subsamp)

    GROUPS = [
        'DataSetInfo',
        'Thumbnail',
        'DataSetTimes',
        'DataSetInfo/Imaris',
        'DataSetInfo/Image',
        'DataSetInfo/TimeInfo'
    ]

    ATTRS = [
        ('/', ('ImarisDataSet', 'ImarisDataSet')),
        ('/', ('ImarisVersion', '5.5.0')),
        ('/', ('DataSetInfoDirectoryName', 'DataSetInfo')),
        ('/', ('ThumbnailDirectoryName', 'Thumbnail')),
        ('/', ('DataSetDirectoryName', 'DataSet')),
        ('DataSetInfo/Imaris', ('Version', '8.0')),
        ('DataSetInfo/Imaris', ('ThumbnailMode', 'thumbnailMIP')),
        ('DataSetInfo/Imaris', ('ThumbnailSize', thumbsize)),
        ('DataSetInfo/Image', ('X', nx)),
        ('DataSetInfo/Image', ('Y', ny)),
        ('DataSetInfo/Image', ('Z', nz)),
        ('DataSetInfo/Image', ('NumberOfChannels', nc)),
        ('DataSetInfo/Image', ('Noc', nc)),
        ('DataSetInfo/Image', ('Unit', 'um')),
        ('DataSetInfo/Image', ('Description', 'description not specified')),
        ('DataSetInfo/Image', ('MicroscopeModality', '',)),
        ('DataSetInfo/Image', ('RecordingDate', '2018-05-24 20:36:07.000')),
        ('DataSetInfo/Image', ('Name', 'name not specified')),
        ('DataSetInfo/Image', ('ExtMin0', '0')),
        ('DataSetInfo/Image', ('ExtMin1', '0')),
        ('DataSetInfo/Image', ('ExtMin2', '0')),
        ('DataSetInfo/Image', ('ExtMax0', nx * dx)),
        ('DataSetInfo/Image', ('ExtMax1', ny * dx)),
        ('DataSetInfo/Image', ('ExtMax2', nz * dz)),
        ('DataSetInfo/Image', ('LensPower', '63x')),
        ('DataSetInfo/TimeInfo', ('DatasetTimePoints', nt)),
        ('DataSetInfo/TimeInfo', ('FileTimePoints', nt)),
    ]

    COLORS = ('0 1 0', '1 0 1', '1 1 0', '0 0 1')
    for c in range(nc):
        grp = 'DataSetInfo/Channel %s' % c
        GROUPS.append(grp)
        ATTRS.append((grp, ('ColorOpacity', 1)))
        ATTRS.append((grp, ('ColorMode', 'BaseColor')))
        ATTRS.append((grp, ('Color', COLORS[c % len(COLORS)])))
        ATTRS.append((grp, ('GammaCorrection', 1)))
        ATTRS.append((grp, ('ColorRange', '0 255')))
        ATTRS.append((grp, ('Name', 'Channel %s' % c)))
        ATTRS.append((grp, ('LSMEmissionWavelength', 0)))
        ATTRS.append((grp, ('LSMExcitationWavelength', '')))
        ATTRS.append((grp, ('Description', '(description not specified)')))

    # TODO: create accurate timestamps
    for t in range(nt):
        m, s = divmod(t, 60)
        h, m = divmod(m, 60)
        strr = '2018-05-24 {:02d}:{:02d}:{:02d}.000'.format(h, m, s)
        ATTRS.append(('DataSetInfo/TimeInfo', ('TimePoint{}'.format(t + 1), strr)))

    with h5py.File(fname, 'a') as hf:
        for grp in GROUPS:
            hf.create_group(grp)

        for grp, (key, value) in ATTRS:
            hf[grp].attrs.create(key, h5str(value))

        try:
            thumb = make_thumbnail(array[0], thumbsize)
            hf.create_dataset('Thumbnail/Data', data=thumb, dtype='u1')
        except Exception:
            logger.warn('Failed to generate Imaris thumbnail')

        # add data
        fmt = '/DataSet/ResolutionLevel {r}/TimePoint {t}/Channel {c}/'
        for t in range(nt):
            for c in range(nc):
                data = np.squeeze(array[t, c])
                for r in range(nr):
                    if any([i > 1 for i in subsamp[r]]):
                        data = subsample_data(data, subsamp[r])
                    hist, edges = np.histogram(data, 256)
                    grp = hf.create_group(fmt.format(r=r, t=t, c=c))
                    print("Writing: %s" % grp)
                    grp.create_dataset('Histogram', data=hist.astype(np.uint64))
                    grp.attrs.create('HistogramMin', h5str(edges[0]))
                    grp.attrs.create('HistogramMax', h5str(edges[-1]))
                    grp.create_dataset('Data', data=data,
                                       chunks=tuple(min(*n) for n in zip(chunks[r], data.shape)),
                                       compression=compression)
                    grp.attrs.create('ImageSizeX', h5str(data.shape[2]))
                    grp.attrs.create('ImageSizeY', h5str(data.shape[1]))
                    grp.attrs.create('ImageSizeZ', h5str(data.shape[0]))

    return fname


def unmap_bdv_from_imaris(hf):
    for i in hf:
        if re.match(r'^t\d{5}$', i) or re.match(r'^s\d{2}$', i):
            del hf[i]
    return

kwargs = {'fname':'myfile.ims',
              'subsamp':((1, 1, 1), (1, 2, 2)),
              'chunks':((16, 128, 128), (64, 64, 64)),
              'compression':'gzip',
              'thumbsize':256,
              'dx':0.1, 'dz':0.25}

testIMS = np_to_ims(B, **kwargs)