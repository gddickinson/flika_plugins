"""
@converted by: Brett Settle
@Department: UCI Neurobiology and Behavioral Science
@Lab: Parker Lab
@Date: August 6, 2015
"""
import numpy as np
import struct
from collections import defaultdict
# MList will be one array of (structures of arrays)

class BinaryReaderEOFException(Exception):
    def __init__(self):
        pass
    def __str__(self):
        return 'Not enough bytes in file to satisfy read request'

class BinaryReader:
    # Map well-known type names into struct format characters.
    typeNames = {
        'int8'   :'b',
        'uint8'  :'B',
        'int16'  :'h',
        'uint16' :'H',
        'int32'  :'i',
        'uint32' :'I',
        'int64'  :'q',
        'uint64' :'Q',
        'float'  :'f',
        'double' :'d',
        'char'   :'s',
        'single' :'f'}

    def __init__(self, fileName):
        self.file = open(fileName, 'rb')

    def seek(self, *args):
        return self.file.seek(*args)

    def tell(self, *args):
        return self.file.tell()

    def read_bytes(self, i):
        return self.file.read(i)

    def read(self, typeName):
        typeFormat = BinaryReader.typeNames[typeName.lower()]
        typeSize = struct.calcsize(typeFormat)
        value = self.file.read(typeSize)
        if typeSize != len(value):
            raise BinaryReaderEOFException
        a =  struct.unpack(typeFormat, value)[0]
        return a

    def close(self):
        self.file.close()


def bin2mat(infile):
    if not infile.endswith('.bin'):
        raise Exception('Not a bin file')
    sizeofminfo=72      # 72 bytes per minfo

    fid = BinaryReader(infile) # % file id or file identifier
    fid.seek(0, 2);
    file_length = fid.tell()
    fid.seek(0, 0);

    # read header
    version = "".join([fid.read('char') for i in range(4)]) # *char % M425
    frames = fid.read('int32') # *int32 ;% number of frames. real frames.
    status = fid.read('int32') # *int32 ;% identified = 2, stormed = 6, traced =3, tracked = 4
    header_length = fid.tell()

    nmol=np.zeros(frames + 1) #% nmol matrix stores the number of molecules in each frame;
    nmol[0]= fid.read('int32') #% number of molecules in the 0th frame (master list)
    fnames = ['x','y','xc','yc','h','a','w','phi','ax','bg','i','c','density',\
        'frame','length','link','z','zc','selfframe']# %length=index. density = valid = fit iterations
    #x, y, xc, yc in pixels.
    #z and zc in nanometer

    ftypes = ['single','single','single','single','single','single','single',\
        'single','single','single','single','int32','int32','int32','int32',\
        'int32','single','single','single']
    lengthfnames=np.size(fnames)
    MList = []
    for f in np.arange(frames):
        fid.seek(sizeofminfo*nmol[f], 1)
        nmol[f+1]=fid.read('int32')

    nmolcum=np.cumsum(nmol)

    if nmolcum[-1]==nmolcum[0]: #% this means molecule lists in 1, 2, ... frames do not exist
        keepframeinfo=0
    else:
        keepframeinfo=1

    # the byte offset of the last molecule
    #testoffset= header_length  + (nmolcum(frames)+nmol(frames+1)-1)*sizeofminfo + (frames+1)*4;

    for index in range(int(nmol[0])):
        fid.seek(header_length+4+(index) *sizeofminfo+14*4, 0)
        length = fid.read('int16')
        if not keepframeinfo:
            length=0

        fid.seek(header_length+4+(index) *sizeofminfo, 0)
        MList.append(dict())
        for k in range(lengthfnames):
            MList[index][fnames[k]] = defaultdict(lambda f: 0)
        for k in range(lengthfnames - 1): #% disregard selfframe for now
            MList[index][fnames[k]][0] = fid.read(ftypes[k])
        MList[index]['selfframe'][0] = 0# % take care of selfframe
        fr = MList[index]['frame'][0]
        lk = MList[index]['link'][0]
        f=1
        while lk != -1: #% link = -1 means there is no "next appearance"
            offset = header_length  + (nmolcum[fr-1]+lk)*sizeofminfo + (fr+1)*4 # % from Insight3: fr is for real. link = 3 means its next appearance is the 4-th molecule in the fr-th frame.
            fid.seek (int(offset), 0)
            for k in range(lengthfnames - 1): #% disregard selfframe for now
                MList[index][fnames[k]][f] = fid.read(ftypes[k])
            MList[index]['selfframe'][f] = fr
            fr = MList[index]['frame'][f]
            lk = MList[index]['link'][f]
            f += 1
    for i in range(len(MList)):
        for k,v in MList[i].items():
            MList[i][k] = list(v.values())
        MList[i]['xmean'] = np.average(MList[i]['x'])
        MList[i]['ymean'] = np.average(MList[i]['y'])
    fid.close()
    print("Loaded %s molecules" % len(MList))
    return MList

if __name__ == "__main__":
    bin2mat('E:\\mousevover-expressed\\over-expressed\\trial_2_before.bin')
