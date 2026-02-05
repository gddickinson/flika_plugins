import numpy as np
from scipy.interpolate import interp2d
from skimage.transform import rescale

def inside_or_outside(x,y,z):
    radius = 40
    origin = [100, 100, 100]
    xr = x-origin[0]
    yr = y-origin[1]
    zr = z-origin[2]
    d=np.sqrt(xr**2+yr**2+zr**2)
    inside_or_ouside_image = d < radius
    inside_or_ouside_image = inside_or_ouside_image * (1+(z/50))  # make the sphere brightest on top
    return inside_or_ouside_image


def get_transformation_matrix(hx=0):
    """
    hx is the horizontal shear factor
    sy is the vertical scaling factor
    Look at the pdf in this folder.
    """
    hx = -hx
    sy = 1/np.sqrt(2)
    S = np.array([[1, hx, 0],
                  [0, sy, 0],
                  [0, 0, 1]])
    #S_inv = np.linalg.inv(S)
    #old_coords = np.array([[2, 2, 1], [6, 6, 1]]).T
    #new_coords = np.matmul(S, old_coords)
    #recovered_coords = np.matmul(S_inv, new_coords)
    #print('new coords: ', new_coords)
    #print('recovered coords: ', recovered_coords)
    return S


def create_circle():
    """ Create circle in original (x,y,z) coordinate system. """
    size = 500
    A = []
    y_idx, x_idx = np.indices((size, size))
    for i in np.arange(size):
        print(i)
        z_idx = np.ones((size,size))*i
        A.append(inside_or_outside(x_idx, y_idx, z_idx))
    A = np.array(A)
    A += 1
    circle = Window(A.swapaxes(1,2), name='Original')  # x and y axes are flipped from their matrix representations when displaying
    return circle


def test_transformation():
    circle = create_circle()
    S = get_transformation_matrix()
    S_inv = np.linalg.inv(S)
    C = circle.image[100, :, :]
    mx, my = C.shape
    new_max = np.matmul(S, np.array([mx, my, 1]))[:-1]
    new_mx, new_my = np.ceil(new_max).astype(np.int)
    all_new_coords = np.meshgrid(np.arange(new_mx), np.arange(new_my))
    new_coords = [all_new_coords[0].flatten(), all_new_coords[1].flatten()]
    new_homog_coords = np.stack([all_new_coords[0].flatten(), all_new_coords[1].flatten(), np.ones(new_mx * new_my)])
    old_coords = np.matmul(S_inv, new_homog_coords)
    old_coords = old_coords[:-1, :]
    old_coords = np.round(old_coords).astype(np.int)
    old_coords[0, old_coords[0, :] >= mx] = mx - 1
    old_coords[1, old_coords[1, :] >= my] = my - 1
    old_coords[0, old_coords[0, :] < 0] = 0
    old_coords[1, old_coords[1, :] < 0] = 0
    A = np.zeros((new_mx, new_my))
    A[new_coords[0], new_coords[1]] = C[old_coords[0], old_coords[1]]
    Window(A)


def get_transformation_coordinates(I, hx):
    negative_new_max = False
    S = get_transformation_matrix(hx)
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


A = g.win.image


A_rescaled = rescale(A, (2.0, 1.0))
A_rescaled = A_rescaled.swapaxes(1,2)
mt, my, mx = A_rescaled.shape
I = A_rescaled[:, :, 0]
old_coords, new_coords = get_transformation_coordinates(I, hx=1)
old_coords = np.round(old_coords).astype(np.int)
new_mx, new_my = np.max(new_coords[0])+1, np.max(new_coords[1])+1
#I_transformed = np.zeros((new_mx, new_my))
#I_transformed[new_coords[0], new_coords[1]] = I[old_coords[0], old_coords[1]]
#Window(I_transformed)

import time
tic = time.time()
D = np.zeros((new_mx, new_my, mx))
D[new_coords[0], new_coords[1], :] = A_rescaled[old_coords[0], old_coords[1], :]
#Window(D)
toc = time.time() - tic
print(toc)
D = D.swapaxes(1, 0)
D = D.swapaxes(1, 2)
D = np.flip(D, 0)
Window(D)






mz_new, _ = zoom(B[0, :, 0, :], (1 / np.sqrt(2), 1)).shape
C = np.zeros((mv, mz_new, mx, my), dtype=B.dtype)
for v in np.arange(mv):
    for x in np.arange(mx):
        C[v, :, x, :] = zoom(B[v, :, x, :], (1 / np.sqrt(2), 1), order=0)  # squash the z axis pixel size by sqrt(2)
mv, mz, mx, my = C.shape

newy = my + mz  # because we will be shifting each x-y plane in the y direction by one pixel, the resulting size will be my plus the number of x-y planes (mz)
D = np.zeros((mv, mz, mx, newy), dtype=A.dtype)
shifted = 0
for z in np.arange(mz):
    minus_z = mz - z
    shifted = minus_z
    D[:, z, :, shifted:shifted + my] = C[:, z, :, :]
D = D[:, ::-1, :, :]  # (mv, mz, mx, my)

g.m.statusBar().showMessage("Successfully generated movie ({} s)".format(time() - t))
w = Window(np.squeeze(D[:, 0, :, :]), name=self.oldname)






















### RECREATE THE MOVIE FROM THE OBJECTIVE'S PERSPECTIVE

size = 500
A = []
zp_idx, x_idx = np.indices((size, size))
zp_idx = np.flipud(zp_idx)
zp_idx = zp_idx / np.sqrt(2)
for c in np.arange(80,250):
    print(c)
    yp_idx = np.flipud(zp_idx) - c
    A.append(inside_or_outside(x_idx, yp_idx, zp_idx))
A = np.array(A)
A = A.swapaxes(1,2)
A += 1
Window(A, name='Perspective from light sheet') # x and y axes are flipped from their matrix representations when displaying



### PUT IT BACK TOGETHER AGAIN



mt, mx, my = A.shape
nSteps = mt  #remove this
shift_factor = 1 #remove this
mv = int(np.floor(mt / nSteps))  # number of volumes
A = A[:mv * nSteps]
B = np.reshape(A, (mv, nSteps, mx, my))
B = B.swapaxes(1,3)  # the direction we step is going to be the new y axis, whereas the old y axis will eventually become the z axis
B = np.repeat(B,shift_factor,axis=3)   # We need to stretch the y axis pixels (which were the step size) so that one new y pixel is the same as a pixel in the x direction. Hopefully before this transformation, the step size (ums) is an integer multiple of the x pixel size (um).
# Now our matrix is in terms of (mv, mz, mx, my).
mv,mz,mx,my = B.shape

C = np.zeros((mv, np.ceil(mz/np.sqrt(2)), mx, my), dtype=B.dtype)
for v in np.arange(mv):
    for x in np.arange(mx):
        C[v,:,x,:] = zoom(B[v,:,x,:], (1/np.sqrt(2), 1), order=0)  # squash the z axis pixel size by sqrt(2)
mv,mz,mx,my = C.shape

newy = my+mz # because we will be shifting each x-y plane in the y direction by one pixel, the resulting size will be my plus the number of x-y planes (mz)
D = np.zeros((mv, mz, mx, newy), dtype=A.dtype)
shifted=0
for z in np.arange(mz):
    minus_z = mz-z
    shifted = minus_z
    D[:,z,:,shifted:shifted+my] = C[:,z,:,:]
#D=D[:,::-1,:,:] # (mv, mz, mx, my)

Window(np.squeeze(D))
