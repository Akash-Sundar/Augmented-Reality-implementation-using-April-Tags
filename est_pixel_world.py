import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordiantes of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    """

    ##### STUDENT CODE START #####
    Pc = np.hstack((pixels, np.ones([pixels.shape[0],1])))
    R_wc = np.linalg.inv(R_wc)
    t_wc = -np.matmul(R_wc,t_wc).reshape(3,1)

    Rt = np.hstack((R_wc[:,0:-1],t_wc))
    Pw = np.zeros([Pc.shape[0],3])

    for i in range(pixels.shape[0]):
        Pw[i,:] = np.transpose(np.linalg.inv(K@Rt)@np.transpose(Pc[i,:]))
        Pw[i,:] = Pw[i,:]/Pw[i,-1]
        Pw[i,-1] = 0
    ##### STUDENT CODE END #####
    return Pw
