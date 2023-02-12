from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Homography Approach
    # Following slides: Pose from Projective Transformation

    #Pw[:,0] = Pw[:,0]/ Pw[:,-1]
    #Pw[:,1] = Pw[:,1]/ Pw[:,-1]
    #Pw = np.array(Pw[:,:2])
  
    H = est_homography(Pw, Pc)

    normalising_const = H[-1,-1]
    H = H/normalising_const

    H_dash = np.linalg.inv(K)@H

    #print(H)
    
    h1 = H_dash[:,0]
    h2 = H_dash[:,1]
    h3 = np.cross(h1, h2, axis = 0)
    

    tmp = np.array([h1, h2, h3]).T

    [U, S, Vt] = np.linalg.svd(tmp)
    V = np.transpose(Vt)
    
    M = np.identity(3)
    M[-1, -1] = np.linalg.det(U @ V.T)
    R = U @ M @ V.T 
      
    #r1 = R[:,0]
    #r2 = R[:,1]

    t = H_dash[:,2] / np.linalg.norm(h1)
    
    #Converting Rcw to Rwc
    #R = np.linalg.inv(R)
    R = np.linalg.inv(R)
    t =  - R @ t

    ##### STUDENT CODE END #####

    return R, t
