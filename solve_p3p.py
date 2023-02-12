import numpy as np
from itertools import combinations

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Invoke Procrustes function to find R, t
    # You may need to select the R and t that could transoform all 4 points correctly. 
    # R,t = Procrustes(Pc_3d, Pw[1:4])
    ##### STUDENT CODE END #####

    a = np.linalg.norm( Pw[2, :] - Pw[3, :] )
    b = np.linalg.norm( Pw[1, :] - Pw[3, :] )
    c = np.linalg.norm( Pw[1, :] - Pw[2, :] )

    u = Pc[:, 0]
    v = Pc[:, 1]

    cal_pts = (np.linalg.inv(K) @ np.hstack((u[:, None], v[:, None], np.ones((4, 1)))).T).T

    j = np.array([ (1 / np.linalg.norm(cal_pts[k, :])) * cal_pts[k, :] for k in range(0, 4) ])

    cos_alpha = j[2] @ j[3]
    cos_beta = j[1] @ j[3]
    cos_gamma = j[1] @ j[2]

    A4 = ((a**2 - c**2) / b**2 - 1)**2 - 4 * (c**2 / b**2) * cos_alpha**2
    A3 = 4 * (((a**2 - c**2) / b**2) * (1 - ((a**2 - c**2) / b**2)) * cos_beta - (1 - ((a**2 + c**2) / b**2)) * cos_alpha * cos_gamma + 2 * (c**2 / b**2) * (cos_alpha**2) * cos_beta)
    A2 = 2 * ( ((a**2 - c**2) / b**2)**2 - 1 + 2 * ((a**2 - c**2) / b**2)**2 * cos_beta**2 + 2 * ((b**2 - c**2) / b**2) * cos_alpha**2 - 4 * ((a**2 + c**2) / b**2) * cos_alpha * cos_beta * cos_gamma + 2 * ((b**2 - a**2) / b**2) * cos_gamma**2 )
    A1 = 4 * ( -((a**2 - c**2) / b**2) * (1 + ((a**2 - c**2) / b**2)) * cos_beta + 2 * (a**2 / b**2) * (cos_gamma**2) * cos_beta - (1 - ((a**2 + c**2) / b**2)) * cos_alpha * cos_gamma )
    A0 = (1 + (a**2 - c**2) / b**2)**2 - 4 * (a**2 / b**2) * cos_gamma**2

    A = np.array([A4, A3, A2, A1, A0])

    roots = np.roots(A)
    roots = roots[np.isreal(roots)]
    roots = roots[roots > 0]

    roots = np.real(roots)
    
    best_R = None
    best_t = None
    best_error = float('inf')
    for i in range(len(roots)):
        u = ((-1 + (a**2 - c**2) / b**2) * np.square(roots[i]) - 2 * ((a**2 - c**2) / b**2) * cos_beta * roots[i] + 1 + ((a**2 - c**2) / b**2)) / (2 * (cos_gamma - roots[i] * cos_alpha))

        s_1 = np.sqrt((b ** 2) / (1 + roots[i]**2 - 2 * roots[i] * cos_beta))
        
        s_2 = u * s_1
        
        s_3 = roots[i] * s_1

        p = np.vstack((s_1*j[1,:], s_2*j[2,:], s_3*j[3,:]))
        R, t = Procrustes(Pw[1:, :], p)

        # R = np.linalg.inv(R)
        # t = (-np.linalg.inv(R) @ t)
        # P = K @ np.hstack((R, t[:, None])) @ np.vstack((Pw[0, None].T, np.ones((1,1))))

        P = K @ (R @ Pw[0, :].T + t) #snp.vstack((R.T, -R.T@t.T)).T @ np.hstack((Pw[0], [1]))
        P = (P / P[-1])[:-1]
        
        # P = K @ np.hstack((np.linalg.inv(R), (-np.linalg.inv(R) @ t)[:, None])) @ np.vstack((Pw[0, None].T, np.ones((1,1))))
        error = np.linalg.norm(P - Pc[0, :])
        
        if error < best_error:
            best_error = error
            best_R = R
            best_t = t

    return np.linalg.inv(best_R), (-np.linalg.inv(best_R) @ best_t)

def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    ##### STUDENT CODE START #####

    X_bar = np.mean(X, axis = 0)
    Y_bar = np.mean(Y, axis = 0)

    X = (X - X_bar).T
    Y = (Y - Y_bar).T

    R = Y @ X.T

    [U, S, Vt] = np.linalg.svd(R)
    V = Vt.T

    M = np.identity(3)
    M[-1, -1] = np.linalg.det(V @ U.T)
    R = U @ M @ V.T
    #print(R)

    t = Y_bar - R @ X_bar
    t = np.reshape(t, [3,])

    ##### STUDENT CODE END #####

    return R, t
