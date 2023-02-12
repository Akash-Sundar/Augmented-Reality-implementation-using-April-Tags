import numpy as np

def est_homography(X, X_prime):
    """
    Calculates the homography of two planes, from the plane defined by X
    to the plane defined by X_prime. In this assignment, X are the coordinates of the
    four corners of the soccer goal while X_prime are the four corners of the penn logo

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        X_prime: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. X_prime ~ H*X

    """

    ##### STUDENT CODE START #####

    #X_prime ~ H X
    #a_x = [-x, -y, -1, 0, 0, 0, x*x_dash, y*x_dash, x_dash]
    #a_y = [0, 0, 0, -x, -y, -1, x*y_dash, y*y_dash, y_dash]
    #h = [h11, h12, h13, h21, h22, h23, h31, h32, h33]

    X = np.array(X)
    X_prime = np.array(X_prime)

    A = np.zeros([X_prime.shape[0]*2, 9])
    
    for i in range(0,X_prime.shape[0]):
        x,y,z= X[i]
        x_dash, y_dash = X_prime[i]
        a_x = [-x, -y, -1, 0, 0, 0, x*x_dash, y*x_dash, x_dash]
        a_y = [0, 0, 0, -x, -y, -1, x*y_dash, y*y_dash, y_dash]
        A[2*i] = a_x
        A[2*i+1] = a_y

    #print(A.shape)
    
    H = np.zeros([9,0])
    
    [U, S, Vt] = np.linalg.svd(A)
    V = np.transpose(Vt)
    H = V[:,-1]

    #print(V.shape)
    #print(np.dot(A,H))
    
    H = np.reshape(H,[3,3])

    ##### STUDENT CODE END #####

    return H
