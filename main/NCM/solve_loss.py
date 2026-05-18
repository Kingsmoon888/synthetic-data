import numpy as np
def solve_loss(y,XL,ZL,XU,ZU,l1,l2,l3,l4,l5):
    """
    XL: N x M
    ZL: N x P
    XU: K x M
    ZU: K x P
    assume prior is flat, need to be updated
    return alpha, beta
    """
    N = np.shape(XL)[0]
    M = np.shape(XL)[1]
    P = np.shape(ZL)[1]
    A_matrix = np.zeros((M+P,M+P))
    A_matrix[:M,:M] = (l1+l3)*np.matmul(XL.T,XL)+l4*np.matmul(XU.T,XU)
    A_matrix[:M,M:] = -1 * (l3 * np.matmul(XL.T,ZL) + l4 * np.matmul(XU.T,ZU))
    A_matrix[M:,:M] = -1 * (l3 * np.matmul(ZL.T,XL) + l4 * np.matmul(ZU.T,XU))
    A_matrix[M:,M:] = (l2+l3)*np.matmul(ZL.T,ZL)+l4*np.matmul(ZU.T,ZU)
    #print(A_matrix)
    b = np.zeros((M+P,1))
    b[:M] = l1*np.matmul(XL.T,y)
    b[M:] = l2*np.matmul(ZL.T,y)
    solution = np.matmul(np.linalg.inv(A_matrix),b)
    return solution[:M],solution[M:]

def solve_loss_l5(y,XL,ZL,XU,ZU,l1,l2,l3,l4,l5):
    """
    XL: N x M
    ZL: N x P
    XU: K x M
    ZU: K x P
    assume prior is l5
    """
    N = np.shape(XL)[0]
    M = np.shape(XL)[1]
    P = np.shape(ZL)[1]
    A_matrix = np.zeros((M+P,M+P))

    A_matrix[:M,:M] = (l1+l3)*np.matmul(XL.T,XL)+l4*np.matmul(XU.T,XU) + np.diag([l5]*M)



    A_matrix[:M,M:] = -1 * (l3 * np.matmul(XL.T,ZL) + l4 * np.matmul(XU.T,ZU))
    A_matrix[M:,:M] = -1 * (l3 * np.matmul(ZL.T,XL) + l4 * np.matmul(ZU.T,XU))
    A_matrix[M:,M:] = (l2+l3)*np.matmul(ZL.T,ZL)+l4*np.matmul(ZU.T,ZU)
    #print(A_matrix)
    b = np.zeros((M+P,1))
    b[:M] = l1*np.matmul(XL.T,y)
    b[M:] = l2*np.matmul(ZL.T,y)
    solution = np.matmul(np.linalg.inv(A_matrix),b)
    return solution[:M],solution[M:]


if __name__=="__main__":
    l1 = 0.1
    l2 = 0.1
    l3 = 0.2
    l4 = 0.5
    l5 = 0.4

    XU = np.array([[1,2,3],[1,4,5]])
    ZU = np.array([[1,2],[1,4]])
    ZL = np.array([[1,3],[1,8],[1,9]])
    XL = np.array([[1,3,9],[1,8,7],[1,9,5]])

    y = np.array([1,2,3]).reshape(3,1)

    alpha,beta = solve_loss_l5(y,XL,ZL,XU,ZU,l1,l2,l3,l4,l5)
    print(alpha,beta)
