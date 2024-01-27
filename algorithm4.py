def algorithm4(A, B, C, Y, UB, VBT, UC, VCT, dividend):
    Ap = A - B @ Y @ C
    At = UB.T @ Ap @ VCT.T 
    return VBT.T @ np.divide(At, dividend) @ UC.T + Y
