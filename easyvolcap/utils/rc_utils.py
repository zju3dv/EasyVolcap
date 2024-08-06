import numpy as np


def ixt_to_rc(K: np.ndarray, D: np.ndarray, H: int, W: int, model='brown3t2'):
    FocalLength35mm = K[0, 0] / W * 36
    Skew = K[0, 1]
    AspectRatio = K[1, 1] / K[0, 0]
    PrincipalPointU = (K[0, 2] / (W / 2) - 1) / 2
    PrincipalPointV = (K[1, 2] / (H / 2) - 1) / 2

    if model == 'brown3t2':
        if D.shape[0] == 1: D = D.T
        if D.shape[0] == 4: D = np.concatenate([D, np.zeros_like(D[:1])]) # 5, 1
        DistortionCoeficients = [D[0, 0], D[1, 0], D[4, 0], 0, D[3, 0], D[2, 0]]
    else:
        raise NotImplementedError(f'Unknown distortion model: {model}')

    return FocalLength35mm, Skew, AspectRatio, PrincipalPointU, PrincipalPointV, DistortionCoeficients


def ext_to_rc(R: np.ndarray, T: np.ndarray):
    C = (-R.T @ T)  # 3, 1
    # C[1, :], C[2, :] = C[2, :], -C[1, :]
    Position = C.ravel().tolist()
    # R[:, 1], R[:, 2] = R[:, 2], -R[:, 1]
    Rotation = R.ravel().tolist()
    return Rotation, Position
