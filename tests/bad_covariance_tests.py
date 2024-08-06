import torch
from easyvolcap.utils.test_utils import my_tests
from easyvolcap.utils.math_utils import normalize
from easyvolcap.utils.net_utils import typed, setup_deterministic
from easyvolcap.utils.fdgs_utils import build_4d_rotation, strip_symmetric
from easyvolcap.utils.console_utils import *


def test_bad_covariance():
    ql = torch.as_tensor([[0.9976122379302979, 0.003511250950396061, -0.028395723551511765, 0.06285887956619263]], device='cuda')
    qr = torch.as_tensor([[0.98433917760849, 0.014462025836110115, 0.0008921733242459595, -0.17568881809711456]], device='cuda')
    # ql = torch.ones_like(torch.as_tensor([[0.9976122379302979, 0.003511250950396061, -0.028395723551511765, 0.06285887956619263]], device='cuda'))
    # qr = torch.ones_like(torch.as_tensor([[0.98433917760849, 0.014462025836110115, 0.0008921733242459595, -0.17568881809711456]], device='cuda'))
    sc = torch.as_tensor([[0.0005611529923044145, 0.03982263803482056, 0.011404299177229404, 24.990943908691406]], device='cuda')

    L = sc.new_zeros(sc.shape[0], 4, 4)
    ql = normalize(ql)
    qr = normalize(qr)

    a, b, c, d = ql.unbind(-1)
    p, q, r, s = qr.unbind(-1)

    M_l = torch.stack([a, -b, -c, -d,
                       b, a, -d, c,
                       c, d, a, -b,
                       d, -c, b, a]).view(4, 4, -1).permute(2, 0, 1)
    M_r = torch.stack([p, q, r, s,
                       -q, p, -s, r,
                       -r, s, p, -q,
                       -s, -r, q, p]).view(4, 4, -1).permute(2, 0, 1)
    R = M_l @ M_r

    L[:, 0, 0] = sc[:, 0]
    L[:, 1, 1] = sc[:, 1]
    L[:, 2, 2] = sc[:, 2]
    L[:, 3, 3] = sc[:, 3]

    L = R @ L

    cov = L @ L.mT
    cov_11 = cov[:, :3, :3]
    cov_12 = cov[:, :3, 3:]
    cov_t = cov[:, 3:, 3:]

    # delta = 1e-5
    # cov[:, 0, 0] = cov[:, 0, 0] + delta
    # cov[:, 1, 1] = cov[:, 1, 1] + delta
    # cov[:, 2, 2] = cov[:, 2, 2] + delta
    # cov[:, 3, 3] = cov[:, 3, 3] + delta

    cov = cov_11 - cov_12 @ cov_12.mT / cov_t
    ms = cov_12 / cov_t

    cov = strip_symmetric(cov)  # 6,
    cov, ms, cov_t = cov, ms.squeeze(-1), cov_t.squeeze(-1)
    log(cov)

if __name__ == '__main__':
    my_tests(globals())
