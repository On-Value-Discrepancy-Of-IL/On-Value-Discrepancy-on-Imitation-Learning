import numpy as np
import cvxopt
import time

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        # print('%r (%r, %r) %2.2f sec' % (method.__name__, args, kw, te-ts))
        print('%r %2.2f sec' % (method.__name__, te - ts))
        return result
    return timed


@timeit
def qp_irl(expert_fe, policy_fes):
    """
    min  1/2 x'Px + q'x
    s.t. Gx <= h
         Ax = b
    :param expert_fe:
    :param policy_fes:
    :return:
    """
    assert isinstance(expert_fe, np.ndarray) and expert_fe.ndim == 1
    m = len(expert_fe)  # w's dim
    for policy_fe in policy_fes:
        assert len(policy_fe) == m and isinstance(policy_fe, np.ndarray) and policy_fe.ndim == 1
    policy_fes = np.array(policy_fes)
    expert_fe = np.array(expert_fe)

    P = cvxopt.matrix(2.0*np.eye(m), tc='d')
    q = cvxopt.matrix(np.zeros(m), tc='d')
    G = cvxopt.matrix(np.concatenate([-expert_fe[None, :], policy_fes], axis=0))
    h = cvxopt.matrix(-np.ones(1 + len(policy_fes)), tc='d')

    solution = cvxopt.solvers.qp(P, q, G, h)
    weight = np.squeeze(np.asarray(solution['x']))
    print('qp solver objective function value: {}'.format(np.linalg.norm(weight)))
    weight /= np.linalg.norm(weight)
    assert weight.shape == (m, )
    return weight


if __name__ == '__main__':
    pass