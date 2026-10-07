import time
import numpy as np

H = W = 64
N = 10
Y, X = np.mgrid[0:H, 0:W]


def render_and_grad(params, target, sharpness):
    p = params.reshape(N, 7)
    q_list, m_list, before_list = [], [], []
    image = np.ones((3, H, W), dtype=np.float64)
    for row in p:
        x, y, r, cr, cg, cb, alpha = row
        z = (r * r - (X - x) ** 2 - (Y - y) ** 2) * sharpness
        # Stable enough for the parameter bounds used here.
        mask = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        q = mask * alpha
        before_list.append(image)
        q_list.append(q)
        m_list.append(mask)
        image = row[3:6, None, None] * q[None] + image * (1.0 - q[None])

    residual = image - target
    loss = np.mean(residual * residual)
    g = (2.0 / residual.size) * residual
    grad = np.zeros((N, 7), dtype=np.float64)
    for i in range(N - 1, -1, -1):
        row = p[i]
        x, y, r, cr, cg, cb, alpha = row
        q, mask, before = q_list[i], m_list[i], before_list[i]
        color = row[3:6, None, None]
        dloss_dq = np.sum(g * (color - before), axis=0)
        grad[i, 3:6] = np.sum(g * q[None], axis=(1, 2))
        dmask_dz = mask * (1.0 - mask)
        dloss_dz = dloss_dq * alpha * dmask_dz
        grad[i, 0] = np.sum(dloss_dz * (2.0 * sharpness * (X - x)))
        grad[i, 1] = np.sum(dloss_dz * (2.0 * sharpness * (Y - y)))
        grad[i, 2] = np.sum(dloss_dz * (2.0 * sharpness * r))
        grad[i, 6] = np.sum(dloss_dq * mask)
        g = g * (1.0 - q[None])
    return loss, grad.ravel()


def opt_adam(target, seed=0, stages=(0.12, 0.2, 0.35, 0.6, 1.0, 2.0, 4.0, 8.0, 20.0), iters=250, lr=0.15):
    rng = np.random.default_rng(seed)
    p = np.empty((N, 7), dtype=np.float64)
    p[:, 0:2] = rng.uniform(6, 58, size=(N, 2))
    p[:, 2] = rng.uniform(5, 17, size=N)
    p[:, 3:6] = rng.uniform(0.02, 0.98, size=(N, 3))
    p[:, 6] = rng.uniform(0.5, 1.0, size=N)
    low = np.array([0, 0, 3, 0, 0, 0, 0.5] * N, dtype=np.float64)
    high = np.array([63, 63, 22, 1, 1, 1, 1] * N, dtype=np.float64)
    reports = []
    for sharp in stages:
        m = np.zeros_like(p.ravel())
        v = np.zeros_like(m)
        best_p, best_loss = p.ravel().copy(), float('inf')
        for t in range(1, iters + 1):
            loss, grad = render_and_grad(p.ravel(), target, sharp)
            if loss < best_loss:
                best_loss, best_p = loss, p.ravel().copy()
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            mh = m / (1.0 - 0.9**t)
            vh = v / (1.0 - 0.999**t)
            p = (p.ravel() - lr * mh / (np.sqrt(vh) + 1e-8)).reshape(N, 7)
            p = np.clip(p.ravel(), low, high).reshape(N, 7)
        p = best_p.reshape(N, 7)
        reports.append((sharp, best_loss))
    return p, reports


def main():
    data = np.load('data/train_validation_sets.npz')
    xs = data['X_validation']
    start = time.time()
    scores = []
    for i, target in enumerate(xs):
        best = (float('inf'), None)
        for seed in range(6):
            p, reports = opt_adam(target.astype(np.float64), seed=seed, iters=100, lr=0.15)
            loss, _ = render_and_grad(p.ravel(), target, 20.0)
            if loss < best[0]: best = (loss, p)
            if best[0] <= 0.0075:
                break
        scores.append(best[0])
        print('sample', i, 'mse', best[0], 'restarts', seed + 1, 'elapsed', round(time.time()-start, 1), flush=True)
    print('VALIDATION mean MSE', float(np.mean(scores)), 'score', max(0.0, min(100.0, 100 * (0.02 - float(np.mean(scores))) / 0.01)), 'per-image', scores)


if __name__ == '__main__':
    main()
