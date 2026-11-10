"""
================================================================================
 24 — n-STEP SEMI-GRADIENT SARSA                        (Sutton & Barto, Ch. 10.2)
================================================================================

BÀI TOÁN
    Ghép n-step return (chương 7) với xấp xỉ hàm (chương 10).

Ý TƯỞNG
    Chỉ thay target của semi-gradient Sarsa bằng n-step return:

        G_{t:t+n} = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n}
                    + γ^n q̂(S_{t+n}, A_{t+n}, w)

        w ← w + α [ G_{t:t+n} − q̂(S_t,A_t,w) ] ∇q̂(S_t,A_t,w)

    Lợi ích: giảm chệch do bootstrap, đồng thời vẫn dùng xấp xỉ hàm.
    Đây là "công thức chuẩn" mà hầu hết thuật toán deep RL hiện đại dùng
    (n-step return + function approximation).

PSEUDOCODE
    Input: q̂ khả vi, α, ε, n
    Khởi tạo w
    Lặp với mỗi episode:
        khởi tạo S0, chọn A0
        T ← ∞
        với t = 0,1,2,...:
            nếu t < T:
                thực hiện A_t, quan sát R_{t+1}, S_{t+1}
                nếu terminal: T ← t+1
                ngược lại: chọn A_{t+1}
            τ ← t - n + 1
            nếu τ ≥ 0:
                G ← Σ_{i=τ+1}^{min(τ+n,T)} γ^{i-τ-1} R_i
                nếu τ+n < T: G ← G + γ^n q̂(S_{τ+n}, A_{τ+n}, w)
                w ← w + α[G − q̂(S_τ,A_τ,w)]∇q̂(S_τ,A_τ,w)
            cho tới khi τ = T-1

CÁCH ĐỌC CODE
    - Vẫn là mảng `states`, `actions`, `rewards` như các file n-step trước.
    - Chỉ một dòng cập nhật khác file 23: target dài n bước thay vì 1 bước.
    - `phi` giữ nguyên one-hot để so sánh công bằng với file 23.

CHẠY THỬ
    python 24_n_step_semi_gradient_sarsa.py
================================================================================
"""

import numpy as np
from common import GridWorld, one_hot


def n_step_semi_gradient_sarsa(env, episodes=50, n=4, alpha=0.2, gamma=0.9,
                               eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    n_features = env.nS
    W = np.zeros((env.nA, n_features))

    def phi(s):
        return np.zeros(n_features) if env.is_terminal(s) else one_hot(s, n_features)

    def q(x, a):
        return float(W[a] @ x)

    def choose(x):
        if rng.random() < eps:
            return int(rng.integers(env.nA))
        vals = np.array([q(x, a) for a in range(env.nA)])
        best = np.flatnonzero(vals == vals.max())
        return int(rng.choice(best))

    for _ in range(episodes):
        s = env.reset()
        x = phi(s)
        states, xs, actions = [s], [x], [choose(x)]
        rewards = [0.0]

        T = float("inf")
        t = 0
        while True:
            if t < T:
                r, s2, done = env.step(states[t], actions[t])
                rewards.append(r)
                states.append(s2)
                xs.append(phi(s2))
                if done:
                    T = t + 1
                else:
                    actions.append(choose(xs[-1]))

            tau = t - n + 1
            if tau >= 0:
                # (1) n-step return
                G = 0.0
                for i in range(tau + 1, int(min(tau + n, T)) + 1):
                    G += gamma ** (i - tau - 1) * rewards[i]
                if tau + n < T:
                    G += gamma ** n * q(xs[tau + n], actions[tau + n])

                # (2) semi-gradient update
                delta = G - q(xs[tau], actions[tau])
                W[actions[tau]] += alpha * delta * xs[tau]

            if tau == T - 1:
                break
            t += 1

    return W


def steps_to_goal(env, W, cap=200):
    s, n = env.reset(), 0
    while not env.is_terminal(s) and n < cap:
        x = one_hot(s, env.nS)
        a = int(np.argmax([W[k] @ x for k in range(env.nA)]))
        r, s2, done = env.step(s, a)
        s = s2
        n += 1
    return n if env.is_terminal(s) else cap


if __name__ == "__main__":
    print("=" * 70)
    print(" n-STEP SEMI-GRADIENT SARSA — lưới 5x5, TB 10 lần chạy")
    print(" Số bước tới đích (tối ưu = 8)")
    print("=" * 70)

    env = GridWorld(rows=5, cols=5, goal=(4, 4), gamma=0.9)
    print(f"\n{'episode':>8}{'n=1':>10}{'n=2':>10}{'n=4':>10}{'n=8':>10}")
    print("-" * 48)
    for episodes in (10, 25, 50):
        row = []
        for n in (1, 2, 4, 8):
            scores = []
            for seed in range(10):
                W = n_step_semi_gradient_sarsa(env, episodes=episodes, n=n, seed=seed)
                scores.append(steps_to_goal(env, W))
            row.append(np.mean(scores))
        print(f"{episodes:>8}" + "".join(f"{v:>10.1f}" for v in row))

    print("\nNhận xét: n-step giúp chính sách tốt lên nhanh hơn ở giai đoạn đầu,")
    print("giống hiện tượng đã thấy ở chương 7. Đây là lý do các thuật toán")
    print("deep RL hiện đại (A3C, PPO, DQN n-step) thường dùng n = 3–5.")
