"""
================================================================================
 30 — TRUE ONLINE SARSA(λ)                              (Sutton & Barto, Ch. 12.7)
================================================================================

BÀI TOÁN
    Sarsa(λ) dùng accumulating/replacing trace là xấp xỉ tốt, nhưng không
    trùng khớp chính xác với λ-return lý thuyết. True online Sarsa(λ) sửa
    điều đó bằng dutch trace, giống như file 28 làm với TD(λ).

Ý TƯỞNG
    Mỗi hành động a có bộ trọng số w_a riêng, và một trace z_a riêng:

        q̂(s,a) = w_aᵀ x(s,a)

        δ = R + γ q̂(S',A') − q̂(S,A)
        z_a ← γλ z_a + (1 − αγλ z_aᵀ x) x
        w_a ← w_a + α(δ + Q − Q_old) z_a − α(Q − Q_old) x

    trong đó Q_old là giá trị của (S,A) được tính ở lần cập nhật trước.

    Kết quả: mỗi bước chỉ O(d) cho hành động vừa chọn (không cần duyệt cả
    bảng như Sarsa(λ)), và khớp chính xác với λ-return.

PSEUDOCODE (sách, hộp 12.4)
    Khởi tạo w (hoặc w_a cho từng a), z = 0
    khởi tạo S, chọn A, x ← x(S,A), Q_old ← 0
    lặp mỗi bước:
        thực hiện A, quan sát R, S'
        chọn A' theo q̂(S',·,w)
        x' ← x(S',A')
        Q ← wᵀx ; Q' ← wᵀx'
        δ ← R + γQ' − Q
        z ← γλz + (1 − αγλ zᵀx) x
        w ← w + α(δ + Q − Q_old) z − α(Q − Q_old) x
        Q_old ← Q'
        x ← x'; A ← A'

CÁCH ĐỌC CODE
    - `W` và `Z` là ma trận (nA × d): mỗi hành động một bộ trọng số/trace.
    - Chỉ khối của hành động A được cập nhật ở mỗi bước — đó là lý do thuật
      toán này rẻ hơn Sarsa(λ) dạng bảng.
    - Với one-hot, thuật toán tương đương Sarsa(λ) về hành vi; khác biệt
      xuất hiện khi feature chồng lấn hoặc α lớn.

CHẠY THỬ
    python 30_true_online_sarsa_lambda.py
================================================================================
"""

import numpy as np
from common import GridWorld, one_hot


def true_online_sarsa_lambda(env, episodes=30, alpha=0.2, gamma=0.9, lam=0.8,
                             eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    d = env.nS
    W = np.zeros((env.nA, d))
    Z = np.zeros((env.nA, d))

    def phi(s):
        return np.zeros(d) if env.is_terminal(s) else one_hot(s, d)

    def q(x, a):
        return float(W[a] @ x)

    def choose(x):
        if rng.random() < eps:
            return int(rng.integers(env.nA))
        vals = [q(x, k) for k in range(env.nA)]
        best = np.flatnonzero(np.isclose(vals, max(vals)))
        return int(rng.choice(best))

    for _ in range(episodes):
        Z[:] = 0.0
        s = env.reset()
        x = phi(s)
        a = choose(x)
        Q_old = 0.0

        while True:
            r, s2, done = env.step(s, a)
            x2 = phi(s2)

            if done:
                Q = q(x, a)
                delta = r - Q
                # (1) dutch trace cho hành động A
                Z[a] = gamma * lam * Z[a] + (1.0 - alpha * gamma * lam * (Z[a] @ x)) * x
                # (2) cập nhật khối trọng số của A
                W[a] += alpha * (delta + Q - Q_old) * Z[a] - alpha * (Q - Q_old) * x
                break

            a2 = choose(x2)
            Q = q(x, a)
            Q_next = q(x2, a2)
            delta = r + gamma * Q_next - Q

            Z[a] = gamma * lam * Z[a] + (1.0 - alpha * gamma * lam * (Z[a] @ x)) * x
            W[a] += alpha * (delta + Q - Q_old) * Z[a] - alpha * (Q - Q_old) * x

            Q_old = Q_next
            x, s, a = x2, s2, a2

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
    print(" TRUE ONLINE SARSA(λ) — lưới 4x4, TB 10 lần chạy (tối ưu = 6)")
    print("=" * 70)

    env = GridWorld(gamma=0.9)
    print(f"\n{'episode':>8}{'λ=0':>10}{'λ=0.5':>10}{'λ=0.8':>10}{'λ=0.95':>10}")
    print("-" * 48)

    for episodes in (5, 10, 20, 40):
        row = []
        for lam in (0.0, 0.5, 0.8, 0.95):
            scores = []
            for seed in range(10):
                W = true_online_sarsa_lambda(env, episodes=episodes, alpha=0.5,
                                             lam=lam, seed=seed)
                scores.append(steps_to_goal(env, W))
            row.append(np.mean(scores))
        print(f"{episodes:>8}" + "".join(f"{v:>10.1f}" for v in row))

    print("\nNhận xét: giống Sarsa(λ) nhưng mỗi bước chỉ cập nhật MỘT khối trọng")
    print("số, không duyệt cả bảng — nhờ vậy dùng được với xấp xỉ hàm thật.")
    print("Đây là một trong những thuật toán mạnh nhất của RL cổ điển.")
