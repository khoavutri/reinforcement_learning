"""
================================================================================
 16 — n-STEP SARSA                                          (Sutton & Barto, Ch. 7.2)
================================================================================

BÀI TOÁN
    Điều khiển (tìm π*) bằng n-step return thay vì 1 bước.

Ý TƯỞNG
    Sarsa 1 bước cập nhật:  Q(S_t,A_t) ← Q + α[R_{t+1} + γQ(S_{t+1},A_{t+1}) - Q]
    n-step Sarsa cập nhật:  Q(S_t,A_t) ← Q + α[G_{t:t+n} - Q]

    với  G_{t:t+n} = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n Q(S_{t+n}, A_{t+n})

    Chú ý: vẫn là ON-POLICY, vì A_{t+n} được chọn theo chính sách hiện tại.

    Lợi ích: giảm chệch do bootstrap, tăng tốc lan truyền phần thưởng về các
    trạng thái trước đó. Trong Cliff Walking, n-step Sarsa thường học nhanh
    hơn Sarsa 1 bước đáng kể.

PSEUDOCODE
    Khởi tạo Q(s,a) tuỳ ý
    Lặp với mỗi episode:
        khởi tạo và lưu S0 ≠ terminal
        chọn và lưu A0 theo π
        T ← ∞
        với t = 0,1,2,...:
            nếu t < T:
                thực hiện A_t, quan sát R_{t+1}, S_{t+1}
                nếu S_{t+1} terminal: T ← t+1
                ngược lại: chọn và lưu A_{t+1}
            τ ← t - n + 1
            nếu τ ≥ 0:
                G ← Σ_{i=τ+1}^{min(τ+n,T)} γ^{i-τ-1} R_i
                nếu τ+n < T: G ← G + γ^n Q(S_{τ+n}, A_{τ+n})
                Q(S_τ,A_τ) ← Q + α[G - Q]
            cho tới khi τ = T-1

CÁCH ĐỌC CODE
    - Ba danh sách `states`, `actions`, `rewards` đóng vai trò bộ nhớ n bước.
    - `tau = t - n + 1` là trạng thái đang được sửa, luôn trễ n bước so với
      hiện tại — đó là lý do n-step phải "nhớ" lịch sử.

CHẠY THỬ
    python 16_n_step_sarsa.py
================================================================================
"""

import numpy as np
from common import GridWorld, epsilon_greedy


def n_step_sarsa(env, episodes=500, n=4, alpha=0.5, gamma=1.0, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.nS, env.nA))

    for _ in range(episodes):
        states = [env.reset()]
        actions = [epsilon_greedy(Q[states[0]], eps, rng)]
        rewards = [0.0]

        T = float("inf")
        t = 0
        while True:
            if t < T:
                r, s2, done = env.step(states[t], actions[t])
                rewards.append(r)
                states.append(s2)
                if done:
                    T = t + 1
                else:
                    actions.append(epsilon_greedy(Q[s2], eps, rng))

            tau = t - n + 1
            if tau >= 0:
                # (1) n-step return
                G = 0.0
                for i in range(tau + 1, int(min(tau + n, T)) + 1):
                    G += gamma ** (i - tau - 1) * rewards[i]
                # (2) bootstrap nếu chưa hết episode
                if tau + n < T:
                    G += gamma ** n * Q[states[tau + n], actions[tau + n]]

                # (3) cập nhật
                s_tau, a_tau = states[tau], actions[tau]
                Q[s_tau, a_tau] += alpha * (G - Q[s_tau, a_tau])

            if tau == T - 1:
                break
            t += 1

    return Q


def steps_to_goal(env, policy, cap=200):
    """Số bước để đi từ start tới đích dưới chính sách tham lam (tối ưu = 6)."""
    s, n = env.reset(), 0
    while not env.is_terminal(s) and n < cap:
        r, s2, done = env.step(s, policy[s])
        s = s2
        n += 1
    return n if env.is_terminal(s) else cap


if __name__ == "__main__":
    print("=" * 72)
    print(" n-STEP SARSA — lưới 4x4 (γ=0.9, α=0.5, ε=0.1), TB 20 lần chạy")
    print(" Số bước tới đích sau khi học; tối ưu = 6")
    print("=" * 72)

    env = GridWorld(gamma=0.9)
    ns = (1, 2, 4, 8)
    print(f"\n{'episode':>8}" + "".join(f"{'n=' + str(n):>12}" for n in ns))
    print("-" * 56)

    for episodes in (5, 10, 20, 40):
        row = []
        for n in ns:
            scores = []
            for seed in range(20):
                Q = n_step_sarsa(env, episodes=episodes, n=n, gamma=0.9, alpha=0.5, seed=seed)
                pol = np.argmax(Q, axis=1)
                pol[env.terminal] = 0
                scores.append(steps_to_goal(env, pol))
            row.append(np.mean(scores))
        print(f"{episodes:>8}" + "".join(f"{v:>12.1f}" for v in row))

    print("\nNhận xét: khi MỚI học, n lớn thắng rõ rệt — return được lan ngược")
    print("về nhiều trạng thái chỉ sau vài episode. Khi đã học đủ lâu, mọi n")
    print("đều đạt 6 bước; lúc đó n nhỏ lại rẻ hơn về bộ nhớ và tính toán.")
    print("Đây chính là lý do sách khuyên chọn n vừa phải (4–10).")
