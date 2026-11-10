"""
================================================================================
 29 — SARSA(λ)                                     (Sutton & Barto, Ch. 12.7)
================================================================================

BÀI TOÁN
    Điều khiển (control) với eligibility trace: học nhanh hơn Sarsa 1 bước
    mà vẫn giữ được tính on-policy.

Ý TƯỞNG
    Áp trace lên Sarsa: mỗi cặp (s,a) có độ "eligible" E(s,a). Khi có TD
    error, cập nhật MỌI cặp vừa góp phần, không chỉ cặp hiện tại.

        δ_t = R_{t+1} + γ Q(S_{t+1},A_{t+1}) − Q(S_t,A_t)
        E(S_t,A_t) ← 1                    (replacing traces)
        Q ← Q + α δ_t E
        E ← γλ E

    "Replacing trace" (đặt =1 thay vì cộng dồn) hoạt động tốt hơn với bài
    toán có vòng lặp, vì nó không để E phình vô hạn khi agent quay lại một
    trạng thái nhiều lần.

    λ = 0 -> Sarsa 1 bước; λ = 1 -> gần Monte Carlo.

PSEUDOCODE (Sarsa(λ) với replacing traces)
    Khởi tạo Q(s,a) tuỳ ý, E(s,a) = 0
    Lặp với mỗi episode:
        khởi tạo S, chọn A
        lặp mỗi bước:
            thực hiện A, quan sát R, S'
            chọn A' theo Q tại S'
            δ ← R + γQ(S',A') − Q(S,A)      (0 nếu S' terminal)
            E(S,A) ← 1
            với mọi s,a: Q(s,a) ← Q(s,a) + αδE(s,a)
            với mọi s,a: E(s,a) ← γλE(s,a)
            S ← S'; A ← A'

CÁCH ĐỌC CODE
    - `E` là ma trận nS × nA cùng kích thước Q.
    - Hai vòng "với mọi s,a" được viết vector hoá bằng numpy: `Q += alpha*delta*E`
      và `E *= gamma*lam`. Đó là toàn bộ chi phí O(|S||A|) mỗi bước — chấp
      nhận được với bài toán nhỏ; với bài toán lớn người ta dùng trace thưa.
    - Chú ý thứ tự: đặt E(S,A)=1 TRƯỚC khi cập nhật Q, và già hoá E SAU.

CHẠY THỬ
    python 29_sarsa_lambda.py
================================================================================
"""

import numpy as np
from common import GridWorld, epsilon_greedy


def sarsa_lambda(env, episodes=30, alpha=0.5, gamma=0.9, lam=0.8, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.nS, env.nA))
    E = np.zeros((env.nS, env.nA))

    for _ in range(episodes):
        E[:] = 0.0
        s = env.reset()
        a = epsilon_greedy(Q[s], eps, rng)

        while True:
            r, s2, done = env.step(s, a)

            if done:
                delta = r - Q[s, a]
            else:
                a2 = epsilon_greedy(Q[s2], eps, rng)
                delta = r + gamma * Q[s2, a2] - Q[s, a]

            # (1) replacing trace
            E[s, a] = 1.0

            # (2) cập nhật toàn bộ bảng theo trace
            Q += alpha * delta * E

            # (3) già hoá trace
            E *= gamma * lam

            if done:
                break
            s, a = s2, a2

    return Q


def steps_to_goal(env, policy, cap=200):
    s, n = env.reset(), 0
    while not env.is_terminal(s) and n < cap:
        r, s2, done = env.step(s, policy[s])
        s = s2
        n += 1
    return n if env.is_terminal(s) else cap


if __name__ == "__main__":
    print("=" * 70)
    print(" SARSA(λ) — lưới 4x4, TB 20 lần chạy (tối ưu = 6 bước)")
    print("=" * 70)

    env = GridWorld(gamma=0.9)
    print(f"\n{'episode':>8}{'λ=0':>10}{'λ=0.5':>10}{'λ=0.8':>10}{'λ=0.95':>10}")
    print("-" * 48)

    for episodes in (5, 10, 20, 40):
        row = []
        for lam in (0.0, 0.5, 0.8, 0.95):
            scores = []
            for seed in range(20):
                Q = sarsa_lambda(env, episodes=episodes, lam=lam, seed=seed)
                pol = np.argmax(Q, axis=1)
                pol[env.terminal] = 0
                scores.append(steps_to_goal(env, pol))
            row.append(np.mean(scores))
        print(f"{episodes:>8}" + "".join(f"{v:>10.1f}" for v in row))

    print("\nNhận xét: khi ít episode, λ lớn học nhanh hơn rõ rệt — trace lan")
    print("ngược phần thưởng về nhiều trạng thái chỉ sau một lần chơi. Khi đã")
    print("học đủ, mọi λ đều đạt 6 bước; lúc đó λ nhỏ lại ổn định hơn.")
