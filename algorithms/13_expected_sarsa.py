"""
================================================================================
 13 — EXPECTED SARSA                                        (Sutton & Barto, Ch. 6.6)
================================================================================

BÀI TOÁN
    Như Sarsa, nhưng giảm phương sai bằng cách lấy KỲ VỌNG thay vì lấy mẫu
    một hành động A'.

Ý TƯỞNG
    Sarsa dùng `Q(S', A')` với một A' ngẫu nhiên. Nhưng ta biết phân phối của
    A' (chính là π). Vậy thay vì tung xúc xắc, hãy lấy trung bình có trọng số:

        Q(S,A) ← Q(S,A) + α [ R + γ Σ_a π(a|S') Q(S',a) - Q(S,A) ]

    Đây là Expected Sarsa. Target của nó là kỳ vọng chính xác, không có
    nhiễu do chọn A' -> phương sai thấp hơn Sarsa, thường học nhanh hơn.

    Đặc biệt: nếu π là THAM LAM thuần (ε=0) thì
        Σ_a π(a|S')Q(S',a) = max_a Q(S',a)
    tức là Expected Sarsa TRÙNG với Q-learning. Nó là "họ hàng" ở giữa:
        ε > 0  -> on-policy, giống Sarsa nhưng ít nhiễu
        ε = 0  -> off-policy, chính là Q-learning

CÔNG THỨC CHO π ε-THAM LAM
    Với |A| hành động, π đặt 1-ε+ε/|A| vào hành động tốt nhất và ε/|A| cho
    các hành động còn lại:

        E_π[Q] = (1 - ε) * max_a Q(s',a) + ε * mean_a Q(s',a)

    (vì tổng trọng số của phần "đều" bằng ε và trung bình của Q là mean Q).

PSEUDOCODE
    Khởi tạo Q(s,a) tuỳ ý
    Lặp với mỗi episode:
        khởi tạo S, chọn A theo π
        lặp với mỗi bước:
            thực hiện A, quan sát R, S'
            Q(S,A) ← Q(S,A) + α[R + γ Σ_a π(a|S')Q(S',a) - Q(S,A)]
            chọn A' theo π tại S'
            S ← S'; A ← A'

CÁCH ĐỌC CODE
    - `expected_q(Q[s2], eps)` chính là công thức trên, viết gọn một dòng.
    - Không cần `rng` để chọn A' cho phần target — đó là nguồn gốc của việc
      giảm phương sai.

CHẠY THỬ
    python 13_expected_sarsa.py
================================================================================
"""

import numpy as np
from common import CliffWalking, ARROWS, epsilon_greedy


def expected_q(q_values, eps):
    """E_π[Q(s',·)] với π là ε-tham lam."""
    return (1.0 - eps) * q_values.max() + eps * q_values.mean()


def expected_sarsa(env, episodes=500, alpha=0.5, gamma=1.0, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.nS, env.nA))

    for _ in range(episodes):
        s = env.reset()
        a = epsilon_greedy(Q[s], eps, rng)

        while True:
            r, s2, done = env.step(s, a)

            # (1) target = kỳ vọng theo π, không lấy mẫu A'
            target = r if done else r + gamma * expected_q(Q[s2], eps)

            # (2) cập nhật
            Q[s, a] += alpha * (target - Q[s, a])

            if done:
                break
            s = s2
            a = epsilon_greedy(Q[s], eps, rng)

    return Q


def evaluate(env, policy, episodes=200, max_steps=500, seed=1):
    total = []
    for _ in range(episodes):
        s, ret = env.reset(), 0.0
        for _ in range(max_steps):
            r, s2, done = env.step(s, policy[s])
            ret += r
            if done:
                break
            s = s2
        total.append(ret)
    return float(np.mean(total))


if __name__ == "__main__":
    print("=" * 70)
    print(" EXPECTED SARSA vs SARSA vs Q-LEARNING — Cliff Walking, 500 episode")
    print("=" * 70)

    env = CliffWalking()

    from importlib import import_module
    sarsa_mod = import_module("11_sarsa")
    q_mod = import_module("12_q_learning")

    results = {}

    # Expected Sarsa
    Q_es = expected_sarsa(env, episodes=500)
    pol_es = np.argmax(Q_es, axis=1)
    pol_es[env.terminal] = 0
    results["Expected Sarsa"] = (pol_es, evaluate(env, pol_es))

    # Sarsa
    Q_sa = sarsa_mod.sarsa(env, episodes=500)
    pol_sa = np.argmax(Q_sa, axis=1)
    pol_sa[env.terminal] = 0
    results["Sarsa"] = (pol_sa, evaluate(env, pol_sa))

    # Q-learning
    Q_ql = q_mod.q_learning(env, episodes=500)
    pol_ql = np.argmax(Q_ql, axis=1)
    pol_ql[env.terminal] = 0
    results["Q-learning"] = (pol_ql, evaluate(env, pol_ql))

    for name, (pol, score) in results.items():
        dist_to_cliff = sum(
            1 for s in range(env.nS)
            if not env.is_terminal(s) and pol[s] in (1, 3)
            and (s // env.cols) == 2
        )
        print(f"{name:<16} thưởng TB = {score:7.1f}   "
              f"số bước đi ngang ở hàng sát vực = {dist_to_cliff}")

    print()
    print("Chính sách của Expected Sarsa:")
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            s = r * env.cols + c
            if s == env.terminal:
                row.append("G")
            elif (r, c) in env.cliff:
                row.append("~")
            else:
                row.append(ARROWS[results['Expected Sarsa'][0][s]])
        print("  " + " ".join(row))

    print("\nNhận xét: Expected Sarsa thường đạt thưởng cao hơn Sarsa/Q-learning")
    print("với cùng số episode, nhờ target ít nhiễu hơn.")
