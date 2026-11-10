"""
================================================================================
 31 — REINFORCE (MONTE CARLO POLICY GRADIENT)            (Sutton & Barto, Ch. 13.3)
================================================================================

BÀI TOÁN
    Thay vì học giá trị rồi suy ra chính sách, hãy học TRỰC TIẾP tham số của
    chính sách θ để tối đa hoá phần thưởng kỳ vọng.

Ý TƯỞNG
    Định lý policy gradient:

        ∇J(θ) ∝ E_π[ G_t · ∇ln π(A_t|S_t,θ) ]

    Nghĩa là: nếu một hành động dẫn tới return G_t cao, hãy TĂNG xác suất
    của nó; nếu thấp, giảm. Không cần biết môi trường, chỉ cần chơi và ghi lại.

    Với chính sách softmax trên bảng θ(s,a):

        π(a|s,θ) = exp θ(s,a) / Σ_b exp θ(s,b)
        ∇ln π(a|s,θ) = onehot(a) − π(·|s,θ)

    nên cập nhật rất đơn giản:

        θ(s,a) ← θ(s,a) + α γ^t G_t (1 − π(a|s))      (hành động đã làm)
        θ(s,b) ← θ(s,b) − α γ^t G_t π(b|s)            (các hành động khác)

PSEUDOCODE (REINFORCE, sách hộp 13.1)
    Input: chính sách khả vi π(a|s,θ), α
    Khởi tạo θ
    Lặp mãi:
        sinh episode S0,A0,R1,...,S_{T-1},A_{T-1},R_T theo π(·|·,θ)
        G ← 0
        với t = T-1 ... 0:
            G ← γG + R_{t+1}
            θ ← θ + α γ^t G ∇ln π(A_t|S_t,θ)

CÁCH ĐỌC CODE
    - `theta` là bảng (nS × nA); `softmax` chuyển thành xác suất.
    - Chỉ MỘT episode mỗi lần cập nhật (Monte Carlo) -> phương sai cao.
    - `gamma ** t` là hệ số chiết khấu của return tại thời điểm t; nếu bỏ đi
      vẫn học được nhưng chệch.

VÌ SAO QUAN TRỌNG
    Đây là tổ tiên trực tiếp của A3C, PPO, TRPO, SAC — và của RLHF dùng để
    huấn luyện các mô hình ngôn ngữ lớn ngày nay. Khác biệt duy nhất giữa
    REINFORCE và PPO là cách giảm phương sai và giới hạn bước cập nhật.

CHẠY THỬ
    python 31_reinforce.py
================================================================================
"""

import numpy as np
from common import GridWorld, softmax


def reinforce(env, episodes=500, alpha=0.1, gamma=0.9, max_steps=200, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.zeros((env.nS, env.nA))

    for _ in range(episodes):
        # (1) sinh một episode theo π(·|·,θ)
        s = env.reset()
        states, actions, rewards = [], [], []
        while len(rewards) < max_steps:
            p = softmax(theta[s])
            a = int(rng.choice(env.nA, p=p))
            r, s2, done = env.step(s, a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = s2
            if done:
                break

        # (2) duyệt ngược, cập nhật θ theo return
        G = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            G = gamma * G + rewards[t]
            s_t, a_t = states[t], actions[t]
            p = softmax(theta[s_t])

            grad = -p                      # ∇ln π cho các hành động khác
            grad[a_t] += 1.0               #   và +1 cho hành động đã làm

            theta[s_t] += alpha * (gamma ** t) * G * grad

    return theta


def greedy_policy(env, theta):
    return np.argmax(theta, axis=1)


def steps_to_goal(env, policy, cap=200):
    s, n = env.reset(), 0
    while not env.is_terminal(s) and n < cap:
        r, s2, done = env.step(s, policy[s])
        s = s2
        n += 1
    return n if env.is_terminal(s) else cap


if __name__ == "__main__":
    print("=" * 70)
    print(" REINFORCE — lưới 4x4, TB 5 lần chạy (tối ưu = 6 bước)")
    print("=" * 70)

    env = GridWorld(gamma=0.9)
    print(f"\n{'episode':>8}{'số bước tới đích':>20}{'tối ưu':>10}")
    print("-" * 38)
    for episodes in (50, 200, 500):
        scores = []
        for seed in range(5):
            theta = reinforce(env, episodes=episodes, alpha=0.2, seed=seed)
            scores.append(steps_to_goal(env, greedy_policy(env, theta)))
        print(f"{episodes:>8}{np.mean(scores):>20.1f}{6:>10}")

    env2 = GridWorld(gamma=0.9)
    theta = reinforce(env2, episodes=500, alpha=0.2, seed=0)
    pol = greedy_policy(env2, theta)
    print("\nChính sách tham lam sau 500 episode:")
    for r in range(env2.rows):
        row = []
        for c in range(env2.cols):
            s = r * env2.cols + c
            row.append("G" if env2.is_terminal(s) else "^v<>"[pol[s]])
        print("  " + " ".join(row))

    print("\nNhận xét: REINFORCE một mình hội tụ CHẬM và dễ kẹt ở chính sách")
    print("vòng lặp — return có phương sai quá lớn. Đây chính là động lực cho")
    print("baseline (file 32) và actor–critic (file 33): cùng bài toán này,")
    print("hai thuật toán đó đạt 6 bước chỉ sau khoảng 200 episode.")
