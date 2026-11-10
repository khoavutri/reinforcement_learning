"""
================================================================================
 32 — REINFORCE VỚI BASELINE                             (Sutton & Barto, Ch. 13.4)
================================================================================

BÀI TOÁN
    REINFORCE dùng return G_t làm "trọng số" cho gradient. Nhưng G_t có
    phương sai rất lớn: có episode may mắn, có episode rủi ro, dù chính sách
    không đổi. Làm sao giảm nhiễu mà không làm chệch gradient?

Ý TƯỞNG — BASELINE
    Trừ đi một hàm b(s) bất kỳ không phụ thuộc hành động:

        ∇J(θ) ∝ E[ (G_t − b(S_t)) ∇ln π(A_t|S_t,θ) ]

    Kỳ vọng KHÔNG đổi, vì:

        E_a[ b(s) ∇ln π(a|s) ] = b(s) Σ_a ∇π(a|s) = b(s) ∇1 = 0

    Nhưng phương sai giảm mạnh nếu b(s) xấp xỉ v_π(s). Trực giác: thay vì
    "return tuyệt đối", ta dùng "return tốt hơn hay kém hơn mức trung bình
    của trạng thái này" — chính là một dạng advantage.

    Baseline tốt nhất (đơn giản): b(s) = v̂(s,w) học bằng TD hoặc Monte Carlo.

PSEUDOCODE (REINFORCE with baseline)
    Khởi tạo θ (chính sách) và w (baseline v̂)
    Lặp mãi:
        sinh episode theo π(·|·,θ)
        G ← 0
        với t = T-1 ... 0:
            G ← γG + R_{t+1}
            δ ← G − v̂(S_t,w)
            w ← w + α_w δ ∇v̂(S_t,w)
            θ ← θ + α_θ γ^t δ ∇ln π(A_t|S_t,θ)

CÁCH ĐỌC CODE
    - `theta` cho chính sách, `v` cho baseline (bảng one-hot, tức là bảng
      giá trị đơn giản nhất).
    - Chỉ một chữ thay đổi so với file 31: `G` -> `G - v[s]`.
    - Baseline học đồng thời, bằng cùng return Monte Carlo.

LIÊN HỆ HIỆN ĐẠI
    Mọi thuật toán policy gradient hiện đại đều dùng baseline/advantage:
    A2C, PPO, SAC... Nếu không có baseline, chúng sẽ dao động không hội tụ.

CHẠY THỬ
    python 32_reinforce_baseline.py
================================================================================
"""

import numpy as np
from common import GridWorld, softmax


def reinforce_baseline(env, episodes=500, alpha=0.2, alpha_v=0.1, gamma=0.9,
                       max_steps=200, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.zeros((env.nS, env.nA))
    v = np.zeros(env.nS)          # baseline v̂(s)

    for _ in range(episodes):
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

        G = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            G = gamma * G + rewards[t]
            s_t, a_t = states[t], actions[t]

            # (1) baseline: return vượt mức trung bình của trạng thái
            advantage = G - v[s_t]

            # (2) học baseline
            v[s_t] += alpha_v * advantage

            # (3) cập nhật chính sách với advantage thay cho G
            p = softmax(theta[s_t])
            grad = -p
            grad[a_t] += 1.0
            theta[s_t] += alpha * (gamma ** t) * advantage * grad

    return theta, v


def steps_to_goal(env, policy, cap=200):
    s, n = env.reset(), 0
    while not env.is_terminal(s) and n < cap:
        r, s2, done = env.step(s, policy[s])
        s = s2
        n += 1
    return n if env.is_terminal(s) else cap


if __name__ == "__main__":
    print("=" * 72)
    print(" REINFORCE + BASELINE vs REINFORCE — lưới 4x4, TB 5 lần chạy")
    print("=" * 72)

    from importlib import import_module
    plain = import_module("31_reinforce")

    env = GridWorld(gamma=0.9)
    print(f"\n{'episode':>8}{'REINFORCE':>14}{'+ baseline':>14}")
    print("-" * 36)

    for episodes in (50, 200, 500):
        s_plain, s_base = [], []
        for seed in range(5):
            th = plain.reinforce(env, episodes=episodes, alpha=0.2, seed=seed)
            s_plain.append(steps_to_goal(env, np.argmax(th, axis=1)))

            th, _ = reinforce_baseline(env, episodes=episodes, alpha=0.2, seed=seed)
            s_base.append(steps_to_goal(env, np.argmax(th, axis=1)))

        print(f"{episodes:>8}{np.mean(s_plain):>14.1f}{np.mean(s_base):>14.1f}")

    print("\nNhận xét: baseline giúp hội tụ nhanh hơn rõ rệt ở cùng số episode.")
    print("Nó không làm thay đổi kỳ vọng gradient — chỉ giảm phương sai.")
    print("Đây là lý do 'advantage' là thành phần bắt buộc trong PPO/A2C.")
