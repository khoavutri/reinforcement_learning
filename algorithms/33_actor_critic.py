"""
================================================================================
 33 — ACTOR–CRITIC (MỘT BƯỚC + ELIGIBILITY TRACE)  (Sutton & Barto, Ch. 13.5)
================================================================================

BÀI TOÁN
    REINFORCE phải đợi hết episode và dùng return đầy đủ (phương sai cao).
    Có thể cập nhật mỗi bước, dùng TD error làm tín hiệu không?

Ý TƯỞNG — HAI MẠNG HỌC CÙNG LÚC
    - ACTOR: chính sách π(a|s,θ) — quyết định hành động.
    - CRITIC: hàm giá trị v̂(s,w) — đánh giá tình hình.

    Critic tính TD error, actor dùng nó làm tín hiệu cập nhật:

        δ_t = R_{t+1} + γ v̂(S_{t+1},w) − v̂(S_t,w)
        w   ← w + α_w δ_t ∇v̂(S_t,w)                  (critic học)
        θ   ← θ + α_θ δ_t ∇ln π(A_t|S_t,θ)          (actor học)

    δ_t đóng vai "advantage": dương nghĩa là hành động vừa làm tốt hơn dự
    đoán -> tăng xác suất; âm -> giảm.

    Đây là sự hợp nhất hoàn hảo của hai dòng thuật toán trong sách:
    value-based (chương 6–12) và policy-based (13.1–13.4).

PSEUDOCODE (One-step Actor–Critic, episodic)
    Khởi tạo θ, w
    Lặp với mỗi episode:
        khởi tạo S
        I ← 1                                (hệ số chiết khấu cho actor)
        lặp khi S chưa terminal:
            A ~ π(·|S,θ)
            thực hiện A, quan sát R, S'
            nếu S' terminal:
                δ ← R − v̂(S,w)
            ngược lại:
                δ ← R + γv̂(S',w) − v̂(S,w)
            w ← w + α_w δ ∇v̂(S,w)
            θ ← θ + α_θ I δ ∇ln π(A|S,θ)
            I ← γI
            S ← S'

CÁCH ĐỌC CODE
    - `theta` (actor) và `v` (critic) như file 32, nhưng cập nhật MỖI BƯỚC.
    - `I` là hệ số chiết khấu của actor; với γ = 1 thì I luôn = 1.
    - Bản có eligibility trace (hộp 13.2 trong sách) chỉ khác ở chỗ thay
      `I` bằng hai vector trace z_θ, z_w; xem phần docstring dưới.

BẢN CÓ TRACE (tóm tắt)
    z_w ← γλ_w z_w + ∇v̂(S,w)
    z_θ ← γλ_θ z_θ + I ∇ln π(A|S,θ)
    w   ← w + α_w δ z_w
    θ   ← θ + α_θ δ z_θ
    I   ← γI
    -> tương đương việc dùng nhiều bước return mà vẫn cập nhật mỗi bước.

CHẠY THỬ
    python 33_actor_critic.py
================================================================================
"""

import numpy as np
from common import GridWorld, softmax


def actor_critic(env, episodes=300, alpha_theta=0.2, alpha_w=0.1, gamma=0.9,
                 max_steps=200, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.zeros((env.nS, env.nA))       # actor
    v = np.zeros(env.nS)                     # critic (bảng)

    for _ in range(episodes):
        s = env.reset()
        I = 1.0
        steps = 0

        while not env.is_terminal(s) and steps < max_steps:
            p = softmax(theta[s])
            a = int(rng.choice(env.nA, p=p))
            r, s2, done = env.step(s, a)

            # (1) TD error — tín hiệu cho cả hai
            if done:
                delta = r - v[s]
            else:
                delta = r + gamma * v[s2] - v[s]

            # (2) critic học giá trị
            v[s] += alpha_w * delta

            # (3) actor học chính sách theo δ
            grad = -p
            grad[a] += 1.0
            theta[s] += alpha_theta * I * delta * grad

            I *= gamma
            s = s2
            steps += 1

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
    print(" ACTOR–CRITIC vs REINFORCE — lưới 4x4, TB 5 lần chạy (tối ưu = 6)")
    print("=" * 72)

    from importlib import import_module
    plain = import_module("31_reinforce")

    env = GridWorld(gamma=0.9)
    print(f"\n{'episode':>8}{'REINFORCE':>14}{'Actor–Critic':>15}")
    print("-" * 38)

    for episodes in (50, 200, 500):
        s_plain, s_ac = [], []
        for seed in range(5):
            th = plain.reinforce(env, episodes=episodes, alpha=0.2, seed=seed)
            s_plain.append(steps_to_goal(env, np.argmax(th, axis=1)))

            th, _ = actor_critic(env, episodes=episodes, seed=seed)
            s_ac.append(steps_to_goal(env, np.argmax(th, axis=1)))

        print(f"{episodes:>8}{np.mean(s_plain):>14.1f}{np.mean(s_ac):>15.1f}")

    env2 = GridWorld(gamma=0.9)
    theta, v = actor_critic(env2, episodes=500, seed=0)
    pol = np.argmax(theta, axis=1)
    print("\nChính sách actor học được sau 500 episode:")
    for r in range(env2.rows):
        row = []
        for c in range(env2.cols):
            s = r * env2.cols + c
            row.append("G" if env2.is_terminal(s) else "^v<>"[pol[s]])
        print("  " + " ".join(row))

    print("\nNhận xét: actor–critic cập nhật MỖI BƯỚC nên học nhanh hơn")
    print("REINFORCE (cập nhật cuối episode). Cái giá: critic có thể chệch,")
    print("và hai mạng phải cân bằng tốc độ học — đây là vấn đề muôn thuở")
    print("của mọi thuật toán actor–critic hiện đại.")
