"""
================================================================================
 19 — n-STEP Q(σ)                                          (Sutton & Barto, Ch. 7.6)
================================================================================

BÀI TOÁN
    Sarsa (sample hoàn toàn), Expected Sarsa (kỳ vọng ở bước cuối) và Tree
    Backup (kỳ vọng hoàn toàn) là ba điểm trên cùng một trục. Có thể nội suy
    giữa chúng một cách mượt mà không?

Ý TƯỞNG
    Đưa vào một tham số σ ∈ [0,1] cho từng bước, quyết định "bước này lấy
    mẫu hay lấy kỳ vọng":

        σ = 1  ->  lấy mẫu  (Sarsa)
        σ = 0  ->  lấy kỳ vọng (Tree Backup)

    σ có thể là hằng số, hoặc ngẫu nhiên từng bước, hoặc phụ thuộc trạng thái.
    Sách gọi đây là "thuật toán thống nhất" vì nó bao trùm cả ba thuật toán
    trước đó chỉ bằng một công thức.

CÔNG THỨC (7.17)
    G_{t:h} = R_{t+1}
              + γ ( σ_{t+1}ρ_{t+1} + (1-σ_{t+1}) π(A_{t+1}|S_{t+1}) )
                ( G_{t+1:h} - Q(S_{t+1},A_{t+1}) )
              + γ V̄(S_{t+1})

    với V̄(s) = Σ_a π(a|s) Q(s,a), và ρ là tỉ số importance sampling.

    Điểm tinh tế: số hạng `(G - Q)` là "control variate" — nó giữ cho ước
    lượng không chệch trong khi vẫn giảm phương sai. Khi σ=1 và on-policy
    (ρ=1), công thức rút gọn thành G = R + γG' — đúng là Sarsa.

PSEUDOCODE
    Lặp với mỗi episode:
        khởi tạo S0, chọn A0 ~ b
        T ← ∞
        với t = 0,1,2,...:
            nếu t < T: thực hiện A_t, quan sát R_{t+1}, S_{t+1}
                       nếu terminal: T ← t+1, ngược lại chọn A_{t+1} ~ b
            chọn σ_{t+1}, lưu ρ_{t+1} = π(A_{t+1}|S_{t+1}) / b(A_{t+1}|S_{t+1})
            τ ← t-n+1
            nếu τ ≥ 0:
                nếu t+1 < T: G ← Q(S_{t+1},A_{t+1})
                với k = min(t+1,T) giảm về τ+1:
                    nếu k = T: G ← R_T
                    ngược lại:
                        V̄ ← Σ_a π(a|S_k)Q(S_k,a)
                        G ← R_k + γ(σ_kρ_k + (1-σ_k)π(A_k|S_k))(G - Q(S_k,A_k)) + γV̄
                Q(S_τ,A_τ) ← Q + α[G - Q]

CÁCH ĐỌC CODE
    - `sigmas` và `rhos` lưu σ, ρ của từng bước (đánh index theo thời điểm).
    - Vòng `for k in range(min(t+1, T), tau, -1)` chạy ngược đúng như sách.
    - Với `sigma=1.0` cố định, kết quả gần với n-step Sarsa; với `sigma=0.0`,
      gần với tree backup. Đây là cách kiểm chứng cài đặt.

CHẠY THỬ
    python 19_n_step_q_sigma.py
================================================================================
"""

import numpy as np
from common import GridWorld, argmax_tie


def n_step_q_sigma(env, episodes=40, n=4, alpha=0.5, gamma=0.9, eps=0.1,
                   sigma=0.5, random_sigma=True, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.nS, env.nA))

    def policy_probs(s):
        p = np.full(env.nA, eps / env.nA)
        p[argmax_tie(Q[s], rng)] += 1.0 - eps
        return p

    def behavior_probs(s):
        return policy_probs(s)          # on-policy: b = π

    for _ in range(episodes):
        states = [env.reset()]
        actions = [int(rng.choice(env.nA, p=behavior_probs(states[0])))]
        rewards = [0.0]
        sigmas = [0.0]
        rhos = [1.0]

        T = float("inf")
        t = 0
        while True:
            if t > 300:              # chặn episode quá dài (chính sách bị kẹt)
                break
            if t < T:
                r, s2, done = env.step(states[t], actions[t])
                rewards.append(r)
                states.append(s2)
                if done:
                    T = t + 1
                else:
                    a2 = int(rng.choice(env.nA, p=behavior_probs(s2)))
                    actions.append(a2)
                    sigmas.append(rng.random() if random_sigma else sigma)
                    rhos.append(policy_probs(s2)[a2] / behavior_probs(s2)[a2])

            tau = t - n + 1
            if tau >= 0:
                # (1) giá trị khởi đầu
                if t + 1 < T:
                    G = Q[states[t + 1], actions[t + 1]]
                else:
                    G = 0.0

                # (2) truy hồi ngược
                T_int = int(T) if T != float("inf") else t + 1
                for k in range(min(t + 1, T_int), tau, -1):
                    if k == T_int:
                        G = rewards[T_int]
                    else:
                        s_k, a_k = states[k], actions[k]
                        p = policy_probs(s_k)
                        v_bar = float(p @ Q[s_k])
                        factor = sigmas[k] * rhos[k] + (1.0 - sigmas[k]) * p[a_k]
                        G = (rewards[k] + gamma * factor * (G - Q[s_k, a_k])
                             + gamma * v_bar)

                # (3) cập nhật
                s_tau, a_tau = states[tau], actions[tau]
                Q[s_tau, a_tau] += alpha * (G - Q[s_tau, a_tau])

            if tau == T - 1:
                break
            t += 1

    return Q


def steps_to_goal(env, policy, cap=200):
    s, n = env.reset(), 0
    while not env.is_terminal(s) and n < cap:
        r, s2, done = env.step(s, policy[s])
        s = s2
        n += 1
    return n if env.is_terminal(s) else cap


if __name__ == "__main__":
    print("=" * 72)
    print(" n-STEP Q(σ) — lưới 4x4, σ thay đổi quyết định mức độ 'lấy mẫu'")
    print(" Số bước tới đích sau 40 episode, TB 10 lần chạy (tối ưu = 6)")
    print("=" * 72)

    env = GridWorld(gamma=0.9)

    configs = [
        ("σ = 0.0  (tree backup)",  dict(sigma=0.0, random_sigma=False)),
        ("σ = 0.5  (pha trộn)",     dict(sigma=0.5, random_sigma=False)),
        ("σ = 1.0  (gần Sarsa)",    dict(sigma=1.0, random_sigma=False)),
        ("σ ~ U[0,1] mỗi bước",     dict(random_sigma=True)),
    ]

    print(f"\n{'cấu hình':<28}{'n=2':>10}{'n=4':>10}{'n=8':>10}")
    print("-" * 58)

    for name, kw in configs:
        row = []
        for n in (2, 4, 8):
            scores = []
            for seed in range(10):
                Q = n_step_q_sigma(env, episodes=40, n=n, seed=seed, **kw)
                pol = np.argmax(Q, axis=1)
                pol[env.terminal] = 0
                scores.append(steps_to_goal(env, pol))
            row.append(np.mean(scores))
        print(f"{name:<28}" + "".join(f"{v:>10.1f}" for v in row))

    print("\nNhận xét: trên bài toán tất định này, σ = 0 (tree backup) học nhanh")
    print("nhất còn σ = 1 (Sarsa) chậm nhất — vì target lấy mẫu có phương sai cao.")
    print("Q(σ) không tự động chọn σ tốt nhất: σ là một siêu tham số, giống α.")
    print("Giá trị của nó là cho phép nội suy có kiểm soát giữa hai cực.")
