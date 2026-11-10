"""
================================================================================
 17 — OFF-POLICY n-STEP SARSA                              (Sutton & Barto, Ch. 7.3)
================================================================================

BÀI TOÁN
    Học giá trị của chính sách MỤC TIÊU π (tham lam) từ dữ liệu sinh bởi
    chính sách HÀNH VI b (ε-mềm), dùng n-step return.

Ý TƯỞNG
    Giống off-policy Monte Carlo (file 09), nhưng thay vì nhân tỉ số cho cả
    episode, ta chỉ nhân cho n bước gần nhất:

        ρ_{t:t+n-1} = Π_{k=t}^{t+n-1} π(A_k|S_k) / b(A_k|S_k)

    và cập nhật:

        Q(S_τ,A_τ) ← Q + α · ρ · [ G_{τ:τ+n} - Q(S_τ,A_τ) ]

    với G dùng `max` ở cuối (vì π là tham lam):

        G = Σ_{i=τ+1}^{τ+n} γ^{i-τ-1} R_i + γ^n max_a Q(S_{τ+n}, a)

    Điểm hay: chỉ n tỉ số thay vì cả episode -> phương sai thấp hơn nhiều so
    với off-policy Monte Carlo, trong khi vẫn giữ được tính off-policy.

PSEUDOCODE
    Lặp với mỗi episode:
        khởi tạo S0, chọn A0 ~ b
        T ← ∞
        với t = 0,1,2,...:
            nếu t < T:
                thực hiện A_t, quan sát R_{t+1}, S_{t+1}
                nếu terminal: T ← t+1, ngược lại chọn A_{t+1} ~ b
            τ ← t - n + 1
            nếu τ ≥ 0:
                ρ ← Π_{i=τ+1}^{min(τ+n-1,T-1)} π(A_i|S_i)/b(A_i|S_i)
                G ← Σ_{i=τ+1}^{min(τ+n,T)} γ^{i-τ-1} R_i
                nếu τ+n < T: G ← G + γ^n max_a Q(S_{τ+n},a)
                Q(S_τ,A_τ) ← Q + α ρ [G - Q]

CÁCH ĐỌC CODE
    - `rho` được tính lại từ đầu mỗi lần cập nhật (n nhỏ nên rẻ và rõ ràng).
    - `rho` bị chặn trên (clip) để tránh bùng nổ khi b và π khác nhau nhiều —
      một mẹo kỹ thuật phổ biến, sách cũng thảo luận ở mục 7.4.
    - π được cập nhật thành tham lam theo Q sau mỗi lần cập nhật (GPI).

VÌ SAO CẦN ρ?
    Dữ liệu đến từ b, nhưng ta muốn ước lượng kỳ vọng dưới π. Tỉ số
    π(a|s)/b(a|s) chính là hệ số "đổi trọng số" để hiệu chỉnh phân phối.
    Nếu π chọn một hành động mà b không bao giờ chọn -> ρ = 0, mẫu đó bị bỏ.

CHẠY THỬ
    python 17_off_policy_n_step_sarsa.py
================================================================================
"""

import numpy as np
from common import GridWorld


def off_policy_n_step_sarsa(env, episodes=40, n=4, alpha=0.2, gamma=0.9,
                            eps_behavior=0.3, rho_clip=10.0, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.nS, env.nA))
    policy = np.zeros(env.nS, dtype=int)       # π: tham lam theo Q

    def behavior_probs(s):
        p = np.full(env.nA, eps_behavior / env.nA)
        p[policy[s]] += 1.0 - eps_behavior
        return p

    def target_prob(s, a):
        return 1.0 if a == policy[s] else 0.0

    for _ in range(episodes):
        states = [env.reset()]
        actions = [int(rng.choice(env.nA, p=behavior_probs(states[0])))]
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
                    actions.append(int(rng.choice(env.nA, p=behavior_probs(s2))))

            tau = t - n + 1
            if tau >= 0:
                # (1) tỉ số quan trọng cho n-1 hành động đầu
                rho = 1.0
                for i in range(tau + 1, int(min(tau + n - 1, T - 1)) + 1):
                    b = behavior_probs(states[i])[actions[i]]
                    if b == 0.0:
                        rho = 0.0
                        break
                    rho *= target_prob(states[i], actions[i]) / b
                rho = min(rho, rho_clip)

                # (2) n-step return, bước cuối lấy max
                G = 0.0
                for i in range(tau + 1, int(min(tau + n, T)) + 1):
                    G += gamma ** (i - tau - 1) * rewards[i]
                if tau + n < T:
                    G += gamma ** n * Q[states[tau + n]].max()

                # (3) cập nhật có trọng số
                s_tau, a_tau = states[tau], actions[tau]
                Q[s_tau, a_tau] += alpha * rho * (G - Q[s_tau, a_tau])
                policy[s_tau] = int(np.argmax(Q[s_tau]))

            if tau == T - 1:
                break
            t += 1

    return Q, policy


def steps_to_goal(env, policy, cap=200):
    s, n = env.reset(), 0
    while not env.is_terminal(s) and n < cap:
        r, s2, done = env.step(s, policy[s])
        s = s2
        n += 1
    return n if env.is_terminal(s) else cap


if __name__ == "__main__":
    print("=" * 74)
    print(" OFF-POLICY n-STEP SARSA — lưới 4x4, α=0.2, ε_hành_vi = 0.3")
    print(" Số bước tới đích (tối ưu = 6), trung bình 10 lần chạy")
    print("=" * 74)

    env = GridWorld(gamma=0.9)
    ns = (1, 2, 4, 8)
    print(f"\n{'episode':>8}" + "".join(f"{'n=' + str(n):>12}" for n in ns))
    print("-" * 56)

    for episodes in (10, 20, 40):
        row = []
        for n in ns:
            scores = []
            for seed in range(10):
                Q, pol = off_policy_n_step_sarsa(env, episodes=episodes, n=n,
                                                 gamma=0.9, alpha=0.2, seed=seed)
                scores.append(steps_to_goal(env, pol))
            row.append(np.mean(scores))
        print(f"{episodes:>8}" + "".join(f"{v:>12.1f}" for v in row))

    print("\nNhận xét: giống n-step Sarsa, n vừa phải (2–4) học nhanh hơn hẳn")
    print("n = 1. Nhưng n = 8 bắt đầu kém ổn định vì tích nhiều tỉ số quan trọng")
    print("-> phương sai tăng (đúng cảnh báo ở mục 7.3).")
    print("Đây là lý do sách đề xuất tree backup / Q(σ) ở các mục sau.")
