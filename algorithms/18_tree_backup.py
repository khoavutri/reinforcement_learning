"""
================================================================================
 18 — n-STEP TREE BACKUP                                   (Sutton & Barto, Ch. 7.5)
================================================================================

BÀI TOÁN
    Học off-policy (hoặc on-policy) với n-step, nhưng KHÔNG dùng importance
    sampling — vì importance sampling có phương sai cao khi π và b khác nhau.

Ý TƯỞNG
    Thay vì nhân tỉ số xác suất, hãy "mở rộng cây": với mỗi bước, lấy KỲ VỌNG
    trên tất cả hành động có thể, trừ hành động đã thực sự xảy ra (thì giữ
    nhánh đó). Kết quả là một cây các khả năng, không cần biết b là gì.

    Công thức truy hồi (7.16):

        G_{t:h} = R_{t+1} + γ Σ_{a≠A_{t+1}} π(a|S_{t+1}) Q(S_{t+1},a)
                        + γ π(A_{t+1}|S_{t+1}) G_{t+1:h}

    Nói cách khác: mọi nhánh "không đi" được lấy kỳ vọng theo π; nhánh
    "đã đi" được giữ nguyên và đệ quy tiếp.

    Khi π là tham lam, tree backup chính là phiên bản nhiều bước của
    Q-learning (một bước); khi π ngẫu nhiên, nó lan toả theo cả cây.

PSEUDOCODE
    Khởi tạo Q(s,a), π (tham lam hoặc cho trước)
    Lặp với mỗi episode:
        khởi tạo S0, chọn A0 tuỳ ý
        T ← ∞
        với t = 0,1,2,...:
            nếu t < T:
                thực hiện A_t, quan sát R_{t+1}, S_{t+1}
                nếu terminal: T ← t+1
                ngược lại: chọn A_{t+1} tuỳ ý
            τ ← t - n + 1
            nếu τ ≥ 0:
                nếu t+1 ≥ T: G ← R_T
                ngược lại:   G ← R_{t+1} + γ Σ_a π(a|S_{t+1})Q(S_{t+1},a)
                với k = min(t, T-1) giảm về τ+1:
                    G ← R_k + γ[ Σ_{a≠A_k} π(a|S_k)Q(S_k,a) + π(A_k|S_k)G ]
                Q(S_τ,A_τ) ← Q + α[G - Q]

CÁCH ĐỌC CODE
    - `backup` xây dựng G bằng cách đi NGƯỢC từ horizon về τ, mỗi bước
      thay một nhánh "đã đi" bằng giá trị đệ quy.
    - Không có biến `rho` nào trong file — đó chính là điểm mạnh của tree backup.
    - Vì π xuất hiện dưới dạng xác suất, thuật toán dùng được cả khi π ngẫu
      nhiên (ví dụ π ε-mềm), điều mà Q-learning không làm được.

CHẠY THỬ
    python 18_tree_backup.py
================================================================================
"""

import numpy as np
from common import GridWorld, argmax_tie, epsilon_greedy


def n_step_tree_backup(env, episodes=40, n=4, alpha=0.5, gamma=0.9, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.nS, env.nA))

    def pi_probs(s):
        """π ε-tham lam theo Q (vừa là chính sách mục tiêu vừa là hành vi)."""
        p = np.full(env.nA, eps / env.nA)
        p[argmax_tie(Q[s], rng)] += 1.0 - eps
        return p

    for _ in range(episodes):
        states = [env.reset()]
        actions = [int(rng.choice(env.nA, p=pi_probs(states[0])))]
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
                    actions.append(int(rng.choice(env.nA, p=pi_probs(s2))))

            tau = t - n + 1
            if tau >= 0:
                # (1) giá trị khởi đầu tại horizon
                if t + 1 >= T:
                    G = rewards[int(T)]
                else:
                    G = rewards[t + 1] + gamma * float(
                        pi_probs(states[t + 1]) @ Q[states[t + 1]]
                    )

                # (2) đi ngược, mỗi bước "mở" một nhánh
                T_int = int(T) if T != float("inf") else t + 1
                for k in range(min(t - 1, T_int - 2), tau - 1, -1):
                    s_k, a_k = states[k + 1], actions[k + 1]
                    p = pi_probs(s_k)
                    not_taken = float(p @ Q[s_k]) - p[a_k] * Q[s_k, a_k]
                    G = rewards[k + 1] + gamma * (not_taken + p[a_k] * G)

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
    print("=" * 70)
    print(" n-STEP TREE BACKUP — lưới 4x4, TB 20 lần chạy")
    print(" Số bước tới đích sau khi học (tối ưu = 6)")
    print("=" * 70)

    env = GridWorld(gamma=0.9)
    print(f"\n{'episode':>8}{'n=1':>10}{'n=2':>10}{'n=4':>10}{'n=8':>10}")
    print("-" * 48)

    for episodes in (10, 20, 40, 80):
        row = []
        for n in (1, 2, 4, 8):
            scores = []
            for seed in range(20):
                Q = n_step_tree_backup(env, episodes=episodes, n=n, seed=seed)
                pol = np.argmax(Q, axis=1)
                pol[env.terminal] = 0
                scores.append(steps_to_goal(env, pol))
            row.append(np.mean(scores))
        print(f"{episodes:>8}" + "".join(f"{v:>10.1f}" for v in row))

    print("\nNhận xét: tree backup học nhanh hơn Q-learning 1 bước (n=1) khi")
    print("mới bắt đầu, và KHÔNG cần importance sampling nên ổn định.")
    print("Đây là lựa chọn tốt khi chính sách mục tiêu là ngẫu nhiên (ε-mềm).")
