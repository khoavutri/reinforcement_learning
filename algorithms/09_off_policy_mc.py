"""
================================================================================
 09 — OFF-POLICY MONTE CARLO + IMPORTANCE SAMPLING        (Sutton & Barto, Ch. 5.5–5.7)
================================================================================

BÀI TOÁN
    Ta muốn đánh giá (hoặc tối ưu) chính sách MỤC TIÊU π, nhưng dữ liệu lại
    được sinh ra bởi chính sách HÀNH VI b (b ≠ π). Ví dụ: dữ liệu là log của
    một hệ thống cũ, còn ta muốn đánh giá chính sách mới.

Ý TƯỞNG — IMPORTANCE SAMPLING
    Xác suất một episode xảy ra dưới π khác dưới b. Để "quy đổi" return quan
    sát được về phân phối của π, nhân nó với tỉ số:

        ρ_{t:T-1} = Π_{k=t}^{T-1}  π(A_k|S_k) / b(A_k|S_k)

    - Nếu episode chứa hành động mà π không bao giờ làm -> ρ = 0 -> bỏ mẫu đó.
    - ORDINARY IS:  V(s) = trung bình của (ρ · G)
      Không chệch nhưng có thể PHƯƠNG SAI VÔ HẠN (sách, ví dụ 5.5).
    - WEIGHTED IS:  V(s) = Σ(ρ·G) / Σρ
      Chệch nhẹ nhưng phương sai thấp; trong thực hành gần như luôn tốt hơn.

CÔNG THỨC CẬP NHẬT TĂNG DẦN (weighted)
        C(s)   ← C(s) + W
        V(s)   ← V(s) + (W / C(s)) · [ G - V(s) ]
        W      ← W · π(A_t|S_t) / b(A_t|S_t)

    `W` là tích các tỉ số từ hiện tại tới cuối episode; `C` là tổng trọng số
    (chứ không phải số lần đếm — đó chính là điểm khác biệt của weighted IS).

PSEUDOCODE (off-policy MC control — sách, mục 5.7)
    Khởi tạo Q(s,a), C(s,a) = 0, π(s) = argmax_a Q(s,a)
    Lặp mãi:
        b ← một chính sách mềm bất kỳ
        sinh episode theo b: S0,A0,R1,...,S_{T-1},A_{T-1},R_T
        G ← 0 ; W ← 1
        với t = T-1 ... 0:
            G ← γG + R_{t+1}
            C(S_t,A_t) ← C(S_t,A_t) + W
            Q(S_t,A_t) ← Q + (W/C)·[G - Q]
            π(S_t) ← argmax_a Q(S_t,a)
            nếu A_t ≠ π(S_t): thoát vòng lặp
            W ← W · 1/b(A_t|S_t)

    Chú ý bước cuối: chỉ nhân với 1/b, vì π mới là tham lam nên
    π(A_t|S_t) = 1 với hành động được chọn.

CÁCH ĐỌC CODE
    - `off_policy_mc_prediction`: dùng cho RandomWalk, so sánh ordinary vs weighted.
    - `off_policy_mc_control`: học điều khiển trên lưới với b là ε-mềm.
    - Trong control, việc `break` khi gặp hành động không tham lam là bắt buộc:
      phần đuôi episode đó không còn "khớp" với π.

NHƯỢC ĐIỂM
    - Chỉ học được từ ĐUÔI episode (khi mọi hành động còn lại đều tham lam).
    - Nếu π và b khác nhau nhiều, trọng số W bùng nổ -> phương sai lớn.

CHẠY THỬ
    python 09_off_policy_mc.py
================================================================================
"""

import numpy as np
from common import Corridor, RandomWalk, argmax_tie


# ---------------------------------------------------------------------------
# 1. ĐÁNH GIÁ (prediction) — ordinary vs weighted importance sampling
# ---------------------------------------------------------------------------
def off_policy_mc_prediction(env, target_policy, behavior_policy,
                             episodes=2000, gamma=1.0, weighted=True, seed=0):
    rng = np.random.default_rng(seed)

    V = np.zeros(env.nS)
    C = np.zeros(env.nS)      # tổng trọng số (weighted IS)

    for _ in range(episodes):
        s = env.reset()
        states, actions, rewards = [], [], []
        while not env.is_terminal(s):
            a = int(rng.choice(env.nA, p=behavior_policy(s)))
            r, s2, done = env.step(s, a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = s2

        G, W = 0.0, 1.0
        for t in range(len(states) - 1, -1, -1):
            G = gamma * G + rewards[t]
            s, a = states[t], actions[t]

            if weighted:
                # (1) weighted IS: trung bình có trọng số
                C[s] += W
                V[s] += (W / C[s]) * (G - V[s])
            else:
                # (2) ordinary IS: trung bình cộng của ρ·G
                C[s] += 1
                V[s] += (W * G - V[s]) / C[s]

            # (3) cập nhật trọng số cho bước trước đó
            W *= target_policy(s)[a] / behavior_policy(s)[a]
            if W == 0.0:
                break     # episode không thể xảy ra dưới π -> bỏ phần còn lại

    return V


# ---------------------------------------------------------------------------
# 2. ĐIỀU KHIỂN (control) — off-policy MC với weighted IS
# ---------------------------------------------------------------------------
def off_policy_mc_control(env, episodes=20_000, gamma=0.9, eps_behavior=0.5,
                          max_steps=200, seed=0):
    rng = np.random.default_rng(seed)

    Q = np.zeros((env.nS, env.nA))
    C = np.zeros((env.nS, env.nA))
    # khởi tạo π NGẪU NHIÊN: nếu khởi tạo tất định (ví dụ toàn "đi lên"),
    # chính sách mục tiêu có thể không bao giờ tới đích -> không có dữ liệu.
    policy = np.array([int(rng.integers(env.nA)) for _ in range(env.nS)])

    def behavior(s):
        p = np.full(env.nA, eps_behavior / env.nA)
        p[policy[s]] += 1.0 - eps_behavior
        return p

    for _ in range(episodes):
        s = env.reset()
        states, actions, rewards = [], [], []
        done, steps = False, 0
        while not done and steps < max_steps:
            a = int(rng.choice(env.nA, p=behavior(s)))
            r, s2, done = env.step(s, a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = s2
            steps += 1

        if not done:
            continue

        G, W = 0.0, 1.0
        for t in range(len(states) - 1, -1, -1):
            G = gamma * G + rewards[t]
            s, a = states[t], actions[t]

            C[s, a] += W
            Q[s, a] += (W / C[s, a]) * (G - Q[s, a])
            policy[s] = argmax_tie(Q[s], rng)

            if a != policy[s]:
                break                      # chỉ học từ đuôi episode
            W /= behavior(s)[a]

    return policy, Q


if __name__ == "__main__":
    print("=" * 66)
    print(" OFF-POLICY MC PREDICTION — RandomWalk")
    print("  π = luôn đi phải (V_thật = 1) ; b = đi trái/phải 50-50")
    print("=" * 66)

    env = RandomWalk(n=5)
    pi_target = lambda s: np.array([0.0, 1.0])          # luôn sang phải
    b_behavior = lambda s: np.array([0.5, 0.5])         # ngẫu nhiên đều

    # Chạy nhiều seed để thấy phương sai
    for name, weighted in [("ordinary", False), ("weighted", True)]:
        estimates = []
        for seed in range(30):
            V = off_policy_mc_prediction(env, pi_target, b_behavior,
                                         episodes=300, weighted=weighted, seed=seed)
            estimates.append(V[env.start])
        estimates = np.array(estimates)
        print(f"{name:<10} trung vị = {np.median(estimates):.3f}   "
              f"min = {estimates.min():.3f}   max = {estimates.max():.3f}   "
              f"(giá trị thật = 1.0)")

    print("\nNhận xét: ordinary IS đúng về kỳ vọng nhưng phương sai vô hạn —")
    print("một mẫu có trọng số khổng lồ đủ làm hỏng cả ước lượng (đúng như")
    print("ví dụ 5.5 và hình 5.4 trong sách). Weighted IS ổn định hơn hẳn.")

    print()
    print("=" * 66)
    print(" OFF-POLICY MC CONTROL — hành lang 5 ô, 5.000 episode")
    print("=" * 66)

    cor = Corridor(n=5, step_reward=-0.1)
    policy, Q = off_policy_mc_control(cor, episodes=5000, gamma=0.9)

    print("Q(s, trái/phải) và hành động tham lam:")
    for s in range(1, cor.n + 1):
        print(f"  s={s}: {Q[s, 0]:+.3f} / {Q[s, 1]:+.3f}   -> "
              f"{'phải' if policy[s] == 1 else 'trái'}")

    print("\nNhận xét: π hội tụ về 'luôn đi phải' — tối ưu.")
    print("Nhưng chú ý: off-policy MC chỉ học được từ ĐUÔI episode, nên nếu")
    print("hành động không tham lam xuất hiện sớm thì phần đầu episode bị bỏ.")
    print("Đây chính là lý do sách chuyển sang TD ở chương 6.")
