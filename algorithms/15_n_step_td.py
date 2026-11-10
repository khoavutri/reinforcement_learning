"""
================================================================================
 15 — n-STEP TD                                            (Sutton & Barto, Ch. 7.1)
================================================================================

BÀI TOÁN
    TD(0) nhìn 1 bước. Monte Carlo nhìn tới cuối episode. Có gì ở giữa không?

Ý TƯỞNG
    Thay vì bootstrap ngay sau 1 bước, hãy đi thêm n bước rồi mới bootstrap:

        G_{t:t+n} = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})

        V(S_t) ← V(S_t) + α [ G_{t:t+n} - V(S_t) ]

    - n = 1  -> TD(0)                       (bootstrap nhiều, chệch, ít nhiễu)
    - n → ∞  -> Monte Carlo                  (không bootstrap, không chệch, nhiều nhiễu)
    - n = 4..10 thường là tốt nhất trong thực tế (sách, hình 7.2).

    Trực giác: đi thêm vài bước thì "phần thật" của target nhiều lên, phần
    "phần dự đoán" ít đi -> giảm chệch, nhưng phương sai tăng lên.

CÔNG THỨC TỔNG QUÁT
    G_{t:t+n} = Σ_{k=0}^{n-1} γ^k R_{t+k+1} + γ^n V(S_{t+n})
    (nếu episode kết thúc trước t+n thì G = return thật tới lúc kết thúc)

PSEUDOCODE
    Input: π, α, n
    Khởi tạo V(s) tuỳ ý, V(terminal) = 0
    Lặp với mỗi episode:
        khởi tạo S0
        T ← ∞
        với t = 0, 1, 2, ...:
            nếu t < T:
                thực hiện A_t theo π, quan sát R_{t+1}, S_{t+1}
                nếu S_{t+1} là terminal: T ← t+1
            τ ← t - n + 1
            nếu τ ≥ 0:
                G ← Σ_{i=τ+1}^{min(τ+n,T)} γ^{i-τ-1} R_i
                nếu τ+n < T: G ← G + γ^n V(S_{τ+n})
                V(S_τ) ← V(S_τ) + α[G - V(S_τ)]
            cho tới khi τ = T-1

CÁCH ĐỌC CODE
    - `G` được tính bằng vòng lặp cộng dồn, sau đó cộng phần bootstrap.
    - `tau = t - n + 1` là thời điểm đang được cập nhật (trễ n bước).
    - Mảng `states` và `rewards` lưu lịch sử n bước gần nhất.

LƯU Ý
    n-step cần NHỚ n trạng thái và n phần thưởng gần nhất, và phải đợi n
    bước mới cập nhật. Đó là cái giá phải trả để có target tốt hơn.

CHẠY THỬ
    python 15_n_step_td.py
================================================================================
"""

import numpy as np
from common import RandomWalk


def n_step_td(env, policy, episodes=100, n=4, alpha=0.1, gamma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    V = np.zeros(env.nS)

    for _ in range(episodes):
        states = [env.reset()]
        rewards = [0.0]
        T = float("inf")
        t = 0

        while True:
            if t < T:
                s = states[t]
                a = int(rng.choice(env.nA, p=policy))
                r, s2, done = env.step(s, a)
                rewards.append(r)
                states.append(s2)
                if done:
                    T = t + 1

            tau = t - n + 1
            if tau >= 0:
                # (1) cộng dồn n phần thưởng
                G = 0.0
                for i in range(tau + 1, int(min(tau + n, T)) + 1):
                    G += gamma ** (i - tau - 1) * rewards[i]
                # (2) bootstrap nếu chưa chạm cuối episode
                if tau + n < T:
                    G += gamma ** n * V[states[tau + n]]

                # (3) cập nhật trạng thái cách đây n bước
                V[states[tau]] += alpha * (G - V[states[tau]])

            if tau == T - 1:
                break
            t += 1

    return V


if __name__ == "__main__":
    print("=" * 68)
    print(" n-STEP TD trên RandomWalk 5 ô (20 lần chạy, 100 episode)")
    print("=" * 68)

    env = RandomWalk(n=5)
    policy = np.array([0.5, 0.5])
    truth = np.array([s / (env.n + 1) for s in range(env.nS)])

    print(f"\n{'n':>4}{'RMSE sau 10 ep':>18}{'RMSE sau 100 ep':>18}   ghi chú")
    print("-" * 62)
    for n in (1, 2, 4, 8, 16):
        e10 = [np.sqrt(np.mean((n_step_td(env, policy, episodes=10, n=n, seed=s)[1:-1]
                                - truth[1:-1]) ** 2)) for s in range(50)]
        e100 = [np.sqrt(np.mean((n_step_td(env, policy, episodes=100, n=n, seed=s)[1:-1]
                                 - truth[1:-1]) ** 2)) for s in range(50)]
        note = "= TD(0)" if n == 1 else ("gần Monte Carlo" if n >= 16 else "")
        print(f"{n:>4}{np.mean(e10):>18.4f}{np.mean(e100):>18.4f}   {note}")

    print("\nNhận xét (đúng như hình 7.2 trong sách):")
    print("  - Khi MỚI học (10 episode), n lớn thắng: phần thưởng cuối episode")
    print("    được lan ngược về nhiều trạng thái chỉ trong vài lần chơi.")
    print("  - Khi đã học lâu (100 episode) với α cố định, n nhỏ lại thắng vì")
    print("    bootstrap ít nhiễu hơn. 'Điểm ngọt' phụ thuộc vào α và số episode.")
