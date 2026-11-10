"""
================================================================================
 27 — TD(λ) VỚI ELIGIBILITY TRACE                       (Sutton & Barto, Ch. 12.2)
================================================================================

BÀI TOÁN
    n-step (chương 7) cần nhớ n bước và đợi n bước mới cập nhật. Có cách nào
    đạt hiệu ứng "trung bình mọi n" mà chỉ cần O(1) bộ nhớ và cập nhật mỗi
    bước không?

Ý TƯỞNG — ELIGIBILITY TRACE
    Ghi lại "trạng thái nào vừa góp phần gây ra TD error". Mỗi trạng thái có
    một độ "eligible" z(s), tăng khi được thăm và giảm dần theo γλ:

        z_t = γλ z_{t-1} + x(S_t)          (accumulating trace)
        w   ← w + α δ_t z_t

    - λ = 0  -> z = x(S_t), chỉ cập nhật trạng thái hiện tại = TD(0)
    - λ = 1  -> z cộng dồn mọi trạng thái đã thăm = gần Monte Carlo
    - 0 < λ < 1 -> trung bình có trọng số của mọi n-step return

    Điểm đẹp: MỘT phép cập nhật mỗi bước, nhưng tương đương với việc trung
    bình hoá tất cả các n-step return với trọng số (1−λ)λ^{n−1}. Không cần
    lưu n bước, không phải đợi.

CÔNG THỨC ĐẦY ĐỦ
    δ_t = R_{t+1} + γ v̂(S_{t+1},w) − v̂(S_t,w)
    z_t = γλ z_{t-1} + ∇v̂(S_t,w)
    w   ← w + α δ_t z_t

PSEUDOCODE (sách, hộp 12.1)
    Khởi tạo w tuỳ ý
    Lặp với mỗi episode:
        z ← 0
        khởi tạo S
        lặp mỗi bước:
            A ← π(·|S)
            thực hiện A, quan sát R, S'
            δ ← R + γv̂(S',w) − v̂(S,w)
            z ← γλz + ∇v̂(S,w)
            w ← w + αδz
            S ← S'

CÁCH ĐỌC CODE
    - `z` cùng kích thước với `w`, khởi tạo lại 0 mỗi episode.
    - Chỉ hai dòng khác TD(0): cập nhật z và nhân delta với z.
    - λ là siêu tham số mạnh: λ=0.8 thường cho kết quả tốt hơn cả 0 và 1.

CHẠY THỬ
    python 27_td_lambda.py
================================================================================
"""

import numpy as np
from common import RandomWalk, one_hot


def td_lambda(env, policy, episodes=20, alpha=0.05, lam=0.8, gamma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    d = env.nS
    w = np.zeros(d)

    for _ in range(episodes):
        z = np.zeros(d)                 # trace khởi tạo lại mỗi episode
        s = env.reset()
        x = one_hot(s, d)

        while not env.is_terminal(s):
            a = int(rng.choice(env.nA, p=policy))
            r, s2, done = env.step(s, a)
            x2 = np.zeros(d) if done else one_hot(s2, d)

            # (1) TD error
            delta = r + gamma * (w @ x2) - (w @ x)

            # (2) trace: già đi rồi cộng thêm trạng thái hiện tại
            z = gamma * lam * z + x

            # (3) cập nhật theo trace
            w += alpha * delta * z

            x = x2
            s = s2

    return w


if __name__ == "__main__":
    print("=" * 68)
    print(" TD(λ) — RandomWalk 5 ô, 10 episode, α=0.05")
    print("=" * 68)

    env = RandomWalk(n=5)
    policy = np.array([0.5, 0.5])
    truth = np.array([s / (env.n + 1) for s in range(env.nS)])

    print(f"\n{'λ':>6}{'RMSE (10 ep)':>18}{'RMSE (50 ep)':>18}")
    print("-" * 42)
    for lam in (0.0, 0.4, 0.8, 1.0):
        e10, e50 = [], []
        for seed in range(20):
            w = td_lambda(env, policy, episodes=10, lam=lam, seed=seed)
            e10.append(np.sqrt(np.mean((w[1:-1] - truth[1:-1]) ** 2)))
            w = td_lambda(env, policy, episodes=50, lam=lam, seed=seed)
            e50.append(np.sqrt(np.mean((w[1:-1] - truth[1:-1]) ** 2)))
        print(f"{lam:>6}{np.mean(e10):>18.4f}{np.mean(e50):>18.4f}")

    print("\nNhận xét: λ = 0 (TD(0)) học chậm nhất ở giai đoạn đầu. Sau 50")
    print("episode, λ ≈ 0.8 cho sai số nhỏ nhất; λ = 1 (gần Monte Carlo) tốt")
    print("lúc đầu nhưng chững lại vì phương sai cao. Đó là 'điểm ngọt' ở hình")
    print("12.3 trong sách: trace là cách đạt n-step với bộ nhớ O(1).")
