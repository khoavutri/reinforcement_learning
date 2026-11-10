"""
================================================================================
 25 — LEAST-SQUARES TD (LSTD)                             (Sutton & Barto, Ch. 9.8)
================================================================================

BÀI TOÁN
    Semi-gradient TD cập nhật từng bước, cần nhiều mẫu. Có cách nào DÙNG HẾT
    dữ liệu cùng lúc để giải trực tiếp không?

Ý TƯỞNG
    Với xấp xỉ tuyến tính, TD(0) có nghiệm "bình phương tối thiểu" thoả:

        A w = b
        A = Σ_t x_t (x_t − γ x_{t+1})ᵀ
        b = Σ_t x_t R_{t+1}

    Vậy chỉ cần cộng dồn A và b, rồi giải hệ:

        w = A⁻¹ b

    Đây là LSTD. Không có step-size α, không cần lặp — nhưng mỗi bước tốn
    O(d²) để cập nhật A, và O(d³) khi giải (thực tế dùng công thức Sherman-
    Morrison để cập nhật nghịch đảo, O(d²)).

    Kết quả: LSTD dùng mẫu hiệu quả hơn TD rất nhiều, nhưng đắt hơn về tính
    toán và bộ nhớ. Sách gọi đây là "con đường thứ hai" của xấp xỉ tuyến tính.

CÔNG THỨC
    A ← A + x_t (x_t − γ x_{t+1})ᵀ
    b ← b + x_t R_{t+1}
    w ← A⁻¹ b        (thêm λI để ổn định số học)

PSEUDOCODE (sách, hộp 9.2)
    Khởi tạo A = 0 (d×d), b = 0 (d)
    Lặp với mỗi episode:
        khởi tạo S; x ← x(S)
        lặp mỗi bước:
            A ← A + x (x − γx(S'))ᵀ
            b ← b + x R
            x ← x(S')
    w ← A⁻¹ b

CÁCH ĐỌC CODE
    - `A` là ma trận dxd, `b` vector d. Với one-hot, d = nS.
    - `np.linalg.solve(A + ridge*I, b)` thay cho nghịch đảo trực tiếp.
    - So sánh RMSE với semi-gradient TD (file 22) ở cùng số episode.

CHẠY THỬ
    python 25_lstd.py
================================================================================
"""

import numpy as np
from common import RandomWalk, one_hot


def lstd(env, policy, episodes=50, gamma=1.0, ridge=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    d = env.nS
    A = np.zeros((d, d))
    b = np.zeros(d)

    for _ in range(episodes):
        s = env.reset()
        x = one_hot(s, d)

        while not env.is_terminal(s):
            a = int(rng.choice(env.nA, p=policy))
            r, s2, done = env.step(s, a)
            x2 = np.zeros(d) if done else one_hot(s2, d)

            # (1) cộng dồn hệ phương trình chuẩn
            A += np.outer(x, x - gamma * x2)
            b += x * r

            x = x2
            s = s2

    # (2) giải A w = b
    return np.linalg.solve(A + ridge * np.eye(d), b)


if __name__ == "__main__":
    print("=" * 68)
    print(" LSTD — RandomWalk 5 ô, feature one-hot")
    print("=" * 68)

    env = RandomWalk(n=5)
    policy = np.array([0.5, 0.5])
    truth = np.array([s / (env.n + 1) for s in range(env.nS)])

    print(f"\nGiá trị thật: {np.round(truth[1:-1], 3)}\n")
    print(f"{'episode':>8}{'RMSE LSTD':>14}{'RMSE TD(0)':>14}")
    print("-" * 36)

    from importlib import import_module
    td = import_module("22_semi_gradient_td")

    for episodes in (10, 50, 200):
        errs_lstd, errs_td = [], []
        for seed in range(10):
            w = lstd(env, policy, episodes=episodes, seed=seed)
            errs_lstd.append(np.sqrt(np.mean((w[1:-1] - truth[1:-1]) ** 2)))

            w_td = td.semi_gradient_td(env, policy, episodes=episodes, alpha=0.05, seed=seed)
            errs_td.append(np.sqrt(np.mean((w_td[1:-1] - truth[1:-1]) ** 2)))

        print(f"{episodes:>8}{np.mean(errs_lstd):>14.4f}{np.mean(errs_td):>14.4f}")

    print("\nNhận xét: LSTD đạt sai số thấp chỉ sau ~10 episode, trong khi TD(0)")
    print("cần hàng trăm. Cái giá: mỗi bước O(d²) thay vì O(d), và phải giải")
    print("hệ phương trình — không phù hợp khi d rất lớn (như mạng nơ-ron).")
