"""
================================================================================
 03 — GRADIENT BANDIT (SOFTMAX)                           (Sutton & Barto, Ch. 2.8)
================================================================================

BÀI TOÁN
    Bandit k tay, nhưng lần này ta KHÔNG ước lượng giá trị q*(a). Thay vào đó
    ta học trực tiếp một "điểm ưa thích" cho mỗi tay, rồi chọn theo xác suất.

VÌ SAO CẦN?
    ε-greedy và UCB đều dựa trên ước lượng giá trị. Gradient bandit cho thấy
    một hướng khác: tối ưu chính sách bằng gradient ascent trên kỳ vọng phần
    thưởng. Đây chính là mầm mống của policy gradient (Chương 13) — cùng một
    công thức, chỉ khác ở chỗ dùng giá trị thay cho phần thưởng tức thời.

CÔNG THỨC
    Điểm ưa thích H_t(a), xác suất chọn theo softmax:

        π_t(a) = exp(H_t(a)) / Σ_b exp(H_t(b))

    Sau khi nhận thưởng R_t, cập nhật:

        H_{t+1}(A_t) = H_t(A_t) + α (R_t - R̄_t)(1 - π_t(A_t))
        H_{t+1}(a)   = H_t(a)   - α (R_t - R̄_t) π_t(a)        với a ≠ A_t

    trong đó R̄_t là phần thưởng trung bình (baseline) — cập nhật cùng α:

        R̄_{t+1} = R̄_t + α (R_t - R̄_t)

    Ý nghĩa: nếu R_t > trung bình, tăng xác suất hành động vừa làm; nếu kém
    hơn trung bình, giảm. Baseline không làm thay đổi kỳ vọng của gradient
    nhưng giảm phương sai rất nhiều — đúng vai trò của `baseline` ở Chương 13.

PSEUDOCODE
    Khởi tạo H(a) = 0, R̄ = 0
    Lặp t = 1, 2, ...:
        π <- softmax(H)
        A ~ π
        R <- Bandit(A)
        R̄ <- R̄ + α(R - R̄)
        với mọi a:
            nếu a == A: H(a) += α(R - R̄)(1 - π(a))
            ngược lại  : H(a) -= α(R - R̄) π(a)

ĐIỂM CẦN NHỚ
    - Gradient bandit KHÔNG cần biết giá trị thật; nó chỉ cần biết hành động
      vừa làm tốt hay kém hơn mức trung bình.
    - Bỏ baseline đi thì vẫn học được, nhưng chậm và nhiễu hơn nhiều.

CHẠY THỬ
    python 03_gradient_bandit.py
================================================================================
"""

import numpy as np
from common import Bandit, softmax


def run_gradient_bandit(k=10, steps=1000, alpha=0.1, baseline=True, seed=0):
    bandit = Bandit(k=k, seed=seed)
    rng = np.random.default_rng(seed + 1)

    H = np.zeros(k)        # điểm ưa thích
    R_bar = 0.0            # phần thưởng trung bình (baseline)

    rewards = np.zeros(steps)
    optimal = np.zeros(steps, dtype=bool)

    for t in range(steps):
        # (1) chuyển điểm ưa thích thành xác suất rồi lấy mẫu hành động
        pi = softmax(H)
        a = int(rng.choice(k, p=pi))

        # (2) nhận thưởng
        r = bandit.step(a)

        # (3) cập nhật baseline
        R_bar += alpha * (r - R_bar)
        advantage = r - (R_bar if baseline else 0.0)

        # (4) gradient ascent trên log-xác suất: chỉ hành động vừa làm được
        #     cộng thêm, các hành động khác bị trừ đi một lượng tỉ lệ π(a).
        onehot = np.zeros(k)
        onehot[a] = 1.0
        H += alpha * advantage * (onehot - pi)

        rewards[t] = r
        optimal[t] = (a == bandit.best)

    return rewards, optimal


def compare(runs=30):
    rows = [
        ("có baseline",    dict(alpha=0.1, baseline=True)),
        ("không baseline", dict(alpha=0.1, baseline=False)),
        ("α = 0.4",        dict(alpha=0.4, baseline=True)),
    ]
    print(f"{'cấu hình':<20}{'thưởng TB':>12}{'% tối ưu':>12}")
    print("-" * 44)
    for name, kw in rows:
        rs, os_ = [], []
        for seed in range(runs):
            r, o = run_gradient_bandit(seed=seed, **kw)
            rs.append(r[-100:].mean())
            os_.append(o[-100:].mean())
        print(f"{name:<20}{np.mean(rs):>12.3f}{100 * np.mean(os_):>11.1f}%")


if __name__ == "__main__":
    print("=" * 44)
    print(" GRADIENT BANDIT (10 tay, 1000 bước, 30 lần chạy)")
    print("=" * 44)
    compare()
    print()
    print("Nhận xét: có baseline thì học nhanh và ổn định hơn rõ rệt.")
    print("Đây là lý do mọi thuật toán policy gradient hiện đại đều dùng baseline.")
