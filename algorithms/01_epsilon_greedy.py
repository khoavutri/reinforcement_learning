"""
================================================================================
 01 — ε-GREEDY VÀ CẬP NHẬT TRUNG BÌNH TĂNG DẦN          (Sutton & Barto, Ch. 2)
================================================================================

BÀI TOÁN
    "Máy đánh bạc k tay" (k-armed bandit): có k cần gạt, mỗi lần kéo một cần
    ta nhận phần thưởng ngẫu nhiên. Mỗi cần có giá trị trung bình q*(a) khác
    nhau nhưng ta KHÔNG biết. Mục tiêu: sau T lần kéo, tổng thưởng càng lớn
    càng tốt.

    Đây là bài toán RL "tối giản": không có trạng thái, chỉ có một quyết định
    lặp lại. Nhờ vậy ta nhìn rõ bản chất của đánh đổi khám phá / khai thác.

Ý TƯỞNG
    - Ước lượng giá trị mỗi tay bằng trung bình cộng các phần thưởng đã nhận.
    - Khi chọn: với xác suất ε chọn ngẫu nhiên (khám phá), còn lại chọn tay
      có ước lượng cao nhất (khai thác).
    - ε = 0  -> luôn tham lam, dễ kẹt ở tay "trông có vẻ tốt" do may mắn.
      ε = 0.1 -> khám phá 10% thời gian, thường tìm được tay tốt nhất.

CÔNG THỨC (sách, mục 2.4)
    Cập nhật trung bình tăng dần — không cần lưu toàn bộ lịch sử:

        Q_{n+1} = Q_n + (1/n) * [ R_n - Q_n ]                    (2.3)

    Dạng tổng quát xuất hiện khắp cuốn sách:

        Ước lượng mới <- Ước lượng cũ + bước * [ Target - Ước lượng cũ ]   (2.4)

    Khi bài toán KHÔNG dừng (phần thưởng đổi theo thời gian), dùng bước cố
    định α thay cho 1/n — đó là trung bình có trọng số mũ (mục 2.5):

        Q_{n+1} = Q_n + α * [ R_n - Q_n ]

PSEUDOCODE TRONG SÁCH
    Khởi tạo Q(a) tuỳ ý, N(a) = 0 cho mọi a
    Lặp mãi:
        A <- argmax_a Q(a)              (phá hoà ngẫu nhiên)
        với xác suất ε: A <- hành động ngẫu nhiên
        R <- Bandit(A)
        N(A) <- N(A) + 1
        Q(A) <- Q(A) + (1/N(A)) * [R - Q(A)]

CÁCH ĐỌC CODE
    - `N` và `Q` là hai mảng k phần tử: số lần đã thử và giá trị ước lượng.
    - Ba dòng cuối vòng lặp chính là công thức (2.3) ở trên.
    - Hàm `run_bandit` chạy nhiều lần với seed khác nhau rồi lấy trung bình,
      vì một lần chạy có thể may/rủi.

CHẠY THỬ
    python 01_epsilon_greedy.py
================================================================================
"""

import numpy as np
from common import Bandit, argmax_tie


def run_bandit(k=10, steps=1000, eps=0.1, alpha=None, seed=0):
    """Chạy một lượt ε-greedy trên bandit k tay.

    alpha = None  -> dùng trung bình cộng thật (1/N), đúng cho bài toán DỪNG
    alpha = số    -> dùng bước cố định, phù hợp bài toán KHÔNG DỪNG
    """
    bandit = Bandit(k=k, seed=seed)
    rng = np.random.default_rng(seed + 1)

    Q = np.zeros(k)      # ước lượng giá trị từng tay
    N = np.zeros(k)      # số lần đã kéo từng tay

    rewards = np.zeros(steps)
    optimal = np.zeros(steps, dtype=bool)

    for t in range(steps):
        # (1) chọn hành động: ε-tham lam
        if rng.random() < eps:
            a = int(rng.integers(k))            # khám phá
        else:
            a = argmax_tie(Q, rng)              # khai thác

        # (2) tương tác với môi trường
        r = bandit.step(a)

        # (3) cập nhật ước lượng — chính là công thức (2.3) / (2.4)
        N[a] += 1
        step_size = alpha if alpha is not None else 1.0 / N[a]
        Q[a] += step_size * (r - Q[a])

        rewards[t] = r
        optimal[t] = (a == bandit.best)

    return rewards, optimal


def experiment(configs, runs=30):
    """Chạy nhiều seed, trả về (thưởng trung bình, % chọn tay tốt nhất) ở 100 bước cuối."""
    print(f"{'cấu hình':<28}{'thưởng TB':>12}{'% tối ưu':>12}")
    print("-" * 52)
    for name, kwargs in configs:
        all_r, all_o = [], []
        for seed in range(runs):
            r, o = run_bandit(seed=seed, **kwargs)
            all_r.append(r[-100:].mean())
            all_o.append(o[-100:].mean())
        print(f"{name:<28}{np.mean(all_r):>12.3f}{100 * np.mean(all_o):>11.1f}%")


if __name__ == "__main__":
    print("=" * 52)
    print(" ε-GREEDY TRÊN 10-ARMED BANDIT (trung bình 30 lần chạy)")
    print("=" * 52)

    experiment([
        ("ε = 0    (tham lam thuần)", dict(eps=0.0)),
        ("ε = 0.01",                  dict(eps=0.01)),
        ("ε = 0.1",                   dict(eps=0.1)),
        ("ε = 0.1, α = 0.1",          dict(eps=0.1, alpha=0.1)),
    ])

    print()
    print("Nhận xét:")
    print("  - ε = 0 kẹt ở tay may mắn ban đầu -> tỉ lệ chọn tay tốt nhất thấp.")
    print("  - ε nhỏ (0.01) học chậm nhưng cuối cùng chọn tay tốt nhất thường xuyên.")
    print("  - α cố định giúp thích nghi khi giá trị các tay thay đổi theo thời gian.")
