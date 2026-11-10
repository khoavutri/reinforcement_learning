"""
================================================================================
 02 — UPPER-CONFIDENCE-BOUND (UCB)                        (Sutton & Barto, Ch. 2.7)
================================================================================

BÀI TOÁN
    Vẫn là bandit k tay. Câu hỏi: ε-greedy khám phá một cách NGẪU NHIÊN và
    không phân biệt tay nào đáng thử hơn. Vậy có cách khám phá "có đầu óc" hơn
    không?

Ý TƯỞNG
    Chọn tay theo cận trên của khoảng tin cậy, không phải theo giá trị ước lượng:

        A_t = argmax_a [ Q_t(a) + c * sqrt( ln(t) / N_t(a) ) ]
                        └────┬────┘   └──────────┬──────────┘
                       khai thác            khám phá (bonus)

    - Tay nào ít được thử (N_t(a) nhỏ) -> bonus lớn -> được ưu tiên thử.
    - Thời gian trôi (t tăng) -> ln(t) tăng -> bonus tăng, giữ cho agent
      không ngừng thử lại các tay (bắt buộc trong môi trường không dừng).
    - Hằng số c điều khiển mức độ khám phá (càng lớn càng tò mò).

    Trực giác: "tay này có thể tốt hơn ta nghĩ bao nhiêu, xét theo mức độ
    chưa biết về nó?"

CÔNG THỨC (2.10)
        A_t = argmax_a [ Q_t(a) + c * sqrt( ln t / N_t(a) ) ]

    (Nếu N_t(a) = 0 thì tay đó được chọn trước, coi bonus = +∞.)

PSEUDOCODE
    Khởi tạo Q(a) = 0, N(a) = 0
    Lặp t = 1, 2, 3, ...:
        Nếu có tay chưa thử: chọn tay đó
        Ngược lại: A <- argmax_a [ Q(a) + c*sqrt(ln(t)/N(a)) ]
        R <- Bandit(A)
        N(A) <- N(A) + 1
        Q(A) <- Q(A) + (1/N(A)) * [R - Q(A)]

ĐIỂM MẠNH / YẾU
    + Khám phá có định hướng, thường tốt hơn ε-greedy trong bài toán dừng.
    + Không cần tham số ε "mò".
    - Giả định các tay độc lập và phân phối phần thưởng dừng; khi môi trường
      thay đổi mạnh, ln(t) tăng quá chậm so với thực tế.
    - Không mở rộng trực tiếp sang bài toán có trạng thái (MDP) — đó là lý do
      sách chỉ trình bày UCB ở chương bandit.

CHẠY THỬ
    python 02_ucb.py
================================================================================
"""

import numpy as np
from common import Bandit, argmax_tie


def run_ucb(k=10, steps=1000, c=2.0, seed=0):
    """UCB1-style với hằng số c."""
    bandit = Bandit(k=k, seed=seed)
    rng = np.random.default_rng(seed + 1)

    Q = np.zeros(k)
    N = np.zeros(k)

    rewards = np.zeros(steps)
    optimal = np.zeros(steps, dtype=bool)

    for t in range(steps):
        # (1) tay chưa thử lần nào thì phải thử trước (bonus = vô cùng)
        untried = np.flatnonzero(N == 0)
        if len(untried) > 0:
            a = int(rng.choice(untried))
        else:
            # (2) chọn theo cận trên: giá trị + phần thưởng khám phá
            bonus = c * np.sqrt(np.log(t + 1) / N)
            a = argmax_tie(Q + bonus, rng)

        r = bandit.step(a)

        # (3) cập nhật trung bình tăng dần
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]

        rewards[t] = r
        optimal[t] = (a == bandit.best)

    return rewards, optimal


def compare(runs=30):
    from importlib import import_module
    eps_mod = import_module("01_epsilon_greedy")

    rows = [
        ("ε-greedy ε=0.1", lambda s: eps_mod.run_bandit(eps=0.1, seed=s)),
        ("UCB c=1",        lambda s: run_ucb(c=1.0, seed=s)),
        ("UCB c=2",        lambda s: run_ucb(c=2.0, seed=s)),
    ]
    print(f"{'thuật toán':<20}{'thưởng TB':>12}{'% tối ưu':>12}")
    print("-" * 44)
    for name, fn in rows:
        rs, os_ = [], []
        for seed in range(runs):
            r, o = fn(seed)
            rs.append(r[-100:].mean())
            os_.append(o[-100:].mean())
        print(f"{name:<20}{np.mean(rs):>12.3f}{100 * np.mean(os_):>11.1f}%")


if __name__ == "__main__":
    print("=" * 44)
    print(" UCB vs ε-GREEDY (10 tay, 1000 bước, 30 lần chạy)")
    print("=" * 44)
    compare()
    print()
    print("Nhận xét: UCB thường đạt tỉ lệ chọn tay tốt nhất cao hơn ε-greedy")
    print("vì nó dồn lượt thử vào những tay còn nhiều bất định.")
