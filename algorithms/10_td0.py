"""
================================================================================
 10 — TABULAR TD(0)                                        (Sutton & Barto, Ch. 6.1)
================================================================================

BÀI TOÁN
    Ước lượng v_π(s) từ trải nghiệm, như Monte Carlo, nhưng KHÔNG cần đợi
    hết episode.

Ý TƯỞNG — BOOTSTRAPPING
    Monte Carlo phải chờ tới cuối episode mới biết G_t. Nhưng ta có sẵn một
    ước lượng V(S_{t+1}) rồi! Vậy hãy thay G_t bằng "phần thưởng thật của
    một bước + dự đoán hiện tại cho phần còn lại":

        Target = R_{t+1} + γ V(S_{t+1})

    và dịch V(S_t) một bước về phía target:

        V(S_t) ← V(S_t) + α [ R_{t+1} + γV(S_{t+1}) - V(S_t) ]        (6.2)

    Đại lượng trong ngoặc là TD ERROR:

        δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)

    δ_t = 0 nghĩa là "mọi thứ diễn ra đúng như ta dự đoán". Đây là tín hiệu
    học quan trọng nhất của cả cuốn sách (và, theo chương 15, của cả não bộ).

SO SÁNH VỚI MONTE CARLO
    MC target  = G_t                (không chệch, phương sai cao, phải đợi)
    TD target  = R + γV(S')         (chệch, phương sai thấp, cập nhật ngay)

    TD học được cả khi chưa có episode kết thúc -> dùng được cho bài toán
    liên tục. Đổi lại, TD "tin" vào ước lượng của chính nó, nên nếu V sai
    thì target cũng sai.

PSEUDOCODE
    Input: π cần đánh giá, α ∈ (0,1]
    Khởi tạo V(s) tuỳ ý, V(terminal) = 0
    Lặp với mỗi episode:
        khởi tạo S
        lặp với mỗi bước của episode:
            A ← hành động theo π tại S
            thực hiện A, quan sát R, S'
            V(S) ← V(S) + α[R + γV(S') - V(S)]
            S ← S'

CÁCH ĐỌC CODE
    - Chỉ một dòng cập nhật, không có chỗ nào lưu cả episode.
    - `V[s2]` của trạng thái kết thúc luôn bằng 0 (khởi tạo và không bao giờ
      cập nhật) — đó là điều kiện biên của bài toán.
    - Trong file dùng RandomWalk 5 ô có giá trị thật V(s) = s/6 để đo sai số.

CHẠY THỬ
    python 10_td0.py
================================================================================
"""

import numpy as np
from common import RandomWalk


def td0(env, policy, episodes=200, alpha=0.1, gamma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    V = np.zeros(env.nS)

    for _ in range(episodes):
        s = env.reset()
        while not env.is_terminal(s):
            a = int(rng.choice(env.nA, p=policy))
            r, s2, done = env.step(s, a)

            # (1) TD error
            delta = r + gamma * V[s2] - V[s]
            # (2) dịch V(S) một bước
            V[s] += alpha * delta

            s = s2

    return V


if __name__ == "__main__":
    print("=" * 70)
    print(" TD(0) trên RandomWalk 5 ô — so với giá trị thật V(s) = s/6")
    print("=" * 70)

    env = RandomWalk(n=5)
    policy = np.array([0.5, 0.5])
    truth = np.array([s / (env.n + 1) for s in range(env.nS)])

    print(f"\nGiá trị thật: {np.round(truth[1:-1], 3)}\n")
    print(f"{'episode':>8}{'RMSE (TB 20 lần chạy)':>26}")
    print("-" * 36)

    for n in (0, 10, 50, 100, 500, 2000):
        errs = []
        for seed in range(20):
            V = td0(env, policy, episodes=n, alpha=0.1, seed=seed) if n > 0 else np.zeros(env.nS)
            errs.append(np.sqrt(np.mean((V[1:-1] - truth[1:-1]) ** 2)))
        print(f"{n:>8}{np.mean(errs):>26.4f}")

    V = td0(env, policy, episodes=2000, alpha=0.1)
    print(f"\nSau 2000 episode, V ≈ {np.round(V[1:-1], 3)}")
    print("\nNhận xét: sai số giảm rất nhanh trong 100 episode đầu, sau đó chững")
    print("lại quanh một mức nhỏ — vì với α cố định, TD(0) chỉ hội tụ tới một")
    print("'quả bóng nhiễu' quanh giá trị thật (đúng như lý thuyết ở mục 9.3).")
