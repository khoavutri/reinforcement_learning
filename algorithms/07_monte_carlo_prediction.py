"""
================================================================================
 07 — MONTE CARLO PREDICTION (FIRST-VISIT / EVERY-VISIT)  (Sutton & Barto, Ch. 5.1)
================================================================================

BÀI TOÁN
    Ước lượng v_π(s) mà KHÔNG cần biết mô hình môi trường.
    Chỉ có một cách: chơi thật, ghi lại phần thưởng, rồi lấy trung bình.

Ý TƯỞNG
    v_π(s) = kỳ vọng của return G_t khi bắt đầu từ s và đi theo π.
    Vậy hãy chơi N episode, mỗi lần ghé thăm s thì ghi lại return từ lúc đó,
    rồi lấy trung bình. Định luật số lớn bảo đảm hội tụ về v_π(s).

    - FIRST-VISIT: mỗi episode chỉ tính lần ghé thăm ĐẦU TIÊN của s.
      Không chệch (unbiased), dễ chứng minh hội tụ.
    - EVERY-VISIT: tính mọi lần ghé thăm s trong episode.
      Chệch nhẹ nhưng dùng nhiều dữ liệu hơn, thực tế hội tụ nhanh hơn.

    Cả hai đều là ước lượng không chệch khi số episode → ∞.

CÔNG THỨC
    V(s) ← trung bình cộng của các return quan sát được sau khi ở s
         = V(s) + (1/N(s)) [ G_t - V(s) ]          (cùng dạng 2.3)

PSEUDOCODE (first-visit)
    Khởi tạo V(s) tuỳ ý, Returns(s) = danh sách rỗng
    Lặp nhiều episode:
        Sinh episode theo π: S0,A0,R1,...,S_{T-1},A_{T-1},R_T
        G ← 0
        với t = T-1, T-2, ..., 0:
            G ← γG + R_{t+1}
            nếu S_t chưa xuất hiện trong episode này:
                thêm G vào Returns(S_t)
                V(S_t) ← trung bình(Returns(S_t))

CÁCH ĐỌC CODE
    - Duyệt NGƯỢC từ cuối episode để tính G bằng một biến duy nhất.
    - `seen` là tập trạng thái đã gặp — chính là điểm khác biệt first/every visit.
    - Trong file dùng RandomWalk 5 ô: đi trái/phải 50-50, thưởng +1 ở đầu phải.
      Giá trị thật là V(s) = s/6, nên ta đo được sai số chính xác.

ƯU / NHƯỢC
    + Không cần mô hình, không chệch, dễ hiểu.
    - Phải đợi hết episode mới cập nhật (không dùng được cho bài toán liên tục).
    - Phương sai cao vì return là tổng nhiều biến ngẫu nhiên.
    - Trong bài toán lớn, nhiều trạng thái có thể không bao giờ được ghé thăm.

CHẠY THỬ
    python 07_monte_carlo_prediction.py
================================================================================
"""

import numpy as np
from common import RandomWalk


def mc_prediction(env, policy, episodes=2000, gamma=1.0, first_visit=True, seed=0):
    """Ước lượng V bằng Monte Carlo. policy(s) -> xác suất [trái, phải]."""
    rng = np.random.default_rng(seed)

    V = np.zeros(env.nS)
    N = np.zeros(env.nS)          # số lần đã dùng để cập nhật mỗi trạng thái

    for _ in range(episodes):
        # (1) sinh một episode theo π
        s = env.reset()
        states, rewards = [], []
        while not env.is_terminal(s):
            a = int(rng.choice(env.nA, p=policy))
            r, s2, done = env.step(s, a)
            states.append(s)
            rewards.append(r)
            s = s2

        # (2) duyệt ngược, tính return G
        G = 0.0
        seen = set()
        for t in range(len(states) - 1, -1, -1):
            G = gamma * G + rewards[t]
            s = states[t]

            # (3) first-visit bỏ qua các lần ghé thăm sau
            if first_visit and s in seen:
                continue
            seen.add(s)

            N[s] += 1
            V[s] += (G - V[s]) / N[s]      # trung bình tăng dần

    return V


def true_values(env):
    """V(s) = s / (n+1) cho random walk công bằng."""
    return np.array([s / (env.n + 1) for s in range(env.nS)])


if __name__ == "__main__":
    print("=" * 66)
    print(" MC PREDICTION trên RandomWalk 5 ô (thưởng +1 ở đầu phải)")
    print("=" * 66)

    env = RandomWalk(n=5)
    policy = np.array([0.5, 0.5])        # đi trái/phải đều nhau
    truth = true_values(env)

    print("\nGiá trị thật:      ", np.round(truth[1:-1], 3))

    for name, fv in [("first-visit", True), ("every-visit", False)]:
        V = mc_prediction(env, policy, episodes=2000, first_visit=fv, seed=1)
        err = np.sqrt(np.mean((V[1:-1] - truth[1:-1]) ** 2))
        print(f"{name:<14} V = {np.round(V[1:-1], 3)}   RMSE = {err:.4f}")

    print("\nTăng số episode lên 20.000:")
    for name, fv in [("first-visit", True), ("every-visit", False)]:
        V = mc_prediction(env, policy, episodes=20_000, first_visit=fv, seed=1)
        err = np.sqrt(np.mean((V[1:-1] - truth[1:-1]) ** 2))
        print(f"{name:<14} V = {np.round(V[1:-1], 3)}   RMSE = {err:.4f}")

    print("\nNhận xét: cả hai đều tiến về giá trị thật, every-visit thường")
    print("hội tụ nhanh hơn vì tận dụng nhiều mẫu hơn trong mỗi episode.")
