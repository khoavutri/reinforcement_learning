"""
================================================================================
 11 — SARSA: ON-POLICY TD CONTROL                          (Sutton & Barto, Ch. 6.4)
================================================================================

BÀI TOÁN
    Tìm chính sách tối ưu (control), không cần mô hình, cập nhật từng bước.

Ý TƯỞNG
    TD(0) học V. Để điều khiển, ta học Q(s,a) — giá trị của việc làm a ở s.
    Nhưng target cần biết hành động TIẾP THEO là gì. Sarsa lấy chính hành
    động mà agent SẼ làm theo chính sách hiện tại:

        Q(S_t,A_t) ← Q(S_t,A_t) + α [ R_{t+1} + γ Q(S_{t+1}, A_{t+1}) - Q(S_t,A_t) ]

    Tên "Sarsa" là viết tắt của bộ 5 thành phần: S, A, R, S', A'.

    Vì A' được chọn theo đúng chính sách đang học (ε-tham lam), Sarsa là
    ON-POLICY: nó đánh giá và cải thiện CÙNG MỘT chính sách.

    Hệ quả quan trọng: Sarsa học giá trị của chính sách CÓ KHÁM PHÁ. Nếu
    chính sách thỉnh thoảng đi liều, Sarsa sẽ học được rằng "đi liều là nguy
    hiểm" và tránh xa vực — kể cả khi đường tối ưu đi sát vực.

PSEUDOCODE
    Khởi tạo Q(s,a) tuỳ ý, Q(terminal,·) = 0
    Lặp với mỗi episode:
        khởi tạo S
        chọn A theo Q tại S (ε-tham lam)
        lặp với mỗi bước:
            thực hiện A, quan sát R, S'
            chọn A' theo Q tại S' (ε-tham lam)
            Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
            S ← S'; A ← A'

CÁCH ĐỌC CODE
    - Hàm `choose` là chính sách ε-tham lam dùng chung cho cả hành động hiện
      tại và hành động kế tiếp — đó chính là điều làm nên "on-policy".
    - `Q[s2, a2]` ở bước cuối episode được bỏ (không có A' sau terminal).

VÍ DỤ CLIFF WALKING (mục 6.5)
    Lưới 4×12. Đi từ S (góc dưới trái) tới G (góc dưới phải). Mỗi bước -1,
    rơi xuống vực -100 và quay về S.
    - Đường TỐI ƯU đi sát mép vực (ngắn nhất) nhưng rất rủi ro.
    - Sarsa học đường VÒNG LÊN TRÊN an toàn, vì nó tính cả xác suất ε bị
      trượt xuống vực.
    - Q-learning (file 12) học đường sát vực, vì nó luôn giả định hành động
      tương lai là tối ưu.
    Đây là ví dụ kinh điển nhất về khác biệt on-policy / off-policy.

CHẠY THỬ
    python 11_sarsa.py
================================================================================
"""

import numpy as np
from common import CliffWalking, ARROWS, epsilon_greedy


def sarsa(env, episodes=500, alpha=0.5, gamma=1.0, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.nS, env.nA))

    for _ in range(episodes):
        s = env.reset()
        a = epsilon_greedy(Q[s], eps, rng)

        while True:
            r, s2, done = env.step(s, a)

            # (1) chọn A' theo CÙNG chính sách ε-tham lam
            a2 = epsilon_greedy(Q[s2], eps, rng) if not done else 0

            # (2) target dùng Q(S', A') — không phải max
            target = r if done else r + gamma * Q[s2, a2]

            # (3) cập nhật
            Q[s, a] += alpha * (target - Q[s, a])

            if done:
                break
            s, a = s2, a2

    return Q


def greedy_policy(env, Q):
    policy = np.argmax(Q, axis=1)
    policy[env.terminal] = 0
    return policy


def print_policy(env, policy, title):
    print(title)
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            s = r * env.cols + c
            if s == env.terminal:
                row.append("G")
            elif (r, c) in env.cliff:
                row.append("~")
            else:
                row.append(ARROWS[policy[s]])
        print("  " + " ".join(row))


if __name__ == "__main__":
    print("=" * 70)
    print(" SARSA — Cliff Walking, 500 episode, α=0.5, ε=0.1")
    print("=" * 70)

    env = CliffWalking()
    Q = sarsa(env, episodes=500)
    policy = greedy_policy(env, Q)

    print()
    print_policy(env, policy, "Chính sách tham lam học được ('~' = vực):")
    print("\nQuan sát: đường đi vòng LÊN TRÊN, tránh xa mép vực.")
    print("Sarsa chấp nhận đi dài hơn để giảm rủi ro bị ε đẩy xuống vực.")

    # Đánh giá không khám phá (ε = 0) để đo chất lượng chính sách
    def evaluate(env, policy, episodes=100, max_steps=500, seed=1):
        rng = np.random.default_rng(seed)
        total = []
        for _ in range(episodes):
            s, ret = env.reset(), 0.0
            for _ in range(max_steps):
                r, s2, done = env.step(s, policy[s])
                ret += r
                if done:
                    break
                s = s2
            total.append(ret)
        return np.mean(total)

    print(f"\nPhần thưởng trung bình (ε=0): {evaluate(env, policy):.1f}")
    print("(Đường tối ưu lý thuyết đạt -13; đường an toàn đạt khoảng -17.)")
