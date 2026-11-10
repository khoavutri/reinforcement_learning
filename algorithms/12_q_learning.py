"""
================================================================================
 12 — Q-LEARNING: OFF-POLICY TD CONTROL                    (Sutton & Barto, Ch. 6.5)
================================================================================

BÀI TOÁN
    Tìm chính sách tối ưu, cập nhật từng bước, KHÔNG cần mô hình.

Ý TƯỞNG
    Sarsa dùng Q(S', A') với A' là hành động mà agent *sẽ* làm. Nhưng nếu ta
    muốn học giá trị của chính sách TỐI ƯU thì không cần biết agent thực sự
    làm gì — cứ giả định nó sẽ làm điều tốt nhất:

        Q(S_t,A_t) ← Q(S_t,A_t) + α [ R_{t+1} + γ max_a Q(S_{t+1},a) - Q(S_t,A_t) ]

    Chỉ khác Sarsa đúng một chữ: `max` thay cho `Q(S',A')`.

    Đây là OFF-POLICY: chính sách hành vi (ε-tham lam, có khám phá) khác
    chính sách mục tiêu (tham lam thuần). Q-learning học thẳng q* bất kể
    agent đang hành động ngu ngốc đến đâu.

    Watkins (1989) đã chứng minh: nếu mọi cặp (s,a) được thăm vô hạn lần và
    α thoả điều kiện Robbins–Monro thì Q hội tụ về q* với xác suất 1.

PSEUDOCODE
    Khởi tạo Q(s,a) tuỳ ý, Q(terminal,·) = 0
    Lặp với mỗi episode:
        khởi tạo S
        lặp với mỗi bước:
            chọn A theo Q tại S (ε-tham lam)
            thực hiện A, quan sát R, S'
            Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
            S ← S'

CÁCH ĐỌC CODE
    - Dòng `target = r + gamma * Q[s2].max()` là toàn bộ khác biệt so với Sarsa.
    - Không cần chọn A' -> code ngắn hơn Sarsa, nhưng đó không phải lý do
      nó mạnh; cái mạnh là nó học thẳng q*.

SO SÁNH VỚI SARSA TRÊN CLIFF WALKING
    - Q-learning: đường sát mép vực (tối ưu -13) nhưng khi chạy với ε > 0
      thì thỉnh thoảng rơi xuống vực -> phần thưởng thực tế tệ hơn.
    - Sarsa: đường vòng an toàn (-17) nhưng ổn định hơn khi có khám phá.
    Không có "cái nào đúng" — tuỳ ta muốn tối ưu khi chạy thật (Sarsa) hay
    tối ưu chính sách thuần (Q-learning).

CẢNH BÁO
    `max` gây ra MAXIMIZATION BIAS: max của các ước lượng nhiễu luôn bị đẩy
    lên cao. Xem file 14 (Double Q-learning) để chữa.

CHẠY THỬ
    python 12_q_learning.py
================================================================================
"""

import numpy as np
from common import CliffWalking, ARROWS, epsilon_greedy


def q_learning(env, episodes=500, alpha=0.5, gamma=1.0, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.nS, env.nA))

    for _ in range(episodes):
        s = env.reset()
        while True:
            # (1) hành vi: ε-tham lam (off-policy: hành vi ≠ mục tiêu)
            a = epsilon_greedy(Q[s], eps, rng)
            r, s2, done = env.step(s, a)

            # (2) mục tiêu: max qua mọi hành động ở S'
            target = r if done else r + gamma * Q[s2].max()

            # (3) cập nhật
            Q[s, a] += alpha * (target - Q[s, a])

            if done:
                break
            s = s2

    return Q


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
    return float(np.mean(total))


if __name__ == "__main__":
    print("=" * 70)
    print(" Q-LEARNING — Cliff Walking, 500 episode, α=0.5, ε=0.1")
    print("=" * 70)

    env = CliffWalking()
    Q = q_learning(env, episodes=500)
    policy = np.argmax(Q, axis=1)
    policy[env.terminal] = 0

    print()
    print_policy(env, policy, "Chính sách tham lam học được ('~' = vực):")
    print("\nQuan sát: đường đi SÁT MÉP VỰC — ngắn nhất, đúng tối ưu lý thuyết.")
    print(f"Phần thưởng trung bình khi chạy ε=0: {evaluate(env, policy):.1f}")

    print("\nSo sánh với Sarsa (file 11):")
    print("  - Q-learning tối ưu chính sách thuần, nhưng rủi ro khi có khám phá.")
    print("  - Sarsa học chính sách an toàn hơn vì tính cả khả năng bị trượt.")
