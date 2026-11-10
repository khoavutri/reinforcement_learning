"""
================================================================================
 05 — POLICY ITERATION                                    (Sutton & Barto, Ch. 4.3)
================================================================================

BÀI TOÁN
    Biết mô hình MDP. Tìm chính sách TỐI ƯU π* và hàm giá trị v*.

Ý TƯỞNG — HAI BƯỚC ĐẨY NHAU
    Bước 1 (đánh giá - Evaluation):  cho π hiện tại, tính v_π.
    Bước 2 (cải thiện - Improvement): thay π bằng chính sách THAM LAM theo v_π.
    Lặp lại. Mỗi vòng, π chỉ tốt lên hoặc giữ nguyên; và vì số chính sách
    tất định là hữu hạn nên phải dừng sau hữu hạn vòng.

        π_0 --E--> v_{π0} --I--> π_1 --E--> v_{π1} --I--> ... --> π* --E--> v*

ĐỊNH LÝ POLICY IMPROVEMENT (mục 4.2)
    Nếu chính sách mới π' được định nghĩa là tham lam theo v_π:

        π'(s) = argmax_a Σ_{s',r} p(s',r|s,a) [ r + γ v_π(s') ]

    thì v_{π'}(s) ≥ v_π(s) với mọi s. Tức là "tham lam theo giá trị của chính
    mình" không bao giờ làm tệ đi, và nếu không có gì thay đổi thì π đã tối ưu.

PSEUDOCODE TRONG SÁCH
    1. Khởi tạo: V(s) và π(s) tuỳ ý; V(terminal) = 0
    2. Policy Evaluation: lặp đến Δ < θ
           V(s) ← Σ_{s',r} p(s',r|s,π(s)) [ r + γ V(s') ]
    3. Policy Improvement:
           policy_stable ← true
           với mỗi s:
               old ← π(s)
               π(s) ← argmax_a Σ_{s',r} p(s',r|s,a) [ r + γ V(s') ]
               nếu old ≠ π(s): policy_stable ← false
    4. Nếu policy_stable: dừng; ngược lại quay về 2

CÁCH ĐỌC CODE
    - `q_value(s, a)` = giá trị của việc làm a ở s rồi đi theo V hiện tại.
      Đây là "một bước nhìn trước", chính là vế phải của phương trình Bellman.
    - `policy_evaluation` được gọi lại mỗi vòng, nhưng KHỞI TẠO từ V cũ
      (warm start) — sách nhấn mạnh chi tiết này giúp hội tụ nhanh hơn nhiều.

ĐỘ PHỨC TẠP
    Mỗi vòng: O(|S|²|A|) nếu evaluation chạy hết |S| sweep. Trong thực tế
    thường chỉ cần vài sweep mỗi vòng (xem Value Iteration ở file 06).

CHẠY THỬ
    python 05_policy_iteration.py
================================================================================
"""

import numpy as np
from common import GridWorld, ARROWS


def q_value(env, V, s, a, gamma):
    """Giá trị hành động một bước: E[ r + γ V(s') ]."""
    return sum(p * (r + gamma * V[s2]) for p, s2, r in env.transitions(s, a))


def policy_evaluation(env, policy, V, gamma, theta=1e-6):
    """Đánh giá π, khởi tạo từ V có sẵn (warm start)."""
    while True:
        delta = 0.0
        for s in range(env.nS):
            if env.is_terminal(s):
                continue
            v_old = V[s]
            V[s] = q_value(env, V, s, policy[s], gamma)
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            return V


def policy_improvement(env, V, policy, gamma):
    """Thay π bằng chính sách tham lam theo V. Trả về True nếu π không đổi."""
    stable = True
    for s in range(env.nS):
        if env.is_terminal(s):
            continue
        old_action = policy[s]
        # (1) với mỗi hành động, tính giá trị nếu đi theo nó một bước
        values = [q_value(env, V, s, a, gamma) for a in range(env.nA)]
        policy[s] = int(np.argmax(values))
        if policy[s] != old_action:
            stable = False
    return stable


def policy_iteration(env, gamma=0.9):
    # (0) khởi tạo tuỳ ý: đi lên hết
    policy = np.zeros(env.nS, dtype=int)
    V = np.zeros(env.nS)

    for it in range(1, 100):
        # (1) đánh giá chính sách hiện tại
        policy_evaluation(env, policy, V, gamma)
        # (2) cải thiện; nếu không đổi gì thì đã tối ưu
        stable = policy_improvement(env, V, policy, gamma)
        print(f"  vòng {it}: {'ổn định -> dừng' if stable else 'chính sách thay đổi'}")
        if stable:
            break

    return policy, V


if __name__ == "__main__":
    print("=" * 74)
    print(" POLICY ITERATION — lưới 4x4")
    print("=" * 74)

    env = GridWorld(gamma=0.9)
    policy, V = policy_iteration(env)

    print("\nGiá trị tối ưu v*:")
    print(env.render(V=V, policy=policy))
    print("\nChính sách tối ưu π* (mũi tên chỉ hướng đi tốt nhất):")
    print("  (mọi mũi tên đều dồn về đích ở góc dưới phải)")
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            s = r * env.cols + c
            row.append("  G  " if env.is_terminal(s) else f"  {ARROWS[policy[s]]}  ")
        print(" ".join(row))
