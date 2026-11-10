"""
================================================================================
 20 — DYNA-Q: HỌC + LẬP KẾ HOẠCH                    (Sutton & Barto, Ch. 8.2)
================================================================================

BÀI TOÁN
    Agent vừa học từ trải nghiệm thật, vừa "tưởng tượng" để học nhanh hơn.
    Có thể không? Có — nếu nó ghi lại một MÔ HÌNH nhỏ của môi trường.

Ý TƯỞNG
    Mỗi khi thực hiện (S,A) và thấy (R,S'), lưu lại cặp đó vào Model(S,A).
    Sau đó, mỗi bước thật, lấy ngẫu nhiên n cặp đã lưu và cập nhật Q như thể
    vừa trải nghiệm lại chúng. Đây là PLANNING — học từ trải nghiệm tưởng tượng.

    Điểm đẹp: planning và learning dùng CHUNG một quy tắc cập nhật Q. Chỉ
    khác nguồn dữ liệu (thật hay model sinh ra).

        (d) Q(S,A) ← Q + α[R + γ max_a Q(S',a) - Q(S,A)]      <- học thật
        (e) Model(S,A) ← (R, S')                              <- ghi nhớ
        (f) lặp n lần: lấy (S,A) ngẫu nhiên đã thấy
                       R,S' ← Model(S,A)
                       Q(S,A) ← Q + α[R + γ max Q(S',a) - Q(S,A)]   <- planning

PSEUDOCODE (Tabular Dyna-Q)
    Khởi tạo Q(s,a) và Model(s,a)
    Lặp mãi:
        (a) S ← trạng thái hiện tại
        (b) A ← ε-tham lam theo Q
        (c) thực hiện A, quan sát R, S'
        (d) Q(S,A) ← Q + α[R + γ max_a Q(S',a) − Q(S,A)]
        (e) Model(S,A) ← (R, S')
        (f) lặp n lần:
                S ← một trạng thái đã từng gặp
                A ← một hành động đã từng làm ở S
                R, S' ← Model(S,A)
                Q(S,A) ← Q + α[R + γ max_a Q(S',a) − Q(S,A)]

CÁCH ĐỌC CODE
    - `model` là dict: (s,a) -> (r, s2). Với môi trường tất định, một mẫu
      là đủ; với môi trường ngẫu nhiên phải lưu phân phối (xem mục 8.3).
    - `n_planning = 0` chính là Q-learning thuần. Tăng n lên là thấy hiệu quả.
    - Vòng planning chọn cặp (s,a) NGẪU NHIÊN trong số đã biết — bản cải
      tiến ở file 21 (prioritized sweeping) sẽ chọn thông minh hơn.

ĐIỀU THÚ VỊ
    Sách ví Dyna-Q như "học trong giấc mơ". Nếu model sai (ví dụ thế giới
    thay đổi sau khi model đã ghi), Dyna-Q sẽ học sai — đó là nội dung mục
    8.3, và là lý do cần cơ chế phát hiện thay đổi (Dyna-Q+).

CHẠY THỬ
    python 20_dyna_q.py
================================================================================
"""

import numpy as np
from common import GridWorld, epsilon_greedy


def dyna_q(env, episodes=50, n_planning=10, alpha=0.5, gamma=0.9, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.nS, env.nA))
    model = {}                       # (s,a) -> (r, s2)
    seen = []                        # các cặp (s,a) đã biết

    steps_per_episode = []

    for _ in range(episodes):
        s = env.reset()
        steps = 0
        while not env.is_terminal(s):
            # (b) chọn hành động
            a = epsilon_greedy(Q[s], eps, rng)

            # (c) tương tác thật
            r, s2, done = env.step(s, a)
            steps += 1

            # (d) học từ trải nghiệm thật (Q-learning)
            target = r if done else r + gamma * Q[s2].max()
            Q[s, a] += alpha * (target - Q[s, a])

            # (e) ghi vào model
            if (s, a) not in model:
                seen.append((s, a))
            model[(s, a)] = (r, s2)

            # (f) PLANNING: học từ trải nghiệm tưởng tượng
            for _ in range(n_planning):
                if not seen:
                    break
                ps, pa = seen[rng.integers(len(seen))]
                pr, ps2 = model[(ps, pa)]
                ptarget = pr + gamma * Q[ps2].max()
                Q[ps, pa] += alpha * (ptarget - Q[ps, pa])

            s = s2

        steps_per_episode.append(steps)

    return Q, steps_per_episode


if __name__ == "__main__":
    print("=" * 70)
    print(" DYNA-Q — lưới 10x10, chỉ 10 episode, α=0.5, ε=0.1")
    print("=" * 70)

    env = GridWorld(rows=10, cols=10, goal=(9, 9), gamma=0.9)
    print(f"\n{'n_planning':>12}{'số bước episode 1':>22}{'TB 3 episode cuối':>22}")
    print("-" * 58)

    for n in (0, 5, 20):
        runs = []
        for seed in range(5):
            Q, steps = dyna_q(env, episodes=10, n_planning=n, seed=seed)
            runs.append((steps[0], np.mean(steps[-3:])))
        first = np.mean([r[0] for r in runs])
        last = np.mean([r[1] for r in runs])
        print(f"{n:>12}{first:>22.1f}{last:>22.1f}")

    print("\nNhận xét: n = 0 là Q-learning thuần (chậm nhất, ~140 bước).")
    print("Chỉ cần thêm 20 bước planning mỗi bước thật, số bước giảm ~5 lần")
    print("dù chỉ học 10 episode. Đây là 'phép màu' của Dyna: tái sử dụng")
    print("dữ liệu cũ thay vì vứt đi sau mỗi lần cập nhật.")
