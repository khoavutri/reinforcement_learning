"""
================================================================================
 21 — PRIORITIZED SWEEPING                          (Sutton & Barto, Ch. 8.4)
================================================================================

BÀI TOÁN
    Dyna-Q chọn cặp (s,a) NGẪU NHIÊN để mô phỏng. Nhưng có những cặp đang
    "sai nhiều" (TD error lớn) — chúng đáng được cập nhật trước. Vậy tại sao
    không ưu tiên chúng?

Ý TƯỞNG
    Dùng hàng đợi ưu tiên theo |TD error|:

        P = | R + γ max_a Q(S',a) − Q(S,A) |

    - Khi vừa trải nghiệm (S,A) mà P lớn -> đẩy vào hàng đợi.
    - Mỗi vòng planning: lấy cặp có P lớn nhất, cập nhật Q, rồi lan truyền
      ngược: mọi (S̄,Ā) có thể dẫn tới S cũng có thể vừa "sai" hơn -> tính
      lại P và đẩy vào hàng đợi.
    - Những cặp có P < θ bị bỏ qua -> tiết kiệm tính toán.

PSEUDOCODE
    Khởi tạo Q, Model, PQueue rỗng
    Lặp mãi:
        (a) S ← trạng thái hiện tại
        (b) A ← chính sách(S, Q)
        (c) thực hiện A, quan sát R, S'
        (d) Model(S,A) ← (R,S')
        (e) P ← |R + γ max_a Q(S',a) − Q(S,A)|
        (f) nếu P > θ: đưa (S,A) vào PQueue với độ ưu tiên P
        (g) lặp n lần, khi PQueue còn:
                S,A ← phần tử ưu tiên cao nhất
                R,S' ← Model(S,A)
                Q(S,A) ← Q + α[R + γ max_a Q(S',a) − Q(S,A)]
                với mọi (S̄,Ā) được dự đoán dẫn tới S:
                    R̄ ← phần thưởng dự đoán
                    P ← |R̄ + γ max_a Q(S,a) − Q(S̄,Ā)|
                    nếu P > θ: đưa (S̄,Ā) vào PQueue

CÁCH ĐỌC CODE
    - `predecessors[s]` = tập các cặp (s', a') có next-state là s. Đây là
      "đồ thị ngược" cần thiết để lan truyền ưu tiên.
    - `heapq` là heap nhị phân; ta lưu (-P) để lấy P lớn nhất.
    - `model` với môi trường tất định chỉ cần một cặp (r, s2).

CHI PHÍ
    Mỗi cập nhật cần duyệt các predecessor -> O(|S||A|) trong trường hợp
    xấu, nhưng thực tế chỉ duyệt những cặp thực sự dẫn tới S.

CHẠY THỬ
    python 21_prioritized_sweeping.py
================================================================================
"""

import heapq
import numpy as np
from collections import defaultdict
from common import GridWorld, epsilon_greedy


def prioritized_sweeping(env, episodes=30, n_planning=10, alpha=0.5, gamma=0.9,
                         eps=0.1, theta=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.nS, env.nA))
    model = {}
    predecessors = defaultdict(set)
    pq = []
    counter = 0

    def push(s, a, p):
        nonlocal counter
        counter += 1
        heapq.heappush(pq, (-p, counter, (s, a)))

    steps_per_episode = []

    for _ in range(episodes):
        s = env.reset()
        steps = 0
        while not env.is_terminal(s):
            a = epsilon_greedy(Q[s], eps, rng)
            r, s2, done = env.step(s, a)
            steps += 1

            # (d) ghi model + đồ thị ngược
            model[(s, a)] = (r, s2)
            predecessors[s2].add((s, a))

            # (e,f) tính độ ưu tiên và đẩy vào hàng đợi
            p = abs(r + (0.0 if done else gamma * Q[s2].max()) - Q[s, a])
            if p > theta:
                push(s, a, p)

            # (g) planning có ưu tiên
            for _ in range(n_planning):
                if not pq:
                    break
                _, _, (ps, pa) = heapq.heappop(pq)
                if (ps, pa) not in model:
                    continue
                pr, ps2 = model[(ps, pa)]
                Q[ps, pa] += alpha * (pr + gamma * Q[ps2].max() - Q[ps, pa])

                # lan truyền ngược về các trạng thái dẫn tới ps
                for (qs, qa) in predecessors[ps]:
                    qr, qs2 = model[(qs, qa)]
                    qp = abs(qr + gamma * Q[qs2].max() - Q[qs, qa])
                    if qp > theta:
                        push(qs, qa, qp)

            s = s2

        steps_per_episode.append(steps)

    return Q, steps_per_episode


if __name__ == "__main__":
    print("=" * 72)
    print(" PRIORITIZED SWEEPING vs DYNA-Q — lưới 10x10, 10 episode")
    print("=" * 72)

    from importlib import import_module
    dyna = import_module("20_dyna_q")

    env = GridWorld(rows=10, cols=10, goal=(9, 9), gamma=0.9)

    result = {}
    for name in ("Dyna-Q (n=20)", "Prioritized (n=20)"):
        runs = []
        for seed in range(5):
            if name.startswith("Dyna"):
                Q, steps = dyna.dyna_q(env, episodes=10, n_planning=20, seed=seed)
            else:
                Q, steps = prioritized_sweeping(env, episodes=10, n_planning=20, seed=seed)
            runs.append(np.mean(steps[-3:]))
        result[name] = np.mean(runs)

    print(f"\n{'thuật toán':<24}{'TB 3 episode cuối':>22}")
    print("-" * 46)
    for name, v in result.items():
        print(f"{name:<24}{v:>22.1f}")

    print("\nNhận xét: cùng ngân sách 20 lần cập nhật/bước, prioritized sweeping")
    print("hội tụ nhanh hơn vì dồn tính toán vào những chỗ đang sai nhiều nhất,")
    print("thay vì cập nhật đều mọi cặp đã biết.")
