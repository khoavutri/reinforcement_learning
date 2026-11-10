# Bộ thuật toán — *Reinforcement Learning: An Introduction* (Sutton & Barto)

Mỗi file là **một thuật toán**, tự chạy được, có giải thích tiếng Việt ở đầu file.
Code được đánh số bước `(1)`, `(2)`… khớp với pseudocode trong sách.

## Cách chạy

```bash
pip install numpy
cd algorithms
python 12_q_learning.py
```

Mỗi file in ra kết quả thực nghiệm nhỏ (bảng số, chính sách học được) để đối chiếu
với lý thuyết. Thời gian chạy mỗi file: vài giây đến ~20 giây.

## Danh mục

### Chương 2 — Multi-armed Bandits
| File | Nội dung | Ý chính |
|---|---|---|
| `01_epsilon_greedy.py` | ε-greedy + cập nhật trung bình tăng dần | công thức (2.3): `Q ← Q + α[R−Q]` |
| `02_ucb.py` | Upper-Confidence-Bound | khám phá có định hướng bằng `√(ln t / N)` |
| `03_gradient_bandit.py` | Gradient bandit (softmax) | tiền thân của policy gradient; vai trò baseline |

### Chương 4 — Dynamic Programming
| File | Nội dung | Ý chính |
|---|---|---|
| `04_policy_evaluation.py` | Iterative Policy Evaluation | quét trạng thái tới khi Δ < θ |
| `05_policy_iteration.py` | Policy Iteration | eval ↔ improve, dừng khi π ổn định |
| `06_value_iteration.py` | Value Iteration | một sweep gộp eval + improve |

### Chương 5 — Monte Carlo
| File | Nội dung | Ý chính |
|---|---|---|
| `07_monte_carlo_prediction.py` | First-visit / every-visit MC | trung bình return, không cần model |
| `08_monte_carlo_control.py` | MC ES + ε-soft control | đảm bảo mọi (s,a) được thăm |
| `09_off_policy_mc.py` | Off-policy MC + importance sampling | ordinary vs weighted IS |

### Chương 6 — TD Learning
| File | Nội dung | Ý chính |
|---|---|---|
| `10_td0.py` | Tabular TD(0) | `V ← V + α[R + γV' − V]` |
| `11_sarsa.py` | Sarsa | on-policy, học đường an toàn |
| `12_q_learning.py` | Q-learning | off-policy, học đường tối ưu |
| `13_expected_sarsa.py` | Expected Sarsa | kỳ vọng thay vì mẫu A' |
| `14_double_q_learning.py` | Double Q-learning | chữa maximization bias |

### Chương 7 — n-step Bootstrapping
| File | Nội dung | Ý chính |
|---|---|---|
| `15_n_step_td.py` | n-step TD | n=1 là TD(0), n→∞ là MC |
| `16_n_step_sarsa.py` | n-step Sarsa | lan return nhanh hơn |
| `17_off_policy_n_step_sarsa.py` | Off-policy n-step Sarsa | tích ρ trong n bước |
| `18_tree_backup.py` | n-step Tree Backup | không cần importance sampling |
| `19_n_step_q_sigma.py` | n-step Q(σ) | nội suy Sarsa ↔ Tree Backup |

### Chương 8 — Planning and Learning
| File | Nội dung | Ý chính |
|---|---|---|
| `20_dyna_q.py` | Dyna-Q | học + planning dùng chung model |
| `21_prioritized_sweeping.py` | Prioritized Sweeping | ưu tiên cặp có TD error lớn |

### Chương 9–11 — Xấp xỉ hàm
| File | Nội dung | Ý chính |
|---|---|---|
| `22_semi_gradient_td.py` | Semi-gradient TD(0) | `w ← w + αδx` |
| `23_semi_gradient_sarsa.py` | Episodic semi-gradient Sarsa | điều khiển + xấp xỉ hàm |
| `24_n_step_semi_gradient_sarsa.py` | n-step semi-gradient Sarsa | n-step + xấp xỉ hàm |
| `25_lstd.py` | Least-Squares TD | giải `Aw = b` trực tiếp |
| `26_differential_sarsa.py` | Differential Sarsa | average reward, không chiết khấu |

### Chương 12 — Eligibility Traces
| File | Nội dung | Ý chính |
|---|---|---|
| `27_td_lambda.py` | TD(λ) | accumulating trace |
| `28_true_online_td_lambda.py` | True online TD(λ) | dutch trace, khớp λ-return |
| `29_sarsa_lambda.py` | Sarsa(λ) | replacing trace cho điều khiển |
| `30_true_online_sarsa_lambda.py` | True online Sarsa(λ) | O(d) mỗi bước |

### Chương 13 — Policy Gradient
| File | Nội dung | Ý chính |
|---|---|---|
| `31_reinforce.py` | REINFORCE | `θ ← θ + αG∇ln π` |
| `32_reinforce_baseline.py` | REINFORCE + baseline | giảm phương sai, không chệch |
| `33_actor_critic.py` | Actor–Critic | cập nhật mỗi bước bằng TD error |

## File dùng chung

`common.py` chứa các môi trường đồ chơi (GridWorld, CliffWalking, RandomWalk,
Corridor, Bandit) và vài hàm tiện ích (`epsilon_greedy`, `argmax_tie`, `softmax`,
`one_hot`). Không có phần nào của thuật toán nằm trong file này.

## Ghi chú khi đọc

- Các bài toán đều nhỏ và tất định để kết quả in ra dễ kiểm chứng bằng tay.
- Với `one_hot`, xấp xỉ hàm tuyến tính tương đương bảng — đó là cách kiểm chứng
  cài đặt trước khi thay bằng tile coding / mạng nơ-ron.
- Con số tuyệt đối phụ thuộc seed và siêu tham số; điều quan trọng là **xu hướng**
  khớp với kết luận của sách.
