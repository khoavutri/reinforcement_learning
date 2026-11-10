# Reinforcement Learning — Sutton & Barto: báo cáo + bộ thuật toán

Tài liệu phục vụ bài tập nghiên cứu cuốn *Reinforcement Learning: An Introduction*
(2nd edition, Sutton & Barto, MIT Press 2018).

## Nộp bài

| Đường dẫn | Nội dung |
|---|---|
| `bao-cao/BAO-CAO-Sutton-Barto.docx` | **Báo cáo tóm tắt** (bản để nộp, mở bằng Word) |
| `bao-cao/BAO-CAO-Sutton-Barto.md` | Cùng nội dung, dạng Markdown (dễ sửa, dễ đọc trên máy) |
| `algorithms/` | **33 file thuật toán**, mỗi file một thuật toán, chạy được |

Báo cáo gồm: thông tin sách, bức tranh tổng thể RL, tóm tắt 17 chương, bảng phân loại
thuật toán theo 3 trục (model-free/model-based, on/off-policy, độ sâu bootstrapping),
đánh giá phê bình và tài liệu tham khảo.

## Chạy thử thuật toán

```bash
pip install numpy
cd algorithms
python 12_q_learning.py
```

Mỗi file có:

1. **Docstring tiếng Việt**: bài toán → ý tưởng → công thức → pseudocode của sách →
   cách đọc code.
2. **Code** đánh số bước `(1)`, `(2)`… khớp với pseudocode.
3. **Phần `__main__`** chạy thử trên bài toán đồ chơi và in kết quả để đối chiếu.

Danh mục đầy đủ nằm trong `algorithms/README.md`.

## Cập nhật báo cáo

Sửa `BAO-CAO-Sutton-Barto.md` rồi chạy lại:

```bash
pip install python-docx
python bao-cao/build_docx.py
```

## Nguồn

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
(2nd ed.). MIT Press. Bản PDF dùng để tham chiếu: `RL_Sutton_Bartol_2020.pdf`.
