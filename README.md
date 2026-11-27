# Caro AI 11×11 - Hệ thống Trí tuệ Nhân tạo Đa Cấp Độ

Đồ án kết thúc học phần **Trí tuệ nhân tạo nâng cao** - Khoa Công nghệ Thông tin, Trường Đại học Sài Gòn.  
Giảng viên hướng dẫn: **TS. Đỗ Như Tài**  
Repository chính thức: https://github.com/SGU-2627-PBL-AdvancedAI/caro_ai_11x11

Dự án xây dựng môi trường trò chơi **Caro / Free Gomoku 11×11**, cài đặt và đánh giá **7 cấp độ tác tử AI** (từ *Random Baseline* đến *AlphaGo-inspired Hybrid PUCT-MCTS*), tổ chức giải đấu đối kháng Round-Robin theo 2 chế độ ngân sách, thực hiện 03 nghiên cứu Ablation/Sensitivity, cung cấp 08 Notebook tái lập bảng/hình và ứng dụng giao diện Web **Streamlit**.

---

## 1. Đặc tả bài toán (`GAME_SPEC`)

- **Bàn cờ:** Kích thước $11 \times 11$ ($121$ ô, ánh xạ hành động phẳng $a = r \cdot 11 + c \in \{0, \dots, 120\}$).
- **Tính chất trò chơi:** Đối kháng 2 người, tổng bằng không (*zero-sum*), quan sát đầy đủ (*perfect information*), tất định (*deterministic*), đánh luân phiên. Quân `X (+1)` đi trước, quân `O (-1)` đi sau.
- **Luật thắng & hòa (Free Gomoku):** Người chơi tạo được chuỗi **$\ge 5$ quân liên tiếp** theo hàng ngang, cột dọc, đường chéo chính hoặc đường chéo phụ sẽ thắng ngay (không áp dụng luật cấm quá 5 hay luật chặn hai đầu). Ván đấu hòa khi cả 121 ô đã kín mà chưa có người thắng.
- **API môi trường ([`src/caro_env.py`](src/caro_env.py)):** `reset()`, `legal_actions()`, `step(action)`, `is_terminal()`, `result(perspective)`, `render()`, `clone()`, `undo()`.

---

## 2. Các cấp độ tác tử AI & Kết quả Giải đấu Round-Robin

Giải đấu chính thức trong [`src/tournament.py`](src/tournament.py) và [`notebooks/08_tournament_benchmark.ipynb`](notebooks/08_tournament_benchmark.ipynb) thi đấu **100 ván/cặp × 21 cặp = 2.100 ván mỗi chế độ** (tổng **4.200 ván Round-Robin + 320 ván Ablation = 4.520 ván**), chia đều **50/50 lượt X–O** trên 20 khai cuộc đối xứng và 5 seed `(42, 102, 202, 302, 402)`.

| Cấp | Agent ID | Phương pháp cài đặt | Score% (`time_budget = 200ms`) | Elo (TB) | `avg / p95` ms (TB) | Score% (`default_config`) | Elo (Def) | Train Time | Checkpoint |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| **1** | `L1_Random` | Chọn ngẫu nhiên đều trên các nước hợp lệ | `0.00%` | 505.4 | `0.03 / 0.04` | `0.00%` | 520.6 | `0 s` | `0 MB` |
| **2** | `L2_RuleBased` | 7 tầng luật chiến thuật + đòn kép + Look-Ahead 3-ply | `73.00%` | 1801.7 | `33.6 / 94.9` | `70.25%` | 1772.2 | `0 s` | `0 MB` |
| **3** | `L3_Minimax` | Minimax + Alpha-Beta + Zobrist TT + Killer + Iterative Deepening | **`73.33%`** | **1804.6** | `154.8 / 207.5` | **`70.58%`** | **1774.9** | `0 s` | `0 MB` |
| **4** | `L4_MCTS` | MCTS-UCT chuẩn $[0,1]$ + Progressive Widening + Heavy Playout | `64.00%` | 1727.9 | `167.8 / 203.0` | `64.67%` | 1727.4 | `0 s` | `0 MB` |
| **5** | `L5_CNN` | PyTorch `ResNet-4×64` (4 kênh canonical, chia tập theo ván + khử trùng $D_4$) | `53.33%` | 1640.6 | `5.4 / 10.7` | `53.25%` | 1635.7 | `185.0 s` | `1.17 MB` |
| **6** | `L6_RL` | Masked Dueling Double-DQN (`ResGN-3×48`) + Potential Shaping + DQfD + Self-Play | `22.58%` | 1293.9 | `3.3 / 5.6` | `23.67%` | 1318.5 | `449.3 s` | `0.62 MB` |
| **7** | `L7_AlphaGo` | AlphaGo-inspired: Ensemble Prior (`L5+L6`) + Blended Value + PUCT + Zobrist Cache | `63.75%` | 1725.9 | `171.6 / 212.5` | `67.58%` | 1750.7 | `634.3 s`* | `1.79 MB`* |

*\*Cấp 7 sử dụng kết hợp 2 mô hình đã huấn luyện ở Cấp 5 (`1.17 MB`) và Cấp 6 (`0.62 MB`). Ở chế độ `default_config`, `L7_AlphaGo` thắng đối đầu trực tiếp trước cả `L3_Minimax` (`59% - 41%`) và `L4_MCTS` (`57% - 43%`). Cấp 7 là mô hình AlphaGo-inspired thu gọn cho bàn 11×11 trên CPU, không phải bản tái tạo đầy đủ hệ thống AlphaGo nguyên bản.*

---

## 3. Cấu trúc thư mục

```text
caro_ai_11x11/
├── .github/workflows/ci.yml              # GitHub Actions kiểm tra cú pháp và chạy pytest
├── checkpoints/
│   ├── 05_cnn_policy_best.pt             # Trọng số mạng Policy ResNet Cấp 5 (1.17 MB)
│   └── 06_rl_best.pt                     # Trọng số mạng Masked Dueling DQN Cấp 6 (0.62 MB)
├── configs/
│   ├── default.yaml                      # Cấu hình luật chơi và tham số thực nghiệm chung
│   └── rl_config.yaml                    # Cấu hình huấn luyện học tăng cường
├── data/
│   └── 05_cnn_resnet_dataset_11x11.npz   # Dữ liệu huấn luyện/kiểm định Cấp 5 (chia theo ván + khử trùng D4)
├── notebooks/
│   ├── 01_level1_random_baseline.ipynb   # Thực nghiệm từng cấp từ 01 đến 07
│   ├── ...
│   ├── 08_tournament_benchmark.ipynb     # Chạy giải đấu Round-Robin & xuất bảng/biểu đồ
│   └── README.md
├── results/
│   ├── figures/                          # Biểu đồ phân tích thực nghiệm (.png)
│   ├── logs/                             # Nhật ký ván đấu thô (.jsonl) của từng cấp, giải đấu và UI
│   └── tables/                           # Các bảng số liệu thống kê (.csv) và metadata (.json)
├── src/
│   ├── caro_env.py                       # Game Engine Caro 11×11
│   ├── heuristic.py                      # Hàm lượng giá trạng thái, đòn kép và lọc nước đi ứng viên
│   ├── agents/                           # BaseAgent và 7 tác tử từ Cấp 1 đến Cấp 7
│   ├── train_cnn.py                      # Kịch bản huấn luyện và đánh giá Cấp 5
│   ├── train_rl.py                       # Kịch bản huấn luyện Cấp 6 (DQfD + Curriculum Double-DQN)
│   ├── tournament.py                     # Bộ điều phối giải đấu đa tiến trình, tính CI95, Elo & Ablation
│   └── ui_app.py                         # Ứng dụng giao diện Web Streamlit
├── tests/                                # Bộ kiểm thử đơn vị tự động (pytest)
├── main.py                               # Điểm khởi chạy nhanh ứng dụng Streamlit
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## 4. Hướng dẫn cài đặt và chạy chương trình

### 4.1. Cài đặt môi trường (Python 3.11+)

```bash
conda create -n caro-ai python=3.11 -y
conda activate caro-ai

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4.2. Chạy kiểm thử tự động

```bash
pytest -v
```

### 4.3. Chạy ứng dụng Demo Web (Người vs Máy, Máy vs Máy & Replay)

```bash
python main.py
# Hoặc:
streamlit run src/ui_app.py
```

Giao diện cung cấp 3 chế độ:
1. **Người vs Máy:** Chọn cấp độ AI (`1 -> 7`), chọn quân `X` hoặc `O`, đi lại (`Undo`), xin gợi ý từ Cấp 7, xem bảng giải thích Top nước đi ứng viên và lưu lịch sử ván đấu.
2. **Máy vs Máy:** Cho 2 cấp AI bất kỳ thi đấu nhiều trận liên tiếp (tùy chọn hoán đổi lượt X/O 50/50), lưu log `.jsonl` và xuất biểu đồ thống kê PNG/CSV.
3. **Lịch sử & Replay:** Nạp lại bất kỳ tệp `.jsonl` nào trong `results/logs/` để xem lại diễn biến từng nước đi.

---

## 5. Hướng dẫn huấn luyện lại và tái lập thực nghiệm

### 5.1. Huấn luyện mạng Policy Network (Cấp 5)

```bash
python -u src/train_cnn.py \
  --train-games 140 --val-games 20 --test-games 20 \
  --teacher mixed --res-channels 64 --num-res-blocks 4 \
  --epochs 15 --batch-size 128 --lr 1.5e-3 --weight-decay 1e-4 \
  --seed 42 \
  --dataset-path data/05_cnn_resnet_dataset_11x11.npz \
  --save-path checkpoints/05_cnn_policy_best.pt
```

### 5.2. Huấn luyện tác tử Học tăng cường (Cấp 6)

```bash
python -u src/train_rl.py \
  --dqfd-demo-games 100 --dqfd-pretrain-steps 1000 \
  --dqfd-margin 0.15 --dqfd-margin-weight 0.15 --dqfd-online-ratio 0.25 \
  --stage1-episodes 100 --stage2-episodes 450 --stage3-episodes 450 \
  --eval-every 150 --eval-games 16 \
  --res-channels 48 --num-res-blocks 3 --batch-size 64 --train-freq 8 \
  --lr 5e-4 --shaping-mode potential --potential-weight 0.15 \
  --seed 42 \
  --save-path checkpoints/06_rl_best.pt \
  --log-path results/tables/06_rl_dqn6a_training_log.csv
```

### 5.3. Tái lập Giải đấu Round-Robin & 03 Ablation

- **Qua Jupyter/Colab Notebook:** Mở [`notebooks/08_tournament_benchmark.ipynb`](notebooks/08_tournament_benchmark.ipynb), chọn `RUN_MODE = "official_100"` (hoặc `"quick"`) và chạy tuần tự các ô để sinh bảng CSV và biểu đồ vào `results/`.
- **Qua dòng lệnh CLI:**
  ```bash
  python -m src.tournament --mode both --games 100 --time-budget 0.20 --ablations --ablation-games 20
  ```

---

## 6. Nguồn tham khảo & Phần kế thừa

- **Phần nhóm tự thực hiện:** Xây dựng lõi `CaroEnv`, hàm lượng giá và lọc ứng viên `heuristic.py`, 7 cấp độ tác tử trong `src/agents/`, quy trình sinh dữ liệu và huấn luyện `train_cnn.py` / `train_rl.py`, bộ tổ chức giải đấu `tournament.py`, giao diện `ui_app.py`, bộ test `tests/` và hệ thống 8 notebook thực nghiệm.
- **Thư viện sử dụng:** `NumPy`, `PyTorch`, `Pandas`, `Matplotlib`, `Streamlit`, `PyYAML`, `pytest`.
- **Tài liệu tham khảo chính:**
  1. Mark Liu, *AlphaGo Simplified: Rule-Based AI and Deep Learning in Everyday Games*, CRC Press.
  2. Stuart Russell & Peter Norvig, *Artificial Intelligence: A Modern Approach* (4th ed., 2021).
  3. Richard S. Sutton & Andrew G. Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018).
  4. Silver et al. (2016), *Mastering the game of Go with deep neural networks and tree search*, Nature; Wang et al. (ICML 2016); van Hasselt et al. (AAAI 2016); Ng, Harada & Russell (ICML 1999); Hester et al. (AAAI 2018).
