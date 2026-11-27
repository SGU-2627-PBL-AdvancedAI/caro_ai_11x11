# Hệ thống Notebook Thực nghiệm, Kiểm thử & Huấn luyện — Caro AI 11×11

Thư mục này tổ chức theo quy ước **1 notebook tương ứng đúng 1 cấp tác tử (`01` → `07`)** cộng thêm **Notebook `08` tổng hợp Giải đấu Round-Robin & Ablation (Mục 3.5)**, dùng chung nguồn mã chuẩn trong `src/` (hỗ trợ chạy trực tiếp trên **VS Code local** lẫn **Google Colab CPU/GPU**).

## 1. Danh mục 8 Notebook chính thức

| Notebook | Cấp độ & Tác tử | Nội dung chính |
|---|---|---|
| `01_level1_random_baseline.ipynb` | **Cấp 1** — `RandomAgent` | Kiểm thử API/seed/telemetry, so sánh `uniform` vs `local_random_r2`, phân bố chọn ô & lợi thế đi trước X/O. |
| `02_level2_rule_based_heuristic.ipynb` | **Cấp 2** — `RuleBasedAgent` + `heuristic.py` | Kiểm thử `heuristic.py` (đối xứng, `broken_4`, đòn kép) + 7 tầng chiến thuật, phá hòa self-play, heatmap đe dọa. |
| `03_level3_minimax_alphabeta.ipynb` | **Cấp 3** — `MinimaxAgent` | Kiểm chứng tương đương toán học 100% `AB + TT + Killer == No-AB` (`d=2, d=3`), Ablation 5 cấu hình, benchmark `d=3` & `d=5 (K=10/7)`. |
| `04_level4_mcts_uct.ipynb` | **Cấp 4** — `MCTSAgent` | Kiểm thử ràng buộc `visits`/ngân sách/Open-3, đo thông lượng rollout/s, Ablation `Upgraded MCTS` vs `Plain-UCT`, heatmap lượt thăm. |
| `05_level5_cnn_policy_resnet.ipynb` | **Cấp 5** — `CNNAgent` (PyTorch ResNet 4 kênh) | **Mục 3.4:** Sinh/nạp dữ liệu Game-Level `Train/Val/Test` + khử trùng D4 (kiểm chứng `0% leakage`), thảo luận giới hạn học từ chuyên gia, train/nạp `checkpoints/05_cnn_policy_best.pt`, đánh giá Test độc lập + 12/12 bài chiến thuật Pure CNN + heatmap xác suất. |
| `06_level6_rl_dueling_dqn.ipynb` | **Cấp 6** — `RLAgent` (Dueling Double-DQN ResGN) | **Mục 3.4:** Đặc tả đầy đủ hàm thưởng (sparse + Ng–Harada–Russell potential-based), exploration, kiến trúc & siêu tham số, kiểm chứng `Masked Dueling Mean`, train/nạp `checkpoints/06_rl_best.pt`, 10/10 chiến thuật, Ablation & Learning Curve + heatmap Q-value. |
| `07_level7_alphago_puct.ipynb` | **Cấp 7 (Bonus)** — `AlphaGoAgent` | Kết hợp `Policy Cấp 5 + DQN Cấp 6 + PUCT-MCTS + Zobrist Eval Cache`, Ablation `value_blend_alpha` (`0.0 / 0.25 / 0.5 / 1.0`), đối kháng Cấp 1–6 dưới `0.8s/nước`, trực quan hóa Explainability (`prior` vs `visits`). |
| `08_tournament_benchmark.ipynb` | **Mục 3.5** — Giải đấu Round-Robin 7 Cấp & 03 Ablation | Vòng tròn đầy đủ 21 cặp × 2 chế độ (`time_budget` 200ms & `default_config`), cân bằng 50/50 X-O, đa seed, lưu log `.jsonl` riêng từng cặp (hỗ trợ `resume`), Wilson & Bootstrap CI95, Elo, FPA, 03 Ablation & replay ván đấu. |

## 2. Chế độ chạy & Tuân thủ Mục 3.4 (Huấn luyện và dữ liệu)

- **2 chế độ chạy (`RUN_MODE`):**
  - `RUN_MODE = "quick"` *(mặc định)*: kiểm thử + nạp checkpoint có sẵn + benchmark gọn trong ~10–40 giây/notebook.
  - `RUN_MODE = "full"` *(hoặc biến môi trường `CARO_NB_MODE=full`)*: re-train đầy đủ mô hình (Cấp 5, Cấp 6) và chạy benchmark quy mô lớn.
- **Tách biệt đánh giá cuối cùng:** tập `Val` chỉ dùng chọn trọng số tốt nhất trong lúc huấn luyện; tập `Test` (Cấp 5) và bộ `FINAL_BENCHMARK_SEEDS = [9001..9005]` hoàn toàn độc lập, không dùng để chọn siêu tham số.
- **Khai báo tài nguyên & tái lập:** mọi notebook tự động in cấu hình `OS / Python / CPU / PyTorch / CUDA`, lưu `seed`, checkpoint và bảng/biểu đồ kết quả vào `results/tables/` và `results/figures/`.

## 3. Bản lưu tạm các Notebook cũ

5 notebook phiên bản trước khi nâng cấp đã được lưu giữ nguyên trạng tại `notebooks/archive/` để đối chiếu khi viết báo cáo:
- `archive/01_rulebase_random.ipynb`
- `archive/02_minimax_alphabeta_ablation.ipynb`
- `archive/03_mcts_uct_analysis.ipynb`
- `archive/04_cnn_policy_analysis.ipynb`
- `archive/06_train_level6_rl.ipynb`
