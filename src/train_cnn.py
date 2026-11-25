from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.cnn_agent import CNNAgent, generate_expert_train_val_test_datasets
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from caro_env import CaroEnv


def _play_arena(
    agent_a,
    make_opponent,
    n_games: int = 10,
    board_size: int = 11,
    win_length: int = 5,
) -> dict[str, float]:
    wins = draws = losses = 0
    times_ms: list[float] = []
    moves_list: list[int] = []

    for g in range(n_games):
        env = CaroEnv(board_size=board_size, win_length=win_length)
        opp = make_opponent(g)
        a_side = 1 if g % 2 == 0 else -1
        while not env.is_terminal():
            if env.current_player == a_side:
                act = agent_a.select_action(env)
                times_ms.append(float(getattr(agent_a, "last_move_time_ms", 0.0)))
            else:
                act = opp.select_action(env)
            env.step(act)
        moves_list.append(len(env.history))
        if env.winner == a_side:
            wins += 1
        elif env.winner == -a_side:
            losses += 1
        else:
            draws += 1

    ts = sorted(times_ms)
    p95 = ts[int(0.95 * (len(ts) - 1))] if ts else 0.0
    score_pct = (wins + 0.5 * draws) / max(1, n_games) * 100.0
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_pct": score_pct,
        "avg_moves": statistics.mean(moves_list) if moves_list else 0.0,
        "avg_ms": statistics.mean(times_ms) if times_ms else 0.0,
        "p95_ms": p95,
    }


def _save_training_csv(history: dict[str, list[float]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history.keys())
    n_epochs = len(history[keys[0]]) if keys else 0
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for ep in range(n_epochs):
            writer.writerow([ep + 1] + [round(float(history[k][ep]), 6) for k in keys])


def _save_tactical_csv(tactical_res: dict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "top_action", "expected", "passed", "target_prob", "top_prob"])
        for row in tactical_res["details"]:
            writer.writerow(
                [
                    row["case"],
                    row["top_action"],
                    " ".join(str(x) for x in row["expected"]),
                    row["passed"],
                    row["target_prob"],
                    row["top_prob"],
                ]
            )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Huấn luyện và đánh giá Level 5B PyTorch Residual Policy CNN cho Caro 11x11."
    )
    parser.add_argument("--train-games", type=int, default=160, help="Số ván chuyên gia cho tập Train.")
    parser.add_argument("--val-games", type=int, default=25, help="Số ván độc lập cho tập Validation.")
    parser.add_argument("--test-games", type=int, default=25, help="Số ván độc lập cho tập Test.")
    parser.add_argument(
        "--teacher",
        choices=["rule_based", "minimax", "mixed", "legacy"],
        default="mixed",
        help="Nguồn chuyên gia sinh nhãn.",
    )
    parser.add_argument("--res-channels", type=int, default=64, help="Số kênh trong các Residual Blocks.")
    parser.add_argument("--num-res-blocks", type=int, default=4, help="Số lượng Residual Blocks.")
    parser.add_argument("--epochs", type=int, default=18, help="Số epoch huấn luyện.")
    parser.add_argument("--batch-size", type=int, default=128, help="Kích thước mini-batch.")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Learning rate ban đầu cho AdamW.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay cho AdamW.")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị tính toán ('auto', 'cpu', 'cuda').")
    parser.add_argument("--seed", type=int, default=42, help="Seed tái lập thực nghiệm.")
    parser.add_argument("--no-aug", action="store_true", help="Tắt tăng cường đối xứng D4 (dùng cho Ablation).")
    parser.add_argument("--rebuild-dataset", action="store_true", help="Sinh lại dataset kể cả khi file cache đã tồn tại.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "05_cnn_resnet_dataset_11x11.npz"),
        help="Đường dẫn lưu/nạp bộ dữ liệu train/val/test.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "05_cnn_policy_best.pt"),
        help="Đường dẫn lưu checkpoint PyTorch (.pt).",
    )
    parser.add_argument("--arena-games", type=int, default=10, help="Số ván đấu kiểm chứng sau huấn luyện.")
    args = parser.parse_args()

    t_start = time.perf_counter()
    print("=== [1/5] Sinh / nạp bộ dữ liệu Game-Level Train / Val / Test (D4-Invariant Dedup) ===")
    X_train, y_train, X_val, y_val, X_test, y_test, meta = generate_expert_train_val_test_datasets(
        num_train_games=args.train_games,
        num_val_games=args.val_games,
        num_test_games=args.test_games,
        teacher=args.teacher,
        board_size=11,
        win_length=5,
        in_channels=4,
        candidate_radius=2,
        opening_random_plies=3,
        opponent_noise_prob=0.25,
        augment_train_symmetry=not args.no_aug,
        seed=args.seed,
        save_path=args.dataset_path,
        load_if_exists=not args.rebuild_dataset,
    )
    print(
        f"Dataset ready in {time.perf_counter() - t_start:.1f}s | teacher={meta['teacher']} | "
        f"train={len(y_train)} (unique={meta['train_unique_samples']}) | "
        f"val={len(y_val)} | test={len(y_test)} | dedup_removed={meta['dedup_removed_total']}"
    )

    print("\n=== [2/5] Khởi tạo và huấn luyện PyTorch Residual Policy CNN (Level 5B) ===")
    agent = CNNAgent(
        name="Level5B_ResNet_Policy",
        model_type="resnet",
        backend="pytorch",
        model_path=None,
        auto_load_default=False,
        board_size=11,
        in_channels=4,
        res_channels=args.res_channels,
        num_res_blocks=args.num_res_blocks,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        use_tactical_shortcut=False,
        device=args.device,
        seed=args.seed,
    )
    n_params = sum(p.numel() for p in agent.torch_model.parameters())
    print(f"Model: TorchPolicyResNet(blocks={args.num_res_blocks}, channels={args.res_channels}, params={n_params:,}) on {agent.device}")

    history = agent.train_from_dataset(
        X=X_train,
        y=y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        learning_rate=args.lr,
        verbose=1,
    )

    print("\n=== [3/5] Đánh giá trên tập Test độc lập (Unseen Games) & Bộ kiểm thử chiến thuật Pure CNN ===")
    test_metrics = agent.evaluate_dataset(X_test, y_test)
    print(
        f"TEST SET (Unseen {len(y_test)} samples): "
        f"loss={test_metrics['loss']:.4f} | "
        f"top1={test_metrics['accuracy']*100:.2f}% | "
        f"top3={test_metrics['top3_accuracy']*100:.2f}% | "
        f"top5={test_metrics['top5_accuracy']*100:.2f}%"
    )

    tactical_res = agent.evaluate_tactical_suite()
    print(
        f"PURE CNN TACTICAL SUITE (no shortcut): "
        f"{tactical_res['passed']}/{tactical_res['total']} ({tactical_res['pass_rate']*100:.1f}%) | "
        f"mean_target_prob={tactical_res['mean_target_prob']:.4f}"
    )
    for row in tactical_res["details"]:
        flag = "PASS" if row["passed"] else "FAIL"
        print(
            f"  [{flag}] {row['case']:18s}: top={row['top_action']:3d} "
            f"(expected={row['expected']}, target_p={row['target_prob']:.4f})"
        )

    print("\n=== [4/5] Lưu checkpoint & bảng log kết quả ===")
    ckpt_metadata = {
        "dataset_meta": meta,
        "test_metrics": test_metrics,
        "tactical_pass_rate": tactical_res["pass_rate"],
        "tactical_mean_target_prob": tactical_res["mean_target_prob"],
    }
    saved_ckpt = agent.save_model(args.save_path, metadata=ckpt_metadata)
    log_csv = PROJECT_ROOT / "results" / "tables" / "05_cnn_resnet_training_log.csv"
    tac_csv = PROJECT_ROOT / "results" / "tables" / "05_cnn_resnet_tactical_checks.csv"
    _save_training_csv(history, log_csv)
    _save_tactical_csv(tactical_res, tac_csv)
    print(f"Saved checkpoint: {saved_ckpt}")
    print(f"Saved logs:       {log_csv} & {tac_csv}")

    if args.arena_games > 0:
        print(f"\n=== [5/5] Đấu kiểm chứng ({args.arena_games} ván cân bằng X/O) ===")
        pure_agent = CNNAgent(
            name="Lv5B_Pure_CNN",
            model_path=saved_ckpt,
            use_tactical_shortcut=False,
            use_mix_mcts=False,
            deterministic=True,
            seed=args.seed,
        )
        r1 = _play_arena(pure_agent, lambda g: RandomAgent(seed=1000 + g), n_games=args.arena_games)
        print(
            f"Pure CNN vs Random:        {r1['wins']}W-{r1['draws']}D-{r1['losses']}L ({r1['score_pct']:.1f}%) | "
            f"avg_moves={r1['avg_moves']:.1f} | avg_ms={r1['avg_ms']:.2f} p95={r1['p95_ms']:.2f}"
        )

        legacy_ckpt = PROJECT_ROOT / "checkpoints" / "cnn_strong_11x11.npz"
        if legacy_ckpt.exists():
            r_old = _play_arena(
                pure_agent,
                lambda g: CNNAgent(
                    name="Old_CNN_Strong_Pure",
                    model_type="strong",
                    backend="numpy",
                    model_path=legacy_ckpt,
                    use_tactical_shortcut=False,
                    deterministic=True,
                    seed=2000 + g,
                ),
                n_games=args.arena_games,
            )
            print(
                f"Pure CNN (5B) vs Old Strong Pure CNN: "
                f"{r_old['wins']}W-{r_old['draws']}D-{r_old['losses']}L ({r_old['score_pct']:.1f}%) | "
                f"avg_moves={r_old['avg_moves']:.1f}"
            )

        r2 = _play_arena(pure_agent, lambda g: RuleBasedAgent(seed=3000 + g), n_games=args.arena_games)
        print(
            f"Pure CNN vs Lv2 RuleBased: {r2['wins']}W-{r2['draws']}D-{r2['losses']}L ({r2['score_pct']:.1f}%) | "
            f"avg_moves={r2['avg_moves']:.1f} | avg_ms={r2['avg_ms']:.2f}"
        )

        puct_agent = CNNAgent(
            name="Lv5B_CNN_PUCT",
            model_path=saved_ckpt,
            use_tactical_shortcut=False,
            use_mix_mcts=True,
            num_rollouts=64,
            time_limit_sec=1.0,
            seed=args.seed,
        )
        r3 = _play_arena(puct_agent, lambda g: RuleBasedAgent(seed=4000 + g), n_games=min(6, args.arena_games))
        print(
            f"CNN+PUCT(64) vs Lv2 RuleBased: {r3['wins']}W-{r3['draws']}D-{r3['losses']}L ({r3['score_pct']:.1f}%) | "
            f"avg_moves={r3['avg_moves']:.1f} | avg_ms={r3['avg_ms']:.1f} p95={r3['p95_ms']:.1f}"
        )

    print(f"\nHoàn tất toàn bộ quy trình Cấp 5B trong {time.perf_counter() - t_start:.1f}s.")


if __name__ == "__main__":
    main()
