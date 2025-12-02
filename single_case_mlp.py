#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
単一ケース (1つの pEqn_*.dat + x_*.dat) を対象に、
グラフ構造を使わず MLP だけで x_true を丸暗記できるかを確認するスクリプト。

- 入力特徴: x, y, z, diag, b, AR, nonOrth, diagContrast, V, h, sizeJump, Co
- 出力: x_true (圧力解)
- 評価指標: ||x_pred - x_true|| / ||x_true||
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ======== 設定パラメータ ========
DATA_DIR = "./gnn"        # pEqn_*.dat, x_*.dat があるディレクトリ
NUM_EPOCHS = 2000
BATCH_SIZE = 1024
LR = 1e-3
PRINT_INTERVAL = 50       # 何エポックごとにログを出すか
SEED = 42
# ==============================


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def auto_select_case(data_dir: str):
    """
    gnn ディレクトリから最初に見つかった pEqn_*.dat を使って
    time, rank を自動抽出する。
    例: pEqn_55010_rank7.dat -> time="55010", rank="7"
    """
    pattern = os.path.join(data_dir, "pEqn_*_rank*.dat")
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"pEqn_*_rank*.dat が見つかりません: {pattern}")

    path = files[0]
    base = os.path.basename(path)  # 例: "pEqn_55010_rank7.dat"

    # "pEqn_", "55010", "rank7.dat" に分割
    parts = base.split("_")
    if len(parts) < 3:
        raise RuntimeError(f"ファイル名形式が想定外です: {base}")

    time_str = parts[1]
    # "rank7.dat" -> "7"
    rank_part = parts[2]
    if not rank_part.startswith("rank"):
        raise RuntimeError(f"rank 部分が見つかりません: {base}")
    rank_str = rank_part[len("rank") :].split(".")[0]

    print(f"[INFO] ケース自動選択: time={time_str}, rank={rank_str}")
    return time_str, rank_str


def load_single_case_mlp(data_dir: str, time_str: str, rank_str: str):
    """
    指定された time, rank に対して
    - pEqn_<time>_rank<rank>.dat から特徴量 X
    - x_<time>_rank<rank>.dat から真値 y (= x_true)
    を読み込み、(X, y) を返す。

    pEqn_*.dat の CELLS 行フォーマット:
      id x y z diag b skew nonOrtho aspect diagContrast V cellSize sizeJump Co
    """
    # ---- pEqn ファイルの読み込み ----
    p_path = os.path.join(data_dir, f"pEqn_{time_str}_rank{rank_str}.dat")
    if not os.path.exists(p_path):
        raise RuntimeError(f"pEqn ファイルが見つかりません: {p_path}")

    with open(p_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # "CELLS" セクションの開始と "EDGES" の位置を探す
    try:
        idx_cells = next(i for i, ln in enumerate(lines) if ln.startswith("CELLS"))
        idx_edges = next(i for i, ln in enumerate(lines) if ln.startswith("EDGES"))
    except StopIteration:
        raise RuntimeError(f"CELLS / EDGES セクションが見つかりません: {p_path}")

    cell_lines = lines[idx_cells + 1 : idx_edges]

    ids = []
    feats = []  # 入力特徴 X
    for ln in cell_lines:
        parts = ln.split()
        if len(parts) < 14:
            raise RuntimeError(f"CELLS 行の列数が足りません: {ln}")

        cell_id = int(parts[0])

        # 位置
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])

        # 係数・メッシュ品質など
        diag = float(parts[4])
        b = float(parts[5])
        skew = float(parts[6])
        non_orth = float(parts[7])
        aspect = float(parts[8])
        diag_con = float(parts[9])
        V = float(parts[10])
        h = float(parts[11])
        size_jump = float(parts[12])
        Co = float(parts[13])

        # 入力特徴ベクトルを構成
        # 必要に応じて項目を取捨選択してください
        feat_vec = [
            x, y, z,
            diag, b,
            aspect, non_orth, skew,
            diag_con, V, h,
            size_jump, Co,
        ]

        ids.append(cell_id)
        feats.append(feat_vec)

    X = np.array(feats, dtype=np.float64)
    ids = np.array(ids, dtype=np.int64)

    # ---- x_true の読み込み ----
    x_path = os.path.join(data_dir, f"x_{time_str}_rank{rank_str}.dat")
    if not os.path.exists(x_path):
        raise RuntimeError(f"x ファイルが見つかりません: {x_path}")

    y_ids = []
    y_vals = []
    with open(x_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 2:
                raise RuntimeError(f"x 行の列数が足りません: {ln}")
            cid = int(parts[0])
            val = float(parts[1])
            y_ids.append(cid)
            y_vals.append(val)

    y_ids = np.array(y_ids, dtype=np.int64)
    y_vals = np.array(y_vals, dtype=np.float64)

    # セル ID でソートして対応を揃える（安全のため）
    sort_idx_X = np.argsort(ids)
    X = X[sort_idx_X]
    ids_sorted = ids[sort_idx_X]

    sort_idx_y = np.argsort(y_ids)
    y_vals = y_vals[sort_idx_y]
    y_ids_sorted = y_ids[sort_idx_y]

    # 一致確認
    if not np.array_equal(ids_sorted, y_ids_sorted):
        raise RuntimeError("pEqn と x のセル ID が一致しません。")

    y = y_vals  # alias

    print(f"[INFO] nCells = {X.shape[0]}, nFeatures = {X.shape[1]}")
    print(f"[INFO] x_true: min={y.min():.3e}, max={y.max():.3e}, mean={y.mean():.3e}")

    return X, y


class MLP(nn.Module):
    """
    シンプルな 3 層 MLP
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        layers = []

        dim_in = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(dim_in, hidden_dim))
            layers.append(nn.ReLU())
            dim_in = hidden_dim
        # 最終出力はスカラー
        layers.append(nn.Linear(dim_in, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (N,) で返す


def main():
    set_seed(SEED)

    # デバイス
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # ケース自動選択
    time_str, rank_str = auto_select_case(DATA_DIR)

    # データ読み込み
    X_np, y_np = load_single_case_mlp(DATA_DIR, time_str, rank_str)

    # --- 特徴量・出力を標準化（学習を安定させるため） ---
    X_mean = X_np.mean(axis=0)
    X_std = X_np.std(axis=0)
    X_std[X_std == 0.0] = 1.0  # ゼロ除算防止
    X_norm = (X_np - X_mean) / X_std

    y_mean = y_np.mean()
    y_std = y_np.std()
    if y_std == 0.0:
        y_std = 1.0
    y_norm = (y_np - y_mean) / y_std

    # Tensor 化
    X = torch.from_numpy(X_norm).float()
    y = torch.from_numpy(y_norm).float()

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # モデル定義
    in_dim = X.shape[1]
    model = MLP(in_dim=in_dim, hidden_dim=128, num_layers=3).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 学習ループ
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(dataset)

        # ログ用に全セルでの相対誤差を計算（x 空間に戻して）
        if epoch % PRINT_INTERVAL == 0 or epoch == 1 or epoch == NUM_EPOCHS:
            model.eval()
            with torch.no_grad():
                X_full = X.to(device)
                y_full = y.to(device)

                y_pred_norm = model(X_full)          # 標準化された空間
                # 元スケールに戻す
                y_pred = y_pred_norm * y_std + y_mean
                y_true = y_np  # NumPy の元値

                # 相対誤差 ||x_pred - x_true|| / ||x_true||
                diff = y_pred.cpu().numpy() - y_true
                num = np.linalg.norm(diff)
                den = np.linalg.norm(y_true) + 1e-30
                rel_err = num / den

            print(f"[Epoch {epoch:4d}] loss={epoch_loss:.4e}, rel_err={rel_err:.4e}")

    # 最終診断
    model.eval()
    with torch.no_grad():
        X_full = X.to(device)
        y_pred_norm = model(X_full)
        y_pred = y_pred_norm * y_std + y_mean
        y_true = y_np

        diff = y_pred.cpu().numpy() - y_true
        num = np.linalg.norm(diff)
        den = np.linalg.norm(y_true) + 1e-30
        rel_err = num / den

    print("\n=== Final diagnostics on this single case (MLP, no graph) ===")
    print(f"  relative error ||x_pred - x_true|| / ||x_true|| = {rel_err:.4e}")

    # いくつかサンプル表示
    print("\nSample of (cellId, x_true, x_pred):")
    n_show = 5
    for i in range(n_show):
        print(
            f"  cell {i:6d}: true={y_true[i]: .6e}, pred={y_pred.cpu().numpy()[i]: .6e}"
        )


if __name__ == "__main__":
    main()

