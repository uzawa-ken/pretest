#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GNN training on 5 cases with relative error-based data loss + PDE loss.

損失関数:
  - データ損失: L_data = ||x_pred - x_true||² / ||x_true||²
  - PDE損失:    L_pde  = ||A·x_pred - b||² / ||b||²
  - 総合損失:   L = λ_data × L_data + λ_pde × L_pde

両方の損失が相対誤差（無次元）なので、λ_data = λ_pde = 1.0 で平等に扱えます。

設定パラメータ:
  - LAMBDA_DATA, LAMBDA_PDE: 損失の重み（デフォルト: 1.0）
  - USE_MESH_QUALITY_WEIGHT: メッシュ品質重み付けを使用するか（デフォルト: False）
  - ALPHA_QUAL, BETA_QUAL: メッシュ品質重み付けのパラメータ
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv
except ImportError:
    raise RuntimeError(
        "torch_geometric がインストールされていません。"
        "pip install torch-geometric などでインストールしてください。"
    )


# ------------------------------------------------------------
# 設定
# ------------------------------------------------------------

DATA_DIR   = "./gnn"  # pEqn_*, x_*, A_csr_* があるディレクトリ
TIME_LIST  = ["10200", "10400", "10600", "10800", "11000"]
RANK_STR   = "0"
NUM_EPOCHS = 1000
LR         = 1e-3
WEIGHT_DECAY = 1e-5

# 損失関数の重み（両方とも相対誤差なので同じスケール）
LAMBDA_DATA = 1.0    # データ損失の重み ||x_pred - x_true||² / ||x_true||²
LAMBDA_PDE  = 1.0    # PDE損失の重み ||A·x_pred - b||² / ||b||²

# メッシュ品質重み付けを使用するかどうか
USE_MESH_QUALITY_WEIGHT = False  # True: 品質の悪いセルを重視, False: 全セル均等

# ------------------------------------------------------------
# メッシュ品質ベース PDE 重み付けの設定
# feats の列インデックスは load_case_with_csr 内の定義に対応
# [x, y, z, diag, b, skew, nonOrth, aspect, diagContrast, V, h, sizeJump, Co]
# ------------------------------------------------------------

QUALITY_IDX = {
    "skew":         5,
    "nonOrth":      6,
    "aspect":       7,
    "diagContrast": 8,
    "sizeJump":    11,
}

# w_i = 1 + ALPHA_QUAL * (q_i ** BETA_QUAL)
ALPHA_QUAL = 4.0
BETA_QUAL  = 1.0
EPS_QUAL   = 1e-12


# ------------------------------------------------------------
# ユーティリティ: pEqn + CSR + x_true の読み込み
# ------------------------------------------------------------

def load_case_with_csr(data_dir: str, time_str: str, rank_str: str):
    """1 ケース分の
      - ノード特徴 (feats_np)
      - エッジ (edge_index_np)
      - 真値 x_true_np
      - PDE 用の A (CSR: row_ptr, col_ind, val) と右辺 b
    を読み込んで返す。
    """
    p_path   = os.path.join(data_dir, f"pEqn_{time_str}_rank{rank_str}.dat")
    x_path   = os.path.join(data_dir, f"x_{time_str}_rank{rank_str}.dat")
    csr_path = os.path.join(data_dir, f"A_csr_{time_str}.dat")

    if not os.path.exists(p_path):
        raise FileNotFoundError(p_path)
    if not os.path.exists(x_path):
        raise FileNotFoundError(x_path)
    if not os.path.exists(csr_path):
        raise FileNotFoundError(csr_path)

    # ---------- pEqn_*.dat をパースして CELLS / EDGES / b を取得 ----------
    with open(p_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # ヘッダ (nCells, nFaces)
    try:
        header_nc = lines[0].split()
        header_nf = lines[1].split()
        assert header_nc[0] == "nCells"
        assert header_nf[0] == "nFaces"
        nCells = int(header_nc[1])
    except Exception as e:
        raise RuntimeError(f"nCells/nFaces ヘッダの解釈に失敗しました: {p_path}\n{e}")

    # セクション位置
    try:
        idx_cells = next(i for i, ln in enumerate(lines) if ln.startswith("CELLS"))
        idx_edges = next(i for i, ln in enumerate(lines) if ln.startswith("EDGES"))
    except StopIteration:
        raise RuntimeError(f"CELLS/EDGES セクションが見つかりません: {p_path}")

    # WALL_FACES セクションの開始位置（あれば）
    idx_wall = None
    for i, ln in enumerate(lines):
        if ln.startswith("WALL_FACES"):
            idx_wall = i
            break
    if idx_wall is None:
        idx_wall = len(lines)

    cell_lines = lines[idx_cells + 1: idx_edges]
    edge_lines = lines[idx_edges + 1: idx_wall]

    if len(cell_lines) != nCells:
        print(f"[WARN] nCells={nCells} と CELLS 行数={len(cell_lines)} が異なります (time={time_str}).")

    # 特徴量: [x, y, z, diag, b, skew, nonOrth, aspect, diagContrast, V, h, sizeJump, Co]
    feats_np = np.zeros((len(cell_lines), 13), dtype=np.float32)
    b_np     = np.zeros(len(cell_lines), dtype=np.float32)

    for ln in cell_lines:
        parts = ln.split()
        if len(parts) < 14:
            raise RuntimeError(f"CELLS 行の列数が足りません: {ln}")
        cell_id = int(parts[0])
        xcoord  = float(parts[1])
        ycoord  = float(parts[2])
        zcoord  = float(parts[3])
        diag    = float(parts[4])
        b_val   = float(parts[5])
        skew    = float(parts[6])
        non_ortho  = float(parts[7])
        aspect     = float(parts[8])
        diag_con   = float(parts[9])
        V          = float(parts[10])
        h          = float(parts[11])
        size_jump  = float(parts[12])
        Co         = float(parts[13])

        if not (0 <= cell_id < len(cell_lines)):
            raise RuntimeError(f"cell_id の範囲がおかしいです: {cell_id}")

        feats_np[cell_id, :] = np.array(
            [
                xcoord, ycoord, zcoord,
                diag, b_val, skew, non_ortho, aspect,
                diag_con, V, h, size_jump, Co
            ],
            dtype=np.float32
        )
        b_np[cell_id] = b_val

    # エッジ（双方向）
    e_src = []
    e_dst = []
    for ln in edge_lines:
        parts = ln.split()
        if parts[0] == "WALL_FACES":
            break
        if len(parts) != 5:
            raise RuntimeError(f"EDGES 行の列数が 5 ではありません: {ln}")
        lower = int(parts[1])
        upper = int(parts[2])
        if not (0 <= lower < len(cell_lines) and 0 <= upper < len(cell_lines)):
            raise RuntimeError(f"lower/upper の cell index が範囲外です: {ln}")

        e_src.append(lower)
        e_dst.append(upper)
        e_src.append(upper)
        e_dst.append(lower)

    edge_index_np = np.vstack([
        np.array(e_src, dtype=np.int64),
        np.array(e_dst, dtype=np.int64)
    ])

    # ---------- x_*.dat から真値 ----------
    x_true_np = np.zeros(len(cell_lines), dtype=np.float32)
    with open(x_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) != 2:
                raise RuntimeError(f"x_*.dat の行形式が想定外です: {ln}")
            cid = int(parts[0])
            val = float(parts[1])
            if not (0 <= cid < len(cell_lines)):
                raise RuntimeError(f"x_*.dat の cell id が範囲外です: {cid}")
            x_true_np[cid] = val

    # ---------- CSR 形式の A を読み込み ----------
    with open(csr_path, "r") as f:
        csr_lines = [ln.strip() for ln in f if ln.strip()]

    try:
        h0 = csr_lines[0].split()
        h1 = csr_lines[1].split()
        h2 = csr_lines[2].split()
        assert h0[0] == "nRows"
        assert h1[0] == "nCols"
        assert h2[0] == "nnz"
        nRows = int(h0[1])
        nCols = int(h1[1])
        nnz   = int(h2[1])
    except Exception as e:
        raise RuntimeError(f"A_csr_{time_str}.dat のヘッダ解釈に失敗しました: {csr_path}\n{e}")

    if nRows != nCells:
        print(f"[WARN] CSR nRows={nRows} と pEqn nCells={nCells} が異なります (time={time_str}).")

    # ROW_PTR, COL_IND, VALUES の位置
    try:
        idx_rowptr = next(i for i, ln in enumerate(csr_lines) if ln.startswith("ROW_PTR"))
        idx_colind = next(i for i, ln in enumerate(csr_lines) if ln.startswith("COL_IND"))
        idx_vals   = next(i for i, ln in enumerate(csr_lines) if ln.startswith("VALUES"))
    except StopIteration:
        raise RuntimeError(f"ROW_PTR/COL_IND/VALUES が見つかりません: {csr_path}")

    row_ptr_str = csr_lines[idx_rowptr + 1].split()
    col_ind_str = csr_lines[idx_colind + 1].split()
    vals_str    = csr_lines[idx_vals + 1].split()

    if len(row_ptr_str) != nRows + 1:
        raise RuntimeError(
            f"ROW_PTR の長さが nRows+1 と一致しません: len={len(row_ptr_str)}, nRows={nRows}"
        )
    if len(col_ind_str) != nnz:
        raise RuntimeError(
            f"COL_IND の長さが nnz と一致しません: len={len(col_ind_str)}, nnz={nnz}"
        )
    if len(vals_str) != nnz:
        raise RuntimeError(
            f"VALUES の長さが nnz と一致しません: len={len(vals_str)}, nnz={nnz}"
        )

    row_ptr_np = np.array(row_ptr_str, dtype=np.int64)
    col_ind_np = np.array(col_ind_str, dtype=np.int64)
    vals_np    = np.array(vals_str,    dtype=np.float32)

    # row_idx (各非ゼロが属する行) を事前構築
    row_idx_np = np.empty(nnz, dtype=np.int64)
    for i in range(nRows):
        start = row_ptr_np[i]
        end   = row_ptr_np[i+1]
        row_idx_np[start:end] = i

    return {
        "time": time_str,
        "feats_np": feats_np,
        "edge_index_np": edge_index_np,
        "x_true_np": x_true_np,
        "b_np": b_np,
        "row_ptr_np": row_ptr_np,
        "col_ind_np": col_ind_np,
        "vals_np": vals_np,
        "row_idx_np": row_idx_np,
    }


# ------------------------------------------------------------
# GNN モデル定義
# ------------------------------------------------------------

class SimpleSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, 1))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x.view(-1)


# ------------------------------------------------------------
# CSR 形式での A x を計算するユーティリティ (PyTorch)
# ------------------------------------------------------------

def matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x):
    """y = A x を CSR (row_ptr, col_ind, vals) から計算する。

    row_ptr: (nRows+1,) long
    col_ind: (nnz,) long
    vals   : (nnz,) float
    row_idx: (nnz,) long  # 各非ゼロが属する行 index
    x      : (nRows,) float
    """
    y = torch.zeros_like(x)
    # y[i] += vals[k] * x[col_ind[k]] for all k with row_idx[k] == i
    y.index_add_(0, row_idx, vals * x[col_ind])
    return y


# ------------------------------------------------------------
# メッシュ品質重み付き PDE loss を計算するユーティリティ
# ------------------------------------------------------------

def compute_pde_loss(case, x_pred, use_mesh_quality_weight=False, eps=1e-12):
    """1 ケース分の PDE loss（相対残差）を計算する。

    損失関数:
        L_pde = ||A·x_pred - b||² / ||b||²  (use_mesh_quality_weight=False)
        L_pde = ||r||²_w / ||b||²_w         (use_mesh_quality_weight=True)

    Args:
        case: b, row_ptr, col_ind, vals, row_idx, w_pde を含む dict
        x_pred: (N,) 物理空間の圧力予測
        use_mesh_quality_weight: メッシュ品質重み付けを使用するか
        eps: ゼロ除算を防ぐための小さな値

    Returns:
        pde_loss: PDE損失（相対残差の二乗）
        R_pred: ログ用の相対残差 ||r|| / ||b||
    """
    b       = case["b"]
    row_ptr = case["row_ptr"]
    col_ind = case["col_ind"]
    vals    = case["vals"]
    row_idx = case["row_idx"]

    # PDE 残差 r = A·x_pred - b
    Ax = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_pred)
    r  = Ax - b

    # 相対残差（ログ用）
    norm_r = torch.norm(r)
    norm_b = torch.norm(b)
    R_pred = norm_r / (norm_b + eps)

    if use_mesh_quality_weight:
        # メッシュ品質重み付き PDE loss
        #   L_pde = ||r||²_w / ||b||²_w
        #        = (Σᵢ wᵢ·rᵢ²) / (Σᵢ wᵢ·bᵢ²)
        w_pde = case["w_pde"]
        r2 = r * r
        b2 = b * b
        weighted_r2 = (w_pde * r2).sum()
        weighted_b2 = (w_pde * b2).sum()
        pde_loss = weighted_r2 / (weighted_b2 + eps)
    else:
        # 通常の相対残差の二乗
        #   L_pde = ||r||² / ||b||²
        pde_loss = (norm_r * norm_r) / (norm_b * norm_b + eps)

    return pde_loss, R_pred


# ------------------------------------------------------------
# メイン: 5 ケース + 相対誤差ベースの data loss + PDE loss
# ------------------------------------------------------------

def train_gnn_5cases_relative_loss(data_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # ---------- ケース読み込み ----------
    raw_cases = []
    print("=== 使用するケース (time, rank) ===")
    for t in TIME_LIST:
        print(f"  time={t}, rank={RANK_STR}")
    print("==============================")

    for t in TIME_LIST:
        print(f"[LOAD] time={t}, rank={RANK_STR} のグラフ+PDE情報を読み込み中...")
        raw = load_case_with_csr(data_dir, t, RANK_STR)
        raw_cases.append(raw)

    # 一貫性チェック
    nCells0 = raw_cases[0]["feats_np"].shape[0]
    nFeat   = raw_cases[0]["feats_np"].shape[1]
    for rc in raw_cases:
        if rc["feats_np"].shape[0] != nCells0 or rc["feats_np"].shape[1] != nFeat:
            raise RuntimeError("全ケースで nCells/nFeatures が一致していません。")

    print(f"[INFO] nCells (1 ケース目) = {nCells0}, nFeatures = {nFeat}")

    # ---------- グローバル正規化 (特徴量 + x_true) ----------
    all_feats = np.concatenate([rc["feats_np"] for rc in raw_cases], axis=0)
    all_xtrue = np.concatenate([rc["x_true_np"] for rc in raw_cases], axis=0)

    feat_mean = all_feats.mean(axis=0, keepdims=True)
    feat_std  = all_feats.std(axis=0, keepdims=True) + 1e-12

    x_mean = all_xtrue.mean()
    x_std  = all_xtrue.std() + 1e-12

    print(f"[INFO] x_true (all cases): min={all_xtrue.min():.3e}, max={all_xtrue.max():.3e}, mean={x_mean:.3e}")

    # torch スカラーとしても保持（GPU/CPU どちらでも使えるように device 上に置く）
    x_mean_t = torch.tensor(x_mean, dtype=torch.float32, device=device)
    x_std_t  = torch.tensor(x_std,  dtype=torch.float32, device=device)

    # ---------- メッシュ品質のグローバル min/max を計算 ----------
    qual_stats = {}
    for name, j in QUALITY_IDX.items():
        v = np.concatenate([rc["feats_np"][:, j] for rc in raw_cases], axis=0).astype(np.float32)
        v_min = float(v.min())
        v_max = float(v.max())
        qual_stats[name] = (v_min, v_max)

    # ---------- Case データを torch Tensor に変換 ----------
    cases = []
    for rc in raw_cases:
        feats_np  = rc["feats_np"]
        x_true_np = rc["x_true_np"]

        # 特徴量の標準化
        feats_norm      = (feats_np  - feat_mean) / feat_std
        x_true_norm_np  = (x_true_np - x_mean)   / x_std

        feats       = torch.from_numpy(feats_norm).float().to(device)
        edge_index  = torch.from_numpy(rc["edge_index_np"]).long().to(device)
        x_true      = torch.from_numpy(x_true_np).float().to(device)
        x_true_norm = torch.from_numpy(x_true_norm_np).float().to(device)

        b       = torch.from_numpy(rc["b_np"]).float().to(device)
        row_ptr = torch.from_numpy(rc["row_ptr_np"]).long().to(device)
        col_ind = torch.from_numpy(rc["col_ind_np"]).long().to(device)
        vals    = torch.from_numpy(rc["vals_np"]).float().to(device)
        row_idx = torch.from_numpy(rc["row_idx_np"]).long().to(device)

        # --- メッシュ品質重み w_pde を計算 (numpy で作ってから torch に) ---
        N = feats_np.shape[0]
        q_accum = np.zeros(N, dtype=np.float32)

        for name, j in QUALITY_IDX.items():
            v = feats_np[:, j].astype(np.float32)
            v_min, v_max = qual_stats[name]
            v_norm = (v - v_min) / (v_max - v_min + EPS_QUAL)
            v_norm = np.clip(v_norm, 0.0, 1.0)
            q_accum += v_norm

        q_mean = q_accum / float(len(QUALITY_IDX))  # [0,1]
        w_pde_np = 1.0 + ALPHA_QUAL * np.power(q_mean, BETA_QUAL)
        w_pde = torch.from_numpy(w_pde_np).float().to(device)

        cases.append({
            "time": rc["time"],
            "feats": feats,
            "edge_index": edge_index,
            "x_true": x_true,
            "x_true_norm": x_true_norm,
            "b": b,
            "row_ptr": row_ptr,
            "col_ind": col_ind,
            "vals": vals,
            "row_idx": row_idx,
            "w_pde": w_pde,
        })

    num_cases = len(cases)

    # ---------- モデル定義 ----------
    model = SimpleSAGE(in_channels=nFeat, hidden_channels=64, num_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    loss_type = "with mesh quality weight" if USE_MESH_QUALITY_WEIGHT else "relative error"
    print(f"=== Training start (data loss + PDE loss [{loss_type}], 5 cases) ===")

    # ---------- 学習ループ ----------
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        total_data_loss = 0.0
        total_pde_loss  = 0.0
        sum_rel_err     = 0.0
        sum_R_pred      = 0.0

        for cs in cases:
            feats       = cs["feats"]
            edge_index  = cs["edge_index"]
            x_true      = cs["x_true"]
            x_true_norm = cs["x_true_norm"]

            # --- forward ---
            x_pred_norm = model(feats, edge_index)

            # 物理空間に戻す
            x_pred = x_pred_norm * x_std_t + x_mean_t  # (N,)

            # データ損失（物理空間での相対誤差）
            #   L_data = ||x_pred - x_true||² / ||x_true||²
            diff = x_pred - x_true
            data_loss_case = torch.sum(diff * diff) / (torch.sum(x_true * x_true) + 1e-12)

            # PDE損失（相対残差）
            pde_loss_case, R_pred_case = compute_pde_loss(cs, x_pred, USE_MESH_QUALITY_WEIGHT)

            total_data_loss = total_data_loss + data_loss_case
            total_pde_loss  = total_pde_loss  + pde_loss_case

            # 評価用 (勾配に影響しないよう detach)
            with torch.no_grad():
                rel_err_case = torch.norm(x_pred.detach() - x_true) / (torch.norm(x_true) + 1e-12)
                sum_rel_err += rel_err_case.item()
                sum_R_pred  += R_pred_case.detach().item()

        total_data_loss = total_data_loss / num_cases
        total_pde_loss  = total_pde_loss  / num_cases
        loss = LAMBDA_DATA * total_data_loss + LAMBDA_PDE * total_pde_loss

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            avg_rel_err = sum_rel_err / num_cases
            avg_R_pred  = sum_R_pred / num_cases
            print(
                f"[Epoch {epoch:5d}] loss={loss.item():.4e}, "
                f"data_loss={LAMBDA_DATA * total_data_loss:.4e}, "
                f"PDE_loss={LAMBDA_PDE * total_pde_loss:.4e}, "
                f"rel_err(avg)={avg_rel_err:.4e}, R_pred(avg)={avg_R_pred:.4e}"
            )

    # ---------- 最終評価 & x_pred 書き出し ----------
    print(f"\n=== Final diagnostics per case (GNN, relative error loss [{loss_type}]) ===")
    model.eval()
    for cs in cases:
        time_str   = cs["time"]
        feats      = cs["feats"]
        edge_index = cs["edge_index"]
        x_true     = cs["x_true"]
        b          = cs["b"]
        row_ptr    = cs["row_ptr"]
        col_ind    = cs["col_ind"]
        vals       = cs["vals"]
        row_idx    = cs["row_idx"]

        with torch.no_grad():
            x_pred_norm = model(feats, edge_index)
            x_pred = x_pred_norm * x_std_t + x_mean_t
            diff = x_pred - x_true
            rel_err = torch.norm(diff) / (torch.norm(x_true) + 1e-12)

            Ax = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_pred)
            r  = Ax - b
            norm_r = torch.norm(r)
            norm_b = torch.norm(b) + 1e-12
            R_pred = norm_r / norm_b

        print(
            f"  Case (time={time_str}, rank={RANK_STR}): "
            f"rel_err = {rel_err.item():.4e}, "
            f"R_pred = {R_pred.item():.4e}"
        )

        # x_pred をファイル出力
        x_pred_np = x_pred.cpu().numpy().reshape(-1)
        out_path = os.path.join(DATA_DIR, f"x_pred_{time_str}_rank{RANK_STR}.dat")
        with open(out_path, "w") as f:
            for i, val in enumerate(x_pred_np):
                f.write(f"{i} {val:.9e}\n")
        print(f"    [INFO] x_pred を {out_path} に書き出しました。")


if __name__ == "__main__":
    train_gnn_5cases_relative_loss(DATA_DIR)

