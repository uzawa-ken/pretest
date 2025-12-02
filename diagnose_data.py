#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データの診断スクリプト
x_trueの値、正規化、モデル予測値を確認
"""

import os
import numpy as np
import torch

DATA_DIR = "./gnn"
TIME_LIST = ["10200", "10400", "10600", "10800", "11000"]
RANK_STR = "0"

print("=" * 60)
print("データ診断")
print("=" * 60)

# 各ケースのx_trueを読み込み
all_x_true = []
for time_str in TIME_LIST:
    x_path = os.path.join(DATA_DIR, f"x_{time_str}_rank{RANK_STR}.dat")
    x_data = []
    with open(x_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                x_data.append(float(parts[1]))
    x_np = np.array(x_data, dtype=np.float32)
    all_x_true.append(x_np)

    print(f"\n時刻 {time_str}:")
    print(f"  要素数: {len(x_data)}")
    print(f"  min:  {x_np.min():.6e}")
    print(f"  max:  {x_np.max():.6e}")
    print(f"  mean: {x_np.mean():.6e}")
    print(f"  std:  {x_np.std():.6e}")
    print(f"  範囲: [{x_np.min():.6e}, {x_np.max():.6e}]")

# 全ケース統合
all_x_concat = np.concatenate(all_x_true, axis=0)
print("\n" + "=" * 60)
print("全ケース統合:")
print(f"  総要素数: {len(all_x_concat)}")
print(f"  min:  {all_x_concat.min():.6e}")
print(f"  max:  {all_x_concat.max():.6e}")
print(f"  mean: {all_x_concat.mean():.6e}")
print(f"  std:  {all_x_concat.std():.6e}")

# 正規化後の値
x_mean = all_x_concat.mean()
x_std = all_x_concat.std() + 1e-12

print("\n" + "=" * 60)
print("正規化パラメータ:")
print(f"  x_mean: {x_mean:.6e}")
print(f"  x_std:  {x_std:.6e}")

# 正規化後のサンプル値
x_sample = all_x_true[0][:10]
x_sample_norm = (x_sample - x_mean) / x_std
print("\n" + "=" * 60)
print("正規化後のサンプル値（最初の10要素）:")
print("  元の値:", x_sample)
print("  正規化後:", x_sample_norm)

# 逆正規化のテスト
x_sample_denorm = x_sample_norm * x_std + x_mean
print("\n逆正規化のテスト:")
print("  元の値:    ", x_sample)
print("  逆正規化後:", x_sample_denorm)
print("  差の最大値:", np.abs(x_sample - x_sample_denorm).max())

# ゼロに近い値の割合
threshold = 1e-10
near_zero_count = np.sum(np.abs(all_x_concat) < threshold)
near_zero_ratio = near_zero_count / len(all_x_concat)
print("\n" + "=" * 60)
print(f"絶対値が {threshold:.1e} 未満の要素:")
print(f"  個数: {near_zero_count} / {len(all_x_concat)}")
print(f"  割合: {near_zero_ratio * 100:.2f}%")

# 相対誤差の計算例
print("\n" + "=" * 60)
print("相対誤差のシミュレーション:")
# ランダムな予測値を生成（実際のモデルではありませんが、スケール感を確認）
for noise_level in [1e-10, 1e-9, 1e-8]:
    x_pred_sim = x_sample + np.random.normal(0, noise_level, size=x_sample.shape)
    rel_err = np.linalg.norm(x_pred_sim - x_sample) / (np.linalg.norm(x_sample) + 1e-12)
    print(f"  ノイズレベル {noise_level:.1e}: 相対誤差 = {rel_err:.4f} ({rel_err*100:.2f}%)")

print("\n" + "=" * 60)
