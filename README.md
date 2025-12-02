# GNN Tiny - グラフニューラルネットワークによる圧力方程式の解法

## 概要

このプロジェクトは、Graph Neural Network (GNN)を用いて流体力学シミュレーションにおける圧力方程式（Pressure Equation）の解を学習・予測するシステムです。OpenFOAMなどのCFD（数値流体力学）ソルバーで使用される圧力方程式の数値解を、GNNによって高速に近似します。

## 主な特徴

- **PyTorch Geometric**を使用したグラフ畳み込みニューラルネットワーク（GraphSAGE）の実装
- **複数時間ステップ**（5ケース）での学習に対応
- **ハイブリッド損失関数**:
  - データ損失（MSE）: 真値との直接的な誤差
  - PDE損失: 物理方程式 `A·x = b` の残差を評価
- **メッシュ品質を考慮した重み付き学習**: 歪度、非直交性、アスペクト比などのメッシュ品質指標に基づいた重み付け
- **CSR形式**による疎行列演算の効率的な実装

## 必要な環境

### Python環境

- Python 3.7以上
- PyTorch
- PyTorch Geometric
- NumPy

### インストール

```bash
pip install torch numpy
pip install torch-geometric
```

または、Conda環境の場合:

```bash
conda install pytorch numpy -c pytorch
conda install pyg -c pyg
```

## データ構造

### データディレクトリ (`./gnn/`)

以下のファイルが各時間ステップごとに含まれています:

#### 1. `pEqn_{time}_rank{rank}.dat`
圧力方程式のセル特徴量とエッジ情報を含むファイル

**ヘッダー:**
- `nCells`: セル数
- `nFaces`: 面数

**CELLS セクション:**
各行の形式: `cell_id x y z diag b skew nonOrth aspect diagContrast V h sizeJump Co`

- `x, y, z`: セル中心座標
- `diag`: 対角成分
- `b`: 右辺ベクトルの値
- `skew`: 歪度（メッシュ品質指標）
- `nonOrth`: 非直交性
- `aspect`: アスペクト比
- `diagContrast`: 対角コントラスト
- `V`: セル体積
- `h`: 特性長さ
- `sizeJump`: サイズジャンプ
- `Co`: クーラン数

**EDGES セクション:**
各行の形式: `face_id lower upper ...`

- セル間の接続関係を定義

#### 2. `x_{time}_rank{rank}.dat`
真値（圧力の解）

各行の形式: `cell_id pressure_value`

#### 3. `A_csr_{time}.dat`
CSR (Compressed Sparse Row)形式の係数行列

**ヘッダー:**
- `nRows`: 行数
- `nCols`: 列数
- `nnz`: 非ゼロ要素数

**データセクション:**
- `ROW_PTR`: 行ポインタ（長さ nRows+1）
- `COL_IND`: 列インデックス（長さ nnz）
- `VALUES`: 非ゼロ要素の値（長さ nnz）

#### 4. その他の補助ファイル
- `divPhi_{time}_rank{rank}.dat`: 発散項
- `rTrue_{time}_rank{rank}.dat`: 真の残差

## 使用方法

### 基本的な実行

```bash
python gnn_tiny.py
```

### パラメータのカスタマイズ

`gnn_tiny.py` の冒頭部分で以下のパラメータを調整できます:

```python
# データ設定
DATA_DIR   = "./gnn"                                      # データディレクトリ
TIME_LIST  = ["10200", "10400", "10600", "10800", "11000"]  # 使用する時間ステップ
RANK_STR   = "0"                                          # ランク番号

# 学習パラメータ
NUM_EPOCHS = 1000       # エポック数
LR         = 1e-3       # 学習率
WEIGHT_DECAY = 1e-5     # 重み減衰

# 損失関数の重み
LAMBDA_DATA = 0.1       # データ損失の重み
LAMBDA_PDE  = 1e-4      # PDE損失の重み

# メッシュ品質重み付けパラメータ
ALPHA_QUAL = 4.0        # 品質重みの強度
BETA_QUAL  = 1.0        # 品質重みの指数
```

## アーキテクチャ

### GNNモデル (SimpleSAGE)

```
入力層 (13特徴量)
    ↓
GraphSAGE層 (64チャネル)
    ↓
GraphSAGE層 (64チャネル) × 2
    ↓
GraphSAGE層 (1チャネル)
    ↓
出力 (圧力予測)
```

### 損失関数

総合損失 = `λ_data × データ損失 + λ_pde × PDE損失`

#### データ損失
```
L_data = MSE(x_pred_normalized, x_true_normalized)
```

#### メッシュ品質重み付きPDE損失
```
L_pde = ||r||²_w / ||b||²_w
```

ここで:
- `r = A·x_pred - b` (残差)
- `w_i = 1 + α_qual × (q_i)^β_qual` (メッシュ品質重み)
- `q_i`: セル i の正規化されたメッシュ品質指標（歪度、非直交性など）

### 特徴量の正規化

すべてのケースから計算されたグローバル平均・標準偏差を使用:

```
x_normalized = (x - x_mean) / x_std
```

## 出力

### 学習中の出力

50エポックごとに以下の指標が表示されます:

- `loss`: 総合損失
- `data_loss`: データ損失成分
- `PDE_loss`: PDE損失成分
- `rel_err(avg)`: 平均相対誤差 `||x_pred - x_true|| / ||x_true||`
- `R_pred(avg)`: 平均PDE残差 `||A·x_pred - b|| / ||b||`

### 最終出力ファイル

学習終了後、各ケースの予測値が以下のファイルに保存されます:

```
./gnn/x_pred_{time}_rank{rank}.dat
```

各行の形式: `cell_id predicted_pressure`

## 実験のヒント

### データ損失の重みを調整する実験

PDE物理制約への依存度を変更:

1. `LAMBDA_DATA = 1.0`: データ駆動型（教師あり学習に近い）
2. `LAMBDA_DATA = 0.3`: バランス型
3. `LAMBDA_DATA = 0.1`: PDE制約重視
4. `LAMBDA_DATA = 0.0`: 完全に物理制約のみ

### メッシュ品質重み付けの調整

品質の悪いセルへの学習強度を変更:

- `ALPHA_QUAL`: 大きくすると品質の悪いセルの損失への寄与が増加
- `BETA_QUAL`: 非線形性を調整（1.0で線形、>1.0で非線形増加）

### モデルサイズの調整

`SimpleSAGE` クラスのパラメータ:

```python
model = SimpleSAGE(
    in_channels=nFeat,
    hidden_channels=64,    # 隠れ層のチャネル数
    num_layers=4          # GraphSAGE層の数
)
```

## 技術的な詳細

### CSR形式での行列-ベクトル積

効率的な疎行列演算のため、CSR形式を使用:

```python
def matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x):
    y = torch.zeros_like(x)
    y.index_add_(0, row_idx, vals * x[col_ind])
    return y
```

この実装により、GPUでの高速な `A·x` 計算が可能です。

### メッシュ品質指標の使用

5つの品質指標を統合:

1. **skew**: 歪度
2. **nonOrth**: 非直交性
3. **aspect**: アスペクト比
4. **diagContrast**: 対角コントラスト
5. **sizeJump**: サイズジャンプ

各指標を[0,1]に正規化し、平均を取って総合品質スコアを計算します。

## ライセンス

（プロジェクトのライセンスを記載してください）

## 参考文献

- GraphSAGE: Hamilton et al. "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)
- Physics-Informed Neural Networks (PINNs): Raissi et al. "Physics-informed neural networks" (JCP 2019)

## 貢献

バグ報告や機能追加の提案は、Issueまたはプルリクエストでお願いします。

## 連絡先

（プロジェクト管理者の連絡先を記載してください）
