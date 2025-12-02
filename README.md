# GNN Tiny - グラフニューラルネットワークによる圧力方程式の解法

## 概要

このプロジェクトは、Graph Neural Network (GNN)を用いて流体力学シミュレーションにおける圧力方程式（Pressure Equation）の解を学習・予測するシステムです。OpenFOAMなどのCFD（数値流体力学）ソルバーで使用される圧力方程式の数値解を、GNNによって高速に近似します。

## 主な特徴

- **PyTorch Geometric**を使用したグラフ畳み込みニューラルネットワーク（GraphSAGE）の実装
- **複数時間ステップ**（5ケース）での学習に対応
- **スケール統一されたハイブリッド損失関数**:
  - データ損失: `||x_pred - x_true||² / ||x_true||²` （相対誤差）
  - PDE損失: `||A·x_pred - b||² / ||b||²` （相対残差）
  - 両方とも無次元量なので、重み `λ = 1.0` で平等に扱える
- **メッシュ品質を考慮した重み付き学習**（オプション）: 歪度、非直交性、アスペクト比などのメッシュ品質指標に基づいた重み付け
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

# 損失関数の重み（両方とも相対誤差なので同じスケール）
LAMBDA_DATA = 1.0       # データ損失の重み
LAMBDA_PDE  = 1.0       # PDE損失の重み

# メッシュ品質重み付けの使用
USE_MESH_QUALITY_WEIGHT = False  # True: 品質の悪いセルを重視

# メッシュ品質重み付けパラメータ（USE_MESH_QUALITY_WEIGHT=True時のみ有効）
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

本プロジェクトでは、**データ損失とPDE損失の両方を相対誤差として定式化**することで、損失のスケールを統一しています。

#### 総合損失

```
L_total = λ_data × L_data + λ_pde × L_pde
```

両方の損失が無次元の相対誤差なので、**λ_data = λ_pde = 1.0** で真に平等に扱えます。

#### データ損失（物理空間での相対誤差）

GNNの予測と真値の相対的な差を評価します：

```
L_data = ||x_pred - x_true||² / ||x_true||²
       = (Σᵢ (x_pred,i - x_true,i)²) / (Σᵢ x_true,i²)
```

**意味**:
- 真値のノルムで正規化した予測誤差の二乗
- 無次元量（相対誤差の二乗）
- 典型的なスケール: O(0.01～1.0)

#### PDE損失（相対残差）

物理方程式 `A·x = b` の満足度を評価します：

```
L_pde = ||A·x_pred - b||² / ||b||²
      = (Σᵢ rᵢ²) / (Σᵢ bᵢ²)
```

ここで、`r = A·x_pred - b` は残差ベクトルです。

**意味**:
- 右辺ベクトルのノルムで正規化した残差の二乗
- 無次元量（相対残差の二乗）
- 典型的なスケール: O(0.01～1.0)

#### メッシュ品質重み付き損失（オプション）

`USE_MESH_QUALITY_WEIGHT = True` の場合、メッシュ品質の悪いセルを重視する重み付きPDE損失を使用します：

```
L_pde = ||r||²_w / ||b||²_w
      = (Σᵢ wᵢ·rᵢ²) / (Σᵢ wᵢ·bᵢ²)
```

重み `w` の計算：

```
wᵢ = 1 + α_qual × (qᵢ)^β_qual
```

ここで:
- `qᵢ`: セル i の総合品質スコア（0～1、1が最悪）
- `α_qual`: 重みの強度（デフォルト: 4.0）
- `β_qual`: 非線形性の指数（デフォルト: 1.0）

総合品質スコア `q` は、以下の5つの指標を正規化して平均したものです：
1. **skew**: 歪度
2. **nonOrth**: 非直交性
3. **aspect**: アスペクト比
4. **diagContrast**: 対角コントラスト
5. **sizeJump**: サイズジャンプ

各指標は [0, 1] に正規化され（1が品質最悪）、その平均が `q` となります。

#### 数学的な整合性

両損失が同じ形式（相対誤差の二乗）なので：

1. **スケールが統一**：両方とも O(0.01～1.0) 程度
2. **無次元量**：物理単位に依存しない
3. **直感的な解釈**：
   - `L_data = 0.01` → 真値から約10%の誤差
   - `L_pde = 0.01` → PDEの右辺から約10%の残差
4. **重みの意味が明確**：
   - `λ_data = 2.0, λ_pde = 1.0` → データ忠実度をPDE制約の2倍重視

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

### データ損失とPDE損失のバランス調整

両方の損失が相対誤差（同じスケール）なので、重みの比率が直感的に解釈できます：

1. **データ駆動型**: `LAMBDA_DATA = 2.0, LAMBDA_PDE = 1.0`
   - 真値への適合を2倍重視
   - 訓練データが十分にある場合に有効

2. **バランス型**: `LAMBDA_DATA = 1.0, LAMBDA_PDE = 1.0`（デフォルト）
   - データ忠実度とPDE制約を平等に扱う
   - 推奨される初期設定

3. **物理制約重視型**: `LAMBDA_DATA = 1.0, LAMBDA_PDE = 2.0`
   - PDE制約の満足を2倍重視
   - 訓練データが少ない、または外挿が必要な場合に有効

4. **PDE損失のみ**: `LAMBDA_DATA = 0.0, LAMBDA_PDE = 1.0`
   - 完全に物理方程式のみで学習
   - 教師なし学習に相当

### メッシュ品質重み付けの実験

`USE_MESH_QUALITY_WEIGHT = True` にすると、メッシュ品質の悪いセルの学習を重視します：

**パラメータ調整:**
- `ALPHA_QUAL = 0.0`: 重み付けなし（全セル均等）
- `ALPHA_QUAL = 2.0`: 軽度の重み付け
- `ALPHA_QUAL = 4.0`: 中程度の重み付け（デフォルト）
- `ALPHA_QUAL = 10.0`: 強い重み付け（品質の悪いセルを大幅に重視）

**非線形性の調整:**
- `BETA_QUAL = 1.0`: 線形（デフォルト）
- `BETA_QUAL = 2.0`: 二次関数的（品質の差を強調）
- `BETA_QUAL = 0.5`: 平方根（品質の差を緩和）

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
