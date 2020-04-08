# %%
import numpy as np

# %%
class Sigmoid:
    def __init__(self):
        # 学習すべきパラメータ
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        out = x @ W + b
        return out

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size
        # np.random.randn(N, M) -> N x M行列を平均0,分散1の要素で初期化
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # レイヤを生成
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        self.params = []
        for layer in self.layers:
            # ["a"] + ["b"] = ["a", "b"]
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

# %%
x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
s = model.predict(x)
s.max(axis=1, keepdims=True)

# %%
# この層自体に入力は複数ある
# 入力の次元と出力の次元は同じ
# 入力一つ一つに対しての確率を出力する
def softmax(x):
    # 入力が行列のとき
    if x.ndim == 2:
        # axisで軸の方向を決める
        # 0の場合は行ごと,1の場合は列ごとの最大値を求める
        # keepdimsをTrueにすることで二次配列のままになる
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x = x / x.sum(axis=1, keepdims=True)
    # 入力がベクトルのとき
    elif x.ndim == 1:
        x = x - np.max(x) # ここはオーバーフロー対策のためであって,理論的にはこの行は必要ない.ただしこの行があっても確率が変わらない?
        x = np.exp(x) / np.sum(np.exp(x))
    return x

def cross_entropy_error(y, t):
    if y.ndim == 1:
        # 行ベクトルに変更?
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
