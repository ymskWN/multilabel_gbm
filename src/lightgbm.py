"""lightgbm for multilabel classification."""
import numpy as np
import optuna.integration.lightgbm as lgb_o

# 評価指標は以下４つ
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train[0].shape)
print(y_train[0].shape)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# 各データを1次元に変換
X_train = X_train.reshape(-1, 784)
X_valid = X_valid.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
print(X_train[0].shape)

# 正規化
X_train = X_train.astype("float32")
X_valid = X_valid.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_valid /= 255
X_test /= 255

# 訓練・検証データの設定
train_data = lgb_o.Dataset(X_train, label=y_train)
eval_data = lgb_o.Dataset(X_valid, label=y_valid, reference=train_data)

# パラメータ設定
params = {
    "task": "train",  # トレーニング用
    "boosting_type": "gbdt",  # 勾配ブースティング決定木
    "objective": "multiclass",  # 目的:多値分類
    "num_class": 10,  # 分類クラス数
    "metric": "multi_logloss",  # 評価指標は多クラスのLog損失
    "early_stopping_rounds": 10,  # 10回連続で改善が見られなければ学習を終了
}

best_params = {}
# モデル作成
gbm = lgb_o.train(
    params,
    train_data,
    valid_sets=[train_data, eval_data],  # ここがチューニングしない場合と違う
    num_boost_round=100,
)
best_params = gbm.params
print(best_params)
# 予測
preds = gbm.predict(X_test, num_iteration=gbm.best_iteration)

y_pred = []
for x in preds:
    y_pred.append(np.argmax(x))

# 正解率など評価指標の計算
print(f"正解率(accuracy_score):{accuracy_score(y_test, y_pred)}")
# 適合率、再現率、F1値はマクロ平均を取る
print("再現率(recall_score):{}".format(recall_score(y_test, y_pred, average="macro")))
print("適合率(precision_score):{}".format(precision_score(y_test, y_pred, average="macro")))
print("F1値(f1_score):{}".format(f1_score(y_test, y_pred, average="macro")))
