from flask import Flask, jsonify, request
import numpy as np
import pickle
import xgboost as xgb
import pkg_resources

app = Flask(__name__)
MODEL_PATH = "./artifact/xgboost_model.pickle"

# この部分で、pickle形式のモデルの読み込みを行う
def load_artifact(filename):
    artifact = None
    with open(filename, mode='rb') as fp:
        artifact = pickle.load(fp)
    if artifact is None:
        raise ValueError        
    return artifact

model = load_artifact(MODEL_PATH)

# "http://<ドメイン名>"へアクセスしたときのブラウザ表示
@app.route("/")
def index():
    return "XGBoost prediction API with App Runner and flask."

# "http://<ドメイン名>/api/v1/predict"へAPI呼び出しを行う際の動作
@app.route("/api/v1/predict", methods=["POST"])
def predict():
    response = {
        "success": False,
        "Content-Type": "application/json"
    }

    if request.get_json().get("feature"):
        feature = request.get_json().get("feature") # リクエストからfeature読み込み
        
        ver = check_version()
        #response["pred"] = model_predict(feature) # model_predict関数を使ってモデル予測
        response["version"] = ver
        response["success"] = True

    return jsonify(response)

def check_version():
    pkglist = ['xgboost', 'lightgbm']
    for dist in pkg_resources.working_set:
        if dist.project_name in pkglist:
            return dist.project_name, dist.version
        else:
            return dist.project_name, "nothing"

def model_predict(feature):
    global model
    feature = np.array(feature)
    app.logger.debug(feature.shape)  # HTTPリクエストのfeatureのnp.ndarrayに変換
    if len(feature.shape) != 2:  # もしデータが1つ（=1次元）であった場合
        feature = feature.reshape((1, -1))
        
    dfeature = xgb.DMatrix(feature)  # XGBoostのデータ形式に変換 
    pred = model.predict(dfeature)  # モデルの予測
    pred_list = pred.tolist()  # 予測結果をpythonのlistに変換

    return pred_list


if __name__ == "__main__":
    app.run()
