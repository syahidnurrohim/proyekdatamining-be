from flask import Flask, request, jsonify
import tensorflow as tf
import gdown
import os

app = Flask(__name__)


@app.route("/health-check", methods=["GET"])
def healthCheck():
    return jsonify({"status": "ok"})


@app.route("/download-model/<fileId>", methods=["GET"])
def downloadModel(fileId):
    try:
        gdown.download(id=fileId, output="./model.h5")
        return jsonify({"status": "ok", "message": "file uploaded"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# Define a route to predict
@app.route("/predict", methods=["POST"])
def predict():
    model_path = './model.h5'
    if os.path.exists(model_path) is False:
        return jsonify({"status": "error", "message": "model does not exists"})
    # Load the TensorFlow model
    model = tf.keras.models.load_model(model_path)

    # Get the JSON data from the request
    data = request.get_json()

    # Perform inference
    prediction = model.predict(data["input"])

    # Return the prediction as JSON
    return jsonify({"prediction": prediction.tolist()})


if __name__ == "__main__":
    app.run(debug=True)
