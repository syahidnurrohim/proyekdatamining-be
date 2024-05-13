from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import gdown
import os

app = Flask(__name__)
CORS(app, origins=["*"])


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
    try:
        file = request.files.get('file')
        if file is None:
            return jsonify({"status": "error", "message": "file does not exists"})
        
        model_path = "./model.h5"
        
        if os.path.exists(model_path) is False:
            return jsonify({"status": "error", "message": "model does not exists"})
        
        model = tf.keras.models.load_model(model_path, compile=False)
        
        image_open = Image.open(file)
        image_open = image_open.resize((150, 150))

        image_array = tf.keras.utils.img_to_array(image_open)
        image_array_resized = np.resize(image_array, (150, 150, 3))
        
        batch_image_array = np.array([image_array_resized])
        arr = model.predict(batch_image_array)
        # Define alphabet
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        # Convert probabilities to alphabet results
        # Find index of maximum probability
        max_index = np.argmax(arr)

        # Convert index to corresponding letter
        result = alphabet[max_index]

        return jsonify({"status": "ok", "data": {
            "result": result
        }})
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
