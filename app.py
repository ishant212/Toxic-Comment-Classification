from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# 🔹 Load model + tokenizer
model = load_model("model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 150
labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']


# 🔹 Prediction Function
def predict_comment(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(pad)[0]

    result = dict(zip(labels, pred))

    # 🔥 Apply threshold
    threshold = 0.5
    detected = [label for label, val in result.items() if val > threshold]

    return {
        "scores": {k: float(v) for k, v in result.items()},  # convert numpy → float
        "labels": detected if detected else ["non-toxic"]
    }


# 🔹 Routes
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    final_labels = None

    if request.method == "POST":
        text = request.form.get("comment", "").strip()

        if text:
            prediction = predict_comment(text)
            result = prediction["scores"]
            final_labels = prediction["labels"]

    return render_template(
        "index.html",
        result=result,
        final_labels=final_labels
    )


# 🔹 Run
if __name__ == "__main__":
    app.run(debug=True)