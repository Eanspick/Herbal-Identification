from flask import Flask, render_template, request
from markupsafe import Markup

from .model import predict_image
from .utils import PLANT_DIC


def home():
    return render_template("index.html")


def predict():
    try:
        file = request.files["file"]
        img = file.read()
        prediction, confidence = predict_image(img)
        print(prediction, confidence)
        res = Markup(PLANT_DIC[prediction])
        return render_template(
            "display.html", status=200, result=res, confidence=confidence
        )
    except ValueError as ve:
        print(ve)
        return render_template(
            "display.html",
            status=400,
            result="Error: Uploaded image is not recognized as a herbal image.",
        )
    except Exception as e:
        print(e)
        return render_template(
            "display.html", status=500, result="Internal Server Error"
        )


def get_app():
    app = Flask(__name__)
    app.route("/", methods=["GET"])(home)
    app.route("/predict", methods=["POST"])(predict)
    return app


if __name__ == "__main__":
    app = get_app()
    app.run(debug=True)
