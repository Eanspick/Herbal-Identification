from flask import redirect, render_template, request
from markupsafe import Markup

from .model import predict_image
from .utils import PLANT_DIC


def home():
    return render_template("index.html")


def predict():
    if request.method != "POST":
        return redirect("/")
    try:
        file = request.files["file"]
        img = file.read()
        prediction, confidence = predict_image(img)
        res = Markup(PLANT_DIC[prediction])
        return render_template(
            "display.html", status=200, result=res, confidence=confidence
        )
    except ValueError as ve:
        print("\n", ve, "\n")
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
