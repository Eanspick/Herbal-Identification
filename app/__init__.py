from flask import Flask

from .main import home, predict


def get_app():
    app = Flask(__name__)
    app.route("/", methods=["GET"])(home)
    app.route("/predict", methods=["GET", "POST"])(predict)
    return app


if __name__ == "__main__":
    app = get_app()
    app.run(debug=True)
