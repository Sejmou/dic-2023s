from flask import Flask, jsonify, make_response
import argparse


def create_app():
    app = Flask(__name__)

    # routing http posts to this method
    @app.route("/", methods=["GET"])
    def main():
        data = {
            "message": "Hello World!",
        }

        return make_response(jsonify(data), 200)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8501, debug=True)
