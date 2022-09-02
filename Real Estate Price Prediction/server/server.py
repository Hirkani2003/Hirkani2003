from flask import Flask, request, jsonify
import util
app = Flask(__name__)


@app.route("/get_location_names")
def get_location_names():
    response = jsonify({
        "locations" : util.get_location_names()
    })
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route("/predict_price", methods=["POST"])
def predict_price():
    sqft = float(request.form["sqft"])
    location = request.form["location"]
    size = int(request.form["size"])
    bath = int(request.form["bath"])
    response = jsonify({
        "estimated_price" : util.predict_price(location, size, sqft, bath)
    })
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == "__main__":
    print("Starting FLask Server for Real Estate Price Prediction...")
    app.debug = True
    app.run()