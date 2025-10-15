from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- add this

app = Flask(__name__)
CORS(app)  # <-- enable CORS

data_storage = []

@app.route('/upload', methods=['POST'])
def upload_data():
    global data_storage
    new_data = request.get_json()
    if new_data:
        data_storage.extend(new_data)
        print(f"[Server] Received {len(new_data)} records")
    return "OK", 200

@app.route('/data', methods=['GET'])
def get_data():
    global data_storage
    return jsonify(data_storage[-100:])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
