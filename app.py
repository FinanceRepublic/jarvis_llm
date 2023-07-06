from flask import Flask, request, jsonify
from llm_functions import generate_response

app = Flask(__name__)


@app.route('/', methods=['POST'])
def main_flask_fn():
    req_as_json = request.get_json(silent=True, force=True)
    userquery = req_as_json.get('userquery')
    context = req_as_json.get('context')
    llm_res = generate_response(userquery, context)
    res = jsonify({'userquery': userquery, 'llm_response': llm_res})
    return res


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
