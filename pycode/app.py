from flask import Flask, request, jsonify

from pycode.main import PhraseSimilarityApp

app = Flask(__name__)

phrase_app = PhraseSimilarityApp('vectors.csv', 'phrases.csv')


@app.route('/similarity', methods=['POST'])
def get_similarity():
    data = request.json
    user_phrase = data['phrase']
    closest_phrase, distance = phrase_app.find_similarity(user_phrase)
    return jsonify({'closest_phrase': closest_phrase, 'distance': distance})


if __name__ == '__main__':
    app.run()
