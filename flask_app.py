from flask import Flask, request, jsonify, send_from_directory, send_file
from graph_model import GraphModel
from audio_node import AudioNode

import os

app = Flask(__name__, static_folder='static')
model = GraphModel()

def node_exists(prompt):
    for node in model.nodes:
        if node.prompt == prompt:
            return True
    return False

def get_node_with_prompt(prompt):
    return_node = AudioNode()

    for node in model.nodes:
        if prompt == node.prompt:
            return_node = node

    return return_node

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/create_node', methods=['POST'])
def create_node():
    prompt = request.json['prompt']
    print("Goo")
    node = model.create_node(prompt)
    return jsonify(node=node.to_dict())

@app.route('/combine_nodes', methods=['POST'])
def combine_nodes():
    node1_prompt = request.json['node1_prompt']
    node2_prompt = request.json['node2_prompt']
    node1 = get_node_with_prompt(node1_prompt)
    node2 = get_node_with_prompt(node2_prompt)

    if node1.audio is not None and node2.audio is not None:
        combined_node = model.combine_nodes(node1, node2)
        return jsonify(node=combined_node.to_dict())
    else:
        print("Cant find one or both nodes!")
        return {}

@app.route('/remix_node', methods=['POST'])
def remix_node():
    node_prompt = request.json['node_prompt']
    prompt = request.json['prompt']

    print(node_prompt)

    node_to_remix = get_node_with_prompt(node_prompt)

    if len(node_to_remix.prompt) > 0:
        remix_node = model.remix_node(node_to_remix, prompt)
        return jsonify(node=remix_node.to_dict())
    else:
        print("Can't find node!")
        return {}

@app.route('/tmp/<path:filename>', methods=['GET'])
def serve_audio(filename):
    audio_dir = os.path.join(app.root_path, 'tmp')
    audio_path = os.path.join(audio_dir, filename)
    if os.path.exists(audio_path):
        return send_file(audio_path, mimetype='audio/wav')
    else:
        return 'Audio file not found', 404


if __name__ == '__main__':
    app.run(debug=False)
