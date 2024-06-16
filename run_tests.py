from audio_node import AudioNode, create_node, combine_nodes, remix_node
from util import save_audio, format_prompt, combine_waveforms
from viz import visualize_graph
from graph_model import GraphModel

import os
import torch

# Tests AudioNode base initialization
def test_audio_node_init():
    node = AudioNode("a basic sawtooth synthesizer")
    assert node.prompt == "a basic sawtooth synthesizer"
    return node

# Tests initialization of our GraphModel and StableAudio model with GPU
def test_graph_model_init():
    model = GraphModel()
    assert model.device == "cuda"
    return model

# Tests the format_prompt method
def test_format_prompt(node):
    node_formatted = format_prompt(node)

    assert node_formatted.prompt.split("|") == list(set(node.prompt.split("|")))

    return node_formatted

# Tests the combine_waveforms method
def test_combine_waveforms():
    random_tensor_1 = torch.randn(2, 32000)
    random_tensor_2 = torch.randn(2, 32000)

    combined = combine_waveforms(random_tensor_1, random_tensor_2)

    assert len(combined) == len(random_tensor_1) and len(combined) == len(random_tensor_2)

# Tests the saving of audio
def test_save_audio(sample_rate):
    random_tensor = torch.randn(2, 32000)
    save_audio(random_tensor, "test", sample_rate)

# Tests basic generation from an AudioNode
def test_generation_basic(node, model):
    node.generate_audio(model)
    assert os.path.exists("./tmp/abasicsawtoothsynthesizer.wav")
    #os.system("rm ./tmp/abasicsawtoothsynthesizer.wav")

    return node

# Test the generation of 'remixed' audio using an existing node
def test_generation_remixed(node, model):
    remix_node = AudioNode("a cat meowing|REMIX|" + node.prompt)
    remix_node.generate_remixed_audio(model, "a cat meowing", node.audio)
    assert len(remix_node.file_path) > 0
    #os.system("rm " + remix_node.file_path)
    return remix_node

# Tests the base method for create_node before abstraction in GraphModel
def test_create_node_base(model):
    node = create_node("a basic sawtooth synthesizer", model)
    assert node.file_path == "./tmp/abasicsawtoothsynthesizer.wav"
    #os.system("rm ./tmp/abasicsawtoothsynthesizer.wav")
    return node

# Tests the base method for combine_nodes before abstraction in GraphModel
def test_combine_nodes_base(model, node_1, node_2):
    combined_node = combine_nodes(node_1, node_2, model.sample_size, model.sample_rate)

    #assert node_1.prompt in combined_node.prompt and node_2.prompt in combined_node.prompt
    return combined_node

# Tests the 'remix' feature base method before abstraction in GraphModel
def test_remix_node_base(model, node):
    remixed_node = remix_node(node, model, "a complex flute melody")
    assert "flute" in remixed_node.prompt

    return remixed_node

# Tests creation of nodes in the GraphModel
def test_create_node_main(model):
    node = model.create_node("a basic sawtooth synthesizer")

    assert os.path.exists("./tmp/abasicsawtoothsynthesizer.wav") and node.file_path == "./tmp/abasicsawtoothsynthesizer.wav"

    return node

# Tests remixing of a node in the GraphModel
def test_remix_node_main(model, node):
    remixed_node = model.remix_node(node, "a cat meowing")
    assert len(remixed_node.file_path) > 0 and "REMIX" in remixed_node.prompt

    return remixed_node

# Tests combination of nodes in the GraphModel
def test_combine_nodes_main(model, node_1, node_2):
    combined_node = model.combine_nodes(node_1, node_2)
    assert len(combined_node.file_path) > 0

    return combined_node

# Tests the base visualizaiton method
def test_visualization_base(nodes):
    dot = visualize_graph(nodes)

    assert dot

def run_test_suite():
    print("Testing AudioNode and GraphModel creation...")

    model = test_graph_model_init()

    test_node = test_audio_node_init()

    test_node = test_format_prompt(test_node)
    
    print("Testing audio utility methods...")

    test_save_audio(model.sample_rate)
    test_combine_waveforms()

    print("Testing AudioNode base methods...")

    # Testing base generation method
    test_node = test_generation_basic(test_node, model)
    test_node_remixed = test_generation_remixed(test_node, model)

    # Testing combination and remix AudioNode methods
    combined_test_node = test_combine_nodes_base(model, test_node, test_node_remixed)
    remixed_combined_node = test_remix_node_base(model, combined_test_node)

    print("Testing GraphModel top level methods...")

    test_node = test_create_node_main(model)

    remixed_node = test_remix_node_main(model, test_node)

    combined_node = test_combine_nodes_main(model, test_node, remixed_node)

    test_visualization_base([test_node, remixed_node, combined_node])

    print("Success!")

if __name__ == "__main__":
    run_test_suite()