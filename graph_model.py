from stable_audio_tools import get_pretrained_model
from graphviz import Digraph
import torch

from audio_node import create_node, remix_node, combine_nodes
from viz import visualize_graph

# Class definition for the GraphModel that contains all generated AudioNodes and their children
# This class mainly wraps the stable-audio-open model and helper methods with top level params
class GraphModel:
    def __init__(self):
        print("Initializing model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        self.sample_rate = self.model_config["sample_rate"]
        self.sample_size = self.model_config["sample_size"]
        self.model_config["model_type"] = "diffusion_cond_inpaint"
        self.model = self.model.to(self.device)
        self.nodes = []

    # create_node: creates an AudioNode
    def create_node(self, prompt):
        print("Generating audio for prompt: " + prompt)
        node = create_node(prompt, self)
        self.nodes.append(node)
        return node

    # remix_node: 'remixes' an AudioNode with a given prompt
    def remix_node(self, node, prompt):
        print("Remixing {audio} with ".format(audio=node.prompt) + prompt)
        node = remix_node(node, self, prompt)
        self.nodes.append(node)
        return node

    # combines_nodes: combines two AudioNodes
    def combine_nodes(self, node_1, node_2):
        print("Combining {audio_1} with {audio_2}".format(audio_1=node_1.prompt, audio_2=node_2.prompt))
        node = combine_nodes(node_1, node_2, self.sample_size, self.sample_rate)
        self.nodes.append(node)
        return node

    # visualize: visualize the graph
    def visualize(self):
        print("Generating visualization...")
        dot = visualize_graph(self.nodes)
        dot.render('graph', format='png', view=True)