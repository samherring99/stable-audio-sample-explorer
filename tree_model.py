from stable_audio_tools import get_pretrained_model
from graphviz import Digraph
import torch

from audio_node import create_node, remix_node, combine_nodes
from viz import visualize_tree

# Class definition for the TreeModel that contains all generated AudioNodes and their children
# This class mainly wraps the Stable-Audio-Open model and helper methods with top level params
class TreeModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        self.sample_rate = self.model_config["sample_rate"]
        self.sample_size = self.model_config["sample_size"]
        self.model_config["model_type"] = "diffusion_cond_inpaint"
        self.model = self.model.to(self.device)
        self.nodes = []

    # create_node: creates an AudioNode
    def create_node(self, prompt):
        node = create_node(prompt, self.model, self.sample_size, self.sample_rate)
        self.nodes.append(node)
        return node

    # remix_node: 'remixes' an AudioNode with a given prompt
    def remix_node(self, node, prompt):
        node = remix_node(node, self.model, prompt, self.sample_size, self.sample_rate)
        self.nodes.append(node)
        return node

    # combines_nodes: combines two AudioNodes
    def combine_nodes(self, node1, node2):
        node = combine_nodes(node1, node2, self.sample_size, self.sample_rate)
        self.nodes.append(node)
        return node

    # visualize: visualize the tree
    def visualize(self):
        dot = visualize_tree(self.nodes)
        dot.render('tree', format='png', view=True)