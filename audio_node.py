import torch
from einops import rearrange
from stable_audio_tools.inference.generation import generate_diffusion_cond

from util import combine_waveforms, format_prompt, save_audio

# Setting global device
device = "cuda" if torch.cuda.is_available() else "cpu"

# AudioNode is an atomic component of the sample exploration tree - the TreeModel
#
# At its core it stores:
# the prompt string 
# a torchaudio tensor 
# a filepath of the audio file in .wav format
# and a list of child AudioNodes
#
# This file stores the class definition and methods used for generation
class AudioNode:

    # Initialization with empty fields
    def __init__(self, prompt=""):
        self.prompt = prompt
        self.audio = None
        self.file_path = ""
        self.children = []

    # generate_audio: called by a Node to generate audio from its prompt. Called in create_node
    # Params: StableAudio model, a sample size, and a sample rate
    # Returns: None
    def generate_audio(self, model, sample_size, sample_rate):
        conditioning = [{"prompt": self.prompt, "seconds_start": 0, "seconds_total": 30}]

        output = generate_diffusion_cond(
            model,
            steps=100,
            cfg_scale=7,
            conditioning=conditioning,
            sample_size=int(sample_size/2),
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=device
        )

        output = rearrange(output, "b d n -> d (b n)")
        self.audio = output
        self.file_path = save_audio(output, self.prompt, sample_rate)


    # generate_audio: called by a Node to generate audio from its prompt. Called in remix_node
    # Params: StableAudio model, a prompt for generation, the torchaudio tensor for init
    # Also takes in sample size and sample rate
    # Returns: None
    def generate_remixed_audio(self, model, prompt, init_audio, sample_size, sample_rate):
        in_sr = sample_rate
        init_audio_pair = (in_sr, init_audio)

        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0, 
            "seconds_total": 30
        }]

        output = generate_diffusion_cond(
            model,
            steps=200,
            cfg_scale=7,
            conditioning=conditioning,
            sample_size=int(sample_size/2),
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            init_audio=init_audio_pair,
            init_noise_level=10.0,
            device=device
        )

        output = rearrange(output, "b d n -> d (b n)")
        self.audio = output
        self.file_path = save_audio(output, self.prompt, sample_rate)

# create_node: Creates a node from a given prompt
# Params: a prompt string to initialize the node and generate 30 audio sample, and the StableAudio model
# Returns: the initiazlied node
def create_node(prompt, model, sample_size, sample_rate):
    node = AudioNode(prompt)
    node.generate_audio(model, sample_size, sample_rate)
    return node

# combines_nodes: Combines two existing nodes and adds itself to each of their children
# Params: Two nodes to combine, plus a sample size and sample rate
# Returns: the child node of the combined nodes with the combined audio
def combine_nodes(node_1, node_2, sample_size, sample_rate):
    new_child = AudioNode(node_1.prompt + "|" + node_2.prompt)
    combined_audio = combine_waveforms(node_1.audio, node_2.audio)
    new_child.audio = combined_audio
    child = format_prompt(new_child)
    child.file_path = save_audio(combined_audio, child.prompt, sample_rate)

    node_1.children.append(child)
    node_2.children.append(child)

    return child

# remix_node: 'Remixes' a given node with a prompt using that node's audio for init
# Params: StableAudio model, a node to remix and a prompt to use for generation, 
# as well as a sample size and sample rate
# Returns: a new child node that has been 'remixed' from the parent node's audio
def remix_node(node, model, prompt, sample_size, sample_rate):
    remixed_node = AudioNode(node.prompt + "|REMIX|" + prompt)
    child = format_prompt(remixed_node)
    child.generate_remixed_audio(model, prompt, node.audio, sample_size, sample_rate)
    node.children.append(child)

    return child