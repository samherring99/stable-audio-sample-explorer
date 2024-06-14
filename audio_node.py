from einops import rearrange
from stable_audio_tools.inference.generation import generate_diffusion_cond

# AudioNode is an atomic component of the sample exploration tree
# At its core it stores:
#   the prompt string 
#   a torchaudio tensor 
#   a filepath of the audio file in .wav format
#   and a list of child AudioNodes
# This file stores the class definition and inherent methods used for generation
class AudioNode:

    # Initialization with empty fields
    def __init__(self, prompt=""):
        self.prompt = prompt
        self.audio = None
        self.file_path = ""
        self.children = []

    # generate_audio: called by a Node to generate audio from its prompt. Called in create_node
    # Params: StableAudio model and a sample size
    # Returns: None
    def generate_audio(self, model, sample_size):
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

        # Rearrange output to be a single sequence
        output = rearrange(output, "b d n -> d (b n)")
        self.audio = output
        self.file_path = save_audio(output, self.prompt)


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
            init_noise_level=7.0,
            device=device
        )

        output = rearrange(output, "b d n -> d (b n)")

        self.audio = output
        self.file_path = save_audio(output, self.prompt)