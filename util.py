import torch
import torchaudio

import os

# Util methods are in this file

# save_audio: saves a torchaudio tensor to the './tmp/ directory with a given filename
# Params: audio (tensor), name (string)
# Returns: the filepath of the saved .wav file (string)
def save_audio(audio, name, sample_rate):

    assert os.path.exists("./tmp/")
    output = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    prompt_string = ''.join(name.split("|"))
    path = "./tmp/{prompt}.wav".format(prompt=prompt_string)
    path = path.replace(' ', '')
    torchaudio.save(path, output, sample_rate)

    return path

# format_prompt: Util method to format a node's prompt by removing duplicates
# Params: a node to format
# Returns: the same node with its prompt string formatted
def format_prompt(node):

    parts = node.prompt.split("|")
    result = list(set(parts))
    node.prompt = "|".join(result)
    
    return node

# combine_waveforms: combines two torchaudio tensors
# Params: two torchaudio tensors
# returns: the torchaudio tensors combined
def combine_waveforms(wave_1, wave_2):

    combined_waveform = wave_1 + wave_2
    # Normalize to avoid clipping
    combined_waveform = combined_waveform / torch.max(torch.abs(combined_waveform))

    return combined_waveform