# Stable Audio Sample Explorer

## Overview

The Sample Explorer is a tool used to explore and iterate on generations with [stable-audio-open](https://huggingface.co/stabilityai/stable-audio-open-1.0). It is a work-in-progress project that serves as both a visualzation tool and iterative sample generation. The bulk of the code used to generate audio is abstracted behind `graph_model.py`, which by default generates 30sec of audio with `cfg_scale` of 7 and using 100-200 steps. These parameters (and others) are tunable in `audio_node.py`, but expect some changes at some point to move to a config file approach for more user control.

## Usage

### Setting up the environment:

```
pip install -r requirements.txt
```

### Running the Flask app

```
python flask_app.py
```

## Limitations

Currently, the sample explorer generates audio in half the sample_size used by the default stable-audio-open model. This is due to the need to fit this model on my 3080, feel free to change this. Other limitations include the messiness of the graph visualization getting a little hard to read longer prompts, and many more with the 'remix' inpainting function.

## Future Work
- Add strength values for combination of nodes
- Combining should be using inpainting also, need to figure this out more (combines waveforms right now)
- 'Remixing' works but is usually messy and unpredictable
- Add better UI details
- Add the ability to delete nodes
- Add the ability to save a project and re-load it
- Better organization of the filesystem for audio files
- and more to come