# 參考下方連結，安裝對應 torch、torchvision、torchaudio 版本
測試成功：

Python 環境為 >=3.8 , <=3.11 下，

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install xformers==0.0.17（或 0.0.20）

參考連結：https://blog.csdn.net/shiwanghualuo/article/details/122860521

# 運行步驟
在主目錄下，
(1) （僅第一次）

    pip install -r requirements.txt

(2) 
    
    pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 -f https://download.pytorch.org/whl/torch_stable.html

    pip install xformers==0.0.17

    pip install audiocraft

(3) 

    python3 main.py

# main.py 是參考下方連結寫成
https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md
https://huggingface.co/facebook/musicgen-melody

# Windows No module named ‘triton‘ 問題解方
從 https://huggingface.co/r4ziel/xformers_pre_built/blob/main/triton-2.0.0-cp310-cp310-win_amd64.whl
下載 triton-2.0.0-cp310-cp310-win_amd64.whl 後放在主目錄中，
並在 Terminal 端 運行指令 python.exe -m pip install triton-2.0.0-cp310-cp310-win_amd64.whl

https://blog.csdn.net/ddrfan/article/details/130127401

# AudioCraft
![docs badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_docs/badge.svg)
![linter badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_linter/badge.svg)
![tests badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_tests/badge.svg)

AudioCraft is a PyTorch library for deep learning research on audio generation. AudioCraft contains inference and training code
for two state-of-the-art AI generative models producing high-quality audio: AudioGen and MusicGen.


## Installation
AudioCraft requires Python 3.9, PyTorch 2.0.0. To install AudioCraft, you can run the following:

```shell
# Best to make sure you have torch installed first, in particular before installing xformers.
# Don't run this if you already have PyTorch installed.
pip install 'torch>=2.0'
# Then proceed to one of the following
pip install -U audiocraft  # stable release
pip install -U git+https://git@github.com/facebookresearch/audiocraft#egg=audiocraft  # bleeding edge
pip install -e .  # or if you cloned the repo locally (mandatory if you want to train).
```

We also recommend having `ffmpeg` installed, either through your system or Anaconda:
```bash
sudo apt-get install ffmpeg
# Or if you are using Anaconda or Miniconda
conda install "ffmpeg<5" -c conda-forge
```

## Models

At the moment, AudioCraft contains the training code and inference code for:
* [MusicGen](./docs/MUSICGEN.md): A state-of-the-art controllable text-to-music model.
* [AudioGen](./docs/AUDIOGEN.md): A state-of-the-art text-to-sound model.
* [EnCodec](./docs/ENCODEC.md): A state-of-the-art high fidelity neural audio codec.
* [Multi Band Diffusion](./docs/MBD.md): An EnCodec compatible decoder using diffusion.

## Training code

AudioCraft contains PyTorch components for deep learning research in audio and training pipelines for the developed models.
For a general introduction of AudioCraft design principles and instructions to develop your own training pipeline, refer to
the [AudioCraft training documentation](./docs/TRAINING.md).

For reproducing existing work and using the developed training pipelines, refer to the instructions for each specific model
that provides pointers to configuration, example grids and model/task-specific information and FAQ.


## API documentation

We provide some [API documentation](https://facebookresearch.github.io/audiocraft/api_docs/audiocraft/index.html) for AudioCraft.


## FAQ

#### Is the training code available?

Yes! We provide the training code for [EnCodec](./docs/ENCODEC.md), [MusicGen](./docs/MUSICGEN.md) and [Multi Band Diffusion](./docs/MBD.md).

#### Where are the models stored?

Hugging Face stored the model in a specific location, which can be overriden by setting the `AUDIOCRAFT_CACHE_DIR` environment variable for the AudioCraft models.
In order to change the cache location of the other Hugging Face models, please check out the [Hugging Face Transformers documentation for the cache setup](https://huggingface.co/docs/transformers/installation#cache-setup).
Finally, if you use a model that relies on Demucs (e.g. `musicgen-melody`) and want to change the download location for Demucs, refer to the [Torch Hub documentation](https://pytorch.org/docs/stable/hub.html#where-are-my-downloaded-models-saved).


## License
* The code in this repository is released under the MIT license as found in the [LICENSE file](LICENSE).
* The models weights in this repository are released under the CC-BY-NC 4.0 license as found in the [LICENSE_weights file](LICENSE_weights).


## Citation

For the general framework of AudioCraft, please cite the following.
```
@article{copet2023simple,
    title={Simple and Controllable Music Generation},
    author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
    year={2023},
    journal={arXiv preprint arXiv:2306.05284},
}
```

When referring to a specific model, please cite as mentioned in the model specific README, e.g
[./docs/MUSICGEN.md](./docs/MUSICGEN.md), [./docs/AUDIOGEN.md](./docs/AUDIOGEN.md), etc.
