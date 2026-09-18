# ME-Dex-1.0

A Video–Action–Tactile policy for robotic manipulation. This repository provides inference code for standardized RoboTwin leaderboard evaluation using XPolicyLib.

## Model weights

Download the model and tactile encoder from [Hugging Face](https://huggingface.co/liuxuetao/ME-Dex-1.0-RoboTwin-Clean2Random-Leaderboard).
Additional VAE, text encoder and tokenizer assets are available from [Wan2.2](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).

## Usage

Install the dependencies with `pip install -r runtime/requirements.txt`.
Evaluation is performed through XPolicyLib.

## Acknowledgements

Built on [Wan2.2](https://github.com/Wan-Video/Wan2.2) and [Motus](https://github.com/motus-robotics/Motus).

## License

[Apache-2.0](LICENSE).
