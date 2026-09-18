# ME-Dex-1.0

ME-Dex-1.0 is a video-action-tactile policy trained on RoboTwin Clean50. This repository provides its inference runtime for standardized RoboTwin leaderboard evaluation using XPolicyLib.

## Model weights

Model and tactile AE weights are available at [liuxuetao/ME-Dex-1.0-RoboTwin-Clean2Random-Leaderboard](https://huggingface.co/liuxuetao/ME-Dex-1.0-RoboTwin-Clean2Random-Leaderboard). The runtime also uses the VAE, T5 encoder, tokenizer and config from [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).

## Usage

Install the dependencies with `pip install -r runtime/requirements.txt`.

The evaluation interface supplies RGB observations. Set `input_color_order: bgr` so the runtime performs the single RGB-to-BGR conversion expected by the checkpoint. The runtime does not apply mean/std image normalization.

Clean50 training included additionally collected three-axis tactile force data. RoboTwin leaderboard observations contain no tactile measurements, so the current-frame tactile force is set to zero while preserving the sensor support mask. Future tactile states are predicted by the model.

## Acknowledgements

Built on [Wan2.2](https://github.com/Wan-Video/Wan2.2) and [Motus](https://github.com/motus-robotics/Motus).

## License

[Apache-2.0](LICENSE).
