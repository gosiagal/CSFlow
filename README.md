# CSFlow: Aligning Flow Matching with Human Contrast Sensitivity

<div align="center">
  <a href="https://vip.mpi-inf.mpg.de/people/Galinska.html" target="_blank">Malgorzata&nbsp;Galinska</a><sup>*</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=Xwe4-J0AAAAJ&hl=en" target="_blank">Bart&nbsp;Pogodzinski</a><sup>*</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=enXCzCgAAAAJ&hl=en" target="_blank">Jan&nbsp;Eric&nbsp;Lenssen</a> <br>
  Max Planck Institute for Informatics, Saarland Informatics Campus &emsp;<br>
  <sup>*</sup>Equal Contribution &emsp; <br>
</div>
<br/>

<p align="center">
  <a href="https://arxiv.org/abs/2606.08833">
    <img src="https://img.shields.io/badge/arXiv-2606.08833-b31b1b.svg">
  </a>
  <a href="https://github.com/gosiagal/CSFlow">
    <img src="https://img.shields.io/badge/GitHub-CSFlow-black.svg?logo=github">
  </a>
  <a href="https://huggingface.co/gosiagalinska/CSFlow">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow.svg">
  </a>

</p>

<p align="center">
  <img src="teaser_figure.jpg" width="100%">
</p>

This is a codebase for the paper [CSFlow: Aligning Flow Matching with Human Contrast Sensitivity](https://arxiv.org/abs/2606.08833).

This project computes CSFlow timestep importance weights, that can be used to bias an image generation model to perceptually more important stages. We also provide a modified weighted-JiT model that can use our weights in inference or training.

## 1. Overview

The CSF stage lives in [main_csflow.py](main_csflow.py) and produces importance weights file under a results directory. The weighted JiT stage lives under [jit](jit) and consumes the file through CLI arguments such as `--inference_weights_path` and `--train_weights_path`.

The CSF code does not directly train or sample an image model. It produces a numerical weighting schedule that reflects how much each diffusion timestep should matter based on perceptual frequency sensitivity and retained signal during the forward process. The weights are generated once, then reused across runs, model variants or models that use the same training dataset and noise schedule.

## 2. Generation of CSFlow weights

Set the environment:
```bash
cd /path/to/csflow

pip install -r requirements.txt
```

Run the main CSFlow code:
```bash
# for ImageNet 256x256 weights:
python main_gpu.py --config configs/imagenet_256.yaml

# for ImageNet 512x512 weights:
python main_gpu.py --config configs/imagenet_512.yaml

# for Blip3o_60k 512x512 weights:
python main_gpu.py --config configs/blip3o_512.yaml
```

### What the script does

The script:

- loads the precomputed RAPSD statistics for the selected dataset
- computes $r_{signal}$, $\partial_t r_{signal}$, $CSF \cdot \partial_t r_{signal}$ curves over timesteps
- integrates $CSF \cdot \partial_t r_{signal}$ over frequency and normalizes it to form scalar timestep weights
- normalizes the weights
- saves output file in the results directory
- reproduces paper figures and saves them in the figures directory

### Output files

The output is created under a directory such as:

```bash
/results/${experiment.name}_${image.resolution}_steps${num_steps}/

# for example:
results/imagenet_256_steps500/
```

File produced:

- `training_weights.pkl` - two numpy arrays: importance weights and corresponding time points.

Since the weights are calculated over time intervals, there is one less weight value than time points. If `weights_calculation.padded` is set to `True` (default), there is a zero-padded value on the last time point 1.0 (unused). The resulting arrays are shapes: `(weights_calculation.num_steps+1)` and `(weights_calculation.num_steps+1)`.

### Figures 
If the `figures.reproduce_figures` is set to `True`, paper figures will be plotted and saved under the specified figures path.

## 3. Requirements for JiT

A suitable conda environment named jit can be created and activated with:

```bash
cd /path/to/csflow

conda env create -f jit/environment.yaml
conda activate jit
```

You will also need ImageNet dataset (available [here](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php)) and JiT-H/16 checkpoint (available [here](https://www.dropbox.com/scl/fo/3ken1avtsd81ip67b9qpi/AK218ZNvXKSv74igVvht4PQ?rlkey=14gjrblmljewpl6ygxzlr3njm&st=ffkl77al&e=1&dl=0)).

## 4. Weighted inference in JiT-H/16

Once the weight files are generated, you can point JiT at them and generate and evaluate images:

```bash
cd /path/to/csflow

python jit/main_jit.py \
  --model JiT-H/16 \
  --img_size 256 --noise_scale 1.0 \
  --gen_bsz 256 --num_images 50000 --cfg 2.2 --interval_min 0.1 --interval_max 1.0 \
  --output_dir ./jit-h-16-csflow-weighted --resume /path/to/checkpoint \
  --data_path /path/to/imagenet --evaluate_gen  \
  --inference_weighted True --inference_interpolated True  \
  --inference_weights_path ./results/imagenet_256_steps500/training_weights.pkl
```

This script will generate 50K test images, evaluate them against test ImageNet dataset [statistics](jit/fid_stats/jit_in256_stats.npz) - calculate FID and IS, and delete all images afterwards. If you wish to save the images, add the `--keep_generated_images` flag and images will stay in the output directory.

## 5. Weighted training in JiT-H/16

For weighted training, enable:

```bash
--train_weighted True --train_interpolated True
--train_weights_path ./results/imagenet_256_steps500/training_weights.pkl
```

Fine-tuned model weights from our paper are available [here](https://huggingface.co/gosiagalinska/CSFlow/tree/main).

## 6. Non-default usage

**General modifications:**

- Computing RAPSD: `imagenet_*.yaml` and `blip3o_*.yaml` default configurations use precomputed RAPSD statistics and do not require providing the `dataset.path`. If you want to compute the statistics yourself, you need to have the ImageNet/BLIP3o dataset locally saved and specitfied in the `dataset.path`. The weights image resolution parameter `image.resolution` and the dataset/JiT image resolution must always stay consistent.
- Changing viewing assumptions: viewing conditions are set using `frequencies.pixel_size` and `frequencies.viewing_distance` parameters in the configuration file (e.g. `imagenet_256.yaml`).

**If you want to use a different dataset:**

- In the current state, `dataset.type` is either `imagenet` or `blip3o`. If you wish to use a different dataset, create and add a corresponding DataLoader and set the `dataset.path` to your designated dataset.
- Image resolution is set using `image.resolution` parameter in the configuration file.

**If you want to use a different model:**

- Note that this implementation uses an inverse-linear noise schedule $x_t = tx_0 + (1-t)\epsilon$: $t=0$ corresponds to noise and $t=1$ corresponds to a clean image. Make sure your model uses the same noise schedule or change the noise scheduler in the implementation.
- `image.resolution` must match the model you are using.
- your model's inference/training algorithm needs to be modified for using CSFlow weights.

## 7. Additional notes

1. In our paper, we use CSFlow weights for 500 steps for accuracy. The modified JiT model can use a discrete weight function with **any** number of steps - it linearly interpolates the discrete weight values. We set the default `weights_calculation.num_steps` to 500 in all config files. 

2. In the paper heatmap figures ($r_{signal}$, $\partial_t r_{signal}$, $CSF \cdot \partial_t r_{signal}$) we use 50 uniform steps for the metric calculations to match the true JiT 50 sampling steps. We set the default `figures.plot_num_steps` to 50 in all config files and it is used only for plotting these heatmaps. 

3. The JiT denoiser supports multiple schedule behaviors:

    - uniform schedule (original)
    - CSFlow schedule
    - combined schedule: $\alpha_2 \cdot w_{\mathrm{CSFlow}} + (1-\alpha_2) \cdot w_{\mathrm{uniform}}$
    - optional timeshift rescaling

    The default option is the combined schedule with interpolation paramter $\alpha_2=0.4$. The timestep grid $t_1, t_2, ..., t_N$ is created through uniform sampling of the inverse CDF of the combined weight function, therefore, smaller steps correspond to larger weights.

4. The JiT training also supports multiple time sampling behaviors:

    - lognorm (original)
    - CSFlow weighting function
    - combined weight function: $\alpha_1 \cdot w_{\mathrm{CSFlow}} + (1-\alpha_1) \cdot w_{\mathrm{lognorm}}$

    The default option is the combined weight function with interpolation parameter $\alpha_1=0.4$. During training, the time value $t$ is a random sample of the inverse CDF of the combined weight function.

## 8. Reproducing figures and results

`figures/` and `results/` are not committed as source going forward; they are generated artifacts of `main_gpu.py` and `reproduce_figures.py` (see [.gitignore](.gitignore)). Regenerate them locally by rerunning the commands in Section 3 with `figures.reproduce_figures: true` in the config (the default). Precomputed RAPSD statistics under `rapsd_cache/` are required inputs and remain tracked in the repo.

## 9. License

This project is released under the [MIT License](LICENSE).

## 10. Citation
```
@misc{csflow2026,
      title={CSFlow: Aligning Flow Matching with Human Contrast Sensitivity}, 
      author={Malgorzata Galinska and Bart Pogodzinski and Jan Eric Lenssen},
      year={2026},
      url={https://arxiv.org/abs/2606.08833}, 
}
```

See also [CITATION.cff](CITATION.cff) for citation metadata.

