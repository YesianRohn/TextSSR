# 🎰TextSSR: Diffusion-based Data Synthesis for Scene Text Recognition

<a href=''><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href=''><img src='https://img.shields.io/badge/Page-Github-green'></a> <a href=''><img src='https://img.shields.io/badge/Demo&Model&Data-ModelScope-lightblue'></a> <a href=''><img src='https://img.shields.io/badge/Demo&Model&Data-HuggingFace-yellow'></a> 

<div style="text-align: left;">
<img src="./imgs/intro.svg" alt="Intro Image">
</div>

$$
TextSSR ~ Capability ~ Showcase.
$$


## 📢News

TBD



## 📝TODOs

- [ ] Provide publicly checkpoints and gradio demo
- [ ] Release TextSSR-benchmark dataset and evaluation code
- [ ] Release processed AnyWord-lmdb dataset 
- [ ] Release our scene text synthesis dataset, TextSSR-F
- [x] Release training and inference code



## 💎Visualization

<div style="text-align: left;">
<img src="./imgs/model.svg" alt="Intro Model">
</div>

$$
Model  ~ Architecture ~ Display.
$$

<div style="text-align: left;">
<img src="./imgs/framework.svg" alt="Intro Framework" width=100%>
</div>

$$
Data  ~ Synthesis ~ Pipeline.
$$

<div style="text-align: left;">
<img src="./imgs/more_viz.svg" alt="Results">
</div>


$$
Results ~ Presentation.
$$



## 🛠Installation

#### Environment Settings

1. **Clone the TextSSR Repository:**
   
    ```bash
    git clone https://github.com/YesianRohn/TextSSR.git
    cd TextSSR
   ```

2. **Create a New Environment for TextSSR:**
   ```bash
   conda create -n textssr python=3.10
   conda activate textssr
   ```

3. **Install Required Dependencies:**
   
   - Install PyTorch, TorchVision, Torchaudio, and the necessary CUDA version:
   ```bash
   conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
   ```
   
   - Install the rest of the dependencies listed in the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```
   - Install our modified diffusers:
   ```bash
   cd diffusers
   pip install -e .
   cd ..
   ```

---

#### Checkpoints/Data Preparation


1. **Data Preparation:**

   - You can use the [`Anyword-3M`](https://www.modelscope.cn/datasets/iic/AnyWord-3M) dataset provided by Anytext. However, you will need to modify the data loading code to use `AnyWordDataset` instead of `AnyWordLmdbDataset`.
   - If you have obtained our `AnyWord-lmdb` dataset, simply place it in the `TextSSR` folder.

2. **Font File Preparation:**

   - You can either download the Alibaba PuHuiTi font from [here](https://www.alibabafonts.com/#/font), which should be named `AlibabaPuHuiTi-3-85-Bold.ttf`, or you can use your own custom font file.
   - Place your font file in the `TextSSR` folder.

3. **Model Preparation:**
- If you want to train the model from scratch, first download the SD2-1 model from [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-2-1).
   - Place the downloaded model in the `model` folder.
   - During the training process, you will obtain several model checkpoints. These should be placed sequentially in the `model` folder as follows:
     - `vae_ft` (trained VAE model)
     - `step1` (trained  CDM after step 1)
     - `step2` (trained CDM after step 2)

After the preparations outlined above, you will have the following file structure:

```
TextSSR/
├── model/
│   ├── stable-diffusion-v2-1
│   ├── vae_ft
│       ├── checkpoint-x/
│       	├── vae/
│       	└── ...
│   ├── step1
│       ├── checkpoint-x/
│       	├── unet/
│       	└── ...
│   ├── step2
│       ├── checkpoint-x/
│       	├── unet/
│       	└── ...
│   └── AnyWord-lmdb/                      
│       ├── step1_lmdb/
│       ├── step2-lmdb/
├── AlibabaPuHuiTi-3-85-Bold.ttf
├── ...(the same as the GitHub code)
```




## 🚂 Training

1. **Step 1: Fine-tune the VAE:**
   ```bash
   accelerate launch --num_processes 8 train_vae.py --config configs/train_vae_cfg.py
   ```

2. **Step 2: First stage of CDM training:**
   ```bash
   accelerate launch --num_processes 8 train_diff.py --config configs/train_diff_step1_cfg.py
   ```

3. **Step 3: Second stage of CDM training:**
   ```bash
   accelerate launch --num_processes 8 train_diff.py --config configs/train_diff_step2_cfg.py
   ```



## 🔍 Inference

- Ensure the `bench` path is correctly set in `infer.py`.
- Run the inference process with:
   ```bash
   python infer.py
   ```

This will start the inference and generate the results.



## 📊Evaluation
```
TBD
```



## 🔗Citation

```
TBD
```



## 🌟 Acknowledgements

Many thanks to these great projects for their contributions, which have influenced and supported our work in various ways: [SynthText](https://arxiv.org/abs/1604.06646), [TextOCR](https://arxiv.org/abs/2105.05486), [DiffUTE](https://arxiv.org/abs/2305.10825), [Textdiffuser](https://arxiv.org/abs/2305.10855) & [Textdiffuser-2](https://arxiv.org/abs/2311.16465), [AnyText](https://arxiv.org/abs/2311.03054), [UDiffText](https://arxiv.org/abs/2312.04884), [SceneVTG](https://arxiv.org/abs/2407.14138), and [SVTRv2](https://arxiv.org/abs/2411.15858).

Special thanks also go to the training frameworks: [STR-Fewer-Labels](https://github.com/ku21fan/STR-Fewer-Labels) and [OpenOCR](https://github.com/Topdu/OpenOCR).


