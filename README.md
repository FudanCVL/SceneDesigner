
<p align="center">
  <h2 align="center">SceneDesigner: Controllable Multi-Object Image Generation with 9-DoF Pose Manipulation</h2>
<p align="center">
    <a href="https://github.com/qqzy/"><strong>Zhenyuan Qin<sup>*</sup></strong></a>
    ·
    <a href="https://github.com/xinchengshuai/"><strong>Xincheng Shuai<sup>*</sup></strong></a>
    ·
    <a href="https://henghuiding.com/"><strong>Henghui Ding </strong><sup>†</sup></a>
</p>

<p align="center">
    Fudan University
</p>
<p align="center">
<a href="https://arxiv.org/abs/"><img src="https://img.shields.io/static/v1?label=Paper&message=2505.19114&color=red&logo=arxiv"></a>
<a href="https://qqzy.github.io/SceneDesigner-Page/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a>
</p>





<!-- ## 🎯 Introduction -->


![](assets/teaser.jpg)

## ⚙️ Quick Start

### 1. Installation

1. Install Python environment (recommended to use uv)
   ```bash
   uv sync
   ```
   Or alternatively:
   ```bash
   pip install -r requirements.txt
   ```

2. Install Blender environment
   ```bash
   cd render
   python install.py
   ```

   If the automatic installation script fails, you can install manually:
   * First download [Blender](https://download.blender.org/release/Blender4.2/) and extract it to the `./render` directory
   * Then locate the Blender Python path and install the Python dependencies for Blender, for example:
    ```bash
    cd render
    blender-4.2.8-linux-x64/4.2/python/bin/python3.11 -m pip install -r blender_requirements.txt
    ```

### 2. Download Checkpoints

1. Download the [SceneDesigner](https://huggingface.co/FudanCVL/SceneDesigner) weights to the `checkpoints` directory
2. Download the [Stable Diffusion 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) base model weights to the `checkpoints` directory

### 3. Run Demo

Launch the Gradio app:
```bash
python app.py \
  --blender_path render/blender/blender \
  --device cuda:0 \
  --port 7861 
```

- Adjust the 9D pose of the cube in the **Cube Controls** panel
- Enter text prompts in the **Generation Config** panel and click the **Generate Images** button to create images



## ✒️ Citation

If you find our work useful for your research and applications, please kindly cite using this BibTeX:

```latex
@inproceedings{SceneDesigner,
        title={SceneDesigner: Controllable Multi-Object Image Generation with 9-DoF Pose Manipulation},
        author={Qin, Zhenyuan and Shuai, Xincheng and Ding, Henghui},
        booktitle={NeurIPS},
        year={2025}
      }
```
