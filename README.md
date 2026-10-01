# Neural-Style-Transfer
Transferring style from one image to another. For some reason the ipynb doesnt render in github. You can download it to run it. 
<Figure size 1440x720 with 2 Axes><img width="1156" height="289" alt="image" src="https://github.com/user-attachments/assets/a64c52a0-20ec-428f-a5d7-2b29ea621383" />

## Streamlit app

`app.py` wraps the notebook in a small web app: upload a photo and a style image (or try the built-in example), watch a live preview as it optimizes, and download the result. It's set up to run on Streamlit Community Cloud, so nothing needs to be installed to use it.

It uses the same method as the notebook (Gatys et al., 2016). A frozen, pretrained VGG19 extracts features; content is matched at `conv4_2` and style with Gram matrices at `conv1_1` to `conv5_1`. Adam then optimizes the image's pixels directly. The app uses fewer steps and a higher learning rate than the notebook so a run finishes in minutes.

### Run it

```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install streamlit numpy pillow
streamlit run app.py
```

The first run downloads the VGG19 weights once and keeps only the convolutional layers it uses (~52 MB), so it stays light on memory. `examples/` has a photo and a style image (Hokusai's *The Great Wave off Kanagawa*) to try.

### Deploy on Streamlit Community Cloud

At [share.streamlit.io](https://share.streamlit.io), create an app from this repo with `main` as the branch and `app.py` as the main file. `requirements.txt` installs the CPU-only build of PyTorch, which is all the free tier needs.

### Settings

- **Output size**: longest side in pixels. Larger is sharper but slower.
- **Steps**: optimisation steps. 150–300 is usually enough.
- **Style strength**: how strongly the style overrides the photo's detail.

On a CPU the defaults (256 px, 200 steps) take about 2–4 minutes. With a CUDA build of PyTorch it uses the GPU automatically.

### Files

- `Style_Transfer.ipynb`: the original notebook
- `style_transfer.py`: the model and optimisation loop from the notebook, as functions (with a memory-lean model loader)
- `app.py`: the Streamlit interface
- `.streamlit/config.toml`: upload size limit and settings
