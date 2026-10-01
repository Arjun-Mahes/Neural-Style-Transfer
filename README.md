# Neural-Style-Transfer
Transferring style from one image to another. For some reason the ipynb doesnt render in github. You can download it to run it. 
<Figure size 1440x720 with 2 Axes><img width="1156" height="289" alt="image" src="https://github.com/user-attachments/assets/a64c52a0-20ec-428f-a5d7-2b29ea621383" />

## Streamlit app

`app.py` wraps the notebook in a small web app: upload a photo and a style image, watch a live preview as it optimizes, and download the result.

It uses the same method as the notebook (Gatys et al., 2016). A frozen, pretrained VGG19 extracts features; content is matched at `conv4_2` and style with Gram matrices at `conv1_1` to `conv5_1`. Adam then optimizes the image's pixels directly. The app uses fewer steps and a higher learning rate than the notebook so a run finishes in minutes.

### Run it

```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install streamlit numpy pillow
streamlit run app.py
```

The first run downloads the VGG19 weights (~550 MB). `examples/` has a photo and a style image (Hokusai's *The Great Wave off Kanagawa*) to try.

### Settings

- **Output size**: longest side in pixels. Larger is sharper but slower.
- **Steps**: optimisation steps. 200–400 is usually enough.
- **Style strength**: how strongly the style overrides the photo's detail.

On a CPU the defaults (256 px, 250 steps) take about 3 minutes. With a CUDA build of PyTorch it uses the GPU automatically.

### Files

- `Style_Transfer.ipynb`: the original notebook
- `style_transfer.py`: the model and optimisation loop from the notebook, as functions
- `app.py`: the Streamlit interface
