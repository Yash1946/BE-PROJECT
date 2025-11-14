import random
import numpy as np
from PIL import Image
import base64
from io import BytesIO

import torch
import torchvision.transforms.functional as F
import gradio as gr

from src.pix2pix_turbo import Pix2Pix_Turbo

# Load the model
model = Pix2Pix_Turbo("sketch_to_image_stochastic")
print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

# Styles list
style_list = [
    {"name": "Cinematic", "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy"},
    {"name": "3D Model", "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting"},
    {"name": "Anime", "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed"},
    {"name": "Digital Art", "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed"},
    {"name": "Photographic", "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed"},
    {"name": "Pixel art", "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics"},
    {"name": "Fantasy art", "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy"},
    {"name": "Neonpunk", "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional"},
    {"name": "Manga", "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style"},
]

styles = {k["name"]: k["prompt"] for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Fantasy art"
MAX_SEED = np.iinfo(np.int32).max

# Convert PIL image to base64 URI
def pil_image_to_data_uri(img, format="PNG"):
    buffered = BytesIO()
    img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"

# Main run function
def run(image, prompt, prompt_template, style_name, seed, val_r):
    if image is None:
        ones = Image.new("L", (512, 512), 255)
        temp_uri = pil_image_to_data_uri(ones)
        return ones, gr.update(link=temp_uri), gr.update(link=temp_uri)
    prompt = prompt_template.replace("{prompt}", prompt)
    image = image.convert("RGB")
    image_t = F.to_tensor(image) > 0.5
    with torch.no_grad():
        c_t = image_t.unsqueeze(0).cuda().float()
        torch.manual_seed(seed)
        B, C, H, W = c_t.shape
        noise = torch.randn((1, 4, H // 8, W // 8), device=c_t.device)
        output_image = model(c_t, prompt, deterministic=False, r=val_r, noise_map=noise)
    output_pil = F.to_pil_image(output_image[0].cpu() * 0.5 + 0.5)
    input_sketch_uri = pil_image_to_data_uri(Image.fromarray(255 - np.array(image)))
    output_image_uri = pil_image_to_data_uri(output_pil)
    return output_pil, gr.update(link=input_sketch_uri), gr.update(link=output_image_uri)

# Update canvas brush
def update_canvas(use_line, use_eraser):
    if use_eraser:
        _color = "#ffffff"
        brush_size = 20
    if use_line:
        _color = "#000000"
        brush_size = 4
    return gr.update(brush_radius=brush_size, brush_color=_color, interactive=True)

# Upload sketch
def upload_sketch(file):
    _img = Image.open(file.name).convert("L")
    return gr.update(value=_img, source="upload", interactive=True)

# JavaScript helper scripts
scripts = """..."""  # Keep your existing JS as-is

# Gradio UI
with gr.Blocks(css="style.css") as demo:

    gr.HTML(
        """
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
            <div>
                <h2><a href="https://github.com/GaParmar/img2img-turbo">One-Step Image Translation with Text-to-Image Models</a></h2>
                <div>By Rudresh, Anurag, Sushant, Jayraj</div>
            </div>
        </div>
        """
    )

    line = gr.Checkbox(label="line", value=False, elem_id="cb-line")
    eraser = gr.Checkbox(label="eraser", value=False, elem_id="cb-eraser")

    with gr.Row(elem_id="main_row"):
        with gr.Column(elem_id="column_input"):
            gr.Markdown("## INPUT", elem_id="input_header")
            image = gr.Image(
                source="canvas",
                tool="color-sketch",
                type="pil",
                image_mode="L",
                invert_colors=True,
                shape=(512, 512),
                brush_radius=4,
                height=440,
                width=440,
                brush_color="#000000",
                interactive=True,
                show_download_button=True,
                elem_id="input_image",
                show_label=False,
            )
            download_sketch = gr.Button("Download sketch", scale=1, elem_id="download_sketch")

            prompt = gr.Textbox(label="Prompt", value="", show_label=True)
            style = gr.Dropdown(label="Style", choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME)
            prompt_temp = gr.Textbox(label="Prompt Style Template", value=styles[DEFAULT_STYLE_NAME], max_lines=1)
            val_r = gr.Slider(label="Sketch guidance: ", minimum=0, maximum=1, value=0.4, step=0.01)
            seed = gr.Textbox(label="Seed", value=42)
            randomize_seed = gr.Button("Random")

        with gr.Column(elem_id="column_process"):
            run_button = gr.Button("Run")

        with gr.Column(elem_id="column_output"):
            gr.Markdown("## OUTPUT")
            result = gr.Image(height=440, width=440, show_label=False, show_download_button=True)
            download_output = gr.Button("Download output")

    # Bind events
    eraser.change(lambda x: gr.update(value=not x), inputs=[eraser], outputs=[line]).then(update_canvas, [line, eraser], [image])
    line.change(lambda x: gr.update(value=not x), inputs=[line], outputs=[eraser]).then(update_canvas, [line, eraser], [image])
    randomize_seed.click(lambda: random.randint(0, MAX_SEED), inputs=[], outputs=seed)
    inputs = [image, prompt, prompt_temp, style, seed, val_r]
    outputs = [result, download_sketch, download_output]
    prompt.submit(fn=run, inputs=inputs, outputs=outputs)
    style.change(lambda x: styles[x], inputs=[style], outputs=[prompt_temp]).then(fn=run, inputs=inputs, outputs=outputs)
    val_r.change(run, inputs=inputs, outputs=outputs)
    run_button.click(fn=run, inputs=inputs, outputs=outputs)
    image.change(run, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.queue().launch(debug=True, share=True, show=True)
