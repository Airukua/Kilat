import sys
from pathlib import Path
import torch
import gradio as gr

# Fix path (biar import jalan di HF Space)

project_root = Path(**file**).parent
sys.path.insert(0, str(project_root))

from arc.model import KilatTransformer
from data.tokenizer import AutoTokenizer
from pipeline.generation.generator import TextGenerator

# Load sekali aja (jangan di dalam function)

model_path = "AiRukua/BabyKilat"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = KilatTransformer.from_pretrained(model_path)
model = model.to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)

generator = TextGenerator(model, tokenizer, device=device)

# Function untuk Gradio

def generate_text(prompt, max_tokens, temperature):
if not prompt.strip():
return "Please enter a prompt."

```
text = generator.generate(
    prompt,
    max_new_tokens=int(max_tokens),
    do_sample=True,
    temperature=float(temperature),
    top_p=0.95,
)

return text
```

# UI

demo = gr.Interface(
fn=generate_text,
inputs=[
gr.Textbox(lines=4, placeholder="Type your prompt here..."),
gr.Slider(10, 200, value=50, label="Max Tokens"),
gr.Slider(0.1, 1.5, value=0.8, label="Temperature"),
],
outputs=gr.Textbox(label="Generated Text"),
title="⚡ BabyKilat",
description="Lightweight MoE Language Model built from scratch",
)

demo.launch()
