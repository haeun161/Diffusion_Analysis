import json
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

PROMPTS = ["airplane", "bird", "car", "clock",
           "couch", "elephant", "train", "umbrella"]
SEEDS = list(range(10))

IMAGE_ROOT = Path("/home/haeun/Diffusion_Analysis/experiment/DIMSIM/output/multi_concept")
OUTPUT_PATH = Path(__file__).parent / "blip_captions.json"
MODEL_NAME = "Salesforce/blip-image-captioning-large"


def load_blip(device: torch.device):
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def caption_image(processor, model, device, image_path: Path) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=50, num_beams=5)
    return processor.decode(out[0], skip_special_tokens=True).strip()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor, model = load_blip(device)

    results: dict[str, dict[str, str]] = {}
    for prompt in PROMPTS:
        results[prompt] = {}
        for seed in SEEDS:
            image_path = IMAGE_ROOT / prompt / "images" / f"{prompt}_seed{seed}.png"
            if not image_path.exists():
                print(f"[skip] missing: {image_path}")
                continue
            caption = caption_image(processor, model, device, image_path)
            results[prompt][str(seed)] = caption
            print(f"{prompt} seed{seed}: {caption}")

    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved captions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
