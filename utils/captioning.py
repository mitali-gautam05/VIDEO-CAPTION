import torch

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)


BLIP_MODEL = "Salesforce/blip-image-captioning-base"


device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained(BLIP_MODEL)

model = BlipForConditionalGeneration.from_pretrained(
    BLIP_MODEL
).to(device)

model.eval()


def generate_captions(frames):

    captions = []

    for img in frames:

        inputs = processor(
            images=img,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():

            output = model.generate(
                **inputs,
                max_length=30
            )

        caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        captions.append(caption)

    return captions