import os
import gradio as gr

from utils.pipeline import process_video

os.makedirs("outputs", exist_ok=True)


def run_pipeline(video):

    output_video, caption = process_video(video)

    return output_video, caption


with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Silent Video Understanding AI"
) as demo:

    gr.Markdown(
        """
        # Silent Video Understanding AI
        
        Upload a silent video and let AI:
        
        - Understand the scene
        - Detect objects
        - Analyze motion
        - Generate captions
        - Render captioned video
        """
    )

    with gr.Row():

        with gr.Column():

            input_video = gr.Video(
                label="Upload Silent Video"
            )

            submit_btn = gr.Button(
                "Generate AI Captioned Video",
                variant="primary"
            )

        with gr.Column():

            output_video = gr.Video(
                label="AI Captioned Output"
            )

            caption_output = gr.Textbox(
                label="AI Understanding",
                lines=6
            )

    submit_btn.click(
        fn=run_pipeline,
        inputs=input_video,
        outputs=[
            output_video,
            caption_output
        ]
    )

demo.launch()