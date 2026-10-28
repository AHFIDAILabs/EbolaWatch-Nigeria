import spaces
import gradio as gr
import uvicorn
import threading
from src.api.main import app as fastapi_app

@spaces.GPU
def _gpu_placeholder():
    """Required by HuggingFace ZeroGPU runtime."""
    pass

def run_fastapi():
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )

thread = threading.Thread(target=run_fastapi, daemon=True)
thread.start()

with gr.Blocks() as demo:
    gr.Markdown("# EbolaWatch Nigeria ML Service")
    gr.Markdown("Ebola misinformation surveillance API")

demo.launch(server_name="0.0.0.0", server_port=7861)
