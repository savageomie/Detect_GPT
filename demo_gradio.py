import os
import gradio as gr
import torch
from transformers import pipeline


def find_candidate_models(root_dir="."):
    """Return list of directories that look like model folders (contain model files or config)."""
    candidates = []
    for entry in os.listdir(root_dir):
        path = os.path.join(root_dir, entry)
        if not os.path.isdir(path):
            continue
        # Heuristics: contains a model file or config/tokenizer
        has_model_file = any(
            os.path.exists(os.path.join(path, name))
            for name in ("pytorch_model.bin", "model.safetensors", "model.pt", "tf_model.h5")
        )
        has_config = os.path.exists(os.path.join(path, "config.json"))
        has_tokenizer = any(
            os.path.exists(os.path.join(path, t))
            for t in ("tokenizer.json", "tokenizer_config.json", "vocab.json")
        )
        if has_model_file or (has_config and has_tokenizer):
            candidates.append(path)
    # Also include common folders inside results/
    results_dir = os.path.join(root_dir, "results")
    if os.path.isdir(results_dir):
        for entry in os.listdir(results_dir):
            path = os.path.join(results_dir, entry)
            if os.path.isdir(path):
                candidates.append(path)

    # Deduplicate and sort
    candidates = sorted(list(dict.fromkeys(candidates)))
    return candidates


CLASSIFIER = None
CURRENT_MODEL = None

# Default friendly label mapping. You can customize in the UI.
LABEL_MAP = {
    "LABEL_0": "Human-written",
    "LABEL_1": "AI-generated",
}


def load_model(model_path):
    global CLASSIFIER, CURRENT_MODEL
    device = 0 if torch.cuda.is_available() else -1
    try:
        CLASSIFIER = pipeline("text-classification", model=model_path, tokenizer=model_path, device=device)
        CURRENT_MODEL = model_path
        return f"Loaded model: {model_path} (device={'cuda' if device==0 else 'cpu'})"
    except Exception as e:
        CLASSIFIER = None
        CURRENT_MODEL = None
        return f"Failed to load model from {model_path}: {e}\n\nHint: the folder may not be a classification model. You can still use the DetectGPT notebook for scoring."


def predict(text):
    if CLASSIFIER is None:
        return "Model not loaded", ""
    try:
        out = CLASSIFIER(text)
        # Pipeline returns list of dicts
        if isinstance(out, list) and len(out) > 0:
            label = out[0].get("label", "")
            score = out[0].get("score", 0.0)
            # Map to friendly name if available
            friendly = LABEL_MAP.get(label, label)
            return friendly, f"confidence: {score:.4f}"
        return str(out), ""
    except Exception as e:
        return f"Error during prediction: {e}", ""


def build_demo():
    model_choices = find_candidate_models()

    with gr.Blocks(title="Model demo") as demo:
        gr.Markdown("# Model demo\nPick a local model folder and run a text classification demo. This demo loads models saved with the Hugging Face Transformers library.")

        with gr.Row():
            model_dropdown = gr.Dropdown(choices=model_choices, label="Detected model folders", interactive=True)
            load_btn = gr.Button("Load model")

        status = gr.Textbox(label="Status", interactive=False)

        # Optional mapping input so users can override LABEL_0/LABEL_1 names
        mapping_input = gr.Textbox(value=",")
        mapping_input.visible = False

        text_in = gr.Textbox(lines=6, placeholder="Type text to classify...", label="Input text")
        predict_btn = gr.Button("Predict")

        label_out = gr.Textbox(label="Predicted label", interactive=False)
        score_out = gr.Textbox(label="Score / confidence", interactive=False)

        def on_load(path):
            if not path:
                return "No model selected"
            return load_model(path)

        load_btn.click(on_load, inputs=model_dropdown, outputs=status)
        predict_btn.click(predict, inputs=text_in, outputs=[label_out, score_out])

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=False)
