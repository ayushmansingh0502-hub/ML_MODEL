"""
Hugging Face Space - Waste Classification Demo
Fast API-only inference, no local model loading.
"""

import gradio as gr
import os
import requests
import io
from PIL import Image

# Configuration
MODEL_ID = "Ayushman0502/waste-classifier"
HF_TOKEN = os.getenv("HF_TOKEN", "")
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

# Class labels
CLASS_DISPLAY = {
    'dry_waste': '♻️ Dry Waste',
    'other_waste': '🔶 Other Waste',
    'wet_waste': '🍃 Wet Waste',
}
CLASS_ACTIONS = {
    'dry_waste': 'Can be recycled → Paper, plastic, metal recovery',
    'other_waste': 'Needs special handling → Hazardous / e-waste processing',
    'wet_waste': 'Compostable → Organic composting / biogas generation',
}

print("✅ Waste Classifier Space Ready")


def predict_waste(image):
    """Classify waste via Hugging Face Inference API."""
    try:
        # Convert image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Call API
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        response = requests.post(
            API_URL,
            headers=headers,
            data=img_bytes.getvalue(),
            timeout=20
        )
        
        if response.status_code != 200:
            msg = response.text if response.text else f"Status {response.status_code}"
            if "loading" in msg.lower():
                return "⏳ Model loading (first use). Retry in 30s.", None
            return f"❌ Error: {msg[:100]}", None
        
        # Parse response
        preds = response.json()
        if not isinstance(preds, list):
            return f"❌ Unexpected response format", None
        
        # Extract best prediction
        best = max(preds, key=lambda x: x.get('score', 0)) if preds else None
        if not best:
            return "❌ No predictions", None
        
        class_name = best['label'].lower()
        score = best['score'] * 100
        
        # Format output
        text = f"**{CLASS_DISPLAY.get(class_name, class_name)}**\n"
        text += f"Confidence: **{score:.1f}%**\n"
        text += f"Action: {CLASS_ACTIONS.get(class_name, 'N/A')}\n\n"
        text += "**All Scores:**\n"
        
        results = {}
        for p in preds:
            label = p['label'].lower()
            perc = p['score'] * 100
            results[CLASS_DISPLAY.get(label, label)] = round(perc, 1)
            text += f"{CLASS_DISPLAY.get(label, label)}: {perc:.1f}%\n"
        
        return text, results
        
    except requests.exceptions.Timeout:
        return "⏳ API timeout. Retry in 30s.", None
    except Exception as e:
        return f"❌ Error: {str(e)[:50]}", None


# Create custom HTML for better styling
HEADER_HTML = """
<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="color: #2c3e50; margin-bottom: 10px;">🌿 Waste Classification AI</h1>
    <p style="color: #7f8c8d; font-size: 16px;">
        Upload an image of waste to classify it as Dry, Wet, or Other waste
    </p>
    <p style="color: #95a5a6; font-size: 14px;">
        Powered by EfficientNetB0 • Supporting Circular Economy
    </p>
</div>
"""

FOOTER_HTML = """
<div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ecf0f1;">
    <p style="color: #7f8c8d; font-size: 13px;">
        🔗 <a href="https://huggingface.co/Ayushman0502/waste-classifier" target="_blank">View Model Card</a> • 
        <a href="https://github.com/ayushmansingh0502-hub/ML_MODEL" target="_blank">GitHub Repository</a>
    </p>
    <p style="color: #95a5a6; font-size: 12px;">
        For questions or feedback, please open an issue on GitHub
    </p>
</div>
"""

# Create Gradio interface with custom CSS
with gr.Blocks(
    title="Waste Classifier",
    theme=gr.themes.Soft(),
    css="""
    .output-image {
        max-width: 400px;
        margin-left: auto;
        margin-right: auto;
    }
    """
) as demo:
    # Header
    gr.HTML(HEADER_HTML)
    
    # Main content
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Step 1: Upload Image")
            image_input = gr.Image(
                label="Upload or Drag & Drop a Waste Image",
                type="pil",
                sources=["upload", "webcam"],
            )
            
            with gr.Row():
                clear_btn = gr.ClearButton(image_input, value="Clear Image")
                submit_btn = gr.Button("🔍 Classify Waste", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Step 2: View Results")
            output_text = gr.Textbox(
                label="Classification Result",
                lines=12,
                interactive=False,
                show_label=True,
            )
    
    # Chart for predictions
    gr.Markdown("### 📈 Confidence Distribution")
    chart = gr.BarChart(
        label="Confidence Scores",
        x="Waste Type",
        y="Confidence (%)",
        show_label=True,
        every=0.5,
    )
    
    # Example images section
    gr.Markdown("---\n### 💡 Tips for Best Results\n"
                "- Use clear, well-lit images\n"
                "- Avoid blurry or partially visible items\n"
                "- Single waste items work best\n"
                "- Try different angles if unsure")
    
    # Example gallery
    examples_html = """
    <div style="padding: 20px; background: #f8f9fa; border-radius: 10px; margin: 20px 0;">
        <h4 style="margin-top: 0;">📸 Example Categories</h4>
        <p><strong>♻️ Dry Waste:</strong> Paper, cardboard, plastic bottles, cans, books, packaging</p>
        <p><strong>🍃 Wet Waste:</strong> Food scraps, fruit peels, leaves, grass, cooked food</p>
        <p><strong>🔶 Other Waste:</strong> Electronics, batteries, broken glass, chemicals</p>
    </div>
    """
    gr.HTML(examples_html)
    
    # Footer
    gr.HTML(FOOTER_HTML)
    
    # Connect button to prediction
    def on_submit(image):
        if image is None:
            return "Upload an image first.", None
        
        text, results = predict_waste(image)
        
        # Format chart data
        chart_data = [{"Waste Type": k.split()[-1], "Confidence (%)": v} for k, v in (results.items() if results else [])]
        
        return text, chart_data if chart_data else None
    
    submit_btn.click(
        on_submit,
        inputs=[image_input],
        outputs=[output_text, chart]
    )


if __name__ == "__main__":
    demo.launch()
