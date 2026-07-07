import os
import gradio as gr
import pandas as pd

from plagiarism import compare_documents

# ==========================================================
# THEME
# ==========================================================

theme = gr.themes.Soft()

# ==========================================================
# BACKEND
# ==========================================================

def plagiarism_checker(original_file, suspected_file):

    if original_file is None:
        raise gr.Error("Please upload the original document.")

    if suspected_file is None:
        raise gr.Error("Please upload the suspected document.")

    # Read original file
    with open(original_file.name, "r", encoding="utf-8") as f:
        original_text = f.read()

    # Read suspected file
    with open(suspected_file.name, "r", encoding="utf-8") as f:
        suspected_text = f.read()

    # Calculate similarity
    similarity = compare_documents(
        original_text,
        suspected_text
    )

    # Determine plagiarism level
    if similarity >= 80:
        status = "🔴 High Similarity"
    elif similarity >= 50:
        status = "🟠 Moderate Similarity"
    elif similarity >= 20:
        status = "🟡 Low Similarity"
    else:
        status = "🟢 Very Low Similarity"

    result = pd.DataFrame(
        {
            "Original Document": [os.path.basename(original_file.name)],
            "Suspected Document": [os.path.basename(suspected_file.name)],
            "Similarity (%)": [similarity],
            "Status": [status],
        }
    )

    return result


# ==========================================================
# CUSTOM CSS
# ==========================================================

css = """
.gradio-container{
    max-width:1000px !important;
    margin:auto;
}

h1{
    text-align:center;
}

footer{
    visibility:hidden;
}
"""


# ==========================================================
# UI
# ==========================================================

with gr.Blocks(theme=theme, css=css) as demo:

    gr.Markdown(
        """
# 📄 Plagiarism Detection System

Compare two text documents using **TF-IDF** and **Cosine Similarity**.
"""
    )

    with gr.Row():

        original_file = gr.File(
            label="📄 Original Document",
            file_types=[".txt"],
            type="filepath"
        )

        suspected_file = gr.File(
            label="📑 Suspected Document",
            file_types=[".txt"],
            type="filepath"
        )

    check_btn = gr.Button(
        "🔍 Check Similarity",
        variant="primary",
        size="lg"
    )

    output = gr.Dataframe(
        headers=[
            "Original Document",
            "Suspected Document",
            "Similarity (%)",
            "Status"
        ],
        interactive=False,
        label="Comparison Result"
    )

    check_btn.click(
        fn=plagiarism_checker,
        inputs=[
            original_file,
            suspected_file
        ],
        outputs=output
    )

    gr.Markdown(
        """
### 📌 Similarity Guide

| Similarity | Interpretation |
|------------|----------------|
| 🟢 0–19% | Very Low Similarity |
| 🟡 20–49% | Low Similarity |
| 🟠 50–79% | Moderate Similarity |
| 🔴 80–100% | High Similarity |
"""
    )

# ==========================================================
# Launch
# ==========================================================

demo.launch()