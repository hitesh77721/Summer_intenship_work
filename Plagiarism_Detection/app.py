import gradio as gr
import pandas as pd

from plagiarism import compare_uploaded_document


def plagiarism_checker(file):

    # Read uploaded file
    with open(file.name, "r", encoding="utf-8") as f:
        text = f.read()

    # Get results from backend
    results = compare_uploaded_document(text)

    # Convert list of dictionaries into DataFrame
    return pd.DataFrame(results)


demo = gr.Interface(
    fn=plagiarism_checker,
    inputs=gr.File(label="Upload a .txt file"),
    outputs=gr.Dataframe(
        headers=["Document", "Similarity (%)"],
        label="Similarity Results"
    ),
    title="📄 Plagiarism Detection System",
    description="Upload a text document to compare it with the dataset."
)

demo.launch()