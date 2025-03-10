import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model
model_path = "./ai_ethics_llm_final"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def generate_response(question):
    prompt = f"Question: {question}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()

# Create Gradio interface
demo = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=3, placeholder="Ask a question about AI ethics..."),
    outputs=gr.Textbox(label="AI Ethics Assistant Response"),
    title="AI Ethics Education Assistant",
    description="Ask questions about AI ethics, responsible AI development, or ethical dilemmas in AI."
)

# Launch the app
if __name__ == "__main__":
    demo.launch() 