import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from prompt_templates import format_llama_prompt, SYSTEM_PROMPT
import os

# Configuration
USE_QLORA = True  # Set to False if using merged model
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
model_path = "./ai_ethics_llm_final"

# Check if merged model exists
if os.path.exists("./ai_ethics_llm_merged"):
    print("Found merged model, loading...")
    model_path = "./ai_ethics_llm_merged"
    USE_QLORA = False

print(f"Loading model from {model_path}")

if USE_QLORA and os.path.exists(model_path):
    # Load base model with quantization
    print("Loading with QLoRA configuration...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
else:
    # Load standard model (merged or original)
    print("Loading standard model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

def generate_response(question, temperature=0.7, max_tokens=512):
    """
    Generate a response to an AI ethics question

    Args:
        question (str): User's question
        temperature (float): Sampling temperature (0.1-1.0)
        max_tokens (int): Maximum response length

    Returns:
        str: Generated response
    """
    # Use Llama 3.1 Instruct format
    prompt = format_llama_prompt(question, include_system=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate with proper stopping criteria for Llama
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode and extract just the assistant's response
    full_response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove the prompt to get just the answer
    # For Llama 3.1, extract text after the assistant header
    if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
        response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    else:
        response = full_response.replace(prompt, "")

    return response.strip()

# Create enhanced Gradio interface with controls
with gr.Blocks(title="AI Ethics Education Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 AI Ethics Education Assistant")
    gr.Markdown(
        "Ask questions about AI ethics principles, frameworks (EU AI Act, OECD, UNESCO, Partnership on AI), "
        "fairness, transparency, accountability, and responsible AI development."
    )

    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                lines=4,
                placeholder="Example: What are the key principles of the EU AI Act?\n\n"
                            "Try asking about: algorithmic bias, fairness metrics, transparency, "
                            "privacy, accountability, or specific use cases...",
                label="Your Question"
            )

            with gr.Row():
                submit_btn = gr.Button("Ask", variant="primary", scale=2)
                clear_btn = gr.Button("Clear", scale=1)

            with gr.Accordion("Advanced Settings", open=False):
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature (creativity)",
                    info="Lower = more focused, Higher = more creative"
                )
                max_tokens = gr.Slider(
                    minimum=128,
                    maximum=1024,
                    value=512,
                    step=128,
                    label="Max Response Length",
                    info="Maximum number of tokens in response"
                )

        with gr.Column(scale=2):
            response_output = gr.Textbox(
                lines=15,
                label="Response",
                show_copy_button=True
            )

    # Example questions
    gr.Examples(
        examples=[
            ["What is algorithmic bias and how can it be mitigated?"],
            ["Explain the key principles of the EU AI Act"],
            ["What are the different definitions of fairness in machine learning?"],
            ["How do ethical frameworks address AI's impact on employment?"],
            ["What is differential privacy and why is it important?"],
            ["What ethical considerations arise with facial recognition technology?"],
            ["How can organizations implement ethical AI governance?"],
            ["What is a Model Card and why is it important for AI ethics?"]
        ],
        inputs=question_input,
        label="Example Questions"
    )

    # Footer
    gr.Markdown(
        "---\n"
        "**Note:** This assistant covers major ethical frameworks including EU AI Act, "
        "OECD AI Principles, Partnership on AI, and UNESCO Guidelines. "
        "Responses are generated by a fine-tuned language model and should be "
        "used as educational guidance, not legal or professional advice."
    )

    # Event handlers
    submit_btn.click(
        fn=generate_response,
        inputs=[question_input, temperature, max_tokens],
        outputs=response_output
    )

    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=None,
        outputs=[question_input, response_output]
    )

    question_input.submit(
        fn=generate_response,
        inputs=[question_input, temperature, max_tokens],
        outputs=response_output
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False) 