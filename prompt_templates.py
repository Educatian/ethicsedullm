"""
Prompt templates for AI Ethics Education LLM
"""

SYSTEM_PROMPT = """You are an AI Ethics Education Assistant, designed to help students, professionals, and researchers understand ethical considerations in AI development and deployment.

Your role is to:
- Provide clear, accurate explanations of AI ethics concepts
- Discuss ethical frameworks and principles (fairness, transparency, accountability, privacy)
- Analyze ethical dilemmas with balanced perspectives
- Reference relevant guidelines and regulations (EU AI Act, IEEE standards, etc.)
- Encourage critical thinking about AI's societal impact

Guidelines:
- Be objective and acknowledge multiple perspectives
- Use concrete examples when helpful
- Cite established frameworks and research when applicable
- Avoid prescriptive judgments; focus on helping users understand tradeoffs
- Acknowledge uncertainty when appropriate
"""

def format_instruction_prompt(question, include_system=True):
    """
    Format a question into the instruction-following format

    Args:
        question (str): The user's question
        include_system (bool): Whether to include system prompt

    Returns:
        str: Formatted prompt
    """
    if include_system:
        return f"{SYSTEM_PROMPT}\n\nQuestion: {question}\nAnswer:"
    else:
        return f"Question: {question}\nAnswer:"

def format_training_example(question, answer):
    """
    Format a training example with instruction and response

    Args:
        question (str): The question/instruction
        answer (str): The expected response

    Returns:
        dict: Formatted training example
    """
    return {
        "instruction": question,
        "response": answer,
        "prompt": format_instruction_prompt(question, include_system=True)
    }

# Llama 3.1 Instruct format
def format_llama_prompt(question, include_system=True):
    """
    Format prompt for Llama 3.1 Instruct model
    Uses the proper chat template format

    Args:
        question (str): The user's question
        include_system (bool): Whether to include system prompt

    Returns:
        str: Formatted prompt for Llama
    """
    if include_system:
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    else:
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

def format_llama_training_example(question, answer):
    """
    Format a complete training example for Llama 3.1 Instruct

    Args:
        question (str): The question
        answer (str): The response

    Returns:
        str: Complete formatted conversation
    """
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{answer}<|eot_id|>"""
