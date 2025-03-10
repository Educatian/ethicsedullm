import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_test_cases(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def evaluate_model(model, tokenizer, test_cases):
    results = []
    
    for case in test_cases:
        prompt = f"Question: {case['question']}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                inputs.input_ids, 
                max_length=512, 
                temperature=0.7,
                top_p=0.9
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        # Compare with expected answer themes
        score = assess_response(response, case['expected_themes'])
        
        results.append({
            "question": case['question'],
            "model_response": response,
            "expected_themes": case['expected_themes'],
            "score": score
        })
    
    return results

def assess_response(response, expected_themes):
    """
    Assess the model's response based on expected themes
    Returns a score between 0 and 1
    """
    score = 0
    for theme in expected_themes:
        if theme.lower() in response.lower():
            score += 1
    
    # Normalize score to be between 0 and 1
    if expected_themes:
        score = score / len(expected_themes)
    
    return score

# Load model
model_path = "./ai_ethics_llm_final"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Run evaluation
test_cases = load_test_cases("ethics_test_cases.json")
results = evaluate_model(model, tokenizer, test_cases)

# Output results
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2) 