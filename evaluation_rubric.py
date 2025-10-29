"""
Enhanced evaluation system for AI Ethics LLM using keyword + rubric hybrid approach
Combines automated keyword matching with structured rubric scoring
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from prompt_templates import format_llama_prompt
import re
from typing import Dict, List, Tuple
import os

# Rubric for evaluating AI ethics responses
EVALUATION_RUBRIC = {
    "accuracy": {
        "description": "Factual correctness and alignment with established frameworks",
        "weight": 0.30,
        "criteria": [
            "Contains accurate information about ethical frameworks",
            "Correctly defines technical concepts",
            "References appropriate standards (EU AI Act, OECD, etc.)",
            "No factual errors or misrepresentations"
        ]
    },
    "completeness": {
        "description": "Comprehensive coverage of the topic",
        "weight": 0.25,
        "criteria": [
            "Addresses all aspects of the question",
            "Includes relevant examples or context",
            "Discusses multiple perspectives when appropriate",
            "Provides sufficient depth"
        ]
    },
    "clarity": {
        "description": "Clear communication and understandability",
        "weight": 0.20,
        "criteria": [
            "Well-structured and organized",
            "Uses clear, accessible language",
            "Explains technical terms",
            "Logically flows from point to point"
        ]
    },
    "ethical_awareness": {
        "description": "Demonstrates ethical reasoning and awareness",
        "weight": 0.15,
        "criteria": [
            "Acknowledges ethical trade-offs and tensions",
            "Shows balanced perspective",
            "Recognizes stakeholder impacts",
            "Avoids prescriptive judgments where appropriate"
        ]
    },
    "actionability": {
        "description": "Provides practical, actionable guidance",
        "weight": 0.10,
        "criteria": [
            "Offers concrete recommendations when appropriate",
            "Explains how principles apply in practice",
            "Mentions specific tools or techniques",
            "Helps reader understand next steps"
        ]
    }
}

def load_model_for_evaluation():
    """Load the fine-tuned model for evaluation"""
    BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    model_path = "./ai_ethics_llm_final"

    if os.path.exists("./ai_ethics_llm_merged"):
        model_path = "./ai_ethics_llm_merged"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif os.path.exists(model_path):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")

    return model, tokenizer

def generate_response(model, tokenizer, question: str) -> str:
    """Generate a response to a question"""
    prompt = format_llama_prompt(question, include_system=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_response = tokenizer.decode(output[0], skip_special_tokens=True)

    if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
        response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    else:
        response = full_response.replace(prompt, "")

    return response.strip()

def keyword_score(response: str, expected_keywords: List[str]) -> Tuple[float, List[str]]:
    """
    Score response based on presence of expected keywords/themes

    Args:
        response: Model's response
        expected_keywords: List of expected keywords or phrases

    Returns:
        Tuple of (score, found_keywords)
    """
    response_lower = response.lower()
    found = []

    for keyword in expected_keywords:
        # Use fuzzy matching for multi-word phrases
        keyword_lower = keyword.lower()

        # Check for exact phrase or individual words from phrase
        if keyword_lower in response_lower:
            found.append(keyword)
        else:
            # Check if most words from the keyword phrase appear
            words = keyword_lower.split()
            if len(words) > 1:
                matches = sum(1 for word in words if word in response_lower)
                if matches >= len(words) * 0.6:  # At least 60% of words present
                    found.append(keyword)

    score = len(found) / len(expected_keywords) if expected_keywords else 0.0
    return score, found

def rubric_score(response: str, question: str) -> Dict[str, float]:
    """
    Score response using rubric criteria (simplified automated version)

    Args:
        response: Model's response
        question: Original question

    Returns:
        Dictionary with scores for each rubric dimension
    """
    scores = {}

    # Accuracy: Check for framework mentions and technical terms
    frameworks = ["eu ai act", "oecd", "unesco", "partnership on ai", "gdpr"]
    framework_mentions = sum(1 for f in frameworks if f in response.lower())
    scores["accuracy"] = min(1.0, framework_mentions / 2.0 + 0.3)  # Base score + bonuses

    # Completeness: Check response length and structure
    word_count = len(response.split())
    has_examples = any(word in response.lower() for word in ["example", "such as", "for instance", "including"])
    has_multiple_points = response.count('.') > 2 or response.count(',') > 3

    completeness = 0.4
    if word_count > 100:
        completeness += 0.2
    if has_examples:
        completeness += 0.2
    if has_multiple_points:
        completeness += 0.2
    scores["completeness"] = min(1.0, completeness)

    # Clarity: Check for structure and readability
    has_structure = bool(re.search(r'\d+\)|firstly|secondly|additionally|furthermore|however', response.lower()))
    avg_sentence_length = word_count / max(1, response.count('.'))
    readable = 10 < avg_sentence_length < 30

    clarity = 0.5
    if has_structure:
        clarity += 0.25
    if readable:
        clarity += 0.25
    scores["clarity"] = clarity

    # Ethical awareness: Check for balanced language
    balance_indicators = [
        "however", "although", "balance", "trade-off", "consider", "depends",
        "context", "stakeholder", "perspective", "tension"
    ]
    balance_mentions = sum(1 for indicator in balance_indicators if indicator in response.lower())
    scores["ethical_awareness"] = min(1.0, balance_mentions / 3.0)

    # Actionability: Check for practical guidance
    action_words = [
        "should", "can", "implement", "use", "apply", "practice",
        "ensure", "establish", "conduct", "monitor"
    ]
    action_mentions = sum(1 for word in action_words if word in response.lower())
    scores["actionability"] = min(1.0, action_mentions / 5.0)

    return scores

def calculate_weighted_score(rubric_scores: Dict[str, float]) -> float:
    """Calculate weighted total score based on rubric"""
    total = 0.0
    for dimension, score in rubric_scores.items():
        weight = EVALUATION_RUBRIC[dimension]["weight"]
        total += score * weight
    return total

def evaluate_response(response: str, question: str, expected_keywords: List[str]) -> Dict:
    """
    Comprehensive evaluation combining keyword and rubric approaches

    Args:
        response: Model's response
        question: Original question
        expected_keywords: Expected keywords/themes

    Returns:
        Dictionary with detailed evaluation results
    """
    # Keyword-based score
    keyword_score_value, found_keywords = keyword_score(response, expected_keywords)

    # Rubric-based scores
    rubric_scores = rubric_score(response, question)
    rubric_weighted = calculate_weighted_score(rubric_scores)

    # Combined score (60% rubric, 40% keywords)
    combined_score = (rubric_weighted * 0.6) + (keyword_score_value * 0.4)

    return {
        "combined_score": combined_score,
        "keyword_score": keyword_score_value,
        "keywords_found": found_keywords,
        "keywords_missing": list(set(expected_keywords) - set(found_keywords)),
        "rubric_scores": rubric_scores,
        "rubric_weighted_score": rubric_weighted,
        "response_length": len(response.split()),
        "detailed_rubric": {
            dim: {
                "score": rubric_scores[dim],
                "weight": EVALUATION_RUBRIC[dim]["weight"],
                "weighted_contribution": rubric_scores[dim] * EVALUATION_RUBRIC[dim]["weight"]
            }
            for dim in rubric_scores
        }
    }

def load_test_cases(file_path: str = "ethics_test_cases.json") -> List[Dict]:
    """Load test cases from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def run_evaluation(model, tokenizer, test_cases: List[Dict]) -> List[Dict]:
    """Run evaluation on all test cases"""
    results = []

    print(f"\nEvaluating {len(test_cases)} test cases...\n")

    for i, case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] Evaluating: {case['question'][:60]}...")

        # Generate response
        response = generate_response(model, tokenizer, case['question'])

        # Evaluate response
        eval_result = evaluate_response(
            response,
            case['question'],
            case.get('expected_themes', [])
        )

        # Compile result
        result = {
            "question": case['question'],
            "response": response,
            "expected_themes": case.get('expected_themes', []),
            "evaluation": eval_result
        }

        results.append(result)

        # Print summary
        print(f"  Combined Score: {eval_result['combined_score']:.2f}")
        print(f"  Keyword: {eval_result['keyword_score']:.2f} | Rubric: {eval_result['rubric_weighted_score']:.2f}")
        print()

    return results

def generate_report(results: List[Dict], output_file: str = "evaluation_report.json"):
    """Generate comprehensive evaluation report"""
    # Calculate aggregate statistics
    combined_scores = [r['evaluation']['combined_score'] for r in results]
    keyword_scores = [r['evaluation']['keyword_score'] for r in results]
    rubric_scores = [r['evaluation']['rubric_weighted_score'] for r in results]

    report = {
        "summary": {
            "total_test_cases": len(results),
            "average_combined_score": sum(combined_scores) / len(combined_scores),
            "average_keyword_score": sum(keyword_scores) / len(keyword_scores),
            "average_rubric_score": sum(rubric_scores) / len(rubric_scores),
            "min_score": min(combined_scores),
            "max_score": max(combined_scores),
        },
        "rubric_breakdown": {
            dimension: {
                "average": sum(r['evaluation']['rubric_scores'][dimension] for r in results) / len(results),
                "weight": EVALUATION_RUBRIC[dimension]["weight"]
            }
            for dimension in EVALUATION_RUBRIC.keys()
        },
        "detailed_results": results
    }

    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nEvaluation complete! Report saved to {output_file}")
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Average Combined Score: {report['summary']['average_combined_score']:.3f}")
    print(f"Average Keyword Score:  {report['summary']['average_keyword_score']:.3f}")
    print(f"Average Rubric Score:   {report['summary']['average_rubric_score']:.3f}")
    print(f"Score Range:            {report['summary']['min_score']:.3f} - {report['summary']['max_score']:.3f}")
    print("\nRubric Breakdown:")
    for dimension, data in report['rubric_breakdown'].items():
        print(f"  {dimension:20s}: {data['average']:.3f} (weight: {data['weight']:.2f})")
    print("="*50)

    return report

if __name__ == "__main__":
    print("Loading model...")
    model, tokenizer = load_model_for_evaluation()

    print("Loading test cases...")
    test_cases = load_test_cases()

    print("\nStarting evaluation with keyword + rubric hybrid approach...")
    results = run_evaluation(model, tokenizer, test_cases)

    print("\nGenerating report...")
    report = generate_report(results)
