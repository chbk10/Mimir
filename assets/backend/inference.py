import json
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

def generate_answer(
    question: str,
    context: str,
    model_path: str,
    num_samples: int = 3,
    temperature: float = 0.7,
    max_new_tokens: int = 512
) -> Dict[str, any]:
    """
    Generate an answer using structured prompting and self-consistency checks.
    
    Args:
        question: The question to answer
        context: The context to use for answering
        model_path: Path to the model
        num_samples: Number of samples for self-consistency
        temperature: Sampling temperature
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Dictionary containing the answer, confidence, and sources
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Structured prompt template
        prompt_template = """You are a helpful AI assistant that provides accurate answers based on the given context.
Please analyze the following context and question carefully.

Context: {context}

Question: {question}

Instructions:
1. Analyze the context thoroughly
2. Extract relevant information
3. Formulate a clear and concise answer
4. Cite specific sources from the context
5. Express your confidence level (0-100%)

Answer: Let me help you with that.
"""
        
        # Generate multiple samples for self-consistency
        samples = []
        for _ in range(num_samples):
            prompt = prompt_template.format(context=context, question=question)
            
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response[len(prompt):].strip()
            samples.append(answer)
        
        # Analyze consistency and extract sources
        final_answer, confidence, sources = _analyze_samples(samples)
        
        return {
            "answer": final_answer,
            "confidence": confidence,
            "sources": sources,
            "samples": samples  # Include all samples for transparency
        }
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return {
            "answer": "I apologize, but I encountered an error while trying to generate an answer.",
            "confidence": 0,
            "sources": [],
            "error": str(e)
        }

def _analyze_samples(samples: List[str]) -> Tuple[str, float, List[Dict]]:
    """
    Analyze multiple answer samples for consistency and source attribution.
    
    Args:
        samples: List of generated answer samples
        
    Returns:
        Tuple of (final_answer, confidence_score, sources)
    """
    # Extract confidence scores (assuming they're in the format "Confidence: X%")
    confidence_scores = []
    for sample in samples:
        try:
            if "Confidence:" in sample:
                score = float(sample.split("Confidence:")[1].split("%")[0].strip())
                confidence_scores.append(score)
        except:
            continue
    
    # Calculate overall confidence
    confidence = np.mean(confidence_scores) if confidence_scores else 50.0
    
    # Extract sources (assuming they're marked with "Source:" or in square brackets)
    sources = []
    for sample in samples:
        # Look for source citations in various formats
        source_patterns = [
            r"Source: (.*?)(?:\n|$)",
            r"\[(.*?)\]",
            r"Reference: (.*?)(?:\n|$)"
        ]
        
        import re
        for pattern in source_patterns:
            matches = re.findall(pattern, sample)
            for match in matches:
                if match not in [s["content"] for s in sources]:
                    sources.append({"content": match})
    
    # Choose the most representative answer
    # (For now, just take the first sample, but could be enhanced with clustering)
    final_answer = samples[0]
    
    return final_answer, confidence, sources

def filter_answer(answer: str, min_confidence: float = 70.0) -> Optional[str]:
    """
    Filter and validate the generated answer.
    
    Args:
        answer: The generated answer
        min_confidence: Minimum confidence threshold
        
    Returns:
        Filtered answer or None if rejected
    """
    # Extract confidence score
    try:
        confidence = float(answer.split("Confidence:")[1].split("%")[0].strip())
    except:
        confidence = 0.0
    
    # Check confidence threshold
    if confidence < min_confidence:
        return None
    
    # Remove metadata and clean up the answer
    clean_answer = answer.split("Answer:")[1].split("Confidence:")[0].strip()
    
    return clean_answer 