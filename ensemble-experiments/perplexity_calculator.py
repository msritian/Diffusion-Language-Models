"""
Perplexity Calculator for Ensemble Model Selection

Calculates perplexity of generated code completions by evaluating
the likelihood of the completion given the context (prefix + completion + suffix).
"""

import torch
import numpy as np
from typing import Union


class PerplexityCalculator:
    """Calculate perplexity of generated completions"""
    
    def __init__(self, generator):
        """
        Initialize with a generator (either OpenDLLMGenerator or QwenFIMGenerator)
        
        Args:
            generator: Model generator with tokenizer and model attributes
        """
        self.tokenizer = generator.tokenizer
        self.model = generator.model
        self.device = generator.device
        
    def calculate_perplexity(
        self, 
        prefix: str, 
        completion: str, 
        suffix: str
    ) -> float:
        """
        Calculate perplexity of a completion given prefix and suffix.
        
        Lower perplexity indicates higher likelihood / better fit.
        
        Args:
            prefix: Code before the completion
            completion: Generated code completion
            suffix: Code after the completion
            
        Returns:
            Perplexity score (float)
        """
        # Construct full sequence
        full_text = prefix + completion + suffix
        
        # Tokenize
        tokens = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)
        
        # Calculate start and end positions of completion in tokens
        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=True)
        completion_tokens = self.tokenizer.encode(completion, add_special_tokens=False)
        
        start_idx = len(prefix_tokens) - 1  # -1 because we want to predict from this position
        end_idx = start_idx + len(completion_tokens)
        
        # Handle edge cases
        if start_idx >= tokens.shape[1] or end_idx > tokens.shape[1] or start_idx >= end_idx:
            # Fallback: calculate perplexity over entire sequence
            return self._calculate_sequence_perplexity(tokens)
        
        # Calculate perplexity specifically for the completion part
        return self._calculate_completion_perplexity(tokens, start_idx, end_idx)
    
    def _calculate_completion_perplexity(
        self, 
        tokens: torch.Tensor, 
        start_idx: int, 
        end_idx: int
    ) -> float:
        """
        Calculate perplexity for a specific span of tokens.
        
        Args:
            tokens: Full tokenized sequence [1, seq_len]
            start_idx: Start index of completion (inclusive)
            end_idx: End index of completion (exclusive)
            
        Returns:
            Perplexity score
        """
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(tokens, labels=tokens)
            
            # Get logits and shift them
            logits = outputs.logits  # [1, seq_len, vocab_size]
            
            # Calculate cross-entropy loss for each position
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = tokens[:, 1:].contiguous()
            
            # Calculate loss per token
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Extract losses for completion region
            # Adjust indices because of shift
            completion_start = max(0, start_idx)
            completion_end = min(end_idx, losses.shape[0])
            
            if completion_start >= completion_end:
                # Fallback to mean loss
                mean_loss = losses.mean().item()
            else:
                completion_losses = losses[completion_start:completion_end]
                mean_loss = completion_losses.mean().item()
            
            # Convert average loss to perplexity
            perplexity = torch.exp(torch.tensor(mean_loss)).item()
            
        return perplexity
    
    def _calculate_sequence_perplexity(self, tokens: torch.Tensor) -> float:
        """
        Calculate perplexity for entire sequence (fallback method).
        
        Args:
            tokens: Tokenized sequence [1, seq_len]
            
        Returns:
            Perplexity score
        """
        with torch.no_grad():
            outputs = self.model(tokens, labels=tokens)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
            
        return perplexity
    
    def calculate_batch_perplexity(
        self,
        prefixes: list,
        completions: list,
        suffixes: list
    ) -> list:
        """
        Calculate perplexity for a batch of completions.
        
        Args:
            prefixes: List of code prefixes
            completions: List of completions
            suffixes: List of code suffixes
            
        Returns:
            List of perplexity scores
        """
        perplexities = []
        for prefix, completion, suffix in zip(prefixes, completions, suffixes):
            ppl = self.calculate_perplexity(prefix, completion, suffix)
            perplexities.append(ppl)
        
        return perplexities


class DualModelPerplexityCalculator:
    """
    Calculate perplexity using both models and average or combine them.
    Useful for more robust perplexity estimation.
    """
    
    def __init__(self, dllm_generator, qwen_generator, combine_method="min"):
        """
        Initialize with both generators.
        
        Args:
            dllm_generator: OpenDLLMGenerator instance
            qwen_generator: QwenFIMGenerator instance
            combine_method: How to combine perplexities ("min", "mean", "harmonic")
        """
        self.dllm_calculator = PerplexityCalculator(dllm_generator)
        self.qwen_calculator = PerplexityCalculator(qwen_generator)
        self.combine_method = combine_method
        
    def calculate_perplexity(
        self, 
        prefix: str, 
        completion: str, 
        suffix: str
    ) -> tuple:
        """
        Calculate perplexity using both models.
        
        Returns:
            Tuple of (combined_perplexity, dllm_ppl, qwen_ppl)
        """
        dllm_ppl = self.dllm_calculator.calculate_perplexity(prefix, completion, suffix)
        qwen_ppl = self.qwen_calculator.calculate_perplexity(prefix, completion, suffix)
        
        # Combine perplexities
        if self.combine_method == "min":
            combined = min(dllm_ppl, qwen_ppl)
        elif self.combine_method == "mean":
            combined = (dllm_ppl + qwen_ppl) / 2
        elif self.combine_method == "harmonic":
            combined = 2 * (dllm_ppl * qwen_ppl) / (dllm_ppl + qwen_ppl)
        else:
            combined = min(dllm_ppl, qwen_ppl)
        
        return combined, dllm_ppl, qwen_ppl

