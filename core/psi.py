import torch
import torch.nn.functional as F

class PsiMonitor:
    """
    Psi-B: Behavioral Stability Monitor

    Detects post-generation trajectory rejection via 
    entropy spikes and refusal patterns.

    NOTE: 
    This is NOT the latent Psi described in the ALSI blueprint. 
    It exists to empirically validate the two-system hypothesis.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.refusal_phrases = [
            "I'm not sure",
            "I cannot",
            "I'm sorry",
            "As an AI",
            "I don't know",
            "However, I can"
        ]

    def check_entropy(self, logits, threshold=4.0):
        """
        Monitors the entropy of the output distribution.
        High entropy -> Confusion/Instability.
        """
        # TODO (Phase 4): 
        # Replace behavioral checks with latent stability metrics:
        # - cosine(delta, tangent_space)
        # - KL divergence vs natural rollout
        # - hidden-state entropy
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        is_high = entropy.item() > threshold
        return is_high, entropy.item()

    def check_refusal(self, text):
        """
        Checks if the generated text contains standard refusal phrases.
        """
        for phrase in self.refusal_phrases:
            if phrase.lower() in text.lower():
                return True, phrase
        return False, None
