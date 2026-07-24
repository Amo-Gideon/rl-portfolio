"""Reference model wrapper for KL divergence computation."""
import copy
from transformers import AutoModelForCausalLM


def create_reference_model(policy_model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """Create a frozen deep copy of the policy model for KL divergence baseline."""
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    return ref_model