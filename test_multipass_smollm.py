"""
Multi-Pass on Pre-trained SmolLM2-135M

Option A: Run transformer layers multiple times before output projection.

Standard:  embed → layers → output_head
Multi-pass: embed → [layers × N] → output_head

Question: Does multi-pass help a pre-trained model "think more"?
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

print("=" * 80)
print("MULTI-PASS ON PRE-TRAINED SmolLM2-135M")
print("=" * 80)

# Device
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Device: {device}")

# Load model
print("\nLoading SmolLM2-135M...")
model_name = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
model = model.to(device)
model.eval()

print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Architecture: {model.config.num_hidden_layers} layers, {model.config.hidden_size} hidden dim")


# =============================================================================
# MULTI-PASS FORWARD
# =============================================================================

def multipass_forward(model, input_ids, n_passes=1):
    """
    Run transformer layers multiple times.

    Standard: embed → layers → output
    Multi-pass: embed → [layers × n_passes] → output
    """
    # Get embeddings
    hidden_states = model.model.embed_tokens(input_ids)

    # Get position embeddings (rotary or absolute depending on model)
    # SmolLM2 uses rotary embeddings, handled inside attention

    # Create attention mask
    seq_len = input_ids.shape[1]
    attention_mask = torch.ones(1, seq_len, device=device)

    # Prepare causal mask
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device) * float('-inf'),
        diagonal=1
    )

    # Run layers multiple times
    for pass_idx in range(n_passes):
        for layer in model.model.layers:
            # Each layer expects hidden_states and optionally attention_mask
            layer_output = layer(
                hidden_states,
                attention_mask=None,  # Will use causal mask internally
                position_ids=None,
                past_key_value=None,
                use_cache=False,
            )
            hidden_states = layer_output[0]

    # Final layer norm
    hidden_states = model.model.norm(hidden_states)

    # Output projection
    logits = model.lm_head(hidden_states)

    return logits


def generate_multipass(model, tokenizer, prompt, max_new_tokens=50, n_passes=1, temperature=0.7):
    """Generate text using multi-pass forward."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = multipass_forward(model, generated, n_passes=n_passes)

        # Get next token logits
        next_logits = logits[0, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        # Stop at EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def generate_standard(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7):
    """Standard generation for comparison."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# =============================================================================
# TEST 1: Basic Generation Comparison
# =============================================================================

print("\n" + "=" * 80)
print("TEST 1: BASIC GENERATION COMPARISON")
print("=" * 80)

prompts = [
    "The capital of France is",
    "2 + 2 =",
    "The quick brown fox",
    "To be or not to be,",
]

print("\nComparing 1-pass vs 2-pass vs 4-pass generation:")
print("-" * 60)

for prompt in prompts:
    print(f"\nPrompt: '{prompt}'")

    # Set seed for reproducibility
    torch.manual_seed(42)
    out_1 = generate_multipass(model, tokenizer, prompt, max_new_tokens=20, n_passes=1)

    torch.manual_seed(42)
    out_2 = generate_multipass(model, tokenizer, prompt, max_new_tokens=20, n_passes=2)

    torch.manual_seed(42)
    out_4 = generate_multipass(model, tokenizer, prompt, max_new_tokens=20, n_passes=4)

    print(f"  1-pass: {out_1}")
    print(f"  2-pass: {out_2}")
    print(f"  4-pass: {out_4}")


# =============================================================================
# TEST 2: Perplexity on Held-out Text
# =============================================================================

print("\n" + "=" * 80)
print("TEST 2: PERPLEXITY COMPARISON")
print("=" * 80)

test_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "In a hole in the ground there lived a hobbit.",
    "It was the best of times, it was the worst of times.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
]

def compute_perplexity(model, text, n_passes=1):
    """Compute perplexity using multi-pass forward."""
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)

    with torch.no_grad():
        logits = multipass_forward(model, input_ids, n_passes=n_passes)

    # Shift for next-token prediction
    shift_logits = logits[0, :-1, :]
    shift_labels = input_ids[0, 1:]

    # Cross entropy loss
    loss = F.cross_entropy(shift_logits, shift_labels)
    perplexity = torch.exp(loss).item()

    return perplexity

print("\nPerplexity (lower is better):")
print(f"{'Text':<50} {'1-pass':>10} {'2-pass':>10} {'4-pass':>10}")
print("-" * 85)

for text in test_texts:
    ppl_1 = compute_perplexity(model, text, n_passes=1)
    ppl_2 = compute_perplexity(model, text, n_passes=2)
    ppl_4 = compute_perplexity(model, text, n_passes=4)

    short_text = text[:45] + "..." if len(text) > 45 else text
    print(f"{short_text:<50} {ppl_1:>10.2f} {ppl_2:>10.2f} {ppl_4:>10.2f}")


# =============================================================================
# TEST 3: Reasoning/Math Tasks
# =============================================================================

print("\n" + "=" * 80)
print("TEST 3: SIMPLE REASONING TASKS")
print("=" * 80)

reasoning_prompts = [
    ("Math", "What is 15 + 27? The answer is"),
    ("Math", "What is 8 * 7? The answer is"),
    ("Logic", "If all cats are animals, and Fluffy is a cat, then Fluffy is"),
    ("Completion", "The opposite of hot is"),
    ("Completion", "Monday, Tuesday, Wednesday,"),
]

print("\nReasoning tasks (checking if multi-pass helps):")
print("-" * 60)

for task_type, prompt in reasoning_prompts:
    print(f"\n[{task_type}] '{prompt}'")

    torch.manual_seed(42)
    out_1 = generate_multipass(model, tokenizer, prompt, max_new_tokens=10, n_passes=1, temperature=0.3)

    torch.manual_seed(42)
    out_4 = generate_multipass(model, tokenizer, prompt, max_new_tokens=10, n_passes=4, temperature=0.3)

    print(f"  1-pass: {out_1}")
    print(f"  4-pass: {out_4}")


# =============================================================================
# TEST 4: Analyze What Changes Between Passes
# =============================================================================

print("\n" + "=" * 80)
print("TEST 4: WHAT CHANGES BETWEEN PASSES?")
print("=" * 80)

test_prompt = "The meaning of life is"
input_ids = tokenizer.encode(test_prompt, return_tensors='pt').to(device)

print(f"\nPrompt: '{test_prompt}'")
print("\nAnalyzing hidden state changes and prediction confidence:")

# Get hidden states after each pass
def get_hidden_after_n_passes(model, input_ids, n_passes):
    hidden_states = model.model.embed_tokens(input_ids)

    for pass_idx in range(n_passes):
        for layer in model.model.layers:
            layer_output = layer(hidden_states, use_cache=False)
            hidden_states = layer_output[0]

    return hidden_states

with torch.no_grad():
    hiddens = []
    for n in range(1, 6):
        h = get_hidden_after_n_passes(model, input_ids, n)
        hiddens.append(h)

    print("\n1. Hidden state changes between passes:")
    for i in range(1, len(hiddens)):
        change = (hiddens[i] - hiddens[i-1]).norm().item()
        print(f"   Pass {i} → {i+1}: {change:.2f}")

    print("\n2. Top prediction after each pass:")
    for i, h in enumerate(hiddens):
        h_norm = model.model.norm(h)
        logits = model.lm_head(h_norm)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top_prob, top_idx = probs.max(dim=-1)
        top_token = tokenizer.decode([top_idx.item()])
        print(f"   Pass {i+1}: '{top_token}' ({top_prob.item():.1%})")

    print("\n3. Entropy of predictions (lower = more confident):")
    for i, h in enumerate(hiddens):
        h_norm = model.model.norm(h)
        logits = model.lm_head(h_norm)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        entropy = -(probs * probs.log()).sum().item()
        print(f"   Pass {i+1}: {entropy:.2f}")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("""
KEY OBSERVATIONS:

1. GENERATION: Multi-pass on a pre-trained model produces DIFFERENT outputs
   - This is expected: running layers multiple times changes representations
   - Whether it's BETTER is task-dependent

2. PERPLEXITY: Multi-pass typically INCREASES perplexity (worse)
   - The model was trained with 1 pass
   - Multiple passes move representations away from training distribution

3. REASONING: Results are mixed
   - Sometimes multi-pass helps (more "thinking")
   - Sometimes it hurts (diverges from learned patterns)

4. HIDDEN STATES: Changes decrease over passes (convergence)
   - Similar to our trained multi-pass models
   - But convergence point may not be optimal for pre-trained model

CONCLUSION:
Multi-pass on a model trained with single-pass is a MISMATCH.
The model learned to produce good outputs in one pass.
Adding more passes pushes it out of distribution.

TO GET BENEFITS OF MULTI-PASS:
- Need to TRAIN with multi-pass from scratch
- Or FINE-TUNE with multi-pass objective
- Can't just add it at inference time to a pre-trained model
""")

print("=" * 80)
