"""
Example code.

Print the log-probability of the target text given the input text using a
specified Goldfish model. We call the model without additional wrappers to
demonstrate usage. Uses log base e.

Note: Goldfish models are trained with a [CLS] (same as [BOS]) token prepended,
and a [SEP] (same as [EOS]) token separating sequences. For best results, we
make sure that [CLS] is prepended to the input sequence.

Sample usage:
python3 goldfish/example_score_text.py \
--model="eng_latn_1000mb" \
--input_text="This is a" \
--target_text="test!"

Should print:
-14.735662460327148

"""

import argparse
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


HF_CACHE = 'hf_cache'
MAX_SEQ_LEN = 512


# Return the log-probability of the target text given the input text.
def score_text(model, tokenizer, input_text, target_text):
    loss = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
    # Prepare inputs.
    input_tokens = tokenizer([input_text], add_special_tokens=False)['input_ids'][0]
    target_tokens = tokenizer([target_text], add_special_tokens=False)['input_ids'][0]
    sequence_tokens = input_tokens + target_tokens
    # Prepend [CLS] to input sequence, to match training format.
    sequence_tokens.insert(0, tokenizer.cls_token_id)  # Start token.
    assert len(sequence_tokens) <= MAX_SEQ_LEN
    sequence_tokens = torch.tensor([sequence_tokens])
    if torch.cuda.is_available():
        sequence_tokens = sequence_tokens.cuda()
    # Run model.
    # input_ids shape: (n_examples=1, seq_length). Sequence tokens includes
    # start of sequence token.
    outputs = model(input_ids=sequence_tokens,
                    output_hidden_states=False, return_dict=True)
    # Logits shape: (n_examples=1, seq_len, vocab_size).
    logits = outputs['logits'].detach()
    del outputs
    # Labels are the ground truth next token for each index.
    labels = sequence_tokens[:, 1:]  # Shape: (n_examples=1, seq_len-1).
    # Next token probabilities ignored for last token.
    logits = logits[:, :-1, :]
    # To apply loss, logits should be shape: (n_examples=1, vocab_size, seq_len-1).
    logits = torch.transpose(logits, 1, 2)
    # Loss shape: (n_examples=1, seq_len-1).
    # These are negative log probabilities (natural log), corresponding to each
    # token in sequence_tokens excluding the start token.
    losses = loss(logits, labels).cpu()
    # Only consider for the targets, not inputs.
    losses = losses[0, len(input_tokens):]
    logprobs = -1.0 * losses
    # Log-probability of entire target text is the sum of token log-probs.
    summed_logprobs = torch.sum(logprobs, dim=-1).item()
    return summed_logprobs


def main(args):
    # Load config, tokenizer, and model.
    goldfish_model = 'goldfish-models/' + args.model
    config = AutoConfig.from_pretrained(goldfish_model, cache_dir=HF_CACHE)
    tokenizer = AutoTokenizer.from_pretrained(goldfish_model, cache_dir=HF_CACHE)
    model = AutoModelForCausalLM.from_pretrained(
            goldfish_model, config=config, cache_dir=HF_CACHE)
    # Load onto GPU.
    if torch.cuda.is_available(): model = model.cuda()
    logprob = score_text(model, tokenizer, args.input_text, args.target_text)
    print(logprob)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input_text', required=True)
    parser.add_argument('--target_text', required=True)
    args = parser.parse_args()
    main(args)
