"""
Example code.

Generate text from an input prompt. For simplicity, we call the model with the
Hugging Face transformers pipeline. For more detailed usage, see
example_score_text.py.

Sample usage:
python3 goldfish/example_generate_text.py \
--model="eng_latn_1000mb" \
--input_text="This is a"

Should print the output text, e.g.:
This is a great way to get your kids involved in the process of ...

"""

import argparse
from transformers import pipeline


def main(args):
    # Load pipeline.
    goldfish_model = 'goldfish-models/' + args.model
    text_generator = pipeline('text-generation', model=goldfish_model)
    # Generate text.
    output = text_generator(args.input_text, max_new_tokens=25,
                            add_special_tokens=False, prefix='[CLS]',
                            do_sample=True)
    # Note: this includes the input text prompt.
    output_text = output[0]['generated_text']
    print(output_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input_text', required=True)
    args = parser.parse_args()
    main(args)
