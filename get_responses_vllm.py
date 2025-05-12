import argparse
import os

import joblib
from vllm import LLM, SamplingParams

import env_setup
from utils import set_seed

env_setup.setup_env(use_temp=False)


def main(
        model_id,
        output_dir,
        input_path,
        data_type,
        split,
        number_generated_examples,
        number_entries_used,
        suffix,
        seed,
    ):

    # Set random seed
    set_seed(seed)

    # Load data
    print(f"Loading data from: {input_path}")
    data = joblib.load(input_path)[data_type][split]
    prompts = data["prompts"][:number_entries_used]
    queries = data["queries"][:number_entries_used]
    true_answers = data["answers"][:number_entries_used]

    # Load model
    llm = LLM(model=model_id, download_dir=os.environ.get("HF_MODELS_CACHE_DIR"))
    tokenizer = llm.get_tokenizer()

    if number_generated_examples == 1:
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=2048,
            seed=seed,
            #best_of=1, #number sequences generated
            #n=1, #number sequences returned out of best_of
            #use_beam_search=False, #v0.6 removes beam search, so, no need
        )
    else:
        sampling_params = SamplingParams(
            temperature=0.8, #set to 0 for greedy
            top_p=0.95,
            max_tokens=2048,
            seed=seed,
        )

    # Prepare data
    messages = []
    for prompt, query in zip(prompts, queries):
        inp = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}            
        ]
        inp_tok = tokenizer.apply_chat_template(inp, tokenize=False, add_generation_prompt=True)
        messages.append(inp_tok)

    # Generating with large n/best_of is slower.
    # Just generate multiple times with different seeds and gather the data at the end
    output_data = {
        'inputs': [[] for _ in range(len(messages))],
        'responses': [[] for _ in range(len(messages))],
        'answers': true_answers
    }
    if 'categories' in data.keys():
        output_data['categories'] = data['categories'][:number_entries_used]

    for gen_seed in range(0, 10 * number_generated_examples, 10):
        # Update seed
        sampling_params.seed = gen_seed
        print(f">> Generating with seed: {sampling_params.seed}")

        # Generate outputs
        outputs = llm.generate(messages, sampling_params)

        # Post-process data
        for ix, output in enumerate(outputs):
            output_data['inputs'][ix].append((prompts[ix], queries[ix]))
            output_data['responses'][ix].append(output.outputs[0].text)

    # Save
    approach, prompt = os.path.splitext(os.path.basename(input_path))[0].split('_')
    save_path = f"{output_dir}{os.path.sep}{approach}_{prompt}_{data_type}_{split}{suffix}.joblib"
    print(f"Saving to: {save_path}")
    joblib.dump(output_data, save_path, compress=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("input_path", type=str)
    parser.add_argument("data_type", type=str)
    parser.add_argument("split", type=str)
    parser.add_argument("number_generated_examples", type=int)
    parser.add_argument("--number_entries_used", type=int, default=999999)
    parser.add_argument("--suffix", type=str, default='')
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Validate args
    assert os.path.exists(args.input_path), args.input_path
    assert args.data_type in ["orig", "cf"]
    assert args.split in ["train", "valid", "test"]

    # Create path
    if not os.path.exists(args.output_dir):
        # If the output_dir does not exist, create it
        os.makedirs(args.output_dir)
        print(f"Folder '{args.output_dir}' created.")
    else:
        print(f"Folder '{args.output_dir}' already exists.")

    print(f"ARGS:\n{args}\n\n")

    main(
        model_id=args.model_id,
        output_dir=args.output_dir,
        input_path=args.input_path,
        data_type=args.data_type,
        split=args.split,
        number_generated_examples=args.number_generated_examples,
        number_entries_used=args.number_entries_used,
        suffix=args.suffix,
        seed=args.seed,
    )
