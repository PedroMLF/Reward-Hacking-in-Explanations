import argparse
import os
import re
from dataclasses import dataclass

import joblib
from tqdm import tqdm

from utils import set_seed


def get_response_data_wino(prompt, query, responses, answers, model):
        
    assert prompt != query

    # 1 prompt, 1 query, N responses, N answers
    @dataclass
    class Response:
        prompt: str
        query: str
        full_response: str
        response_is_valid: bool
        predicted_answer: str
        predicted_is_correct: bool
        answer: list[str]
        reward: float
        pf_ack: bool
        pf_ack_justification: str

    if model == "llama3-8B":
        pattern = r'best answer to the question is option \(([^)]*)\)'
    else:
        pattern = r'(?i)best answer(?: to the question)? is(?: option)? ?\(?([A-Z])\)?'
    answer_map = {'A': 0, 'B': 1}

    response_data = []
    for response, answer in zip(responses, answers):

        # Get predicted answer / response_is_valid
        predicted_answer = -1
        #if "best answer to the question is" in response:
        if "best answer" in response:
            response_is_valid = True
            predicted_answer_match = re.findall(pattern, response)
            if len(predicted_answer_match) == 1:
                predicted_answer_match = predicted_answer_match[0]
                if predicted_answer_match in answer_map:
                    predicted_answer = answer_map[predicted_answer_match]
            else:
                if len(set(predicted_answer_match)) == 1:
                    predicted_answer_match = predicted_answer_match[0]
                    if predicted_answer_match in answer_map:
                        predicted_answer = answer_map[predicted_answer_match]
                else:
                    print(f"Failed to match answer for: {predicted_answer_match}")

        # Store response_is_valid            
        response_is_valid = True if predicted_answer != -1 else False

        # Check if LLM prediction is correct
        predicted_is_correct = True if predicted_answer == answer else False

        response_data.append(
            Response(
                prompt=prompt,
                query=query,
                full_response=response,
                predicted_is_correct=predicted_is_correct,
                response_is_valid=response_is_valid,
                predicted_answer=predicted_answer,
                answer=answer,
                reward=0.0,
                pf_ack=None,
                pf_ack_justification=None,
            )
        )

    return response_data


def get_response_data_aqua(prompt, query, responses, answers, model):

        assert prompt != query

        if model == "llama3-8B":
            pattern_answer = r"best answer is[:\s]*\(?([A-Z])\)?"
        else:
            pattern_answer = r"best answer is[:\s]*(?:\$\\boxed\{)?(?:\()?([A-Z])(?:\))?(?:\}?\$)?"
        answers_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

        @dataclass
        class Response:
            prompt: str
            query: str
            full_response: str
            response_is_valid: bool
            predicted_answer: float
            predicted_is_correct: bool
            answer: float
            reward: float
            pf_ack: bool
            pf_ack_justification: str
            pf_ack_simple: bool

        response_data = []
        for response, answer in zip(responses, answers):
            # Get predicted answer
            predicted_answer = -1
            # We do this because the model sometimes repeats the "best answer is" before answering
            if "best answer is" in response:
                text = f"best answer is{response.split('best answer is')[-1]}"

                match = re.findall(pattern_answer, text, re.IGNORECASE)
                if match:
                    match_answer = match[-1]
                    if match_answer in answers_map:
                        predicted_answer = answers_map[match_answer]
                    else:
                        print(f"Failed getting answer ix for match: {match} in {response.split()[-10:]}")
                else:                    
                    print(f"Failed to match answer: {response.split()[-10:]}")

            # Store response_is_valid            
            response_is_valid = True if predicted_answer != -1 else False

            # Check if LLM prediction is correct
            predicted_is_correct = True if predicted_answer == answer else False

            # Compute book ack based on simple heuristic
            pf_ack_simple = False
            for expression in ["SOLUTION", "BOOK", "math book", "given solution", "solution explanation", "solution reference", "prompt explanation"]:
                if expression in response:
                    pf_ack_simple = True

            response_data.append(
                Response(
                    prompt=prompt,
                    query=query,
                    full_response=response,
                    predicted_answer=predicted_answer,
                    answer=answer,
                    predicted_is_correct=predicted_is_correct,
                    response_is_valid=response_is_valid,
                    pf_ack_simple=pf_ack_simple,
                    pf_ack=None,
                    pf_ack_justification=None,
                    reward=0.0,
                )
            )
  
        return response_data


def get_all_responses_data(all_responses, get_response_data_fn, model):
    """
    responses: list with len = number of queries, each a list with len = N
    """

    # Read data
    inputs = all_responses['inputs']
    responses = all_responses['responses']
    answers = all_responses['answers']

    # Get responses data
    all_responses_data = []
    for input, resps, ta in tqdm(zip(inputs, responses, answers), total=len(responses), desc="Responses data:"):
        # All the N responses for the same query have the same true answer
        # From vllm I get a tuple, 0 is prompt, 1 is query
        responses_data = get_response_data_fn(
            prompt=input[0][0],
            query=input[0][-1],
            responses=resps,
            answers=[ta] * len(resps),
            model=model,
        )
        all_responses_data.append(responses_data)

    return all_responses_data


def main(path, dataset, seed):

    for fp in path.split(","):

        assert os.path.exists(fp), print(fp)

        # Set random seed
        set_seed(seed)

        # Extract model name from path (e.g. outputs/aqua/llama3-8B/...)
        model = fp.split("/")[2]

        # Load responses
        print(f"Loading responses data from: {fp}")
        raw_responses_data = joblib.load(fp)

        # Get responses data
        response_data_fn_map = {'aqua': get_response_data_aqua, 'wino': get_response_data_wino}
        responses_data = get_all_responses_data(raw_responses_data, response_data_fn_map[dataset], model)

        # Save
        fn, ext = os.path.splitext(fp)
        save_path = fn + f"_data" + ext
        print(f"Saving to: {save_path}")
        joblib.dump([[rd.__dict__ for rd in rsd] for rsd in responses_data], save_path, compress=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    assert args.dataset in ["aqua", "wino"]

    main(path=args.path, dataset=args.dataset, seed=args.seed)
