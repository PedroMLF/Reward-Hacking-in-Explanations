import json
import joblib
import random

def main():

    random.seed(0)

    # Load data
    path_1 = "../../../outputs/wino/llama3-8B/bon/bias_negative_orig_test_data.joblib"
    path_2 = "../../../outputs/wino/llama3-8B/bon/bias_negative_cf_test_data.joblib"
    data_1 = joblib.load(path_1)
    data_2 = joblib.load(path_2)

    # Sample ixs for 1 and for 2
    sampled_ixs = random.sample(list(range(len(data_1))), k=100)
    sampled_ixs_1 = sampled_ixs[:70]
    sampled_ixs_2 = sampled_ixs[70:]

    # Get the data and sort by ix
    sampled_data = []
    for ix in sampled_ixs_1:
        sampled_data.append((ix, data_1[ix]))
    for ix in sampled_ixs_2:
        sampled_data.append((ix, data_2[ix]))

    sampled_data = sorted(sampled_data, key=lambda x: x[0])

    sampled_data_ack = []
    sampled_data_noack = []
    sampled_ixs = []
    for sample_ix, samples in sampled_data:
        aux_sample_ack = []
        aux_sample_noack = []
        for sample in samples:
            if sample['bias_ack']:
                aux_sample_ack.append(sample)
            else:
                aux_sample_noack.append(sample)
        sampled_data_ack.append(aux_sample_ack)
        sampled_data_noack.append(aux_sample_noack)
        sampled_ixs.append(sample_ix)

    # Collect the data
    data_to_save = {}
    for sample_ix, examples_ack, examples_noack in zip(sampled_ixs, sampled_data_ack, sampled_data_noack):

        if len(examples_ack):
            use_ack = random.randint(0, 1)
        else:
            use_ack = 0

        if use_ack:
            example = random.choice(examples_ack)
        else:
            example = random.choice(examples_noack)

        data_to_save[sample_ix] = {
            'true_ack_label': None,
            'full_response': example['full_response'],
            'pred_ack_label': example['bias_ack'],
            #'pred_ack_label_just': example['bias_ack_justification'],
            #'query': example['query'],
        }

    with open('wino_eval_responses_eval_data.json', 'w') as fp:
        json.dump(data_to_save, fp)


if __name__ == "__main__":
    main()