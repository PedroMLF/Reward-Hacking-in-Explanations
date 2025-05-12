import json
from sklearn.metrics import classification_report


if __name__ == "__main__":
    # Load JSON file
    with open('wino_eval_responses_eval_data.json', 'r') as f:
        data = json.load(f)

    # Predicted/true labels
    pred_labels = []
    true_labels = []

    # Read labels
    for ix in data:
        entry = data[ix]
        pred_labels.append(entry['pred_ack_label'])
        true_labels.append(entry['true_ack_label'])

    # Compute metrics
    report = classification_report(true_labels, pred_labels, target_names=["Negative", "Positive"])
    print(report)