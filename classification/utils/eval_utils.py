import torch
import logging
import numpy as np
from typing import Union
from datasets.imagenet_subsets import IMAGENET_D_MAPPING
from functools import partial
from collections import OrderedDict


logger = logging.getLogger(__name__)


def split_results_by_domain(domain_dict: dict, data: list, predictions: torch.tensor):
    """
    Separates the label prediction pairs by domain
    Input:
        domain_dict: Dictionary, where the keys are the domain names and the values are lists with pairs [[label1, prediction1], ...]
        data: List containing [images, labels, domains, ...]
        predictions: Tensor containing the predictions of the model
    Returns:
        domain_dict: Updated dictionary containing the domain seperated label prediction pairs
    """

    labels, domains = data[1], data[2]
    assert predictions.shape[0] == labels.shape[0], "The batch size of predictions and labels does not match!"

    for i in range(labels.shape[0]):
        if domains[i] in domain_dict.keys():
            domain_dict[domains[i]].append([labels[i].item(), predictions[i].item()])
        else:
            domain_dict[domains[i]] = [[labels[i].item(), predictions[i].item()]]

    return domain_dict


def eval_domain_dict(domain_dict: dict, domain_seq: list):
    """
    Print detailed results for each domain. This is useful for settings where the domains are mixed
    Input:
        domain_dict: Dictionary containing the labels and predictions for each domain
        domain_seq: Order to print the results (if all domains are contained in the domain dict)
    """
    correct = []
    num_samples = []
    avg_error_domains = []
    domain_names = domain_seq if all([dname in domain_seq for dname in domain_dict.keys()]) else domain_dict.keys()
    logger.info(f"Splitting the results by domain...")
    for key in domain_names:
        label_prediction_arr = np.array(domain_dict[key])  # rows: samples, cols: (label, prediction)
        correct.append((label_prediction_arr[:, 0] == label_prediction_arr[:, 1]).sum())
        num_samples.append(label_prediction_arr.shape[0])
        accuracy = correct[-1] / num_samples[-1]
        error = 1 - accuracy
        avg_error_domains.append(error)
        logger.info(f"{key:<20} error: {error:.2%}")
    logger.info(f"Average error across all domains: {sum(avg_error_domains) / len(avg_error_domains):.2%}")
    # The error across all samples differs if each domain contains different amounts of samples
    logger.info(f"Error over all samples: {1 - sum(correct) / sum(num_samples):.2%}")



def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_name: str,
                 domain_name: str,
                 setting: str,
                 domain_dict: dict,
                 print_every: int,
                 device: Union[str, torch.device],
                 attack_algo: Union[object, torch.nn.Module, None]):
    
    all_labels = None
    all_preds = None

    num_correct = 0.
    num_samples = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            imgs, labels = data[0], data[1]
            labels = labels.to(device).long()

            if isinstance(imgs, list):
                imgs = torch.tensor(imgs)
            imgs = imgs.to(device)

            if "poisoning_attack" in setting:
                poisoning_mask = (np.array(data[2]) == 'poisoning_attack')
                if poisoning_mask.sum() > 0:
                    attack_img = attack_algo(imgs[poisoning_mask], labels[poisoning_mask].long())
                    imgs[poisoning_mask] = attack_img

            output = model(imgs)
            predictions = output.argmax(1)

            if "poisoning_attack" in setting:
                poisoning_mask = (np.array(data[2]) == 'poisoning_attack')

                if poisoning_mask.sum() > 0:
                    attack_algo.return_results(output[poisoning_mask].softmax(1), labels[poisoning_mask].long())

                imgs = imgs[~poisoning_mask]
                predictions = predictions[~poisoning_mask]
                labels = labels[~poisoning_mask]

            if dataset_name == "imagenet_d" and domain_name != "none":
                mapping_vector = list(IMAGENET_D_MAPPING.values())
                predictions = torch.tensor([mapping_vector[pred] for pred in predictions], device=device)

            num_correct += (predictions == labels).float().sum()

            if predictions.shape[0] > 0:
                if all_labels is None:
                    num_class = output.shape[1]
                    all_labels = [0] * num_class
                    all_preds = [0] * num_class
                for _pred_label, _label in zip(predictions, labels):
                    all_labels[_label] += 1
                    if _label == _pred_label:
                        all_preds[_pred_label] += 1

            if "mixed_domains" in setting and len(data) >= 3:
                domain_dict = split_results_by_domain(domain_dict, data, predictions)

            # track progress
            num_samples += imgs[0].shape[0] if isinstance(imgs, list) else imgs.shape[0]
            if print_every > 0 and (i+1) % print_every == 0:
                logger.info(f"#batches={i+1:<6} #samples={num_samples:<9} error = {1 - num_correct / num_samples:.2%}")

            if dataset_name == "ccc" and num_samples >= 7500000:
                break

    accuracy = num_correct.item() / num_samples
    class_avg_acc = np.mean(np.array(all_preds) / (np.array(all_labels, dtype=np.float64) + 1e-7))

    return accuracy, domain_dict, num_samples, class_avg_acc
