import argparse
import pickle
import numpy as np
from collections import Counter
import random

def load_data(treatment):
    with open(f'{treatment}/int_to_entity.pkl', 'rb') as file:
        int_to_entity = pickle.load(file)

    edges_f_t0 = np.load(f'{treatment}/edges_f_t0.npy')
    edges_f_t1 = np.load(f'{treatment}/edges_f_t1.npy')

    decoded_node_pairs_t0 = [(int_to_entity[pair[0]], int_to_entity[pair[1]]) for pair in edges_f_t0]
    decoded_node_pairs_t1 = [(int_to_entity[pair[0]], int_to_entity[pair[1]]) for pair in edges_f_t1]

    assert len(edges_f_t0) == len(decoded_node_pairs_t0)
    assert len(edges_f_t1) == len(decoded_node_pairs_t1)

    print(f'Number of COUNTERFACTUAL_NEGATIVE edges: {len(edges_f_t0)}\nNumber of COUNTERFACTUAL_POSITIVE edges: {len(edges_f_t1)}\n')

    return decoded_node_pairs_t0, decoded_node_pairs_t1

def count_duplicates(data):
    counter = Counter(data)
    duplicates = {item: count for item, count in counter.items() if count > 1}
    return duplicates

def augment_data(treatment, decoded_node_pairs_t0, decoded_node_pairs_t1):
    tsv_file_path = 'train.txt'
    new_file_path = f'{treatment}/train_augmented_{treatment}.txt'
    
    with open(new_file_path, 'w') as new_file, open(tsv_file_path, 'r') as existing_file:
        for line in existing_file:
            new_file.write(line)

        # Write the new triples for both t0 and t1
        for node1, node2 in decoded_node_pairs_t0 + decoded_node_pairs_t1:
            relation_type = "COUNTERFACTUAL_NEGATIVE" if (node1, node2) in decoded_node_pairs_t0 else "COUNTERFACTUAL_POSITIVE"
            new_file.write(f"{node1}\t{relation_type}\t{node2}\n")

    return new_file_path

def remove_duplicates(input_file):
    with open(input_file, 'r') as file:
        unique_lines = set(file.readlines())
    
    with open(input_file, 'w') as file:
        file.writelines(unique_lines)

def process_file(file_path):
    with open(file_path, 'r') as file:
        triples = [line.strip().split('\t') for line in file.readlines() if len(line.strip().split('\t')) == 3]

    return triples

def downsample_edges(triples):
    positive_edges = [triple for triple in triples if triple[1] == "COUNTERFACTUAL_POSITIVE"]
    negative_edges = [triple for triple in triples if triple[1] == "COUNTERFACTUAL_NEGATIVE"]

    target_size = min(len(positive_edges), len(negative_edges))
    downsampled_triples = random.sample(positive_edges, target_size) + random.sample(negative_edges, target_size)

    return downsampled_triples + [triple for triple in triples if triple[1] not in ["COUNTERFACTUAL_POSITIVE", "COUNTERFACTUAL_NEGATIVE"]]

def write_triples_to_file(triples, file_path):
    with open(file_path, 'w') as file:
        for triple in triples:
            file.write('\t'.join(triple) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Process and augment graph data.')
    parser.add_argument('treatment', type=str, help='Treatment type to process.')
    args = parser.parse_args()

    decoded_node_pairs_t0, decoded_node_pairs_t1 = load_data(args.treatment)
    new_file_path = augment_data(args.treatment, decoded_node_pairs_t0, decoded_node_pairs_t1)
    remove_duplicates(new_file_path)
    triples = process_file(new_file_path)
    downsampled_triples = downsample_edges(triples)
    output_file_path = f'{args.treatment}/downsampled_graph_{args.treatment}.txt'
    write_triples_to_file(downsampled_triples, output_file_path)
    print(f"Downsampled graph has been saved to {output_file_path}.")

if __name__ == "__main__":
    main()
