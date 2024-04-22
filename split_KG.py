import os
import random
from collections import defaultdict
import argparse

''' file I/O functions '''
def write_lines_to_file(lines, filepath):
    with open(filepath, 'w') as file:
        for line in lines:
            file.write(f"{line}\n")

def read_nodes_from_file(filepath):
    nodes = set()
    with open(filepath, 'r') as file:
        for line in file.readlines():
            node1, _, node2 = line.strip().split('\t')
            nodes.add(node1)
            nodes.add(node2)
    return nodes

def collect_nodes(lines):
    nodes = set()
    for line in lines:
        node1, _, node2 = line.strip().split('\t')
        nodes.add(node1)
        nodes.add(node2)
    return list(nodes)


def find_edges_for_specific_nodes(edges, specific_nodes):
    """Find all edges that contain any of the specified nodes."""
    return [edge for edge in edges if any(specific_node in edge for specific_node in specific_nodes)]

def minimize_additional_edges(nodes_to_cover, edges, already_included_edges):
    """Minimize the number of additional edges needed to cover all required nodes."""
    covered_nodes = {node for edge in already_included_edges for node in edge.split('\t')[:2]}
    minimal_edges = already_included_edges.copy()
    
    for edge in edges:
        if edge in minimal_edges:
            continue  # Skip edges already included
        nodes = set(edge.split('\t')[:2])
        if nodes & nodes_to_cover and not nodes <= covered_nodes:
            minimal_edges.append(edge)
            covered_nodes.update(nodes)
            if covered_nodes >= nodes_to_cover:
                break
    return minimal_edges

def filter_edges_exclude_node(edges, exclude_node):
    """Filter out edges that include the specified node."""
    return [edge for edge in edges if exclude_node not in edge.split('\t')]

def delete_test_data_files(fp):
    test_data_dir = os.path.join(os.path.dirname(__file__), f'data/GPKG/{fp}')
    
    if not os.path.exists(test_data_dir):
        print(f"The directory {test_data_dir} does not exist.")
        return

    for filename in os.listdir(test_data_dir):
        file_path = os.path.join(test_data_dir, filename)
        try:
            if os.path.isfile(file_path):
                if filename != 'ad_pre.txt':  # Skip deleting 'ad_pre.txt'
                    os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def read_nodes_and_edges_from_file(filepath):
    """Read edges from a file and return both edges and nodes."""
    with open(filepath, 'r') as file:
        lines = [line.strip() for line in file.readlines()]
    nodes = set()
    for line in lines:
        node1, node2 = line.split('\t')[:2]
        nodes.add(node1)
        nodes.add(node2)
    return nodes, set(lines)

def identify_new_nodes_edges(all_edges, training_edges):
    # Extract nodes from the training set
    training_nodes = set()
    for edge in training_edges:
        node1, _, node2 = edge.split('\t')  # Assuming tab-separated values
        training_nodes.add(node1)
        training_nodes.add(node2)

    # Identify edges with at least one new node
    new_nodes_edges = []
    for edge in all_edges:
        node1, _, node2 = edge.split('\t')
        if node1 not in training_nodes or node2 not in training_nodes:
            new_nodes_edges.append(edge)

    return new_nodes_edges

def save_splits(training_edges, edges_t, edges_si):#, edges_fi):
    ## save train
    base_output_dir = os.path.join(os.path.dirname(__file__), f'data/GPKG/non_aug')
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    write_lines_to_file(training_edges, os.path.join(base_output_dir, 'train.txt'))

    ## save transductive
    base_output_dir = os.path.join(os.path.dirname(__file__), f'data/GPKG/non_aug/transd')
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    write_lines_to_file(edges_t[0], os.path.join(base_output_dir, 'valid.txt'))
    write_lines_to_file(edges_t[1], os.path.join(base_output_dir, 'test.txt'))

    ## save semi-ind
    base_output_dir = os.path.join(os.path.dirname(__file__), f'data/GPKG/non_aug/semi_ind')
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    write_lines_to_file(edges_si[0], os.path.join(base_output_dir, 'inference.txt'))
    write_lines_to_file(edges_si[1], os.path.join(base_output_dir, 'valid.txt'))
    write_lines_to_file(edges_si[2], os.path.join(base_output_dir, 'test.txt'))

    ## save fully-ind
    #base_output_dir = os.path.join(os.path.dirname(__file__), f'data/GPKG/non_aug/fully_ind')
    #if not os.path.exists(base_output_dir):
    #    os.makedirs(base_output_dir)
    #
    #write_lines_to_file(edges_fi[0], os.path.join(base_output_dir, 'valid.txt'))
    #write_lines_to_file(edges_fi[1], os.path.join(base_output_dir, 'test.txt'))

def read_lines_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file]
    return lines


def find_non_overlapping_lines(file_path_1, file_path_2):
    """Find lines in file 2 that don't overlap with file 1."""
    lines_file_1 = set(read_lines_from_file(file_path_1))
    lines_file_2 = set(read_lines_from_file(file_path_2))
    
    # Get lines in file 2 that aren't in file 1
    non_overlapping_lines = lines_file_2 - lines_file_1
    return list(non_overlapping_lines), list(lines_file_1)


'''main logic'''
def split_kg(kg_filepath, train_fn = None):

    #1. Get train split
    if train_fn is not None:
        all_edges, training_edges = find_non_overlapping_lines(train_fn, kg_filepath)
        
    else:
        all_edges, training_edges = generate_train_split(kg_filepath)


    #2. Get validation and test for TRANSDUCTIVE SPLIT
    validation_edges_t, test_edges_t = create_transductive_splits(all_edges, training_edges, validation_ratio=0.15, test_ratio=0.15)

    #3. Get validation and test for SEMI-INDUCTIVE SPLIT
    specific_nodes = ['C0002395'] + ['DB' + str(i).zfill(3) for i in range(1, 100)]  # Example 'DB' nodes
    new_nodes_edges = identify_new_nodes_edges(all_edges, training_edges)
    inference_edges_si, validation_edges_si, test_edges_si = create_modified_splits(new_nodes_edges, 0.25, 0.25, specific_nodes) #create_semi_inductive_splits(all_edges, training_edges, new_nodes_edges, validation_ratio=0.15, test_ratio=0.15)
    
    #4. Get validation and test for FULLY-INDUCTIVE SPLIT
    #validation_edges_fi, test_edges_fi = create_fully_inductive_splits(all_edges, training_edges)

    #5. Save splits
    ## save train
    base_output_dir = os.path.join(os.path.dirname(__file__), f'data/GPKG/non_aug')
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    write_lines_to_file(training_edges, os.path.join(base_output_dir, 'train.txt'))

    ## save splits
    save_splits(training_edges, (validation_edges_t, test_edges_t), (inference_edges_si, validation_edges_si, test_edges_si))#, (validation_edges_fi, test_edges_fi))
    print('splits saved.')

''' generating train split function '''
def generate_train_split(kg_filepath):
    try:
        # Step 1: Read the file
        file_path = os.path.join(os.path.dirname(__file__), kg_filepath)
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file.readlines()]

        total_edges = len(lines)
        print(f"Total edges in the graph: {total_edges}")

        # Step 2: Identify all unique nodes
        all_nodes = set()
        for line in lines:
            node1, _, node2 = line.split('\t')
            all_nodes.add(node1)
            all_nodes.add(node2)

        print(f"Total unique nodes: {len(all_nodes)}")
        print('========================================')

        # Step 3: Shuffle and split edges for training
        random.seed(42)  # Ensure reproducibility
        random.shuffle(lines)  # Shuffle the lines to randomize edge selection

        # Assuming a 70-30 split for simplicity; adjust as needed
        train_split_index = int(0.7 * total_edges)
        training_edges = lines[:train_split_index]

        # Calculate statistics for the training split
        training_nodes = set()
        training_relations = set()
        for edge in training_edges:
            node1, relation, node2 = edge.split('\t')
            training_nodes.add(node1)
            training_nodes.add(node2)
            training_relations.add(relation)

        # Output statistics
        print(f"Training set created with {len(training_edges)} edges.")
        print(f"Unique nodes in training set: {len(training_nodes)}")
        print(f"Unique relations in training set: {len(training_relations)}")
        percent_of_data = (len(training_edges) / total_edges) * 100
        print(f"Percent of data allocated to training: {percent_of_data:.2f}%")
        print('======================================\n')

        return lines, training_edges  # or write to a file as needed

    except Exception as e:
        print(f"An error occurred: {e}")

def create_transductive_splits(all_edges, training_edges, validation_ratio=0.15, test_ratio=0.15):
    """
    Adjusts validation and test splits to ensure they only contain nodes present in the training set,
    suitable for a transductive setting.

    Parameters:
    - all_edges: List of all edges in the dataset.
    - training_edges: List of edges designated for training.
    - validation_ratio: Proportion of total edges for validation.
    - test_ratio: Proportion of total edges for testing.

    Returns:
    - validation_edges: List of edges for validation.
    - test_edges: List of edges for testing.
    """
    # Identify all nodes in the training set
    training_nodes = set()
    for edge in training_edges:
        node1, _, node2 = edge.split('\t')
        training_nodes.add(node1)
        training_nodes.add(node2)

    # Filter all_edges to only those that contain nodes present in the training set
    filtered_edges = [edge for edge in all_edges if edge.split('\t')[0] in training_nodes and edge.split('\t')[2] in training_nodes]

    # Exclude training edges from filtered_edges
    non_training_edges = list(set(filtered_edges) - set(training_edges))

    # Shuffle non-training edges for random split
    random.shuffle(non_training_edges)

    # Calculate split sizes
    validation_size = int(validation_ratio * len(non_training_edges))
    test_size = int(test_ratio * len(non_training_edges))

    # Create validation and test splits
    validation_edges = non_training_edges[:validation_size]
    test_edges = non_training_edges[validation_size:validation_size + test_size]

    # Statistics for validation and test sets
    validation_nodes = set()
    test_nodes = set()
    for edge in validation_edges:
        node1, _, node2 = edge.split('\t')
        validation_nodes.add(node1)
        validation_nodes.add(node2)
    for edge in test_edges:
        node1, _, node2 = edge.split('\t')
        test_nodes.add(node1)
        test_nodes.add(node2)

    print('=========TRANSDUCTIVE SETTING=========')
    print(f"Validation set: {len(validation_edges)} edges, {len(validation_nodes)} unique nodes")
    print(f"Test set: {len(test_edges)} edges, {len(test_nodes)} unique nodes")
    print(f"Total non-training edges used: {len(validation_edges) + len(test_edges)}")
    print(f"Percent of non-training data allocated to validation: {(len(validation_edges) / len(non_training_edges)) * 100:.2f}%")
    print(f"Percent of non-training data allocated to test: {(len(test_edges) / len(non_training_edges)) * 100:.2f}%")
    print('======================================\n')

    return validation_edges, test_edges

def create_semi_inductive_splits(training_edges, new_nodes_edges, validation_ratio=0.15, test_ratio=0.15):

    """
    Creates validation and test splits for a semi-inductive setting. The validation and test sets
    will include some overlap with the training set (in terms of nodes and edges) but also introduce
    new nodes and edges not present in the training set.

    Parameters:
    - all_edges: List of all edges in the dataset.
    - training_edges: List of edges designated for training.
    - new_nodes_edges: List of edges involving new nodes, not included in the training set.
    - validation_ratio: Proportion of total edges for validation, including new nodes.
    - test_ratio: Proportion of total edges for testing, including new nodes.

    Returns:
    - validation_edges: List of edges for validation, including edges from training and new edges.
    - test_edges: List of edges for testing, including edges from validation and new edges.
    """
    # Separate new_nodes_edges into validation and test portions
    random.shuffle(new_nodes_edges)
    total_new = len(new_nodes_edges)
    new_validation_size = int(validation_ratio * total_new)
    new_test_size = int(test_ratio * total_new)

    new_validation_edges = new_nodes_edges[:new_validation_size]
    new_test_edges = new_nodes_edges[new_validation_size:new_validation_size + new_test_size]

    # Combine new edges with a subset of training edges for validation and test sets
    # Ensure that the combination still respects the original validation and test ratios
    remaining_edges = list(set(all_edges) - set(training_edges) - set(new_nodes_edges))
    random.shuffle(remaining_edges)

    existing_validation_size = int(validation_ratio * len(remaining_edges))
    existing_test_size = int(test_ratio * len(remaining_edges))

    existing_validation_edges = remaining_edges[:existing_validation_size]
    existing_test_edges = remaining_edges[existing_validation_size:existing_validation_size + existing_test_size]

    # Final validation and test sets combine existing and new node edges
    validation_edges = existing_validation_edges + new_validation_edges
    test_edges = existing_test_edges + new_test_edges
    
    # Calculate the total non-training edges including new nodes for percentages
    total_non_training_edges = len(remaining_edges) + len(new_nodes_edges)

    # Statistics for validation and test sets
    validation_nodes = set(edge.split('\t')[0] for edge in validation_edges) | set(edge.split('\t')[2] for edge in validation_edges)
    test_nodes = set(edge.split('\t')[0] for edge in test_edges) | set(edge.split('\t')[2] for edge in test_edges)
    new_validation_nodes = set(edge.split('\t')[0] for edge in new_validation_edges) | set(edge.split('\t')[2] for edge in new_validation_edges)
    new_test_nodes = set(edge.split('\t')[0] for edge in new_test_edges) | set(edge.split('\t')[2] for edge in new_test_edges)

    print('=========SEMI-INDUCTIVE SETTING=========')
    print(f"Validation set: {len(validation_edges)} edges, {len(validation_nodes)} unique nodes (including {len(new_validation_nodes)} new nodes)")
    print(f"Test set: {len(test_edges)} edges, {len(test_nodes)} unique nodes (including {len(new_test_nodes)} new nodes)")
    print(f"Percent of non-training data allocated to Validation: {(len(validation_edges) / total_non_training_edges) * 100:.2f}%")
    print(f"Percent of non-training data allocated to Test: {(len(test_edges) / total_non_training_edges) * 100:.2f}%")
    print('========================================\n')

    return validation_edges, test_edges

import random

def minimize_set1_edges(nodes_to_cover, all_edges):
    covered_nodes = set()
    minimal_edges = []
    
    for edge in all_edges:
        nodes = set(edge.split('\t')[:2])
        if not nodes <= covered_nodes:  # If the edge adds new nodes
            minimal_edges.append(edge)
            covered_nodes.update(nodes)
            if covered_nodes >= nodes_to_cover:  # All required nodes are covered
                break
    return minimal_edges


def find_specific_edges(triples, specific_nodes):
    """Find triples that contain any of the specified nodes in head or tail."""
    return [triple for triple in triples if any(node in triple.split('\t') for node in specific_nodes)]

def minimize_set1_with_specific_nodes(all_triples, required_nodes, specific_triples):
    """Ensure Set 1 covers all required nodes and includes specific triples with minimal additional triples."""
    covered_nodes = {node for triple in specific_triples for node in triple.split('\t')[::2]}  # Heads and Tails
    set1_triples = specific_triples.copy()

    # Minimize triples while covering all required nodes
    for triple in all_triples:
        if triple in set1_triples:
            continue
        triple_nodes = set(triple.split('\t')[::2])  # Heads and Tails
        if not triple_nodes <= covered_nodes:
            set1_triples.append(triple)
            covered_nodes.update(triple_nodes)
        if covered_nodes >= required_nodes:
            break

    return set1_triples

def create_splits_with_all_db_nodes(new_triples, additional_triples, set2_ratio, set3_ratio):
    total_triples = len(new_triples) + len(additional_triples)
    combined_triples = new_triples + additional_triples

    # Identify 'DB'-prefixed nodes and 'C0002395' in both triple sets
    specific_nodes = {node for triple in combined_triples for node in triple.split('\t')[::2] if node.startswith('DB') or node == 'C0002395'}

    # Find triples including 'DB'-prefixed nodes and 'C0002395'
    specific_triples = find_specific_edges(combined_triples, specific_nodes)

    # Allocate nodes to Set 2 and Set 3 without specific criteria
    all_nodes = {node for triple in combined_triples for node in triple.split('\t')[::2]} - specific_nodes
    all_nodes_list = list(all_nodes)
    random.shuffle(all_nodes_list)

    # Split nodes for Set 2 and Set 3
    split_index = int(len(all_nodes_list) * set2_ratio)
    set2_nodes = set(all_nodes_list[:split_index])
    set3_nodes = set(all_nodes_list[split_index:split_index + int(len(all_nodes_list) * set3_ratio)])
    nodes_to_cover = set2_nodes.union(set3_nodes).union(specific_nodes)

    # Minimize Set 1 triples while covering all required nodes
    set1_triples = minimize_set1_with_specific_nodes(combined_triples, nodes_to_cover, specific_triples)

    # Derive Sets 2 and 3 triples from new_triples only, to respect the original sets' definitions
    set2_triples = [triple for triple in new_triples if set(triple.split('\t')[::2]) & set2_nodes]
    set3_triples = [triple for triple in new_triples if set(triple.split('\t')[::2]) & set3_nodes]

    # Output statistics
    print('=========CUSTOM SPLITS STATISTICS (With All DB Nodes in Set 1)=========')
    print(f"Total triples in combined triple sets: {total_triples}")
    print(f"Set 1 (Inf): {len(set1_triples)} triples (Includes 'C0002395' and all 'DB' nodes)")
    print(f"Set 2 (Valid): {len(set2_triples)} triples")
    print(f"Set 3 (Test): {len(set3_triples)} triples")
    print('======================================================================\n')

    return set1_triples, set2_triples, set3_triples

def create_modified_splits(all_edges, set2_ratio, set3_ratio, specific_nodes):
    new_nodes_edges = all_edges
    total_edges = len(new_nodes_edges)
    random.shuffle(new_nodes_edges)

    # Extract all nodes and identify specific nodes
    all_nodes = {node for edge in new_nodes_edges for node in edge.split('\t')[:2]}
    
    # Allocate nodes to Set 2 and Set 3
    all_nodes_list = [node for node in all_nodes if node not in specific_nodes]
    random.shuffle(all_nodes_list)
    split_index_2 = int(len(all_nodes_list) * set2_ratio)
    split_index_3 = split_index_2 + int(len(all_nodes_list) * set3_ratio)

    set2_nodes = set(all_nodes_list[:split_index_2]).union(specific_nodes)
    set3_nodes = set(all_nodes_list[split_index_2:split_index_3]).union(specific_nodes)
    nodes_to_cover = set2_nodes.union(set3_nodes)

    # Pre-select edges that include 'C0002395' and 'DB'-prefixed nodes
    specific_edges = find_edges_for_specific_nodes(new_nodes_edges, specific_nodes)

    # Generate Sets 2 and 3
    set2_edges = [edge for edge in new_nodes_edges if set(edge.split('\t')[:2]) & set2_nodes]
    set3_edges = [edge for edge in new_nodes_edges if set(edge.split('\t')[:2]) & set3_nodes]

    # Minimize Set 1 by including specific edges and adding more to cover all nodes
    set1_edges = minimize_additional_edges(nodes_to_cover, new_nodes_edges, specific_edges)

    # Statistics calculation
    percent_set1 = (len(set1_edges) / total_edges) * 100 #total edges indicate total NEW edges
    percent_set2 = (len(set2_edges) / total_edges) * 100
    percent_set3 = (len(set3_edges) / total_edges) * 100

    print('=========MODIFIED SEMI-IND SPLITS STATISTICS=========')
    #print(f"Set 1 (Inference): {len(set1_edges)} edges, {len(unique_nodes_in_set1)} unique nodes (Superset containing entities from Set 2 and Set 3)")
    #print(f"Set 2 (Valid): {len(set2_edges)} edges, {len(unique_nodes_in_set2)} unique nodes (Entities not in Set 3)")
    #print(f"Set 3 (Test): {len(set3_edges)} edges, {len(unique_nodes_in_set3)} unique nodes (Entities not in Set 2)")
    print(f"Total edges (triples) in new_nodes_edges: {total_edges}")
    print(f"Set 1: {len(set1_edges)} edges, {percent_set1:.2f}% of total")
    print(f"Set 2: {len(set2_edges)} edges, {percent_set2:.2f}% of total")
    print(f"Set 3: {len(set3_edges)} edges, {percent_set3:.2f}% of total")
    print('=============================================\n')

    return set1_edges, set2_edges, set3_edges



def create_fully_inductive_splits(all_edges, training_edges):
    """
    Adjusts validation and test splits to ensure they are completely disjoint from the training set,
    in terms of both nodes and edges, suitable for a fully-inductive setting. This revision also ensures
    that validation and test sets are disjoint from each other.

    Parameters:
    - all_edges: List of all edges in the dataset.
    - training_edges: List of edges designated for training.

    Returns:
    - validation_edges: List of edges for validation, disjoint from training set.
    - test_edges: List of edges for testing, disjoint from training and validation sets.
    """
    # Extract nodes from the training set
    training_nodes = set()
    for edge in training_edges:
        node1, _, node2 = edge.split('\t')  # Assuming tab-separated values
        training_nodes.add(node1)
        training_nodes.add(node2)

    # Filter for edges that do not involve training nodes
    non_training_edges = [edge for edge in all_edges if edge.split('\t')[0] not in training_nodes and edge.split('\t')[2] not in training_nodes]

    # Shuffle to ensure randomness
    random.shuffle(non_training_edges)

    # Initialize sets to track validation and test nodes
    validation_nodes = set()
    test_nodes = set()
    validation_edges = []
    test_edges = []

    # Distribute non-training edges between validation and test sets ensuring node disjointness
    for edge in non_training_edges:
        node1, _, node2 = edge.split('\t')
        if node1 not in validation_nodes and node2 not in validation_nodes and \
           node1 not in test_nodes and node2 not in test_nodes:
            # Arbitrarily assign to validation or test if both sets are options
            if len(validation_edges) <= len(test_edges):
                validation_edges.append(edge)
                validation_nodes.update([node1, node2])
            else:
                test_edges.append(edge)
                test_nodes.update([node1, node2])
        elif node1 not in test_nodes and node2 not in test_nodes:
            # Can safely add to test set
            test_edges.append(edge)
            test_nodes.update([node1, node2])
        elif node1 not in validation_nodes and node2 not in validation_nodes:
            # Can safely add to validation set
            validation_edges.append(edge)
            validation_nodes.update([node1, node2])

    print('=========FULLY-INDUCTIVE SETTING=========')
    print(f"Validation set: {len(validation_edges)} edges, {len(validation_nodes)} unique nodes")
    print(f"Test set: {len(test_edges)} edges, {len(test_nodes)} unique nodes")
    print(f"Percent of non-training data allocated to Validation: {(len(validation_edges) / len(non_training_edges)) * 100:.2f}%")
    print(f"Percent of non-training data allocated to Test: {(len(test_edges) / len(non_training_edges)) * 100:.2f}%")
    print('========================================\n')

    return validation_edges, test_edges



############TESTS##############
''' testing functions (testing whether all nodes in the validation and test sets also appear in the training set) '''
def test_splits_transductive(train_filepath, valid_filepath, test_filepath):
    # Read nodes from each file
    train_nodes = read_nodes_from_file(train_filepath)
    valid_nodes = read_nodes_from_file(valid_filepath)
    test_nodes = read_nodes_from_file(test_filepath)

    # Check if all nodes in valid and test sets are also in the train set
    valid_diff = valid_nodes.difference(train_nodes)
    test_diff = test_nodes.difference(train_nodes)

    if len(valid_diff) == 0 and len(test_diff) == 0:
        print("✔ Test Passed: All nodes in the validation and test sets also appear in the training set.")
    else:
        print("✘ Test Failed")
        if len(valid_diff) > 0:
            print(f"Nodes in validation set not found in training set: {len(valid_diff)}")
        if len(test_diff) > 0:
            print(f"Nodes in test set not found in training set: {len(test_diff)}")

'''Includes checks for:
- New nodes in both validation and test sets.
- Additional new nodes in the test set that are not in the validation set.
- The ratio of new nodes in the test set being larger than in the validation set.'''
def test_splits_semi_inductive(train_filepath, valid_filepath, test_filepath):
    
    # Read nodes from each file
    train_nodes = read_nodes_from_file(train_filepath)
    valid_nodes = read_nodes_from_file(valid_filepath)
    test_nodes = read_nodes_from_file(test_filepath)

    # Identify new nodes in validation and test sets
    valid_new_nodes = valid_nodes.difference(train_nodes)
    test_new_nodes = test_nodes.difference(train_nodes)

    # Additional nodes in test set not in validation set
    additional_test_nodes = test_new_nodes.difference(valid_new_nodes)

    if valid_new_nodes or test_new_nodes:
        print("✔ Test Passed: There are new nodes in validation and/or test sets.")
        if valid_new_nodes:
            print(f"New nodes in validation set: {len(valid_new_nodes)}")
        if test_new_nodes:
            print(f"New nodes in test set: {len(test_new_nodes)}")
        if additional_test_nodes:
            print(f"Additional new nodes in test set that are not in validation set: {len(additional_test_nodes)}")
        #if len(test_new_nodes) > len(valid_new_nodes):
        #    print("✔ Test Passed: The number of new nodes in the test set is larger than in the validation set.")
        #else:
        #    print("✘ Test Failed: The number of new nodes in the test set is not larger than in the validation set.")
    else:
        print("✘ Test Failed: No new nodes in validation and test sets. This does not align with semi-inductive setup.")

def test_splits_fully_inductive(train_filepath, valid_filepath, test_filepath):
    train_nodes, train_edges = read_nodes_and_edges_from_file(train_filepath)
    valid_nodes, valid_edges = read_nodes_and_edges_from_file(valid_filepath)
    test_nodes, test_edges = read_nodes_and_edges_from_file(test_filepath)

    # Test 1: Disjoint Node Sets
    if train_nodes.isdisjoint(valid_nodes) and train_nodes.isdisjoint(test_nodes) and valid_nodes.isdisjoint(test_nodes):
        print("✔ Node sets are disjoint.")
    else:
        print("✘ Node sets are not disjoint.")

    # Test 2: Correct Edge Allocation
    correct_train_allocation = all(node1 in train_nodes and node2 in train_nodes for node1, node2 in (edge.split('\t')[:2] for edge in train_edges))
    correct_valid_allocation = all(node1 in valid_nodes and node2 in valid_nodes for node1, node2 in (edge.split('\t')[:2] for edge in valid_edges))
    correct_test_allocation = all(node1 in test_nodes and node2 in test_nodes for node1, node2 in (edge.split('\t')[:2] for edge in test_edges))
    
    if correct_train_allocation and correct_valid_allocation and correct_test_allocation:
        print("✔ All edges are correctly allocated.")
    else:
        print("✘ Some edges are incorrectly allocated.")

    # Test 3: Non-Empty Sets (Optional, based on requirements)
    if train_edges and valid_edges and test_edges:
        print("✔ All sets contain edges.")
    else:
        print("✘ One or more sets are empty.")

def run_test(fp):
    base_dir = f'data/GPKG/{fp}'  # Replace with the path where your txt files are stored
    train_filepath = os.path.join(base_dir, 'train.txt')
    valid_filepath = os.path.join(base_dir, 'valid.txt')
    test_filepath = os.path.join(base_dir, 'test.txt')

    print('\n______testing transductive splits:______')
    test_splits_transductive(train_filepath, os.path.join(base_dir, 'transd/valid.txt'), os.path.join(base_dir, 'transd/test.txt'))
    print('\n______testing semi-ind splits:______')
    test_splits_semi_inductive(train_filepath, os.path.join(base_dir, 'semi_ind/valid.txt'), os.path.join(base_dir, 'semi_ind/test.txt'))
    #print('\n______testing fully-ind splits:______')
    #test_splits_fully_inductive(train_filepath, os.path.join(base_dir, 'fully_ind/valid.txt'), os.path.join(base_dir, 'fully_ind/test.txt'))
    #print('tests ran.')

''' main '''

''' split_kg uses the same training set to create the trands and semi-ind splits '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split a knowledge graph into training, validation, and test sets.")
    parser.add_argument('-kg_filepath', type=str, default=None, help="Path to the knowledge graph file.")
    parser.add_argument('-train_filepath', type=str, default=None, help="Path to the training graph file.")

    args = parser.parse_args()
    kg_filepath = args.kg_filepath
    train_filepath = args.train_filepath

    delete_test_data_files('non_aug')
    delete_test_data_files('non_aug/transd')
    delete_test_data_files('non_aug/semi_ind')
    #delete_test_data_files('non_aug/fully_ind')
    print('files deleted.')
    #quit()

    if kg_filepath:
        print(f'Splitting {kg_filepath}...\n========================================')
        print(f'Train set: {train_filepath}...\n========================================')
        split_kg(kg_filepath, train_filepath)

    else:
        split_kg(kg_filepath, train_filepath)

    run_test('non_aug')
