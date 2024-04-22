import os
import sys
import math
import pprint
from itertools import islice
import copy
import itertools

import torch
import torch_geometric as pyg
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util
from ultra.models import Ultra
from tqdm import tqdm

import datetime

separator = ">" * 30
line = "-" * 30

torch.cuda.set_device(2)
torch.cuda.empty_cache()

def train_and_validate(cfg, model, train_data, valid_data, device, logger, filtered_data=None, batch_per_epoch=None):
    if cfg.train.num_epoch == 0:
        return
    
    # Early stopping parameters
    best_val_loss = float('inf')
    best_avg_loss = float('inf')
    best_model_state = None  # Initialize to None or the initial model state
    epochs_no_improve = 0
    patience = 5
    stop_training = False

    world_size = util.get_world_size()
    rank = util.get_rank()

    train_triplets = torch.cat([train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler)

    batch_per_epoch = batch_per_epoch or len(train_loader)

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    num_params = sum(p.numel() for p in model.parameters())
    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1

    batch_id = 0
    for i in range(0, cfg.train.num_epoch, step):
        parallel_model.train()
        
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            sampler.set_epoch(epoch)
            for batch in islice(train_loader, batch_per_epoch):
                batch = tasks.negative_sampling(train_data, batch, cfg.task.num_negative,
                                                strict=cfg.task.strict_negative)
                pred = parallel_model(train_data, batch)
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                neg_weight = torch.ones_like(pred)
                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
                else:
                    neg_weight[:, 1:] = 1 / cfg.task.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % loss)
                losses.append(loss.item())
                batch_id += 1

            # After each epoch, check for improvement
            if util.get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning("average binary cross entropy: %g" % avg_loss)
                
                # Check if the average loss improved
                if avg_loss < best_avg_loss:
                    best_avg_loss = avg_loss
                    epochs_no_improve = 0
                    # Optionally, save the model as the best so far
                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    epochs_no_improve += 1
                
                # Early stopping check
                if epochs_no_improve >= patience:
                    logger.warning("Early stopping triggered. Restoring best model...")
                    model.load_state_dict(best_model_state)
                    stop_training = True
                    ##
                    epoch = min(cfg.train.num_epoch, i + step)
                    if rank == 0:
                        logger.warning("-> Save checkpoint to model_epoch_%d.pth" % epoch)
                        state = {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict()
                        }
                        torch.save(state, "model_epoch_%d.pth" % epoch)
                    util.synchronize()
                    if rank == 0:
                        logger.warning(separator)
                        logger.warning("Evaluate on valid")
                    result = test(cfg, model, valid_data, filtered_data=filtered_data, device=device, logger=logger)
                    if result > best_result:
                        best_result = result
                        best_epoch = epoch
                    ##
                    break  # Exit the epoch loop, but continue with post-loop code

            ##########    
        if stop_training:
            break  # Clean exit from the loop if early stopping was triggered

        epoch = min(cfg.train.num_epoch, i + step)
        if rank == 0:
            logger.warning("-> Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()
        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        result = test(cfg, model, valid_data, filtered_data=filtered_data, device=device, logger=logger)
        if result > best_result:
            best_result = result
            best_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()

@torch.no_grad()
def infer(cfg, model, test_data, device, logger, filtered_data=None, return_metrics=False):
    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, shuffle=False)  # Ensure consistent order

    model.eval()
    triplet_to_score = {}  # Dictionary to map triplets to their actual tail scores

    triplet_scores = []

    for batch_idx, batch in enumerate(test_loader):
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        t_pred = model(test_data, t_batch)
        # No need to process h_batch for tail scores

        # Assuming batch contains [head, tail, relation] triplets,
        # and each row in t_pred corresponds to the scores for all possible tails for a given head, relation pair.
        for i, (head, tail, relation) in enumerate(batch):
            # Assuming tail is the index in the list of all entities that corresponds to the actual tail entity.
            # This index should match the column in t_pred that corresponds to this tail entity.
            actual_tail_score = t_pred[i, tail.item()]  # Convert tail to index if not already
            
            # Store the score using the triplet as a key (convert to tuple to make it hashable)
            triplet_key = (head.item(), relation.item(), tail.item())  # Convert to items if they are tensors
            triplet_to_score[triplet_key] = actual_tail_score.item()  # Convert to Python scalar

    # At this point, triplet_to_score contains the scores for the actual tails
    # You can print or process these scores further as needed
    for triplet, score in triplet_to_score.items():
        #print(f"Triplet: {triplet}, Score: {score}")
        triplet_scores.append(score)
    
    print('len of triplet_scores:', len(triplet_scores))

    # save ranks of test triplets to new file
    # Open the original test file and a new file for output
    with open('/data/ad_pre_new.tsv', 'r') as test_file, open('/data/test_with_ranks.txt', 'w') as output_file:
        for line, score in zip(test_file, triplet_scores):
            # Strip newline characters from the original line and append the rank
            output_line = f"{line.strip()}\t{score}\n"
            output_file.write(output_line)

    print('SAVED')

@torch.no_grad()
def test(cfg, model, test_data, device, logger, filtered_data=None, return_metrics=False, rank_tails=False, save_ranks=False):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank, shuffle=False)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)
    
    model.eval()

    rankings = []
    num_negatives = []
    tail_scores = []
    tail_rankings, num_tail_negs = [], []  # for explicit tail-only evaluation needed for 5 datasets
    i = 0
    for batch in tqdm(test_loader, desc="Processing test batches", leave=True, ncols=100, unit="batch"): ##
    
        no_of_triples = len(test_triplets)

        t_batch, h_batch = tasks.all_negative(test_data, batch, no_of_triples=None) #make no_of_triples None for general test (IMPORTANT)

        t_pred = model(test_data, t_batch)
        h_pred = model(test_data, h_batch)

        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)

        pos_h_index, pos_t_index, pos_r_index = batch.t()
        
        
        t_ranking, t_scores = tasks.compute_ranking(t_pred, pos_t_index, t_mask)        
        h_ranking, _ = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

        tail_rankings += [t_ranking]
        tail_scores.append(t_scores)
        num_tail_negs += [num_t_negative]
    
    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)

    # ugly repetitive code for tail-only ranks processing
    tail_ranking = torch.cat(tail_rankings)
    num_tail_neg = torch.cat(num_tail_negs)
    all_size_t = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_t[rank] = len(tail_ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_t, op=dist.ReduceOp.SUM)

    # obtaining all ranks 
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative

    # the same for tails-only ranks
    cum_size_t = all_size_t.cumsum(0)
    all_ranking_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_ranking_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = tail_ranking
    all_num_negative_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_num_negative_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = num_tail_neg
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_ranking_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative_t, op=dist.ReduceOp.SUM)

    metrics = {}
    if rank == 0:
        for metric in tqdm(cfg.task.metric, desc="Processing metrics", leave=True, unit="metric"): ###
            #if "-tail" in metric:
            if rank_tails:
                #_metric_name, direction = metric.split("-")
                #if direction != "tail":
                #    raise ValueError("Only tail metric is supported in this mode")
                _ranking = all_ranking_t
                _num_neg = all_num_negative_t
            else:
                _ranking = all_ranking 
                _num_neg = all_num_negative 
            _metric_name = metric

            if _metric_name == "mr":
                score = _ranking.float().mean()
            elif _metric_name == "mrr":
                score = (1 / _ranking.float()).mean()
            elif _metric_name.startswith("hits@"):
                values = _metric_name[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (_ranking - 1).float() / _num_neg
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample - 1 negatives
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (_ranking <= threshold).float().mean()

            logger.warning("%s: %g" % (metric, score))
            metrics[metric] = score
    
    if rank_tails:
        all_ranking = _ranking

        flattened_tail_scores = list(itertools.chain.from_iterable(tail_scores))
        '''
        if rank_tails:
        flattened_tail_rankings = torch.cat(tail_rankings).flatten()
        flattened_tail_scores = list(itertools.chain.from_iterable(tail_scores))
        assert len(flattened_tail_rankings) == len(flattened_tail_scores)
        
        for r, s in zip(flattened_tail_rankings, flattened_tail_scores):
            if r.item() == 1:
                print(f"Most confident score for triple with ranking {r.item()}: {s}")
        '''

        if save_ranks:
            # Read the existing content of the file
            with open('/data/ad_pre_newer.txt', 'r') as file:
                triple_lines = file.readlines()

            # Assuming `all_scores` exists and matches `all_ranking` in length
            if len(triple_lines) != len(all_ranking.float()) or len(triple_lines) != len(flattened_tail_scores):
                print("The number of lines in the file and the tensor lengths do not match.")
            else:
                # Append each tensor value (ranking and score) to the end of each line
                modified_lines = [f"{line.strip()}\t{r}\t{s}\n" for line, r, s in zip(triple_lines, all_ranking.float().tolist(), flattened_tail_scores)]

                now = datetime.datetime.now()
                formatted_now = now.strftime("%Y%m%d_%H%M%S")

                # Write the modified lines back to the file or a new file
                with open(f'/data/test_with_ranks_{formatted_now}.txt', 'w') as modified_file:
                    modified_file.writelines(modified_lines)

                print(f'new file SAVED-> /data/test_with_ranks_{formatted_now}.txt')

    mrr = (1 / all_ranking.float()).mean()

    return mrr if not return_metrics else metrics


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    
    task_name = cfg.task["name"]
    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)
    
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)

    model = Ultra(
        rel_model_cfg=cfg.model.relation_model,
        entity_model_cfg=cfg.model.entity_model,
    )

    if "checkpoint" in cfg and cfg.checkpoint is not None:
        print(cfg.checkpoint)

        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    model = model.to(device)
    
    if task_name == "InductiveInference":
        # filtering for inductive datasets
        # Grail, MTDEA, HM datasets have validation sets based off the training graph
        # ILPC, Ingram have validation sets from the inference graph
        # filtering dataset should contain all true edges (base graph + (valid) + test) 
        if "ILPC" in cfg.dataset['class'] or "Ingram" in cfg.dataset['class']:
            # add inference, valid, test as the validation and test filtering graphs
            full_inference_edges = torch.cat([valid_data.edge_index, valid_data.target_edge_index, test_data.target_edge_index], dim=1)
            full_inference_etypes = torch.cat([valid_data.edge_type, valid_data.target_edge_type, test_data.target_edge_type])
            test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)
            val_filtered_data = test_filtered_data
        else:
            # test filtering graph: inference edges + test edges
            full_inference_edges = torch.cat([test_data.edge_index, test_data.target_edge_index], dim=1)
            full_inference_etypes = torch.cat([test_data.edge_type, test_data.target_edge_type])
            test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)

            # validation filtering graph: train edges + validation edges
            val_filtered_data = Data(
                edge_index=torch.cat([train_data.edge_index, valid_data.target_edge_index], dim=1),
                edge_type=torch.cat([train_data.edge_type, valid_data.target_edge_type])
            )
    else:
        # for transductive setting, use the whole graph for filtered ranking
        filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=dataset[0].num_nodes)
        val_filtered_data = test_filtered_data = filtered_data
    
    val_filtered_data = val_filtered_data.to(device)
    test_filtered_data = test_filtered_data.to(device)
    
    train_and_validate(cfg, model, train_data, valid_data, filtered_data=val_filtered_data, device=device, batch_per_epoch=cfg.train.batch_per_epoch, logger=logger)
    print('>> out of train_and_validate <<')
    is_this_a_case_study = args.this_is_a_case_study
    print(f'This is a case study: {is_this_a_case_study}')

    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    test(cfg, model, test_data, filtered_data=val_filtered_data, device=device, logger=logger, rank_tails=is_this_a_case_study) #make rank_tails False for general test 

    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    test(cfg, model, test_data, filtered_data=test_filtered_data, device=device, logger=logger, rank_tails=is_this_a_case_study, save_ranks=is_this_a_case_study) #make rank_tails False for general test 