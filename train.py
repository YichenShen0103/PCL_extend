import os
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import torchvision
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from modules import transform, resnet, network, contrastive_loss, attention
from utils import yaml_config_hook, save_model
from utils.query_store import (
    ConstraintStore,
    EpochQuerySelector,
    GlobalQuerySelector,
    MemoryBank,
    global_pair_feature_loss,
)
from custom_datasets import *
from evaluation import evaluation


def inference(loader, model, device):
    model.eval()
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to("cuda")
        with torch.no_grad():
            c = model.forward_cluster(x, device)
        c = c.detach()
        feature_vector.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    return feature_vector, labels_vector


def eval(args, data_loader, model, device, epoch="final"):
    X, Y = inference(data_loader, model, device)
    if args.dataset == "CIFAR-10":  # super-class
        super_label = (
            [0, 1, 0, 1, 1, 1, 0, 1, 0, 1]
            if args.label_strategy
            else [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
        )  # 根据参数值设定类别
        for i in range(len(Y)):
            Y[i] = super_label[Y[i]]
    elif args.dataset == "CIFAR-100":  # super-class
        if args.label_strategy == 0:
            super_label = [
                0,
                3,
                3,
                3,
                1,
                2,
                1,
                0,
                0,
                2,
                0,
                3,
                2,
                2,
                1,
                1,
                2,
                2,
                1,
                1,
                2,
                3,
                0,
                3,
                0,
                2,
                0,
                1,
                0,
                0,
                3,
                1,
                3,
                1,
                1,
                3,
                0,
                2,
                1,
                0,
                2,
                0,
                1,
                1,
                1,
                1,
                3,
                1,
                1,
                2,
                1,
                1,
                1,
                0,
                0,
                3,
                1,
                0,
                2,
                1,
                2,
                0,
                0,
                1,
                3,
                3,
                3,
                3,
                2,
                2,
                0,
                2,
                3,
                3,
                1,
                3,
                2,
                1,
                1,
                0,
                1,
                2,
                0,
                0,
                2,
                2,
                0,
                2,
                1,
                2,
                2,
                3,
                0,
                3,
                2,
                3,
                1,
                3,
                3,
                0,
            ]
        else:
            super_label = [
                2,
                1,
                0,
                0,
                0,
                3,
                1,
                1,
                3,
                3,
                3,
                0,
                3,
                3,
                1,
                0,
                3,
                3,
                1,
                0,
                3,
                0,
                3,
                2,
                1,
                3,
                1,
                1,
                3,
                1,
                0,
                0,
                1,
                2,
                0,
                0,
                1,
                3,
                0,
                3,
                3,
                3,
                0,
                0,
                1,
                1,
                0,
                2,
                3,
                2,
                1,
                2,
                2,
                2,
                2,
                0,
                2,
                2,
                3,
                2,
                2,
                3,
                2,
                0,
                0,
                1,
                0,
                2,
                3,
                3,
                2,
                2,
                0,
                1,
                1,
                0,
                3,
                1,
                1,
                1,
                1,
                3,
                2,
                2,
                3,
                3,
                3,
                3,
                0,
                3,
                3,
                1,
                2,
                1,
                3,
                0,
                2,
                0,
                0,
                1,
            ]
        for i in range(len(Y)):
            Y[i] = super_label[Y[i]]
    elif args.dataset == "ImageNet-10":  # super-class
        super_label = (
            [0, 1, 1, 1, 0, 1, 0, 0, 0, 1]
            if args.label_strategy
            else [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        )
        for i in range(len(Y)):
            Y[i] = super_label[Y[i]]
    elif args.dataset == "ImageNet-dogs":  # super-class
        super_label = (
            [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0]
            if args.label_strategy
            else [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0]
        )
        for i in range(len(Y)):
            Y[i] = super_label[Y[i]]
    nmi, ari, f, acc = evaluation.evaluate(Y, X)
    print(
        "epoch = {}, NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}".format(
            epoch, nmi, ari, f, acc
        )
    )


def build_super_labels(raw_labels, mapping):
    return torch.tensor([mapping[l] for l in raw_labels], dtype=torch.long)


def build_batch_queries(indices, constraint_store, device):
    if isinstance(indices, torch.Tensor):
        indices_list = indices.cpu().tolist()
    else:
        indices_list = [int(i) for i in indices]
    return constraint_store.build_batch_matrix(indices_list, device)


def update_memory_banks(feature_bank, cluster_bank, indices, z_i, z_j, c_i, c_j):
    avg_z = 0.5 * (z_i.detach() + z_j.detach())
    avg_c = 0.5 * (c_i.detach() + c_j.detach())
    feature_bank.update(indices, avg_z)
    cluster_bank.update(indices, avg_c)


def maybe_request_global_query(selector, super_labels, constraint_store, query_state):
    if query_state["remaining"] <= 0:
        return False
    if not selector.ready():
        return False
    pair = selector.propose_pair(query_state["remaining"])
    if pair is None:
        return False
    a, b = pair
    is_same = bool(super_labels[a] == super_labels[b])
    constraint_store.add(a, b, is_same)
    query_state["remaining"] -= 1
    return True


def run_epoch_queries(selector, super_labels, constraint_store, query_state):
    if selector is None:
        return 0, []
    if query_state["remaining"] <= 0:
        return 0, []
    max_pairs = min(selector.queries_per_epoch, query_state["remaining"])
    if max_pairs <= 0:
        return 0, []
    pairs, debug_records = selector.propose_epoch_pairs(max_pairs)
    added = 0
    logs = []
    for idx, (a, b) in enumerate(pairs):
        if query_state["remaining"] <= 0:
            break
        is_same = bool(super_labels[a] == super_labels[b])
        constraint_store.add(a, b, is_same)
        query_state["remaining"] -= 1
        added += 1
        hardness = float("nan")
        if idx < len(debug_records):
            hardness = float(debug_records[idx].get("hardness", float("nan")))
        logs.append(
            {
                "anchor": a,
                "partner": b,
                "hardness": hardness,
                "same_cluster": int(is_same),
            }
        )
    return added, logs


def train(
    data_loader,
    model,
    optimizer,
    criterion_instance,
    criterion_cluster,
    constraint_store,
    feature_bank,
    cluster_bank,
    selector,
    super_labels,
    query_state,
    args,
    current_epoch,
):
    device = next(model.parameters()).device
    loss_epoch = 0.0
    scheduler_mode = getattr(args, "query_scheduler", "batch")
    use_batch_queries = isinstance(scheduler_mode, str) and scheduler_mode.lower() == "batch"
    for step, ((x_i, x_j), label, indices) in enumerate(data_loader):
        batch_size = x_i.size(0)
        optimizer.zero_grad()
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        indices = indices.long().to(device)
        flag = torch.ones(batch_size, device=device)

        z_i, z_j, c_i, c_j = model(x_i, x_j, flag)

        batch_queries = build_batch_queries(indices.cpu(), constraint_store, device)
        q_truth = batch_queries.clone()
        positives = torch.count_nonzero(q_truth == 1)
        proportion = max(4 * args.batch_size / (positives + 1), 2)

        loss_instance = criterion_instance(z_i, z_j, q_truth, batch_queries, proportion)
        loss_cluster, c1, c2, c3 = criterion_cluster(
            c_i, c_j, q_truth, batch_queries, proportion
        )
        loss = loss_instance + loss_cluster

        if args.global_pairs_per_batch > 0:
            ready_mask = (
                (feature_bank.ready_mask & cluster_bank.ready_mask).detach().cpu()
            )
            pairs = constraint_store.sample_pairs(
                args.global_pairs_per_batch, ready_mask=ready_mask
            )
            global_loss = global_pair_feature_loss(
                pairs, feature_bank, cluster_bank, device, args.global_pair_margin
            )
            if global_loss is not None:
                loss = loss + args.global_pair_weight * global_loss

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        optimizer.step()

        update_memory_banks(feature_bank, cluster_bank, indices, z_i, z_j, c_i, c_j)

        if use_batch_queries and current_epoch >= args.global_query_warmup:
            maybe_request_global_query(
                selector, super_labels, constraint_store, query_state
            )

        if step % 100 == 0:
            print(
                f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}"
            )
            print(
                f"Step [{step}/{len(data_loader)}]\t loss: {c1.item()}\t acc_loss: {c2.item()}\t ne_loss: {c3.item()}"
            )
        loss_epoch += loss.item()

    return loss_epoch


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    query_file = args.model_path + "/query.npy"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_trans = transform.Transforms_train(
        size=args.image_size,
        s=0.5,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    test_trans = transform.Transforms_test(
        size=args.image_size,
        s=0.5,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if args.dataset == "CIFAR-10":
        base_train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=False,
            train=True,
            transform=train_trans,
        )
        train_labels = base_train_dataset.targets
        train_dataset = IndexedDataset(base_train_dataset)
        train_dataset_test = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=False,
            train=True,
            transform=test_trans,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=False,
            train=False,
            transform=test_trans,
        )
        super_label = (
            [0, 1, 0, 1, 1, 1, 0, 1, 0, 1]
            if args.label_strategy
            else [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
        )  # 根据参数值设定类别
        class_num = 2
    elif args.dataset == "CIFAR-100":
        base_train_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=False,
            train=True,
            transform=train_trans,
        )
        train_labels = base_train_dataset.targets
        train_dataset = IndexedDataset(base_train_dataset)
        train_dataset_test = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=False,
            train=True,
            transform=test_trans,
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=False,
            train=False,
            transform=test_trans,
        )
        if args.label_strategy == 0:
            super_label = [
                0,
                3,
                3,
                3,
                1,
                2,
                1,
                0,
                0,
                2,
                0,
                3,
                2,
                2,
                1,
                1,
                2,
                2,
                1,
                1,
                2,
                3,
                0,
                3,
                0,
                2,
                0,
                1,
                0,
                0,
                3,
                1,
                3,
                1,
                1,
                3,
                0,
                2,
                1,
                0,
                2,
                0,
                1,
                1,
                1,
                1,
                3,
                1,
                1,
                2,
                1,
                1,
                1,
                0,
                0,
                3,
                1,
                0,
                2,
                1,
                2,
                0,
                0,
                1,
                3,
                3,
                3,
                3,
                2,
                2,
                0,
                2,
                3,
                3,
                1,
                3,
                2,
                1,
                1,
                0,
                1,
                2,
                0,
                0,
                2,
                2,
                0,
                2,
                1,
                2,
                2,
                3,
                0,
                3,
                2,
                3,
                1,
                3,
                3,
                0,
            ]
        else:
            super_label = [
                2,
                1,
                0,
                0,
                0,
                3,
                1,
                1,
                3,
                3,
                3,
                0,
                3,
                3,
                1,
                0,
                3,
                3,
                1,
                0,
                3,
                0,
                3,
                2,
                1,
                3,
                1,
                1,
                3,
                1,
                0,
                0,
                1,
                2,
                0,
                0,
                1,
                3,
                0,
                3,
                3,
                3,
                0,
                0,
                1,
                1,
                0,
                2,
                3,
                2,
                1,
                2,
                2,
                2,
                2,
                0,
                2,
                2,
                3,
                2,
                2,
                3,
                2,
                0,
                0,
                1,
                0,
                2,
                3,
                3,
                2,
                2,
                0,
                1,
                1,
                0,
                3,
                1,
                1,
                1,
                1,
                3,
                2,
                2,
                3,
                3,
                3,
                3,
                0,
                3,
                3,
                1,
                2,
                1,
                3,
                0,
                2,
                0,
                0,
                1,
            ]
        class_num = 4
    elif args.dataset == "ImageNet-10":
        train_data, test_data = ImageNetData(
            args.dataset_dir + "/imagenet-10", args.seed, split_ratio=0.2
        ).get_data()
        base_train_dataset = ImageNet(train_data, train_trans)
        train_labels = base_train_dataset.labels()
        train_dataset = IndexedDataset(base_train_dataset)
        train_dataset_test = ImageNet(train_data, test_trans)
        test_dataset = ImageNet(test_data, test_trans)
        super_label = (
            [0, 1, 1, 1, 0, 1, 0, 0, 0, 1]
            if args.label_strategy
            else [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        )
        class_num = 2
    else:
        raise NotImplementedError
    train_super_labels = build_super_labels(train_labels, super_label)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2,
        num_workers=args.workers,
    )
    train_data_test_loader = DataLoader(
        train_dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2,
        num_workers=args.workers,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2,
        num_workers=args.workers,
    )

    res = resnet.get_resnet(args.resnet)
    my_atten = attention.MultiHeadSelfAttention(res.rep_dim)
    # my_atten = None
    model = network.BaNet(res, my_atten, args.feature_dim, class_num)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to("cuda")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    loss_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion_instance = contrastive_loss.InstanceLoss(
        args.batch_size, args.instance_temperature, loss_device
    ).to(loss_device)
    criterion_cluster = contrastive_loss.ClusterLoss(
        args.batch_size, class_num, args.cluster_temperature, loss_device
    ).to(loss_device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_num = len(train_data_loader)

    constraint_store = ConstraintStore(len(train_dataset))
    feature_bank = MemoryBank(
        len(train_dataset), args.feature_dim, loss_device, momentum=args.memory_momentum
    )
    cluster_bank = MemoryBank(
        len(train_dataset), class_num, loss_device, momentum=args.memory_momentum
    )
    selector = GlobalQuerySelector(
        constraint_store,
        feature_bank,
        cluster_bank,
        candidate_pool=args.global_candidate_pool,
        sup_epsilon=args.query_sup_epsilon,
        minmax_eps=args.query_minmax_eps,
        total_queries=args.query_num,
    )
    epoch_selector = EpochQuerySelector(
        constraint_store,
        feature_bank,
        cluster_bank,
        num_clusters=class_num,
        queries_per_epoch=args.epoch_queries_per_epoch,
        candidate_pool=args.global_candidate_pool,
        neighbor_k=args.epoch_query_neighbor_k,
        kmeans_iters=args.epoch_query_kmeans_iters,
    )
    query_state = {"remaining": args.query_num}
    scheduler_mode = getattr(args, "query_scheduler", "batch")
    use_epoch_queries = isinstance(scheduler_mode, str) and scheduler_mode.lower() == "epoch"

    if args.reload:
        model_fp = os.path.join(
            args.model_path, "checkpoint_{}.tar".format(args.start_epoch)
        )
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        args.start_epoch = checkpoint["epoch"] + 1
        eval(args, test_data_loader, model, device, epoch=args.start_epoch)

    if args.retrain:
        current_epoch = args.start_epoch
        # while current_epoch< args.start_epoch+10 :
        #     lr = optimizer.param_groups[0]["lr"]
        #     loss_epoch = train(train_data_loader, model, current_epoch, isquery=False, query_strategy=args.query_strategy)
        #     if current_epoch % 100 == 0:
        #         save_model(args, model, optimizer, current_epoch)
        #     print(f"Epoch [{current_epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_data_loader)}")
        #     eval(args, train_data_test_loader, model, device, epoch="Train:"+str(current_epoch))
        #     eval(args, test_data_loader, model, device, epoch="Test:"+str(current_epoch))
        #     current_epoch+=1

        while current_epoch < args.epochs + 1:
            lr = optimizer.param_groups[0]["lr"]
            loss_epoch = train(
                train_data_loader,
                model,
                optimizer,
                criterion_instance,
                criterion_cluster,
                constraint_store,
                feature_bank,
                cluster_bank,
                selector,
                train_super_labels,
                query_state,
                args,
                current_epoch,
            )
            if use_epoch_queries and current_epoch >= args.global_query_warmup:
                added, debug_logs = run_epoch_queries(
                    epoch_selector,
                    train_super_labels,
                    constraint_store,
                    query_state,
                )
                if added:
                    print(
                        f"Epoch [{current_epoch}] added {added} epoch queries. Remaining: {query_state['remaining']}"
                    )
                    for idx, info in enumerate(debug_logs):
                        hardness_val = info.get("hardness", float("nan"))
                        relation = "same" if info.get("same_cluster", 0) == 1 else "diff"
                        print(
                            f"\tPair {idx+1}: anchor={info['anchor']} partner={info['partner']} "
                            f"hardness={hardness_val:.4f} relation={relation}"
                        )
            if current_epoch % 100 == 0:
                save_model(args, model, optimizer, current_epoch)
            print(
                f"Epoch [{current_epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_data_loader)}\t remaining_queries: {query_state['remaining']}"
            )
            eval(
                args,
                train_data_test_loader,
                model,
                device,
                epoch="Train:" + str(current_epoch),
            )
            eval(
                args,
                test_data_loader,
                model,
                device,
                epoch="Test:" + str(current_epoch),
            )
            current_epoch += 1

    eval(args, test_data_loader, model, device)
