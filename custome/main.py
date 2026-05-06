import copy
import datetime
import json
import logging
import os
import pickle
import random
import sys
import time
from math import sqrt
from statistics import mean
import gc
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch_npu
from absl import app, flags
from sklearn.metrics import roc_auc_score
import subprocess
from const_global import myclass
from data.data_loader import create_data_loader, create_data_loader_ep
from data.eval import gather_all_list, gather_all_dict, avg_eval, MetricsCalculator, ag_process_rank_scores
from data.reco_dataset import get_reco_dataset
from modeling.generic.executors.executor import Executor
from modeling.generic.executors.local_executor import LocalExecutor
from modeling.generic.sequential.embedding_modules import EmbeddingType
from modeling.generic.utils.constants import Const
from modeling.model_initializer import ModelInitializer
from modeling.model_registry import ModelRegistry
from utils.common_utils import get_config, refine_feat_and_model_conf
from torch.distributed._shard.sharded_tensor import ShardedTensor
from modeling.model_initializer import ModelInitializer
from utils.model_saver import ModelSaver

# 初始化传入参数
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

flags.DEFINE_string("config_file", "./train.config", "Path to the config file.")
flags.DEFINE_integer("master_port", 12355, "Master port.")
flags.DEFINE_string("data_dir", "/home/ma-user/work/AG/data/GRs_dataset_daily/20250610/out_1file", "Path to data.")
flags.DEFINE_string("save_dir", "/home/ma-user/work/AG/data/ag_rec_GRs_dataset_v7/HSTU_V1_export2", "Path to save.")
flags.DEFINE_string("feature_map_dir", "/home/ma-user/work/AG/data/ag_rec_GRs_dataset_v7/feature_map/featureMaxIndexMap.json",
                    "Path of feature_map.")
flags.DEFINE_string("feature_map_max_index_path", "/home/ma-user/work/AG/data/ag_rec_GRs_dataset_v7/feature_map/featureMaxIndexMap.json",
                    "Path of feature map max index")
flags.DEFINE_string("period", "20250610-000000", "Period of task execution.")
flags.DEFINE_boolean("is_train", True, "If the model is training or testing.")
flags.DEFINE_boolean("is_finetune", False, "If the model is doing incremental finetune.")
flags.DEFINE_string("load_data_dir", "/home/ma-user/work/AG/data/GRs_dataset_daily/20250609/out_1file",
                    "Path to previous-period data, used to derive cut_off_time_lower for finetune.")
flags.DEFINE_string("load_dir", "/home/ma-user/work/AG/data/ag_rec_GRs_dataset_v7/HSTU_V1_export2",
                    "Path to load checkpoint from for finetune (previous period model dir).")

FLAGS = flags.FLAGS

def log_pid_localrank_cpu(rank):
    pid = os.getpid()

    try:
        result = subprocess.run(['taskset', '-cp', str(pid)],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        cpu_binding = result.stdout.strip()
    except Exception as e:
        cpu_binding = f"无法获取 CPU 绑定信息: {e}"

    print(f"[PID {pid}] Local Rank: {rank} | CPU 绑定: {cpu_binding}")
    
    if rank == 0:
        # 打印 lscpu 信息
        try:
            lscpu_result = subprocess.run(['lscpu'],
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE,
                                          text=True)
            print("[lscpu 输出]:")
            print(lscpu_result.stdout.strip())
        except Exception as e:
            print(f"获取 lscpu 信息失败: {e}")

        # 打印 npu-smi topo 信息
        try:
            npu_result = subprocess.run(['npu-smi', 'info', '-t', 'topo'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
            print("[npu-smi info -t topo 输出]:")
            print(npu_result.stdout.strip())
        except Exception as e:
            print(f"获取 npu-smi topo 信息失败: {e}")
            
def init_ddp_info_params_mtp(already_init=False):
    """
    初始化加速器参数
    """

    addr = os.getenv("MASTER_ADDR")
    port = os.getenv("MASTER_PORT")
    rank_id = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    node_num = int(os.environ["NNODES"])
    logging.info(f"tcp://{addr}:{port}, nnodes={node_num}, rank={rank_id}")

    if not already_init:
        # initialize the process group
        dist.init_process_group("hccl", init_method=f"tcp://{addr}:{port}", rank=rank_id,
                                world_size=world_size)
    log_pid_localrank_cpu(rank_id)
    return f"npu:{local_rank}", rank_id, local_rank, world_size, node_num


def init_ddp_info_params(already_init=False):
    """
    初始化加速器参数
    """

    addr = os.getenv("MASTER_ADDR", "localhost")
    port = os.getenv("MASTER_PORT", "12356")

    rank_id = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    node_num = int(os.getenv("NNODES", "1"))  # 默认为 1 台机器

    logging.info(f"tcp://{addr}:{port}, nnodes={node_num}, rank={rank_id}")

    if not already_init:
        # initialize the process group
        dist.init_process_group("hccl", init_method=f"tcp://{addr}:{port}", rank=rank_id,
                                world_size=world_size)

    return f"npu:{local_rank}", rank_id, local_rank, world_size, node_num


def init_random_seed(config):
    """
    设置全局随机数以使训练结果更确定
    """
    random_seed = config.get('seed_conf', {}).get("global_seed", '1234')
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch_npu.npu.manual_seed(random_seed)
    torch_npu.npu.manual_seed_all(random_seed)


def init_torch_config(local_rank, train_conf):
    """
    设置torch配置
    """
    use_tf32 = train_conf.get("use_tf32", False)
    torch_npu.npu.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    logging.info("cuda.matmul.allow_tf32: %s", use_tf32)
    logging.info("cudnn.allow_tf32: %s", use_tf32)


def get_data_loaders(dataset, data_dir, rank, world_size, train_config, model_conf, 
                     feature_config, dataloader_config, config):
    """
    获取训练和验证数据 dataloader
    """
    dataset, eval_data_loader, train_data_loader = get_train_eval_dataloader(
        dataset, data_dir, rank, world_size, train_config, model_conf, feature_config, dataloader_config,
        data_pth=(['train', 'eval'] if dataloader_config.get('split_train_eval', False) else '')
    )
    return dataset, eval_data_loader, train_data_loader


def train_step(executor: Executor, train_conf, export_conf, batch_id, world_size, rank, epoch, device, save_dir):
    """
    训练模型
    """
    train_costs, batch_group = [], train_conf['eval_interval']
    last_training_time = time.time()
    is_output_loss_csv_file = train_conf.get("is_output_loss_csv_file", False)
    step_list, loss_list = [], []

    while True:
        try:
            loss, _ = executor.execute(batch_id)
        except StopIteration:
            logging.info("finished")
            break

        if rank == 0 and is_output_loss_csv_file:
            loss_list.append(loss.detach().cpu().item())
            step_list.append(batch_id)

        if (batch_id % batch_group) == 0:
            train_cost = time.time() - last_training_time
            logging.info("rank %s; batch-stat (train): step %s "
                         "(epoch %s in %.2fs): %.6f", rank, batch_id, epoch, train_cost, loss)
            if batch_id > 0:
                train_costs.append(train_cost)
            last_training_time = time.time()
        batch_id += 1

    if train_costs:
        logging.info(f"rank {rank} epoch {epoch}; mean cost per step: {mean(train_costs) / batch_group:.2f}s")
    else:
        logging.info(f"rank {rank} epoch {epoch}; less than {batch_group} steps")

    epoch_loss = avg_eval(torch.tensor([loss]).to(device), world_size=world_size)

    # 训练完成，保存模型
    model_save_pth = os.path.join(save_dir, export_conf["save_dir_name"])
    os.makedirs(model_save_pth, exist_ok=True)
    model_path = os.path.join(model_save_pth, "model_hstu.pth")
    if isinstance(executor, LocalExecutor):
        # 新版代码只支持保存state_dict
        if rank == 0:
            torch.save(executor.model.module.state_dict(), model_path)
    else:
        # torchrec保存分布式模型，用于后续执行Eval
        dist_model_path = os.path.join(model_save_pth, f"model_hstu_{rank}.pth")
        torch.save(executor.model.state_dict(), dist_model_path)
        # 合并分布式模型权重并保存
        gather_state_dict = executor.gather_state_dict()
        if rank == 0:
            torch.save(gather_state_dict, model_path)
    logging.info("Saving model to %s", model_path)
    return batch_id, epoch_loss


def save_loss_csv_file(loss_list, step_list):
    """
    保存loss.csv文件
    """
    save_dir = FLAGS.save_dir
    if not loss_list:
        logging.info("loss_list is None!!!")
        pass

    step_total = len(step_list) # 101 m/n
    all_loss_num = len(loss_list)  # 404 n
    step_length = all_loss_num / step_total  # 4 m
    # 分割成m组，每组长度为k
    sublists = [loss_list[i * step_total: (i + 1) * step_total] for i in range(int(step_length))]
    # 转置后计算每列的平均值
    score_avg_list = [sum(values) / step_length for values in zip(*sublists)]
    setp_avg_dict = {k: v for k, v in zip(step_list, score_avg_list)}

    df = pd.DataFrame({
        'step': list(setp_avg_dict.keys()),
        'loss': [round(v, 6) for v in setp_avg_dict.values()]
    })
    # 按键排序
    df = df.sort_values('step')
    # 保存为 CSV
    save_loss_path = "%s/modelfile" % save_dir
    loss_file_name = "step_vs_loss.csv"
    loss_file_path = os.path.join(save_loss_path, loss_file_name)
    if not os.path.exists(save_loss_path):
        os.makedirs(save_loss_path, exist_ok=True)
    logging.info("Saving step_vs_loss.csv to %s", loss_file_path)
    df.to_csv(loss_file_path, index=False, chunksize=100000)


def save_input_output(
    executor: Executor,
    save_dir,
    model_input_export_list,
    export_conf,
    feature_conf,
    commom_conf,
    feature_map_dir_or_path,
    cut_off_timestamp,
    device,
    rank,
    feature_map=None
):
    export_mode = export_conf.get("export_mode", "fake")

    model_input_export_list_new = []

    export_batch_size = export_conf.get('export_batch_size', 1)
    num_rerank = export_conf.get("num_rerank", 256)
    his_lengths = commom_conf.get("history_length", 150)
    shape_checked = False

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    logging.info("export_batch_size is %s", export_batch_size)
    logging.info("num_rerank is %s", num_rerank)
    for inputs in model_input_export_list:
        for k in inputs:
            if k.startswith("history_"):
                if k in feature_conf["history_item_feature_columns"]:
                    history_dtype = feature_conf["history_item_feature_columns"][k]["dtype"]
                    if history_dtype == "int" or history_dtype == "context":
                        lengths = sum(inputs["history_lengths"][:export_batch_size])
                        inputs[k] = inputs[k][:lengths]
                    elif history_dtype == 'multi':
                        lengths = sum(inputs["history_lengths"][:export_batch_size])
                        lengths_multi = sum(inputs[k + "_multi_sum_len"][:export_batch_size])
                        max_len = feature_conf["history_item_feature_columns"][k]["max_len"]
                        inputs[k] = inputs[k][:lengths_multi]
                        inputs[k + "_multi_sum_len"] = torch.tensor([lengths_multi]).to(device)
                        inputs[k + "_multi_len"] = inputs[k + "_multi_len"][:lengths].to(device)
                elif k.endswith("_multi_sum_len") or k.endswith("_multi_len"):
                    continue
                else:
                    inputs[k] = inputs[k][:export_batch_size]
            elif k.startswith("candidate_"):                
                if export_mode == "real":
                        _, real_length = inputs[k].size()
                        inputs[k] = torch.nn.functional.pad(inputs[k], (0, num_rerank - real_length))
                elif export_mode == "fake":
                    new_k = k
                    if k == "candidate_timestamps":
                        len_candidate_timestamps_fake = num_rerank * export_batch_size
                        candidate_timestamps_fake = [int(cut_off_timestamp + 86399)] * len_candidate_timestamps_fake
                        candidate_timestamps_fake = torch.tensor(candidate_timestamps_fake, dtype=torch.int64).view(
                            export_batch_size, -1).to(device)
                        inputs[k] = candidate_timestamps_fake
                    elif k == "candidate_labels" or k == "candidate_ratings" or k == "candidate_action_type":
                        candidate_labels_fake = torch.randint(0, 2, (export_batch_size, num_rerank), dtype=torch.int64).to(
                            device)
                        inputs[k] = candidate_labels_fake
                    elif k == "candidate_date":
                        len_candidate_date_fake = num_rerank * export_batch_size
                        ts = cut_off_timestamp + 86399
                        dt = datetime.datetime.fromtimestamp(ts).date()
                        candidate_date_fake = [int(dt.strftime("%Y%m%d"))] * len_candidate_date_fake
                        candidate_date_fake = torch.tensor(candidate_date_fake, dtype=torch.int64).view(
                            export_batch_size, -1).to(device)
                        inputs[k] = candidate_date_fake
                    elif k == "candidate_item_id":
                        candidate_feature_count = feature_conf["candidate_item_feature_columns"]["candidate_item_id"][
                            "feature_count"]
                        candidate_item_id_fake = torch.randint(0, candidate_feature_count, (export_batch_size * num_rerank,),
                                                               dtype=torch.int64).to(device)
                        inputs[k] = candidate_item_id_fake
                        inputs["candidate_ids"] = candidate_item_id_fake.reshape(export_batch_size, num_rerank)
                    elif k in feature_conf["candidate_item_feature_columns"]:
                        candidate_dtype = feature_conf["candidate_item_feature_columns"][new_k]["dtype"]
                        if candidate_dtype == "int" or candidate_dtype == "context":
                            candidate_feature_count = feature_conf["candidate_item_feature_columns"][new_k][
                                "feature_count"]
                            candidate_feat_fake = torch.randint(0, candidate_feature_count,
                                                                (export_batch_size * num_rerank,), dtype=torch.int64).to(
                                device)
                        elif candidate_dtype == "con":
                            candidate_feat_fake = torch.rand((export_batch_size * num_rerank), dtype=torch.float32).to(
                                device)
                        elif candidate_dtype == "multi":
                            candidate_feature_count = feature_conf["candidate_item_feature_columns"][new_k][
                                "feature_count"]
                            max_len = feature_conf["candidate_item_feature_columns"][new_k]["max_len"]
                            candidate_feat_fake = torch.randint(0, candidate_feature_count,
                                                                (export_batch_size * num_rerank * max_len,),
                                                                dtype=torch.int64).to(device)
                            inputs[k + "_multi_sum_len"] = torch.tensor([num_rerank * max_len]).repeat(export_batch_size).to(device)
                            print('inputs[k + "_multi_sum_len"]', inputs[k + "_multi_sum_len"])
                            inputs[k + "_multi_len"] = torch.tensor([max_len]).repeat(export_batch_size * num_rerank).to(device)
                            print('inputs[k + "_multi_len"]', inputs[k + "_multi_len"])
                        inputs[k] = candidate_feat_fake
                    elif k == "candidate_lengths":
                        inputs[k] = torch.tensor([num_rerank] * export_batch_size, dtype=torch.int64).to(device)
                    elif k in feature_conf["user_feature_columns"]:
                        inputs[k] = inputs[k][:export_batch_size]
                else:
                    logging.error("export_mode in export_conf should be either real or fake")
            elif k == "labels" or k == "loss_weights":
                labels_fake = torch.randint(0, 2, (export_batch_size, num_rerank), dtype=torch.int64).to(device)
                inputs[k] = labels_fake
            else:
                inputs[k] = inputs[k][:export_batch_size]


        if "history_timestamps" in inputs:
            print("found hist ts")
            inputs["timestamps"] = inputs["history_timestamps"]
        input_keys = list(inputs.keys())
        for k in input_keys:
            new_k = k
            if k in feature_conf["candidate_item_feature_columns"]:
                enabled = feature_conf["candidate_item_feature_columns"][new_k].get("enabled", True)
                if not enabled:
                    inputs.pop(k, None)
            elif k in feature_conf["history_item_feature_columns"]:
                enabled = feature_conf["history_item_feature_columns"][new_k].get("enabled", True)
                if not enabled:
                    inputs.pop(k, None)
            elif k in feature_conf["user_feature_columns"]:
                enabled = feature_conf["user_feature_columns"][k].get("enabled", True)
                if not enabled:
                    inputs.pop(k, None)
            elif k in ["candidate_item_id"] and dataset_name == "music-scalingraw-rank":
                inputs.pop(k, None)
        if not shape_checked and rank == 0:
            for k, v in inputs.items():
                print(k, "|", to_numpy(v).shape)
            shape_checked = True
        model_input_export_list_new.append(inputs)
    infer_item_id_name = "candidate_item_id"
    y_true_eval_all = []
    y_score_eval_all = []
    output_all = None
    input_all = {}
    input_keys = list(inputs.keys())
    candidate_label_csv = []
    score_csv = []
    
    for inputs in model_input_export_list_new:
        with torch.no_grad():
            model_out = executor.evaluate_once(inputs)["rerank_score"]

        if rank == 0:
            candidate_labels = inputs["labels"]
            candidate_ids = inputs[infer_item_id_name].unsqueeze(0)
            
            y_true_eval = to_numpy(candidate_labels[candidate_ids != 0])
            y_score_eval = to_numpy(model_out[candidate_ids != 0])
            for candidate_label_row, candidate_id_row, model_out_row in zip(candidate_labels, candidate_ids, model_out):
                candidate_label_csv.append("^".join(str(x) for x in to_numpy(candidate_label_row[candidate_id_row != 0])))
                score_csv.append("^".join(str(x) for x in to_numpy(model_out_row[candidate_id_row != 0])))
            model_output = to_numpy(model_out)
            inputs = {k: to_numpy(inputs[k]) for k in inputs}
            y_true_eval_all.append(y_true_eval)
            y_score_eval_all.append(y_score_eval)
            if output_all is None:
                output_all = model_output
            else:
                output_all = np.concatenate([output_all, model_output], axis=0)
            for k in input_keys:
                if k not in input_all:
                    input_all[k] = inputs[k]
                else:
                    input_all[k] = np.concatenate([input_all[k], inputs[k]], axis=0)
                    
    if rank == 0:
        is_compute_export_auc = export_conf.get('compute_auc', False)
        if is_compute_export_auc:
            y_true_eval_all = np.concatenate(y_true_eval_all, axis=None)
            y_score_eval_all = np.concatenate(y_score_eval_all, axis=None)
            export_pctr = np.mean(y_true_eval_all)
            export_pctr_gt_ctr = np.mean(y_score_eval_all)
            export_auc = roc_auc_score(y_true_eval_all, y_score_eval_all)
            logging.info("export auc is %s", export_auc)

        with open("%s/%s/pth_input" % (save_dir, export_conf["save_dir_name"]), 'wb') as fp:
            pickle.dump(input_all, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
        with open("%s/%s/pth_output" % (save_dir, export_conf["save_dir_name"]), 'wb') as fp:
            pickle.dump(output_all, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
        uid_csv = input_all["uid"].tolist()
        need_export_csv = export_conf.get("need_export_csv", False)
        if need_export_csv:
            export_csv = pd.DataFrame({"column_id": uid_csv, "label_id": candidate_label_csv, "score": score_csv})
            with open("%s/%s/result.csv" % (save_dir, export_conf["save_dir_name"]), 'wb') as fp:
                export_csv.to_csv(fp, index=False)

def save_feature_map(export_conf, save_dir, feature_map_dir_or_path, feature_map=None):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists("%s/%s.config" % (save_dir, export_conf["save_dir_name"])):
        os.mkdir("%s/%s.config" % (save_dir, export_conf["save_dir_name"]))

    if feature_map is not None:
        feature_dict = feature_map
    else:
        feature_map_files = os.listdir(feature_map_dir_or_path)
        feature_dict = dict()
        for feature_map_file in feature_map_files:
            feature_map_df = pd.read_orc(os.path.join(feature_map_dir_or_path, feature_map_file))
            indices_to_remove = [i for i, value in enumerate(feature_map_df['feature_name']) if 'user_id' in value]
            feature_map_df = feature_map_df.drop(indices_to_remove)
            feature_name = feature_map_df['feature_name'].tolist()
            feature_id = feature_map_df['feature_id'].tolist()
            feature_id = [int(x) for x in feature_id]
            feature_dict.update(zip(feature_name, feature_id))
    with open("%s/%s.config/feature_map.json" % (save_dir, export_conf["save_dir_name"]), 'w', encoding="utf-8") as fp:
        json.dump(feature_dict, fp, indent=4, ensure_ascii=False)
    logging.info("Saved feature map to %s/%s.config/feature_map.json", save_dir, export_conf["save_dir_name"])


def evaluate_step(executor: Executor, feature_conf, common_conf, feature_map, train_conf, save_dir, feature_map_dir_or_path,
                  export_conf, world_size, rank, save_after_eval, device, dataset,
                  model_saver: ModelSaver):
    """
    评估模型
    """
    logging.info("rank %s starting evaluation...", rank)

    save_score_csv_item_cols = feature_conf.get('save_score_csv_item_cols', {})
    save_score_csv_user_cols = feature_conf.get('save_score_csv_user_cols', {})

    scores_all, ground_truth_all, eval_weights_all = [], [], []
    group_keys_all = {
        'uid': [],
        'scores': [],
        'ground_truth': []
    }
    for key in save_score_csv_item_cols:
        group_keys_all[key] = []
    for key in save_score_csv_user_cols:
        group_keys_all[key] = []

    eval_iter = 0
    last_eval_time = time.perf_counter()

    is_input_copied = 0
    model_input_export_list = []

    while True:
        try:
            with torch.no_grad():
                scores_dict, model_input = executor.execute(eval_iter)
        except StopIteration:
            logging.info("finished")
            break

        scores, ground_truth, group_keys = ag_process_rank_scores(
            scores_dict, model_input, feature_conf.get('save_score_csv_item_cols', {}),
            feature_conf.get('save_score_csv_user_cols', {})
        )

        scores_all.extend(scores.view(-1).detach().cpu().tolist())
        ground_truth_all.extend(ground_truth.view(-1).detach().cpu().tolist())
        group_keys_all["scores"].extend(scores.view(-1).detach().cpu().tolist())
        group_keys_all["ground_truth"].extend(ground_truth.view(-1).detach().cpu().tolist())
        for key in group_keys_all.keys():
            if key not in ["scores", "ground_truth"]:
                items = group_keys.get(key, torch.tensor([]))
                group_keys_all[key].extend(items.detach().view(-1).cpu().tolist())

        if (eval_iter % train_conf["eval_interval"]) == 0:
            torch.distributed.barrier()
            cost = time.perf_counter() - last_eval_time
            logging.info(
                f"rank {rank}; batch-stat (eval): step {eval_iter} (EVAL in {cost:.2f}s)")
            last_eval_time = time.perf_counter()

        export_num_batch = export_conf.get("export_num_batch", 1)
        if is_input_copied < export_num_batch:
            model_input_export = copy.deepcopy(model_input)
            model_input_export_list.append(model_input_export)
            is_input_copied += 1

        eval_iter += 1
        del scores
        del ground_truth
        del group_keys

    torch_npu.npu.empty_cache()
    logging.info("rank %s start gather...", rank)
    ground_truth_all = gather_all_list(ground_truth_all, world_size=world_size)
    scores_all = gather_all_list(scores_all, world_size=world_size)
    group_keys_all = gather_all_dict(group_keys_all, world_size=world_size)

    if is_input_copied == 0:
        logging.error("No input sample can be saved because all the candidates are zero.")
    save_result_to_local = feature_conf.get("save_result_to_local", True)

    state_dict = executor.gather_state_dict()
    if rank == 0:
        save_path = "%s/modelfile" % save_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        logging.info("Test data count: %d", len(group_keys_all["ground_truth"]))
        metrics_file_name = "metrics_report.csv"
        metrics_path = os.path.join(save_path, metrics_file_name)
        if save_result_to_local:
            logging.info("Saving metrics to %s", metrics_path)
        eval_df = pd.DataFrame({
            **group_keys_all,
        })
        calculator = MetricsCalculator(eval_df, feature_map, feature_conf.get('metric_list', []),
                                       feature_conf.get('group_metric_cols_values', []),
                                       feature_conf.get('bias_evaluation_conf', []), save_result_to_local)
        eval_result = calculator.calculate(metrics_path)

        auc = eval_result.get("global").get("auc")

        if auc > model_saver.best_auc:
            model_saver.best_auc = auc
            logging.info("Update best AUC: %f, save model", model_saver.best_auc)

            model_saver.save_model(state_dict, "model_hstu.pth")
            model_saver.save_metric()

    if save_after_eval:
        save_dir_export = save_dir
        if not os.path.exists(save_dir_export):
            os.mkdir(save_dir_export)
        save_input_output(executor, save_dir_export, model_input_export_list, export_conf, feature_conf, common_conf, 
                          feature_map_dir_or_path, feature_conf["cut_off_time"], device, rank, feature_map)

    logging.info(
        'Number of samples in eval dataset: %s, rank %s, number of positive samples %s, '
        'number of negative samples %s',
        len(group_keys_all["scores"]), rank, sum(group_keys_all["ground_truth"]),
        len(group_keys_all["ground_truth"]) - sum(group_keys_all["ground_truth"]))

    return


def cleanup():
    gc.collect()
    if torch.npu.is_available():
        torch.npu.synchronize()
        torch.npu.empty_cache()


def safe_barrier():
    """安全的分布式屏障"""
    if torch.distributed.is_initialized():
        try:
            torch.distributed.barrier()
        except Exception as e:
            print(f"屏障同步失败: {e}")

def train_fn(config, data_dir, save_dir, feature_map_dir_or_path, period, model_saver: ModelSaver) -> None:
    """
    训练函数
    
    :param config: 配置文件字典
    :param data_dir: 数据集路径
    :param save_dir: 模型保存路径
    :return:
    """
    global dataset_name
    common_config = config[Const.COMMON_HP]
    init_random_seed(common_config)
    device, rank, local_rank, world_size, node_num = init_ddp_info_params()

    train_conf = common_config['train_conf']
    feature_conf = common_config['feature_conf']
    learning_rate = train_conf['learning_rate']
    lr_scaling = train_conf['lr_scaling']
    export_conf = common_config['export_conf']
    data_loader_conf = common_config['data_loader_conf']
    model_conf = common_config['model_conf']
    dataset_name = data_loader_conf["dataset_name"]
    feature_conf["period"] = period
    train_conf['learning_rate'] = init_learning_rate(learning_rate, lr_scaling, world_size)

    if dataset_name == "ag-rank":
        myclass.set_path(rootdir=save_dir, filename="analysis.csv")

    logging.info("Training model on rank %s (local rank: %s); device: %s; world_size: %s;",
                 rank, local_rank, device, world_size)

    init_torch_config(local_rank, train_conf)

    feature_conf, model_conf, feature_map = refine_feat_and_model_conf(dataset=dataset_name,
                                                                       feature_conf=feature_conf,
                                                                       model_conf=model_conf,
                                                                       feature_map_dir_or_path=feature_map_dir_or_path)
    if rank == 0:
        save_feature_map(export_conf=export_conf,
                         save_dir=save_dir,
                         feature_map_dir_or_path=feature_map_dir_or_path,
                         feature_map=feature_map)
        with open(os.path.join(save_dir, "gr_module_config.json"), "w", encoding="utf-8") as f:
            json.dump({"gr_module_config": config}, f, ensure_ascii=False, indent=4)
            logging.info("train config has been stored in %s", os.path.join(save_dir, "gr_module_config.json"))

    ModelRegistry.register_all_modules(
        modul_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "modeling/generic/sequential"))
    )
    model = ModelInitializer.init(gr_module_cfg=config)
    

    if dataset_name == "ag-rank":
        train_conf["find_unused_parameters"] = True
    else:
        train_conf["find_unused_parameters"] = False

    from modeling.generic.executors.torchrec_executor import TorchrecExecutor
    executor = TorchrecExecutor(model, train_conf, world_size, node_num, device, feature_conf)

    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"Parameter {name} has no gradient")

    for name, param in model.named_parameters():
        print(f"层名称: {name} | 形状: {param.shape} | 参数量: {param.numel()}")

    batch_id = 0
    for epoch in range(train_conf['num_epochs']):
        print('-----start epoch-----')
        batch_id = 0
        dataset, eval_data_loader, train_data_loader = get_data_loaders(dataset=dataset_name,
                                                                    data_dir=data_dir,
                                                                    rank=rank,
                                                                    world_size=world_size,
                                                                    train_config=train_conf,
                                                                    model_conf=model_conf,
                                                                    feature_config=feature_conf,
                                                                    dataloader_config=data_loader_conf,
                                                                    config=config)
        executor.train(train_data_loader)
        train_step_para = {
            "executor": executor,
            "train_conf": train_conf,
            "export_conf": export_conf,
            "batch_id": batch_id,
            "world_size": world_size,
            "rank": rank,
            "epoch": epoch,
            "device": device,
            "save_dir": save_dir
        }

        batch_id, epoch_loss = train_step(**train_step_para)

        logging.info("loss at epoch %s is %s", epoch, epoch_loss)

        if dataset_name == "ag-rank":
            logging.info("saving count results")
            myclass.save_data(rank)

        executor.eval(eval_data_loader)
        evaluate_step(executor=executor,
                      feature_conf=feature_conf,
                      common_conf=common_config,
                      feature_map=feature_map,
                      train_conf=train_conf,
                      save_dir=save_dir,
                      feature_map_dir_or_path=feature_map_dir_or_path,
                      export_conf=export_conf,
                      world_size=world_size,
                      rank=rank,
                      save_after_eval=True,
                      device=device,
                      dataset=dataset_name,
                      model_saver=model_saver)
    del executor


def eval_fn(config, data_dir, save_dir, feature_map_dir_or_path, period,
            model_saver: ModelSaver, already_init=False) -> None:
    """
    单独评估函数
    """
    common_config = config[Const.COMMON_HP]
    init_random_seed(common_config)
    device, rank, local_rank, world_size, node_num = init_ddp_info_params(already_init)

    train_conf = common_config['train_conf']
    feature_conf = common_config['feature_conf']
    learning_rate = train_conf['learning_rate']
    lr_scaling = train_conf['lr_scaling']
    export_conf = common_config['export_conf']
    data_loader_conf = common_config['data_loader_conf']
    model_conf = common_config['model_conf']
    dataset_name = data_loader_conf["dataset_name"]
    feature_conf["period"] = period
    train_conf['learning_rate'] = init_learning_rate(learning_rate, lr_scaling, world_size)

    logging.info("Testing model on rank %s (local rank: %s); device: %s; world_size: %s;",
                 rank, local_rank, device, world_size)

    init_torch_config(local_rank, train_conf)

    feature_conf, model_conf, feature_map = refine_feat_and_model_conf(dataset=dataset_name,
                                                                       feature_conf=feature_conf,
                                                                       model_conf=model_conf,
                                                                       feature_map_dir_or_path=feature_map_dir_or_path)
    dataset, eval_data_loader, train_data_loader = get_data_loaders(dataset=dataset_name,
                                                                    data_dir=data_dir,
                                                                    rank=rank,
                                                                    world_size=world_size,
                                                                    train_config=train_conf,
                                                                    model_conf=model_conf,
                                                                    feature_config=feature_conf,
                                                                    dataloader_config=data_loader_conf,
                                                                    config=config)

    ModelRegistry.register_all_modules(modul_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "modeling/generic/sequential")))
    model = ModelInitializer.init(gr_module_cfg=config)

    from modeling.generic.executors.torchrec_executor import TorchrecExecutor
    shardedTensor_patched_setstate()
    executor = TorchrecExecutor(model, train_conf, world_size, node_num, device, feature_conf)
    model_path = os.path.join(save_dir, export_conf["save_dir_name"], f"model_hstu_{rank}.pth")
    executor.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))

    if dataset_name == "ag-rank":
        myclass.save_data(rank)
    executor.eval(eval_data_loader)
    evaluate_step(executor=executor,
                  feature_conf=feature_conf,
                  common_conf=common_config,
                  feature_map=feature_map,
                  train_conf=train_conf,
                  save_dir=save_dir,
                  feature_map_dir_or_path=feature_map_dir_or_path,
                  export_conf=export_conf,
                  world_size=world_size,
                  rank=rank,
                  save_after_eval=True,
                  device=device,
                  dataset=dataset_name,
                  model_saver=model_saver)


def finetune_fn(config, data_dir, load_data_dir, save_dir, load_dir, feature_map_dir_or_path,
                period, model_saver: ModelSaver) -> None:
    """
    增量微调函数。

    数据策略(双 cut_off):
      - cut_off_time       = data_dir 路径中的 8 位日期 (例如 20260220)
                             作为 train 上界(不含)和 eval 下界(含)。
      - cut_off_time_lower = load_data_dir 路径中的 8 位日期 - cut_off_bias (例如 20260214)
                             作为 train 下界(含)。
      - 微调和评估都使用 data_dir,不依赖 load_data_dir 的样本文件。
        微调样本: cut_off_time_lower <= candidate_date < cut_off_time
        评估样本: candidate_date >= cut_off_time

    模型策略:
      - 从 load_dir 加载上一周期的 checkpoint。
      - 提取 ckpt_info(每个 emb_*.weight 的 size/mean/std),通过 config 注入,
        让 LocalEmbeddingModuleOnlySideInfo 在新增 ID 行时能用同分布初始化,
        同时支持 embedding 表的扩容(新词表 >= 旧词表)。
      - LocalExecutor 路径走 adaptive_load_weights(允许 shape 不严格匹配的部分加载)。
      - TorchrecExecutor 路径走 model.load_state_dict(要求新旧词表大小一致)。
    """
    global dataset_name
    common_config = config[Const.COMMON_HP]
    init_random_seed(common_config)
    device, rank, local_rank, world_size, node_num = init_ddp_info_params()

    train_conf = common_config['train_conf']
    feature_conf = common_config['feature_conf']
    learning_rate = train_conf['learning_rate']
    lr_scaling = train_conf['lr_scaling']
    export_conf = common_config['export_conf']
    data_loader_conf = common_config['data_loader_conf']
    model_conf = common_config['model_conf']
    dataset_name = data_loader_conf["dataset_name"]
    feature_conf["period"] = period
    train_conf['learning_rate'] = init_learning_rate(learning_rate, lr_scaling, world_size)

    if dataset_name == "ag-rank":
        myclass.set_path(rootdir=save_dir, filename="analysis.csv")

    logging.info("Finetuning model on rank %s (local rank: %s); device: %s; world_size: %s;",
                 rank, local_rank, device, world_size)

    init_torch_config(local_rank, train_conf)

    # 1. refine_feat_and_model_conf 会基于 period 计算一个初始 cut_off_time,后面会被覆盖
    feature_conf, model_conf, feature_map = refine_feat_and_model_conf(dataset=dataset_name,
                                                                       feature_conf=feature_conf,
                                                                       model_conf=model_conf,
                                                                       feature_map_dir_or_path=feature_map_dir_or_path)

    if rank == 0:
        save_feature_map(export_conf=export_conf,
                         save_dir=save_dir,
                         feature_map_dir_or_path=feature_map_dir_or_path,
                         feature_map=feature_map)
        with open(os.path.join(save_dir, "gr_module_config.json"), "w", encoding="utf-8") as f:
            json.dump({"gr_module_config": config}, f, ensure_ascii=False, indent=4)
            logging.info("train config has been stored in %s", os.path.join(save_dir, "gr_module_config.json"))

    # 2. 加载旧 checkpoint,提取 ckpt_info 用于 embedding 仿生初始化和扩容
    model_path = os.path.join(load_dir, export_conf["save_dir_name"], "model_hstu.pth")
    print(f"zxp finetune loading checkpoint from: {model_path}")
    load_dict = torch.load(model_path, map_location=device, weights_only=True)
    ckpt_info = {}
    for k, v in load_dict.items():
        if "emb_" in k and k.endswith(".weight"):
            parts = k.split('.')
            if len(parts) >= 3:
                feat_name = parts[2]
                ckpt_info[feat_name] = {
                    'size': v.shape[0],
                    'mean': float(v.mean()),
                    'std': float(v.std())
                }
                if rank == 0:
                    print("zxp finetune ckpt_info entry:", feat_name, ckpt_info[feat_name])

    # 3. 注入 ckpt_info 到 config,供 model_initializer/embedding_modules 读取
    config['ckpt_info'] = ckpt_info

    # 4. 初始化模型
    ModelRegistry.register_all_modules(
        modul_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "modeling/generic/sequential"))
    )
    model = ModelInitializer.init(gr_module_cfg=config)

    # 5. 设置双 cut_off
    from pathlib import Path
    from datetime import datetime as _dt, timedelta as _td

    def _extract_8digit_date(p):
        for part in Path(p).parts:
            if part.isdigit() and len(part) == 8:
                return int(part)
        return None

    cut_off_main = _extract_8digit_date(data_dir)
    if cut_off_main is None:
        raise ValueError(f"finetune: 无法从 data_dir 解析 8 位日期: {data_dir}")

    load_date = _extract_8digit_date(load_data_dir)
    if load_date is None:
        raise ValueError(f"finetune: 无法从 load_data_dir 解析 8 位日期: {load_data_dir}")

    cut_off_bias = int(feature_conf.get("cut_off_bias", 1))
    cut_off_lower = int((_dt.strptime(str(load_date), "%Y%m%d")
                         - _td(days=cut_off_bias)).strftime("%Y%m%d"))

    feature_conf["cut_off_time"] = cut_off_main
    feature_conf["cut_off_time_lower"] = cut_off_lower
    print(f"zxp finetune data_dir       = {data_dir}")
    print(f"zxp finetune load_data_dir  = {load_data_dir}")
    print(f"zxp finetune cut_off_time       = {cut_off_main}   (train 上界/eval 下界)")
    print(f"zxp finetune cut_off_time_lower = {cut_off_lower}  (train 下界, 含)")

    if dataset_name == "ag-rank":
        train_conf["find_unused_parameters"] = True
    else:
        train_conf["find_unused_parameters"] = False

    # 6. 构造 executor 并加载旧权重
    if model.embedding_type == EmbeddingType.TORCHREC:
        from modeling.generic.executors.torchrec_executor import TorchrecExecutor
        # 跳过跨机场景 worldsize 校验
        shardedTensor_patched_setstate()
        executor = TorchrecExecutor(model, train_conf, world_size, node_num, device, feature_conf)
        # TorchRec 路径:走 partial-load 以支持 num_embeddings 扩容(item_id 等词表常变)。
        # 不能直接 load_state_dict,因为 ShardedTensor metadata 严格匹配会在 size 变化时失败。
        # 该函数从 rank 0 加载 model_hstu.pth(gather 后的完整权重),通过 broadcast +
        # 按行 partial copy 把旧权重前 N 行填到新 ShardedTensor 的对应位置;新行保留 init_fn
        # 设置的仿生分布(由 InitEmbeddingConfig.init_mean/init_std 控制)。
        adaptive_load_torchrec_partial(
            model=executor.model,
            ckpt_dir=os.path.join(load_dir, export_conf["save_dir_name"]),
            rank=rank,
            world_size=world_size,
            device=device,
        )
    else:
        # LocalExecutor 路径,走 adaptive_load_weights 支持 shape 不严格匹配
        adaptive_load_weights(model, load_dict)
        executor = LocalExecutor(model, train_conf, local_rank, device)

    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"FAILED: Parameter '{name}' contains NaN/Inf right after loading!")
        if param.grad is None:
            print(f"Parameter {name} has no gradient")

    for name, param in model.named_parameters():
        print(f"层名称: {name} | 形状: {param.shape} | 参数量: {param.numel()}")

    # 7. 训练 + 评估循环(仿照 train_fn 结构,dataloader 放进 epoch 循环)
    batch_id = 0
    for epoch in range(train_conf['num_epochs']):
        print('-----start finetune epoch-----')
        batch_id = 0
        # 微调和评估同源,都用 data_dir;train_dataset 用 cut_off_time_lower 做下界过滤
        dataset, eval_data_loader, train_data_loader = get_data_loaders(dataset=dataset_name,
                                                                    data_dir=data_dir,
                                                                    rank=rank,
                                                                    world_size=world_size,
                                                                    train_config=train_conf,
                                                                    model_conf=model_conf,
                                                                    feature_config=feature_conf,
                                                                    dataloader_config=data_loader_conf,
                                                                    config=config)
        executor.train(train_data_loader)
        train_step_para = {
            "executor": executor,
            "train_conf": train_conf,
            "export_conf": export_conf,
            "batch_id": batch_id,
            "world_size": world_size,
            "rank": rank,
            "epoch": epoch,
            "device": device,
            "save_dir": save_dir
        }

        batch_id, epoch_loss = train_step(**train_step_para)

        logging.info("loss at epoch %s is %s", epoch, epoch_loss)

        if dataset_name == "ag-rank":
            logging.info("saving count results")
            myclass.save_data(rank)

        executor.eval(eval_data_loader)
        evaluate_step(executor=executor,
                      feature_conf=feature_conf,
                      common_conf=common_config,
                      feature_map=feature_map,
                      train_conf=train_conf,
                      save_dir=save_dir,
                      feature_map_dir_or_path=feature_map_dir_or_path,
                      export_conf=export_conf,
                      world_size=world_size,
                      rank=rank,
                      save_after_eval=True,
                      device=device,
                      dataset=dataset_name,
                      model_saver=model_saver)
    del executor


def shardedTensor_patched_setstate():
    def patched_setstate(self, state):
        self._sharded_tensor_id = None
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "Need to initialize default process group using "
                '"init_process_group" before loading ShardedTensor'
            )

        (
            self._local_shards,
            self._metadata,
            pg_state,
            self._sharding_spec,
            self._init_rrefs,
        ) = state

        from torch.distributed._shard.api import _get_current_process_group
        self._process_group = _get_current_process_group()

        self._post_init()

    ShardedTensor.__setstate__ = patched_setstate


def adaptive_load_weights(model, load_dict):
    """
    LocalExecutor 路径下的"扩容感知"权重加载,从 utils.common_utils 内联过来,
    避免对该模块的依赖(用户新版 common_utils.py 可能没有此函数)。
    
    手动将 load_dict 中的权重拷贝到模型中。
    如果 shape 不一致,仅拷贝重合部分(交集),保留模型原有的初始化值
    (即 reset_params_load 设置的仿生分布)。
    """
    model_state = model.state_dict()
    
    for key, load_param in load_dict.items():
        if key not in model_state:
            continue
            
        model_param = model_state[key]
        
        if model_param.shape != load_param.shape:
            if model_param.dim() >= 1:
                # 构造多维切片,确保能处理 [Rows, Cols] 都不一致的情况
                slices = tuple(slice(0, min(m, l)) for m, l in zip(model_param.shape, load_param.shape))
                model_param[slices].copy_(load_param[slices])
                print(f"Layer {key} adapted: {list(load_param.shape)} -> {list(model_param.shape)}")
            else:
                print(f"Layer {key} shape mismatch, skipping.")
        else:
            model_param.copy_(load_param)

    print("Adaptive weight loading completed via manual copy.")


def adaptive_load_torchrec_partial(model, ckpt_dir, rank, world_size, device):
    """
    TorchRec 路径下的"扩容感知"权重加载,替代 model.load_state_dict。
    
    设计动机:
    - 业务上 item_id / user_id 等特征的 num_embeddings 跨周期会变化(item 个数常增长)
    - TorchRec 默认的 model.load_state_dict 要求 ShardedTensor metadata 严格一致,
      num_embeddings 变化会直接报错
    - 该函数实现按行 partial copy:旧权重前 N 行覆盖新表前 N 行,新行(N 之后)
      保留 InitEmbeddingConfig.init_fn 设置的仿生分布
    
    加载源:
    - ckpt_dir/model_hstu.pth (gather 后的完整权重,由旧版 train_step rank 0 保存)
    - 不使用 model_hstu_{rank}.pth(分片版本),因为分片版本在 num_embeddings 变化时
      跨 rank 行号映射混乱,无法直接用于扩容
    
    通信流程:
    - rank 0 从磁盘加载完整权重,通过 dist.broadcast 分发到所有 rank
    - 每个 rank 拿到完整旧权重后,根据自己 ShardedTensor local_shards 的 (offset, size)
      元数据切出对应行,行内拷贝
    
    限制:
    - 假设 ROW_WISE sharding(EmbeddingCollection 默认就是,通常 OK)
    - 假设 emb 维度 (dim) 不变,只 num_embeddings 可变
    - non-emb 参数(MLP 等)要求 shape 严格相同,否则跳过
    """
    import torch.distributed as dist
    from torch.distributed._shard.sharded_tensor import ShardedTensor as _ST

    full_path = os.path.join(ckpt_dir, "model_hstu.pth")
    logging.info("adaptive_load_torchrec_partial: loading from %s on rank %s", full_path, rank)

    # rank 0 从磁盘加载完整 state_dict;其他 rank 仅持引用占位
    if rank == 0:
        old_full = torch.load(full_path, map_location='cpu', weights_only=False)
        keys = list(old_full.keys())
    else:
        old_full = None
        keys = None

    # 把 keys 列表 broadcast 到所有 rank
    keys_holder = [keys]
    dist.broadcast_object_list(keys_holder, src=0)
    keys = keys_holder[0]

    new_state_dict = dict(model.state_dict())

    n_loaded_emb = 0
    n_loaded_param = 0
    n_skipped = 0

    for key in keys:
        if key not in new_state_dict:
            if rank == 0:
                logging.info("adaptive_load_torchrec_partial: skip %s (not in new model)", key)
            n_skipped += 1
            continue

        new_value = new_state_dict[key]

        # 拿到旧权重的 dense tensor 表示。rank 0 才有数据,其他 rank 占位接收 broadcast。
        if rank == 0:
            old_value = old_full[key]
            # gather_state_dict 通常存的就是 dense Tensor;若是 ShardedTensor 取本地分片
            if isinstance(old_value, _ST):
                local_shards = old_value.local_shards()
                if local_shards:
                    old_value = local_shards[0].tensor
                else:
                    old_value = None
            if old_value is not None:
                old_value = old_value.detach().cpu().contiguous().float()
                old_shape = list(old_value.shape)
                old_dtype_str = "float32"
            else:
                old_shape = None
                old_dtype_str = None
        else:
            old_value = None
            old_shape = None
            old_dtype_str = None

        # 把 shape 元信息 broadcast 到所有 rank
        meta_holder = [(old_shape, old_dtype_str)]
        dist.broadcast_object_list(meta_holder, src=0)
        old_shape, old_dtype_str = meta_holder[0]

        if old_shape is None:
            if rank == 0:
                logging.info("adaptive_load_torchrec_partial: skip %s (no data)", key)
            n_skipped += 1
            continue

        # 创建 buf,broadcast 实际数据
        if rank == 0:
            buf = old_value.to(device).contiguous()
        else:
            buf = torch.empty(old_shape, dtype=torch.float32, device=device)
        dist.broadcast(buf, src=0)

        # 根据 new_value 的类型做 partial copy
        with torch.no_grad():
            if isinstance(new_value, _ST):
                # ShardedTensor:遍历每个本地分片,根据 shard_offsets/shard_sizes 切 buf
                local_shards = new_value.local_shards()
                for shard in local_shards:
                    meta = shard.metadata
                    offsets = list(meta.shard_offsets)
                    sizes = list(meta.shard_sizes)
                    
                    # 简化处理:仅支持 ROW_WISE(行方向切分),即 col_offset = 0
                    # col 方向 partial 暂不处理(emb_dim 不应该变)
                    if len(offsets) >= 2 and offsets[1] != 0:
                        if rank == 0:
                            logging.warning(
                                "adaptive_load_torchrec_partial: %s shard has col_offset=%s, "
                                "non-row-wise sharding not fully supported, doing best-effort",
                                key, offsets[1])
                    
                    row_start = offsets[0]
                    row_size = sizes[0]
                    row_end = row_start + row_size

                    # 旧权重在该分片对应行段中能覆盖的范围
                    src_start = row_start
                    src_end = min(row_end, buf.shape[0])

                    if src_end > src_start:
                        n_rows_copy = src_end - src_start
                        target_tensor = shard.tensor.data
                        # local 内的索引从 0 开始
                        target_tensor[:n_rows_copy].copy_(
                            buf[src_start:src_end].to(target_tensor.dtype).to(target_tensor.device)
                        )
                        # row_end 之后的行(超出旧 num_embeddings 的部分)保持 init_fn 的值不动

                    # padding_idx=0 行强制清零(防止覆盖了非零值)
                    if row_start == 0 and shard.tensor.size(0) > 0:
                        shard.tensor.data[0].zero_()

                n_loaded_emb += 1
                if rank == 0:
                    logging.info(
                        "adaptive_load_torchrec_partial: emb %s loaded "
                        "(old_rows=%d, new_rows=%d)",
                        key, buf.shape[0], new_value.shape[0]
                    )
            else:
                # 普通 Tensor:要求 shape 严格相同,否则跳过(MLP/bias 这类不应该变 shape)
                if list(new_value.shape) == list(buf.shape):
                    new_value.data.copy_(buf.to(new_value.dtype).to(new_value.device))
                    n_loaded_param += 1
                else:
                    if rank == 0:
                        logging.warning(
                            "adaptive_load_torchrec_partial: param %s shape mismatch "
                            "(old=%s, new=%s), skip",
                            key, buf.shape, new_value.shape
                        )
                    n_skipped += 1

        # 释放 buf
        del buf

    if rank == 0:
        logging.info(
            "adaptive_load_torchrec_partial done: emb=%d, param=%d, skipped=%d",
            n_loaded_emb, n_loaded_param, n_skipped
        )

    # 同步,确保所有 rank 完成
    dist.barrier()


def get_train_eval_dataloader(dataset, data_dir, rank, world_size, train_conf, model_conf, feature_conf,
                              dataloader_conf, data_pth=''):
    max_sequence_length = model_conf.get("max_sequence_length", 100)
    local_batch_size = train_conf.get("local_batch_size", 256)
    eval_batch_size = train_conf.get("eval_batch_size", 256)
    prefetch_factor = dataloader_conf['prefetch_factor']
    num_workers = dataloader_conf['num_workers']

    dataset = get_reco_dataset(
        dataset=dataset,
        data_dir=data_dir,
        pth=data_pth,
        rank=rank,
        world_size=world_size,
        max_sequence_length=max_sequence_length,
        chronological=feature_conf.get("chronological", True),
        feature_conf=feature_conf,
        num_rerank=dataloader_conf.get("num_rerank", 256),
        history_length=dataloader_conf.get("history_length", 400))

    train_data_loader = create_data_loader_ep(
        dataset.train_dataset,
        batch_size=local_batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers
    )

    eval_data_loader = create_data_loader_ep(
        dataset.eval_dataset,
        batch_size=eval_batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers
    )
    return dataset, eval_data_loader, train_data_loader


def init_learning_rate(learning_rate, lr_scaling, world_size):
    lr_scaling = lr_scaling.strip().lower()
    if lr_scaling == "linear":
        learning_rate *= world_size
    elif lr_scaling == "sqrt":
        learning_rate *= sqrt(world_size)
    else:
        raise ValueError("'%s' is not a supported scaling strategy for the learning rate." % lr_scaling)
    return learning_rate


def main_torchrun(argv):
    if FLAGS.config_file is None:
        raise ValueError("you have to assign the train config file")
    config = get_config(FLAGS.config_file)
    config = config["gr_module_config"]

    export_save_dir_name = config[Const.COMMON_HP]['export_conf']['save_dir_name']
    model_saver = ModelSaver(FLAGS.save_dir, export_save_dir_name)

    if FLAGS.is_finetune:
        finetune_fn(config,
                    FLAGS.data_dir,
                    FLAGS.load_data_dir,
                    FLAGS.save_dir,
                    FLAGS.load_dir,
                    FLAGS.feature_map_dir,
                    str(FLAGS.period),
                    model_saver)
    elif FLAGS.is_train:
        train_fn(config,
                 FLAGS.data_dir,
                 FLAGS.save_dir,
                 FLAGS.feature_map_dir,
                 str(FLAGS.period),
                 model_saver)
    else:
        eval_fn(config,
                FLAGS.data_dir,
                FLAGS.save_dir,
                FLAGS.feature_map_dir,
                str(FLAGS.period),
                model_saver)


if __name__ == "__main__":
    app.run(main_torchrun)
