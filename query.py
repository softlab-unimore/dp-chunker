import faiss
import argparse
import json
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
from functools import lru_cache
from typing import List
from tqdm import tqdm
import warnings
from multiprocessing import cpu_count
from sentence_transformers import SentenceTransformer, models
from copy import deepcopy

import gc

from data_processor import NQProcessor, index_paths, index_root_path, TriviaQAProcessor, \
    WebQProcessor, SquadProcessor, EQProcessor

warnings.filterwarnings("ignore")
faiss.omp_set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count())))
print(f"Using {faiss.omp_get_max_threads()} instead of value given by cpu_count: {cpu_count()}")

cls = ["sup-simcse-bert-base-uncased"]

def get_embedding(texts: List[str], model=None) -> np.ndarray:
    with torch.no_grad():
        token_embs = model.encode(texts, convert_to_tensor=True)
        sent_embs = F.normalize(token_embs, p=2, dim=1)

    return sent_embs.cpu().numpy().astype('float32') #faiss requires float32

@lru_cache()
def load_model(model_name_full: str):
    model_name = model_name_full.split("/")[-1]
    word_emb_model = models.Transformer(model_name_full, max_seq_length=512)
    if model_name not in cls:
        pooling_model_mean = models.Pooling(
            word_emb_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )
    else:
        pooling_model_mean = models.Pooling(
            word_emb_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=False,
            pooling_mode_cls_token=True,
            pooling_mode_max_tokens=False
        )
    model = SentenceTransformer(modules=[word_emb_model, pooling_model_mean]).to("cuda").half()
    model.eval()

    return model

def query_faiss_index(query_texts, model_name, index_path, meta_path, top_k=5, save_path=""):
    print("loading model")
    model = load_model(model_name)

    print("reading index...")
    index = faiss.read_index(index_path)
    index.nprobe = 1024

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    print("computing query embedding...")
    embeddings = get_embedding(query_texts, model)

    print("index search...")
    D, I = index.search(embeddings, top_k)  # D = distances, I = indices

    print("mapping the results...")
    # mapping FAISS int IDs to original string IDs
    results = []
    for i, (distances, indices) in enumerate(zip(D, I)):
        query_results = []
        for d, idx in zip(distances, indices):
            original_id = metadata.get(str(idx), None)
            query_results.append(([int(idx), original_id], float(d)))

        results.append(query_results)

    save(results, query_texts, top_k, save_path)
    return results

def get_cluster_mask(I, metadata):
    mask = []
    for i in range(len(I)):
        row = []
        for j in range(len(I[i])):
            if str(I[i][j]) != "-1" and metadata[str(I[i][j])]["type"] == "normal":
                row.append(0)
            else:
                row.append(-float("inf"))
        mask.append(row)

    return np.array(mask)

def save(predictions, questions, k, save_path):
    results = []

    for i, r in enumerate(predictions):
        tmp = []
        scores = []

        tmp.append(questions[i])
        for original_id, score in r:
            tmp.append(original_id)
            scores.append(score)

        tmp.append(scores)
        results.append(tmp)

    columns = ["question"] + ["ID"+str(i+1) for i in range(k)] + ["scores"]
    df = pd.DataFrame(results, columns=columns)

    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()

    if args.dataset == "nq":
        processor = NQProcessor()
    elif args.dataset == "eq":
        processor = EQProcessor()
    elif args.dataset == "squad":
        processor = SquadProcessor()
    elif args.dataset == "triviaqa":
        processor = TriviaQAProcessor()
    elif args.dataset == "webq":
        processor = WebQProcessor()
    else:
        raise NotImplementedError()

    questions, _ = processor.read_data()

    index_path = os.path.join(index_root_path,
                              index_paths[args.model.split("/")[-1]][args.method]["index"])
    meta_path = os.path.join(index_root_path,
                             index_paths[args.model.split("/")[-1]][args.method]["meta"])

    save_dir = os.path.join("predictions", args.dataset, args.model.split("/")[-1], args.method)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.k}.csv")

    print("Querying...")
    predictions = query_faiss_index(
        questions,
        args.model,
        index_path,
        meta_path,
        top_k=args.k,
        save_path=save_path
    )
