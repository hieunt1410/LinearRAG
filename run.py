import argparse
import json
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from src.config import LinearRAGConfig
from src.LinearRAG import LinearRAG
import os
import warnings
from src.evaluate import Evaluator
from src.utils import LLM_Model
from src.utils import setup_logging
from datetime import datetime


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spacy_model",
        type=str,
        default="en_core_web_trf",
        help="The spacy model to use",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="model/all-mpnet-base-v2",
        help="The path of embedding model to use",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="novel", help="The dataset to use"
    )
    parser.add_argument(
        "--llm_model", type=str, default="gpt-4o-mini", help="The LLM model to use"
    )
    parser.add_argument(
        "--max_workers", type=int, default=16, help="The max number of workers to use"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=3,
        help="The max number of iterations to use",
    )
    parser.add_argument(
        "--iteration_threshold",
        type=float,
        default=0.4,
        help="The threshold for iteration",
    )
    parser.add_argument(
        "--passage_ratio", type=float, default=2, help="The ratio for passage"
    )
    parser.add_argument(
        "--top_k_sentence", type=int, default=3, help="The top k sentence to use"
    )
    return parser.parse_args()


def load_dataset(dataset_name):
    questions_path = os.path.join("datasets", dataset_name, "questions.json")
    chunks_path = os.path.join("datasets", dataset_name, "chunks.json")
    with open(chunks_path, "r") as f:
        passages = json.load(f)
    with open(questions_path, "r") as f:
        questions = json.load(f)
    return questions, passages


def load_embedding_model(embedding_model_path):
    model = SentenceTransformer(embedding_model_path)
    return model


def main():
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    args = parse_arguments()
    embedding_model = load_embedding_model(args.embedding_model)
    questions, passages = load_dataset(args.dataset_name)
    setup_logging(f"results/{args.dataset_name}/{time}/log.txt")
    llm_model = LLM_Model(args.llm_model)
    config = LinearRAGConfig(
        spacy_model=args.spacy_model,
        embedding_model=embedding_model,
        llm_model=llm_model,
        max_workers=args.max_workers,
        max_iterations=args.max_iterations,
        iteration_threshold=args.iteration_threshold,
        passage_ratio=args.passage_ratio,
        top_k_sentence=args.top_k_sentence,
    )
