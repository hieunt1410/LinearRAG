from src.embedding_store import EmbeddingStore
from src.utils import min_max_normalize
import os
import json
from collections import defaultdict
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.ner import SpacyNER
import igraph as ig
import re
import logging

logger = logging.getLogger(__name__)


class LinearRAG:
    def __init__(self, config):
        self.config = config
        logger.info(f"Initializing LinearRAG with config: {self.config}")
        self.dataset_name = config.dataset_name
        self.llm_model = config.llm_model
        self.spacy_ner = SpacyNER(model_name=config.spacy_model)
        self.graph = ig.Graph(directed=False)

        self.load_embedding_store()

    def load_embedding_store(self):
        self.passage_embedding_store = EmbeddingStore(
            embedding_model=self.config.embedding_model,
            db_filename=f"datasets/{self.dataset_name}_passage_embeddings.parquet",
            batch_size=16,
            namespace=f"{self.dataset_name}_passage",
        )
        self.entity_embedding_store = EmbeddingStore(
            embedding_model=self.config.embedding_model,
            db_filename=f"datasets/{self.dataset_name}_entity_embeddings.parquet",
            batch_size=16,
            namespace=f"{self.dataset_name}_entity",
        )
        self.sentence_embedding_store = EmbeddingStore(
            embedding_model=self.config.embedding_model,
            db_filename=f"datasets/{self.dataset_name}_sentence_embeddings.parquet",
            batch_size=16,
            namespace=f"{self.dataset_name}_sentence",
        )

    def load_existing_data(self, passage_hash_ids):
        self.ner_results_path = os.path.join(
            self.config.working_dir, "datasets", self.dataset_name, "ner_results.json"
        )
        if os.path.exists(self.ner_results_path):
            existing_ner_results = json.load(open(self.ner_results_path, "r"))
            existing_passage_hash_id_to_entities = existing_ner_results[
                "passage_hash_id_to_entities"
            ]
            existing_sentences_to_entities = existing_ner_results[
                "sentences_to_entities"
            ]
            existing_passage_hash_ids = set(existing_passage_hash_id_to_entities.keys())
            new_passage_hash_ids = set(passage_hash_ids) - existing_passage_hash_ids

            return (
                existing_passage_hash_id_to_entities,
                existing_sentences_to_entities,
                new_passage_hash_ids,
            )
        else:
            return {}, {}, set(passage_hash_ids)

    def index(self, passages):
        self.nodes_to_nodes_edges = defaultdict(dict)
        self.entity_to_sentence_edges = defaultdict(dict)
        self.passage_embedding_store.insert_text(passages)
        hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text()
        (
            existing_passage_hash_id_to_entities,
            existing_sentences_to_entities,
            new_passage_hash_ids,
        ) = self.load_existing_data(list(hash_id_to_passage.keys()))
        logger.info(
            f"Loaded existing NER results for {len(existing_passage_hash_id_to_entities)} passages."
        )
        if len(new_passage_hash_ids):
            new_hash_id_to_passage = {
                h: hash_id_to_passage[h] for h in new_passage_hash_ids
            }
            new_passage_hash_id_to_entities, new_sentences_to_entities = (
                self.spacy_ner.batch_ner(
                    new_hash_id_to_passage, max_workers=self.config.max_workers
                )
            )

            existing_passage_hash_id_to_entities.update(new_passage_hash_id_to_entities)
            existing_sentences_to_entities.update(new_sentences_to_entities)
        self.save_ner_results(
            existing_passage_hash_id_to_entities, existing_sentences_to_entities
        )
        logger.info(
            f"Total NER results for {len(existing_passage_hash_id_to_entities)} passages after processing new data."
        )

        (
            entity_nodes,
            sentence_nodes,
            self.entity_to_sentence,
            self.sentence_to_entity,
        ) = self.extract_nodes_and_edges(
            existing_passage_hash_id_to_entities, existing_sentences_to_entities
        )
        self.entity_embedding_store.insert_text(list(entity_nodes))
        self.sentence_embedding_store.insert_text(list(sentence_nodes))
        self.entity_hash_id_to_sentence_hash_ids = defaultdict(list)
        for entity, sentences in self.entity_to_sentence.items():
            entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
            sentence_hash_ids = [
                self.sentence_embedding_store.text_to_hash_id[sent]
                for sent in sentences
            ]
            self.entity_hash_id_to_sentence_hash_ids[entity_hash_id] = sentence_hash_ids
        for sentence, entities in self.sentence_to_entity.items():
            sentence_hash_id = self.sentence_embedding_store.text_to_hash_id[sentence]
            entity_hash_ids = [
                self.entity_embedding_store.text_to_hash_id[ent] for ent in entities
            ]
            self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id] = entity_hash_ids
        self.add_entity_to_passage_edges(existing_passage_hash_id_to_entities)
        self.add_adjacent_passage_edges()
        self.augment_graph()
        output_graphml_path = os.path.join(
            self.config.working_dir, self.dataset_name, "LinearRAG.graphml"
        )
        os.makedirs(os.path.dirname(output_graphml_path), exist_ok=True)
        self.graph.write_graphml(output_graphml_path)
        logger.info(f"Saved graph to {output_graphml_path}")

    def retrieve(self, questions):
        self.entity_hash_ids = list(self.entity_embedding_store.hash_id_to_text.keys())
        self.entity_embeddings = np.array(self.entity_embedding_store.embeddings)
        self.passage_hash_ids = list(
            self.passage_embedding_store.hash_id_to_text.keys()
        )
        self.passage_embeddings = np.array(self.passage_embedding_store.embeddings)
        self.sentence_hash_ids = list(
            self.sentence_embedding_store.hash_id_to_text.keys()
        )
        self.sentence_embeddings = np.array(self.sentence_embedding_store.embeddings)
        self.node_name_to_vertex_idx = {
            v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()
        }
        self.vertex_idx_to_node_name = {
            v.index: v["name"] for v in self.graph.vs if "name" in v.attributes()
        }

        retrieval_results = []
        for question_info in tqdm(questions, desc="Retrieving"):
            question = question_info["question"]
            question_embedding = self.config.embedding_model.encode(
                question,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.config.batch_size,
            )
            (
                seed_entity_indices,
                seed_entities,
                seed_entity_hash_ids,
                seed_entity_scores,
            ) = self.get_seed_entities(question)
            if len(seed_entities) != 0:
                sorted_passage_hash_ids, sorted_passage_scores = (
                    self.graph_search_with_seed_entities(
                        question_embedding,
                        seed_entity_indices,
                        seed_entities,
                        seed_entity_hash_ids,
                        seed_entity_scores,
                    )
                )
                final_passage_hash_ids = sorted_passage_hash_ids[
                    : self.config.retrieval_top_k
                ]
                final_passage_scores = sorted_passage_scores[
                    : self.config.retrieval_top_k
                ]
                final_passages = [
                    self.passage_embedding_store.hash_id_to_text[passage_hash_id]
                    for passage_hash_id in final_passage_hash_ids
                ]
            else:
                sorted_passage_indices, sorted_passage_scores = (
                    self.dense_passage_retrieval(question_embedding)
                )
                final_passage_indices = sorted_passage_indices[
                    : self.config.retrieval_top_k
                ]
                final_passage_scores = sorted_passage_scores[
                    : self.config.retrieval_top_k
                ]
                final_passages = [
                    self.passage_embedding_store.texts[idx]
                    for idx in final_passage_indices
                ]
            result = {
                "question": question,
                "sorted_passage": final_passages,
                "sorted_passage_scores": final_passage_scores,
                "gold_answer": question_info["answer"],
            }
            retrieval_results.append(result)
        return retrieval_results

    def graph_search_with_seed_entities(
        self,
        question_embedding,
        seed_entity_indices,
        seed_entities,
        seed_entity_hash_ids,
        seed_entity_scores,
    ):
        entity_weights, actived_entities = self.calculate_entity_scores(
            question_embedding,
            seed_entity_indices,
            seed_entities,
            seed_entity_hash_ids,
            seed_entity_scores,
        )
        passage_weights = self.calculate_passage_scores(
            question_embedding, actived_entities
        )
        node_weights = entity_weights + passage_weights
        ppr_sorted_passage_indices, ppr_sorted_passage_scores = self.run_ppr(
            node_weights
        )
        return ppr_sorted_passage_indices, ppr_sorted_passage_scores

    def run_ppr(self, node_weights):
        reset_prob = np.where(
            np.isnan(node_weights) | (node_weights < 0), 0, node_weights
        )
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=self.config.damping,
            directed=False,
            weights="weight",
            reset=reset_prob,
            implementation="prpack",
        )

        doc_scores = np.array(
            [pagerank_scores[idx] for idx in self.passage_node_indices]
        )
        sorted_indices_in_doc_scores = np.argsort(doc_scores)[::-1]
        sorted_passage_scores = doc_scores[sorted_indices_in_doc_scores]

        sorted_passage_hash_ids = [
            self.vertex_idx_to_node_name[self.passage_node_indices[i]]
            for i in sorted_indices_in_doc_scores
        ]

        return sorted_passage_hash_ids, sorted_passage_scores.tolist()

    def calculate_passage_scores(self, question_embedding, actived_entities):
        passage_weights = np.zeros(len(self.graph.vs["name"]))
        dpr_passage_indices, dpr_passage_scores = self.dense_passage_retrieval(
            question_embedding
        )
        dpr_passage_scores = min_max_normalize(dpr_passage_scores)
        for i, dpr_passage_index in enumerate(dpr_passage_indices):
            total_entity_bonus = 0
            passage_hash_id = self.passage_embedding_store.hash_ids[dpr_passage_index]
            dpr_passage_score = dpr_passage_scores[i]
            passage_text_lower = self.passage_embedding_store.hash_id_to_text[
                passage_hash_id
            ].lower()
            for entity_hash_id, (
                entity_id,
                entity_score,
                tier,
            ) in actived_entities.items():
                entity_lower = self.entity_embedding_store.hash_id_to_text[
                    entity_hash_id
                ].lower()
                entity_occurrences = passage_text_lower.count(entity_lower)
                if entity_occurrences > 0:
                    denom = tier if tier >= 1 else 1
                    entity_bonus = (
                        entity_score * math.log(1 + entity_occurrences) / denom
                    )
                    total_entity_bonus += entity_bonus
            passage_score = self.config.passage_ratio * dpr_passage_score + math.log(
                1 + total_entity_bonus
            )
            passage_node_idx = self.node_name_to_vertex_idx[passage_hash_id]
            passage_weights[passage_node_idx] = passage_score
        return passage_weights

    def dense_passage_retrieval(self, question_embedding):
        question_emb = question_embedding.reshape(1, -1)
        question_passage_similarities = np.dot(
            self.passage_embeddings, question_emb.T
        ).flatten()
        sorted_passage_indices = np.argsort(question_passage_similarities)[::-1]
        sorted_passage_scores = question_passage_similarities[
            sorted_passage_indices
        ].tolist()
        return sorted_passage_indices, sorted_passage_scores

    def calculate_entity_scores(
        self,
        question_embedding,
        seed_entity_indices,
        seed_entities,
        seed_entity_hash_ids,
        seed_entity_scores,
    ):
        actived_entities = {}
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        for seed_entity_idx, seed_entity, seed_entity_hash_id, seed_entity_score in zip(
            seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores
        ):
            actived_entities[seed_entity_hash_id] = (
                seed_entity_idx,
                seed_entity_score,
                1,
            )
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score
        used_sentence_hash_ids = set()
        current_entities = actived_entities.copy()
        iteration = 1
        while len(current_entities) > 0 and iteration < self.config.max_iterations:
            new_entities = {}
            for entity_hash_id, (
                entity_id,
                entity_score,
                tier,
            ) in current_entities.items():
                if entity_score < self.config.iteration_threshold:
                    continue
                sentence_hash_ids = [
                    sid
                    for sid in list(
                        self.entity_hash_id_to_sentence_hash_ids[entity_hash_id]
                    )
                    if sid not in used_sentence_hash_ids
                ]
                if not sentence_hash_ids:
                    continue
                sentence_indices = [
                    self.sentence_embedding_store.hash_id_to_idx[sid]
                    for sid in sentence_hash_ids
                ]
                sentence_embeddings = self.sentence_embeddings[sentence_indices]
                question_emb = (
                    question_embedding.reshape(-1, 1)
                    if len(question_embedding.shape) == 1
                    else question_embedding
                )
                sentence_similarities = np.dot(
                    sentence_embeddings, question_emb
                ).flatten()
                top_sentence_indices = np.argsort(sentence_similarities)[::-1][
                    : self.config.top_k_sentence
                ]
                for top_sentence_index in top_sentence_indices:
                    top_sentence_hash_id = sentence_hash_ids[top_sentence_index]
                    top_sentence_score = sentence_similarities[top_sentence_index]
                    used_sentence_hash_ids.add(top_sentence_hash_id)
                    entity_hash_ids_in_sentence = (
                        self.sentence_hash_id_to_entity_hash_ids[top_sentence_hash_id]
                    )
                    for next_entity_hash_id in entity_hash_ids_in_sentence:
                        next_entity_score = entity_score * top_sentence_score
                        if next_entity_score < self.config.iteration_threshold:
                            continue
                        next_enitity_node_idx = self.node_name_to_vertex_idx[
                            next_entity_hash_id
                        ]
                        entity_weights[next_enitity_node_idx] += next_entity_score
                        new_entities[next_entity_hash_id] = (
                            next_enitity_node_idx,
                            next_entity_score,
                            iteration + 1,
                        )
            actived_entities.update(new_entities)
            current_entities = new_entities.copy()
            iteration += 1
        return entity_weights, actived_entities

    def get_seed_entities(self, question):
        question_entities = list(self.spacy_ner.question_ner(question))
        if len(question_entities) == 0:
            return [], [], [], []
        question_entity_embeddings = self.config.embedding_model.encode(
            question_entities,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.config.batch_size,
        )
        similarities = np.dot(self.entity_embeddings, question_entity_embeddings.T)
        seed_entity_indices = []
        seed_entity_texts = []
        seed_entity_hash_ids = []
        seed_entity_scores = []
        for query_entity_idx in range(len(question_entities)):
            entity_scores = similarities[:, query_entity_idx]
            best_entity_idx = np.argmax(entity_scores)
            best_entity_score = entity_scores[best_entity_idx]
            best_entity_hash_id = self.entity_hash_ids[best_entity_idx]

            seed_entity_indices.append(best_entity_idx)
            seed_entity_texts.append(question_entities[best_entity_idx])
            seed_entity_hash_ids.append(best_entity_hash_id)
            seed_entity_scores.append(best_entity_score)
        return (
            seed_entity_indices,
            seed_entity_texts,
            seed_entity_hash_ids,
            seed_entity_scores,
        )

    def add_entity_to_passage_edges(self, passage_hash_id_to_entities):
        passage_to_entity_count = defaultdict(int)
        passage_to_all_entities_count = defaultdict(int)

        for passage_hash_id, entities in passage_hash_id_to_entities.items():
            passage = self.passage_embedding_store.hash_id_to_text[passage_hash_id]
            for entity in entities:
                entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
                count = passage.count(entity)
                passage_to_entity_count[(passage_hash_id, entity_hash_id)] = count
                passage_to_all_entities_count[passage_hash_id] += count

        for (passage_hash_id, entity_hash_id), count in passage_to_entity_count.items():
            score = count / passage_to_all_entities_count[passage_hash_id]
            self.nodes_to_nodes_edges[passage_hash_id][entity_hash_id] = score

    def add_adjacent_passage_edges(self):
        passage_id_to_text = self.passage_embedding_store.hash_id_to_text
        index_pattern = re.compile(r"^(\d+):")
        indexed_items = [
            (int(match.group(1)), node_key)
            for node_key, text in passage_id_to_text.items()
            if (match := index_pattern.match(text.strip()))
        ]
        indexed_items.sort(key=lambda x: x[0])
        for i in range(len(indexed_items) - 1):
            current_node = indexed_items[i][1]
            next_node = indexed_items[i + 1][1]
            self.node_to_node_stats[current_node][next_node] = 1.0

    def augment_graph(self):
        self.add_nodes()
        self.add_edges()

    def add_nodes(self):
        existing_nodes = {v["name"] for v in self.graph.vs if "name" in v.attributes()}
        entity_hash_id_to_text = self.entity_embedding_store.get_hash_id_to_text()
        passage_hash_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        all_hash_id_to_text = {**entity_hash_id_to_text, **passage_hash_id_to_text}
        passage_hash_ids = set(passage_hash_id_to_text.keys())
        for hash_id, text in all_hash_id_to_text.items():
            if hash_id not in existing_nodes:
                self.graph.add_vertex(name=hash_id, content=text)
        self.node_name_to_vertex_idx = {
            v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()
        }
        self.passage_node_indices = [
            self.node_name_to_vertex_idx[passage_id]
            for passage_id in passage_hash_ids
            if passage_id in self.node_name_to_vertex_idx
        ]

    def add_edges(self):
        edges = []
        weights = []

        for node_hash_id, node_to_node_edges in self.nodes_to_nodes_edges.items():
            for neighbor_hash_id, weight in node_to_node_edges.items():
                edges.append((node_hash_id, neighbor_hash_id))
                weights.append(weight)
        self.graph.add_edges(edges)
        self.graph.es["weight"] = weights

    def extract_nodes_and_edges(
        self, passage_hash_id_to_entities, sentences_to_entities
    ):
        entity_nodes = set()
        sentence_nodes = set()
        entity_to_sentence = defaultdict(set)
        sentence_to_entity = defaultdict(set)

        for entities in passage_hash_id_to_entities.values():
            for entity in entities:
                entity_nodes.add(entity)
        for sentence, entities in sentences_to_entities.items():
            sentence_nodes.add(sentence)
            for entity in entities:
                entity_to_sentence[entity].add(sentence)
                sentence_to_entity[sentence].add(entity)

        return entity_nodes, sentence_nodes, entity_to_sentence, sentence_to_entity

    def save_ner_results(self, passage_hash_id_to_entities, sentences_to_entities):
        ner_results = {
            "passage_hash_id_to_entities": passage_hash_id_to_entities,
            "sentences_to_entities": sentences_to_entities,
        }
        with open(self.ner_results_path, "w") as f:
            json.dump(ner_results, f)
        logger.info(f"Saved NER results to {self.ner_results_path}")
