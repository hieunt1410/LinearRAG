import spacy
from collections import defaultdict


class SpacyNER:
    def __init__(self, model_name="en_core_web_sm"):
        self.nlp = spacy.load(model_name)

    def batch_ner(self, hash_id_to_passage: dict, max_workers: int = 4) -> dict:
        passage_list = list(hash_id_to_passage.values())
        batch_size = len(passage_list) // max_workers + 1
        docs_list = self.nlp.pipe(
            passage_list,
            batch_size=batch_size,
            n_process=max_workers,
        )

        passage_hash_id_to_entities = defaultdict(set)
        sentences_to_entities = defaultdict(list)
        for idx, doc in enumerate(docs_list):
            passage_hash_id = list(hash_id_to_passage.keys())[idx]
            single_passage_hash_id_to_entities, single_sentence_to_entities = (
                self.extract_entities_from_sentences(doc, passage_hash_id)
            )
            passage_hash_id_to_entities.update(single_passage_hash_id_to_entities)
            for sent, entities in single_sentence_to_entities.items():
                sentences_to_entities[sent].extend(entities)

        return passage_hash_id_to_entities, sentences_to_entities

    def extract_entities_from_sentences(self, doc, passage_hash_id: str) -> dict:
        sentence_to_entities = defaultdict(list)
        unique_entities = set()
        passage_hash_id_to_entities = defaultdict(set)

        for ent in doc.ents:
            if ent.label_ in ["CARDINAL", "ORDINAL"]:
                continue
            sent_text = ent.sent.text.strip()
            entity_text = ent.text.strip()
            if entity_text not in sentence_to_entities[sent_text]:
                sentence_to_entities[sent_text].append(entity_text)
            unique_entities.add(entity_text)

        passage_hash_id_to_entities[passage_hash_id] = list(unique_entities)
        return passage_hash_id_to_entities, sentence_to_entities

    def question_ner(self, question: str):
        doc = self.nlp(question)
        question_entities = set()
        for ent in doc.ents:
            if ent.label_ == "ORDINAL" or ent.label_ == "CARDINAL":
                continue
            question_entities.add(ent.text.strip().lower())
        return question_entities
