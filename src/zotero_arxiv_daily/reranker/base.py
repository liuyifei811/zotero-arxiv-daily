from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..protocol import Paper, CorpusPaper
import numpy as np
from typing import Type
class BaseReranker(ABC):
    def __init__(self, config:DictConfig):
        self.config = config

    def rerank(self, candidates: list[Paper], corpus: list[CorpusPaper]) -> list[Paper]:
        corpus = sorted(corpus, key=lambda x: x.added_date, reverse=True)
    
        time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
        time_decay_weight = time_decay_weight / time_decay_weight.sum()
    
        sim = self.get_similarity_score(
            [c.abstract for c in candidates],
        [c.abstract for c in corpus]
    )

    assert sim.shape == (len(candidates), len(corpus))

    scores = (sim * time_decay_weight).sum(axis=1) * 10

    medical_keywords = {
        "vision transformer": 4,
        "vit": 4,
        "swin": 4,
        "medical image": 3,
        "medical imaging": 3,
        "radiomics": 5,
        "coronary": 5,
        "coronary artery": 5,
        "coronary ct": 5,
        "cardiac ct": 5,
        "ccta": 5,
        "pcat": 8,
        "pericoronary adipose tissue": 8,
        "epicardial adipose tissue": 6,
        "fat attenuation index": 6,
        "plaque": 4,
        "stent": 6,
        "restenosis": 8,
        "in-stent restenosis": 10,
        "pci": 6,
        "ischemia": 4,
        "cardiovascular": 4,
        "heart": 3,
        "segmentation": 2,
        "classification": 2,
        "foundation model": 2,
    }

    bad_keywords = {
        "large language model": -6,
        "llm": -6,
        "agent": -5,
        "chatgpt": -6,
        "robot": -4,
        "robotics": -4,
        "autonomous driving": -8,
        "diffusion": -5,
        "stable diffusion": -5,
        "image generation": -5,
        "speech": -4,
        "nlp": -4,
        "recommendation system": -4,
    }

    for s, c in zip(scores, candidates):

        text = (
            f"{getattr(c, 'title', '')} "
            f"{getattr(c, 'abstract', '')}"
        ).lower()

        keyword_score = 0

        for k, v in medical_keywords.items():
            if k in text:
                keyword_score += v

        for k, v in bad_keywords.items():
            if k in text:
                keyword_score += v

        c.score = s * 0.7 + keyword_score * 0.3

    candidates = sorted(
        candidates,
        key=lambda x: x.score,
        reverse=True,
    )

    return candidates
    
    @abstractmethod
    def get_similarity_score(self, s1:list[str], s2:list[str]) -> np.ndarray:
        raise NotImplementedError

registered_rerankers = {}

def register_reranker(name:str):
    def decorator(cls):
        registered_rerankers[name] = cls
        return cls
    return decorator

def get_reranker_cls(name:str) -> Type[BaseReranker]:
    if name not in registered_rerankers:
        raise ValueError(f"Reranker {name} not found")
    return registered_rerankers[name]
