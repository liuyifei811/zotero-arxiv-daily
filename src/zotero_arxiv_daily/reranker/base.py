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
            # === 课题核心：支架再狭窄 (ISR) ===
            "in-stent restenosis": 10,
            "stent restenosis": 8,
            "ISR": 8,
            "coronary stent": 7,
            "stent": 5,
            "restenosis": 7,
            "drug-eluting stent": 6,
            "DES": 4,
            "PCI": 6,

            # === 课题核心：冠周脂肪 (PCAT) ===
            "pericoronary adipose tissue": 8,
            "PCAT": 8,
            "pericoronary fat": 8,
            "epicardial adipose": 6,
            "fat attenuation index": 6,
            "FAI": 6,

            # === 课题核心：冠心病 / CCTA ===
            "coronary artery disease": 7,
            "coronary": 5,
            "CCTA": 7,
            "coronary CT angiography": 7,
            "coronary computed tomography angiography": 7,
            "cardiac CT": 6,
            "CT-FFR": 7,
            "fractional flow reserve": 7,
            "MACE": 6,
            "major adverse cardiovascular events": 6,

            # === 课题核心：Vision Transformer ===
            "vision transformer": 7,
            "ViT": 6,
            "swin transformer": 6,
            "transformer": 4,
            "attention mechanism": 4,
            "self-attention": 4,

            # === 方法相关：影像组学 / 分割 ===
            "radiomics": 6,
            "image-based": 4,
            "dual source CT": 5,
            "dual-source CT": 5,
            "metal artifact": 5,
            "metal artifact reduction": 5,
            "coronary plaque": 5,
            "plaque analysis": 5,
            "plaque": 3,
            "vessel segmentation": 5,
            "coronary segmentation": 6,
            "coronary artery segmentation": 6,

            # === 方法相关：多模态 / 可解释性 ===
            "multi-modal": 5,
            "multimodal": 5,
            "feature fusion": 5,
            "grad-cam": 4,
            "SHAP": 3,
            "explainability": 4,
            "interpretable": 4,

            # === 中等权重：CV+医学交叉 ===
            "deep learning": 3,
            "CNN": 2,
            "neural network": 2,
            "cardiovascular": 4,
            "cardiac": 4,
            "myocardial": 3,
            "atherosclerosis": 4,
            "intravascular": 4,
            "angiography": 3,
            "CTA": 3,
            "computed tomography": 2,
            "medical image": 3,
            "medical imaging": 3,
            "image segmentation": 3,
            "semantic segmentation": 2,
            "classification": 2,
        }

        bad_keywords = {
            # === 不相关器官/部位 ===
            "brain tumor": -5,
            "brain MRI": -4,
            "brain CT": -4,
            "lung nodule": -5,
            "lung cancer": -4,
            "pulmonary": -4,
            "liver tumor": -5,
            "liver segmentation": -4,
            "kidney": -4,
            "prostate": -4,
            "retinal": -4,
            "retina": -4,
            "skin lesion": -4,
            "breast cancer": -4,
            "mammogram": -4,
            "fetal": -4,

            # === 不相关领域 ===
            "autonomous driving": -8,
            "object detection": -2,
            "pedestrian": -4,
            "traffic": -4,
            "large language model": -6,
            "LLM": -6,
            "agent": -5,
            "chatgpt": -6,
            "NLP": -4,
            "language model": -4,
            "chatbot": -4,
            "speech recognition": -4,
            "recommendation system": -4,
            "robot": -3,
            "robotics": -3,
            "reinforcement learning": -3,
            "game": -3,

            # === 不相关 CV 方向 ===
            "GAN": -3,
            "diffusion": -5,
            "stable diffusion": -5,
            "image generation": -5,
            "text-to-image": -4,
            "speech": -4,
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
