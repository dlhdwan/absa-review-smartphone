import html
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForTokenClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.pipelines import pipeline as hf_pipeline
from underthesea import word_tokenize

APP_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ONE_STAGE_MODEL_NAME = "vinai/phobert-base-v2"
PARALLEL_MODEL_NAME = "vinai/phobert-base-v2"
ONE_STAGE_MAX_LEN = 256
PARALLEL_MAX_LEN = 256

ONE_STAGE_CHECKPOINT_PATH = Path(r"D:\absa review smartphone\checkpoints\one_stage\phobert_best.pt")
ATE_CHECKPOINT_PATH = Path(r"D:\absa review smartphone\checkpoints\two_stage\ate_best.pt")
ACSC_CHECKPOINT_PATH = Path(r"D:\absa review smartphone\checkpoints\two_stage\acsc_best.pt")
INTEGRATED_CHECKPOINT_PATH = Path(r"D:\absa review smartphone\checkpoints\absa_phobert_bilstm_tclstm\absa_phobert_bilstm_tclstm.pt")

ATE_LABEL_LIST = ["O", "B-ASPECT", "I-ASPECT"]
ATE_ID2LABEL = {i: label for i, label in enumerate(ATE_LABEL_LIST)}
ATE_LABEL2ID = {label: i for i, label in ATE_ID2LABEL.items()}

ONE_STAGE_ASPECT_KEYS = [
    "BATTERY",
    "CAMERA",
    "DESIGN",
    "FEATURES",
    "GENERAL",
    "PERFORMANCE",
    "PRICE",
    "SCREEN",
    "SER_ACC",
    "STORAGE",
]

ACSC_ASPECT_LIST = [
    "SCREEN",
    "CAMERA",
    "FEATURES",
    "BATTERY",
    "PERFORMANCE",
    "STORAGE",
    "DESIGN",
    "PRICE",
    "GENERAL",
    "SER&ACC",
]

ACSC_SENTIMENT_MAP = {
    0: "none",
    1: "positive",
    2: "neutral",
    3: "negative",
}

PIPELINE_OPTIONS = {
    "Pipeline 1: PhoBERT one-stage": {
        "kind": "one_stage",
        "checkpoint": ONE_STAGE_CHECKPOINT_PATH,
        "model_name": ONE_STAGE_MODEL_NAME,
    },
    "Pipeline 2: ATE + ACSC song song": {
        "kind": "parallel_two_branch",
        "ate_checkpoint": ATE_CHECKPOINT_PATH,
        "acsc_checkpoint": ACSC_CHECKPOINT_PATH,
        "model_name": PARALLEL_MODEL_NAME,
    },
    "Pipeline 3: PhoBERT + BiLSTM + TC-LSTM": {
        "kind": "integrated_two_stage",
        "checkpoint": INTEGRATED_CHECKPOINT_PATH,
    },
}

SPACE_RE = re.compile(r"\s+")
ZERO_WIDTH_RE = re.compile(r"[​‌‍﻿]")
WORD_RE = re.compile(r"\S+")


class MultiTaskACSC(nn.Module):
    def __init__(
        self,
        model_name: str = PARALLEL_MODEL_NAME,
        num_aspects: int = len(ACSC_ASPECT_LIST),
        num_sentiments: int = 4,
        dropout: float = 0.3,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.num_aspects = num_aspects
        self.num_sentiments = num_sentiments
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, hidden // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden // 2, num_sentiments),
                )
                for _ in range(num_aspects)
            ]
        )

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits = torch.stack([head(cls) for head in self.classifiers], dim=1)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = 0.0
            for idx in range(self.num_aspects):
                loss = loss + loss_fct(logits[:, idx, :], labels[:, idx])
            loss = loss / self.num_aspects

        return {"loss": loss, "logits": logits}


class FrozenPhoBERT(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state


class PhoBERTBiLSTMATE(nn.Module):
    def __init__(self, encoder, num_tags, hidden=128, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        h_size = encoder.model.config.hidden_size
        self.lstm = nn.LSTM(h_size, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 2, num_tags)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None):
        h = self.encoder(input_ids, attention_mask)
        out, _ = self.lstm(h)
        logits = self.classifier(self.dropout(out))
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits}


class PhoBERTMaskTCLSTMACSC(nn.Module):
    def __init__(self, encoder, num_labels, num_aspects, hidden=128, aspect_dim=64, dropout=0.3, class_weights=None):
        super().__init__()
        self.encoder = encoder
        h_size = encoder.model.config.hidden_size
        self.aspect_emb = nn.Embedding(num_aspects, aspect_dim)
        self.target_proj = nn.Linear(h_size + aspect_dim, h_size)
        self.lstm = nn.LSTM(h_size * 2, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 2, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids, attention_mask, evidence_mask, aspect_id, labels=None):
        h = self.encoder(input_ids, attention_mask)

        em = evidence_mask.float()
        valid = attention_mask.float()
        has_evidence = (em.sum(dim=1, keepdim=True) > 0).float()
        pool_mask = em * has_evidence + valid * (1 - has_evidence)

        evidence_vec = (h * pool_mask.unsqueeze(-1)).sum(dim=1)
        evidence_vec = evidence_vec / pool_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        aspect_vec = self.aspect_emb(aspect_id)
        target_info = self.target_proj(torch.cat([evidence_vec, aspect_vec], dim=-1))
        target_expand = target_info.unsqueeze(1).expand(-1, h.size(1), -1)

        x = torch.cat([h, target_expand], dim=-1)
        out, _ = self.lstm(x)

        pooled = (out * pool_mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / pool_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        logits = self.classifier(self.dropout(pooled))
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


def expand_local_context_mask(evidence_mask, attention_mask, window=2):
    em = (evidence_mask.float() > 0).float() * attention_mask.float()
    if window <= 0:
        return em
    expanded = torch.nn.functional.max_pool1d(
        em.unsqueeze(1),
        kernel_size=2 * window + 1,
        stride=1,
        padding=window,
    ).squeeze(1)
    return expanded * attention_mask.float()


class MaskAwareTCLSTMAttentionACSC(nn.Module):
    def __init__(self, encoder, num_labels, num_aspects, hidden=128, aspect_dim=64,
                 dropout=0.3, local_window=2, class_weights=None):
        super().__init__()
        self.encoder = encoder
        self.local_window = local_window
        h_size = encoder.model.config.hidden_size

        self.aspect_emb = nn.Embedding(num_aspects, aspect_dim)
        self.target_proj = nn.Linear(h_size + aspect_dim, h_size)
        self.lstm = nn.LSTM(h_size * 2, hidden, batch_first=True, bidirectional=True)
        self.attn_h = nn.Linear(h_size, h_size, bias=False)
        self.attn_t = nn.Linear(h_size, h_size, bias=False)
        self.attn_v = nn.Linear(h_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2 + h_size + h_size, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_labels),
        )
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights.float() if class_weights is not None else None)

    def forward(self, input_ids, attention_mask, evidence_mask, aspect_id, labels=None):
        h = self.encoder(input_ids, attention_mask)

        valid = attention_mask.float()
        em = evidence_mask.float() * valid
        has_evidence = (em.sum(dim=1, keepdim=True) > 0).float()

        target_pool_mask = em * has_evidence + valid * (1 - has_evidence)
        evidence_vec = (h * target_pool_mask.unsqueeze(-1)).sum(dim=1)
        evidence_vec = evidence_vec / target_pool_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        aspect_vec = self.aspect_emb(aspect_id)
        target_info = self.target_proj(torch.cat([evidence_vec, aspect_vec], dim=-1))

        local_mask = expand_local_context_mask(em, attention_mask, window=self.local_window)
        pool_mask = local_mask * has_evidence + valid * (1 - has_evidence)

        target_expand = target_info.unsqueeze(1).expand(-1, h.size(1), -1)
        x = torch.cat([h, target_expand], dim=-1)
        lstm_out, _ = self.lstm(x)

        attn_hidden = torch.tanh(self.attn_h(h) + self.attn_t(target_info).unsqueeze(1))
        attn_scores = self.attn_v(attn_hidden).squeeze(-1)
        attn_scores = attn_scores.masked_fill(pool_mask <= 0, -1e4)
        attn_weights = torch.softmax(attn_scores, dim=1)

        context_vec = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)
        lstm_vec = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        final_vec = torch.cat([lstm_vec, context_vec, target_info], dim=-1)

        logits = self.classifier(self.dropout(final_vec))
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


def normalize_aspect_key(aspect: str) -> str:
    return aspect.replace("SER&ACC", "SER_ACC")


def format_aspect_label(aspect: str) -> str:
    return aspect.replace("SER_ACC", "SER&ACC")


def build_one_stage_label_maps(aspect_keys: List[str]):
    one_stage_aspects = [format_aspect_label(a) for a in aspect_keys]
    sentiments = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    labels = []
    for prefix in ["B", "I"]:
        for aspect in sorted(one_stage_aspects):
            for sentiment in sentiments:
                labels.append(f"{prefix}-{aspect}#{sentiment}")
    labels.append("O")
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


@st.cache_resource(show_spinner=False)
def load_artifacts(pipeline_name: str):
    pipeline_cfg = PIPELINE_OPTIONS[pipeline_name]

    if pipeline_cfg["kind"] == "one_stage":
        checkpoint_path = pipeline_cfg["checkpoint"]
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint one-stage: {checkpoint_path}")

        one_stage_state = torch.load(checkpoint_path, map_location=DEVICE)
        one_stage_num_labels = int(one_stage_state["classifier.bias"].shape[0])
        one_stage_label2id, one_stage_id2label = build_one_stage_label_maps(ONE_STAGE_ASPECT_KEYS)
        tokenizer = AutoTokenizer.from_pretrained(pipeline_cfg["model_name"], use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(
            pipeline_cfg["model_name"],
            num_labels=one_stage_num_labels,
            id2label=one_stage_id2label,
            label2id=one_stage_label2id,
            ignore_mismatched_sizes=True,
        ).to(DEVICE)
        model.load_state_dict(one_stage_state, strict=True)
        model.eval()
        pipe = hf_pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=0 if DEVICE.type == "cuda" else -1,
        )

        return {
            "kind": "one_stage",
            "model_name": pipeline_cfg["model_name"],
            "checkpoint_source": str(checkpoint_path),
            "tokenizer": tokenizer,
            "one_stage_model": model,
            "one_stage_pipeline": pipe,
            "max_len": ONE_STAGE_MAX_LEN,
            "pipeline_name": pipeline_name,
        }

    if pipeline_cfg["kind"] == "parallel_two_branch":
        ate_checkpoint = pipeline_cfg["ate_checkpoint"]
        acsc_checkpoint = pipeline_cfg["acsc_checkpoint"]
        if not ate_checkpoint.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint ATE: {ate_checkpoint}")
        if not acsc_checkpoint.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint ACSC: {acsc_checkpoint}")

        tokenizer = AutoTokenizer.from_pretrained(pipeline_cfg["model_name"], use_fast=True)

        ate_model = AutoModelForTokenClassification.from_pretrained(
            pipeline_cfg["model_name"],
            num_labels=len(ATE_LABEL_LIST),
            id2label=ATE_ID2LABEL,
            label2id=ATE_LABEL2ID,
            ignore_mismatched_sizes=True,
        ).to(DEVICE)
        ate_state = torch.load(ate_checkpoint, map_location=DEVICE)
        ate_model.load_state_dict(ate_state, strict=True)
        ate_model.eval()

        acsc_state = torch.load(acsc_checkpoint, map_location=DEVICE)
        class_weights = acsc_state.get("class_weights")
        acsc_model = MultiTaskACSC(
            model_name=pipeline_cfg["model_name"],
            num_aspects=len(ACSC_ASPECT_LIST),
            num_sentiments=4,
            class_weights=class_weights,
        ).to(DEVICE)
        acsc_model.load_state_dict(acsc_state, strict=True)
        acsc_model.eval()

        return {
            "kind": "parallel_two_branch",
            "model_name": pipeline_cfg["model_name"],
            "checkpoint_source": f"ATE={ate_checkpoint} | ACSC={acsc_checkpoint}",
            "tokenizer": tokenizer,
            "ate_model": ate_model,
            "acsc_model": acsc_model,
            "max_len": PARALLEL_MAX_LEN,
            "pipeline_name": pipeline_name,
        }

    checkpoint_path = pipeline_cfg["checkpoint"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy integrated checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    config = checkpoint.get("config", {})
    model_name = config.get("model_name", "vinai/phobert-base")
    max_len = int(config.get("max_len", 192))
    hidden = int(config.get("lstm_hidden", 128))
    dropout = float(config.get("dropout", 0.3))

    mapping_dict = checkpoint.get("mapping", checkpoint) # Fallback lấy thẳng checkpoint nếu không có key mapping
    tag2id = mapping_dict["tag2id"]
    id2tag = {int(k): v for k, v in mapping_dict["id2tag"].items()} if isinstance(next(iter(mapping_dict["id2tag"].keys())), str) else mapping_dict["id2tag"]
    pol2id = mapping_dict["pol2id"]
    id2pol = {int(k): v for k, v in mapping_dict["id2pol"].items()} if isinstance(next(iter(mapping_dict["id2pol"].keys())), str) else mapping_dict["id2pol"]
    aspect2id = mapping_dict["aspect2id"]
    id2aspect = {int(k): v for k, v in mapping_dict["id2aspect"].items()} if isinstance(next(iter(mapping_dict["id2aspect"].keys())), str) else mapping_dict["id2aspect"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    encoder_stage1 = FrozenPhoBERT(model_name).to(DEVICE)
    encoder_stage2 = FrozenPhoBERT(model_name).to(DEVICE)

    stage1 = PhoBERTBiLSTMATE(
        encoder_stage1,
        num_tags=len(tag2id),
        hidden=hidden,
        dropout=dropout,
    ).to(DEVICE)

    stage1_state = dict(checkpoint["stage1_state_dict"])
    stage1_tag_weights = stage1_state.pop("loss_fn.weight", None)
    if stage1_tag_weights is not None:
        stage1.register_buffer("stage1_tag_weights", stage1_tag_weights.to(DEVICE))
        stage1.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, weight=stage1.stage1_tag_weights)

    stage2_state = dict(checkpoint["stage2_state_dict"])
    stage2_class_weights = stage2_state.get("loss_fn.weight")
    local_context_window = int(config.get("local_context_window", 0))
    has_attention_head = any(key.startswith("attn_") or key.startswith("classifier.0") for key in stage2_state.keys())
    if has_attention_head:
        stage2 = MaskAwareTCLSTMAttentionACSC(
            encoder_stage2,
            num_labels=len(pol2id),
            num_aspects=len(aspect2id),
            hidden=hidden,
            aspect_dim=64,
            dropout=dropout,
            local_window=local_context_window,
            class_weights=stage2_class_weights,
        ).to(DEVICE)
    else:
        stage2 = PhoBERTMaskTCLSTMACSC(
            encoder_stage2,
            num_labels=len(pol2id),
            num_aspects=len(aspect2id),
            hidden=hidden,
            aspect_dim=64,
            dropout=dropout,
            class_weights=stage2_class_weights,
        ).to(DEVICE)

    stage1.load_state_dict(stage1_state, strict=False)
    stage2.load_state_dict(stage2_state, strict=True)
    stage1.eval()
    stage2.eval()

    threshold_dict = checkpoint.get("thresholds", {})
    
    evidence_thresholds_by_aspect = threshold_dict.get("evidence_aspect", checkpoint.get("evidence_thresholds_by_aspect"))
    if isinstance(evidence_thresholds_by_aspect, dict):
        evidence_thresholds_by_aspect = {str(k): float(v) for k, v in evidence_thresholds_by_aspect.items()}

    best_neutral_threshold = threshold_dict.get("neutral", checkpoint.get("best_neutral_threshold"))
    
    default_evidence_threshold = checkpoint.get("default_evidence_threshold")
    best_global_threshold = checkpoint.get("best_global_threshold")
    scalar_evidence_threshold = checkpoint.get("evidence_threshold")

    effective_evidence_threshold = evidence_thresholds_by_aspect or best_global_threshold or default_evidence_threshold or scalar_evidence_threshold or 0.4
    
    return {
        "kind": "integrated_two_stage",
        "model_name": model_name,
        "checkpoint_source": str(checkpoint_path),
        "tokenizer": tokenizer,
        "stage1": stage1,
        "stage2": stage2,
        "tag2id": tag2id,
        "id2tag": id2tag,
        "pol2id": pol2id,
        "id2pol": id2pol,
        "aspect2id": aspect2id,
        "id2aspect": id2aspect,
        "max_len": max_len,
        "evidence_threshold": scalar_evidence_threshold,
        "default_evidence_threshold": default_evidence_threshold,
        "best_global_threshold": best_global_threshold,
        "evidence_thresholds_by_aspect": evidence_thresholds_by_aspect,
        "best_neutral_threshold": best_neutral_threshold,
        "effective_evidence_threshold": effective_evidence_threshold,
        "pipeline_name": pipeline_name,
    }


def preprocess_shared_text(text: str) -> Dict[str, str]:
    clean_text = unicodedata.normalize("NFC", str(text))
    clean_text = ZERO_WIDTH_RE.sub("", clean_text)
    clean_text = SPACE_RE.sub(" ", clean_text).strip()
    segmented = word_tokenize(clean_text, format="text") if clean_text else ""
    return {
        "clean_text": clean_text,
        "segmented": segmented,
    }


def preprocess_integrated_text(text: str) -> str:
    clean_text = unicodedata.normalize("NFC", str(text))
    clean_text = ZERO_WIDTH_RE.sub("", clean_text)
    clean_text = SPACE_RE.sub(" ", clean_text).strip()
    return clean_text


def prepare_ate_inputs(segmented_text: str, tokenizer, max_length: int):
    words = segmented_text.split()
    input_ids = [tokenizer.cls_token_id]
    attention_mask = [1]
    is_first_subword = [False]
    kept_words = []

    for word in words:
        sub_ids = tokenizer.encode(word, add_special_tokens=False)
        if not sub_ids:
            continue
        if len(input_ids) + len(sub_ids) + 1 > max_length:
            break
        kept_words.append(word)
        for idx, sub_id in enumerate(sub_ids):
            input_ids.append(sub_id)
            attention_mask.append(1)
            is_first_subword.append(idx == 0)

    input_ids.append(tokenizer.sep_token_id)
    attention_mask.append(1)
    is_first_subword.append(False)

    return {
        "words": kept_words,
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        "is_first_subword": is_first_subword,
    }


def get_word_spans(text: str):
    return [(match.group(0), match.start(), match.end()) for match in WORD_RE.finditer(text)]


def manual_encode_integrated(
    text: str,
    tokenizer,
    tag2id: Dict[str, int],
    max_len: int,
    target_spans: Optional[List[Tuple[int, int]]] = None,
    return_evidence_mask: bool = False,
):
    word_spans = get_word_spans(text)
    target_spans = target_spans or []

    sub_tokens, sub_offsets, sub_evidence = [], [], []

    for word, start, end in word_spans:
        toks = tokenizer.tokenize(word) or [tokenizer.unk_token]
        in_evidence = any(start < target_end and end > target_start for target_start, target_end in target_spans)

        for tok in toks:
            sub_tokens.append(tok)
            sub_offsets.append((start, end))
            sub_evidence.append(1.0 if in_evidence else 0.0)

    sub_tokens = sub_tokens[: max_len - 2]
    sub_offsets = sub_offsets[: max_len - 2]
    sub_evidence = sub_evidence[: max_len - 2]

    input_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(sub_tokens) + [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)
    offsets = [(-1, -1)] + sub_offsets + [(-1, -1)]
    evidence_mask = [0.0] + sub_evidence + [0.0]

    pad_len = max_len - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        offsets += [(-1, -1)] * pad_len
        evidence_mask += [0.0] * pad_len

    output = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "offsets": offsets,
    }
    if return_evidence_mask:
        output["evidence_mask"] = torch.tensor(evidence_mask, dtype=torch.float)
    return output


def mask_to_text(text: str, offsets, mask, threshold: float = 0.5):
    ranges = []
    for (start, end), value in zip(offsets, mask):
        if start < 0 or end < 0 or float(value) < threshold:
            continue
        if not ranges or start > ranges[-1][1] + 1:
            ranges.append([start, end])
        else:
            ranges[-1][1] = max(ranges[-1][1], end)
    return " | ".join(text[start:end] for start, end in ranges)


def get_valid_token_positions(enc) -> List[int]:
    positions = []
    attention_values = enc["attention_mask"].tolist() if hasattr(enc["attention_mask"], "tolist") else enc["attention_mask"]
    for idx, ((start, end), attn) in enumerate(zip(enc["offsets"], attention_values)):
        if int(attn) == 1 and start >= 0 and end >= 0:
            positions.append(idx)
    return positions


def resolve_integrated_threshold(threshold: Union[float, Dict[str, float], None], category: str, artifacts) -> float:
    if threshold is None:
        threshold = artifacts.get("evidence_thresholds_by_aspect")
        if threshold is None:
            threshold = artifacts.get("best_global_threshold")
        if threshold is None:
            threshold = artifacts.get("default_evidence_threshold")
        if threshold is None:
            threshold = artifacts.get("evidence_threshold")
        if threshold is None:
            threshold = 0.4

    if isinstance(threshold, dict):
        default_value = artifacts.get("default_evidence_threshold")
        if default_value is None:
            default_value = artifacts.get("evidence_threshold")
        if default_value is None:
            default_value = 0.4
        return float(threshold.get(category, default_value))

    return float(threshold)


def integrated_threshold_label(artifacts) -> str:
    threshold = artifacts.get("evidence_thresholds_by_aspect")
    if isinstance(threshold, dict) and threshold:
        return "per-aspect"

    scalar = artifacts.get("best_global_threshold")
    if scalar is None:
        scalar = artifacts.get("default_evidence_threshold")
    if scalar is None:
        scalar = artifacts.get("evidence_threshold")
    if scalar is None:
        scalar = 0.4
    return f"{float(scalar):.2f}"


def predict_polarity_ids_from_probs(probs: np.ndarray, pol2id: Dict[str, int], neutral_threshold: Optional[float] = None):
    probs = np.asarray(probs)
    if probs.ndim == 1:
        probs = probs[None, :]

    neutral_id = pol2id.get("neutral")
    if neutral_threshold is None or neutral_id is None:
        return probs.argmax(axis=1)

    non_neutral = probs.copy()
    non_neutral[:, neutral_id] = -1.0
    non_neutral_best = non_neutral.argmax(axis=1)
    return np.where(probs[:, neutral_id] >= float(neutral_threshold), neutral_id, non_neutral_best)


def predict_ate_spans(segmented_text: str, artifacts) -> List[str]:
    if not segmented_text:
        return []

    prepared = prepare_ate_inputs(segmented_text, artifacts["tokenizer"], artifacts["max_len"])
    batch = {
        "input_ids": prepared["input_ids"].to(DEVICE),
        "attention_mask": prepared["attention_mask"].to(DEVICE),
    }

    model = artifacts["ate_model"]
    model.eval()
    with torch.no_grad():
        pred_ids = model(**batch).logits[0].argmax(-1).detach().cpu().tolist()

    word_preds = [pred for pred, flag in zip(pred_ids, prepared["is_first_subword"]) if flag]
    spans = []
    current = None

    for word, pred in zip(prepared["words"], word_preds):
        if pred == ATE_LABEL2ID["B-ASPECT"]:
            if current:
                spans.append(" ".join(current))
            current = [word]
        elif pred == ATE_LABEL2ID["I-ASPECT"] and current:
            current.append(word)
        else:
            if current:
                spans.append(" ".join(current))
                current = None

    if current:
        spans.append(" ".join(current))

    return [span.replace("_", " ") for span in spans]


def predict_acsc_items(segmented_text: str, artifacts) -> List[Dict]:
    if not segmented_text:
        return []

    tokenizer = artifacts["tokenizer"]
    encoded = tokenizer(
        segmented_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=artifacts["max_len"],
    )
    batch = {key: value.to(DEVICE) for key, value in encoded.items()}

    model = artifacts["acsc_model"]
    model.eval()
    with torch.no_grad():
        logits = model(**batch)["logits"][0]
        probs = torch.softmax(logits, dim=-1).detach().cpu()

    outputs = []
    for aspect, prob_row in zip(ACSC_ASPECT_LIST, probs):
        pred_id = int(prob_row.argmax().item())
        if pred_id == 0:
            continue
        aspect_key = normalize_aspect_key(aspect)
        sentiment = ACSC_SENTIMENT_MAP[pred_id]
        outputs.append(
            {
                "aspect": aspect_key,
                "polarity": sentiment,
                "score": float(prob_row[pred_id].item()),
                "pair": f"{aspect_key}#{sentiment}",
            }
        )

    return sorted(outputs, key=lambda item: item["score"], reverse=True)


def predict_integrated_evidence_masks(text: str, artifacts, threshold: Union[float, Dict[str, float], None] = None):
    stage1 = artifacts["stage1"]
    tokenizer = artifacts["tokenizer"]
    tag2id = artifacts["tag2id"]
    max_len = artifacts["max_len"]

    stage1.eval()
    enc = manual_encode_integrated(text, tokenizer, tag2id, max_len)
    batch = {
        "input_ids": enc["input_ids"].unsqueeze(0).to(DEVICE),
        "attention_mask": enc["attention_mask"].unsqueeze(0).to(DEVICE),
    }

    with torch.no_grad():
        logits = stage1(**batch)["logits"][0]
        probs = torch.softmax(logits, dim=-1).detach().cpu()

    valid_positions = get_valid_token_positions(enc)
    outputs = []
    for category in artifacts["aspect2id"].keys():
        category_threshold = resolve_integrated_threshold(threshold, category, artifacts)
        b_id = tag2id[f"B-{category}"]
        i_id = tag2id[f"I-{category}"]
        scores = probs[:, b_id] + probs[:, i_id]

        mask = torch.zeros(max_len, dtype=torch.float)
        for pos in valid_positions:
            if float(scores[pos]) >= category_threshold:
                mask[pos] = float(scores[pos])

        if mask.sum().item() == 0:
            continue

        outputs.append(
            {
                "category": category,
                "mask": mask,
                "threshold": category_threshold,
                "confidence": float(mask[mask > 0].mean().item()),
                "evidence_text": mask_to_text(text, enc["offsets"], mask, threshold=category_threshold),
            }
        )

    return outputs


def predict_integrated_acsc_polarity(
    text: str,
    category: str,
    evidence_mask: torch.Tensor,
    artifacts,
    neutral_threshold: Optional[float] = None,
):
    stage2 = artifacts["stage2"]
    tokenizer = artifacts["tokenizer"]
    tag2id = artifacts["tag2id"]
    max_len = artifacts["max_len"]
    aspect2id = artifacts["aspect2id"]
    id2pol = artifacts["id2pol"]
    pol2id = artifacts["pol2id"]

    if neutral_threshold is None:
        neutral_threshold = artifacts.get("best_neutral_threshold")

    stage2.eval()
    enc = manual_encode_integrated(text, tokenizer, tag2id, max_len)
    batch = {
        "input_ids": enc["input_ids"].unsqueeze(0).to(DEVICE),
        "attention_mask": enc["attention_mask"].unsqueeze(0).to(DEVICE),
        "evidence_mask": evidence_mask.unsqueeze(0).to(DEVICE),
        "aspect_id": torch.tensor([aspect2id.get(category, 0)], dtype=torch.long).to(DEVICE),
    }

    with torch.no_grad():
        logits = stage2(**batch)["logits"][0]
        probs = torch.softmax(logits, dim=-1)

    probs_np = probs.detach().cpu().numpy()[None, :]
    pred_id = int(predict_polarity_ids_from_probs(probs_np, pol2id, neutral_threshold=neutral_threshold)[0])

    return {
        "polarity": id2pol[pred_id],
        "score": float(probs[pred_id].item()),
        "neutral_threshold": neutral_threshold,
    }


def predict_one_stage(text: str, artifacts) -> Dict:
    processed = preprocess_shared_text(text)
    predictions = artifacts["one_stage_pipeline"](processed["segmented"])
    outputs = []

    for pred in predictions:
        label = pred["entity_group"]
        word = pred["word"].replace("@@", "").replace("_", " ").strip()
        if "#" not in label:
            continue
        aspect, sentiment = label.split("#", 1)
        aspect_key = normalize_aspect_key(aspect)
        outputs.append(
            {
                "aspect": aspect_key,
                "evidence": word,
                "ate_score": float(pred["score"]),
                "polarity": sentiment.lower(),
                "score": float(pred["score"]),
                "pair": f"{aspect_key}#{sentiment.lower()}",
            }
        )

    return {
        "raw_text": text,
        "processed_text": processed["clean_text"],
        "segmented_text": processed["segmented"],
        "outputs": outputs,
    }


def predict_parallel_pipeline(text: str, artifacts) -> Dict:
    processed = preprocess_shared_text(text)
    segmented_text = processed["segmented"]
    ate_spans = predict_ate_spans(segmented_text, artifacts)
    acsc_outputs = predict_acsc_items(segmented_text, artifacts)

    return {
        "raw_text": text,
        "processed_text": processed["clean_text"],
        "segmented_text": segmented_text,
        "ate_spans": ate_spans,
        "acsc_outputs": acsc_outputs,
    }


def predict_integrated_pipeline(
    text: str,
    artifacts,
    threshold: Union[float, Dict[str, float], None] = None,
    neutral_threshold: Optional[float] = None,
) -> Dict:
    model_input_text = preprocess_integrated_text(text)
    evidence_items = predict_integrated_evidence_masks(model_input_text, artifacts, threshold=threshold)

    outputs = []
    for item in evidence_items:
        polarity = predict_integrated_acsc_polarity(
            model_input_text,
            item["category"],
            item["mask"],
            artifacts,
            neutral_threshold=neutral_threshold,
        )
        outputs.append(
            {
                "aspect": item["category"],
                "evidence": item["evidence_text"],
                "threshold": item["threshold"],
                "ate_score": item["confidence"],
                "polarity": polarity["polarity"],
                "score": polarity["score"],
                "neutral_threshold": polarity["neutral_threshold"],
                "pair": f"{item['category']}#{polarity['polarity']}",
            }
        )

    return {
        "raw_text": text,
        "model_input_text": model_input_text,
        "threshold_label": integrated_threshold_label(artifacts),
        "neutral_threshold": artifacts.get("best_neutral_threshold"),
        "outputs": outputs,
    }


def sentiment_meta(label: str):
    label = label.lower()
    if label == "positive":
        return {"emoji": "🟢", "color": "#16a34a", "bg": "#ecfdf5", "border": "#86efac", "name": "Tích cực"}
    if label == "negative":
        return {"emoji": "🔴", "color": "#dc2626", "bg": "#fef2f2", "border": "#fca5a5", "name": "Tiêu cực"}
    return {"emoji": "⚪", "color": "#6b7280", "bg": "#f9fafb", "border": "#d1d5db", "name": "Trung tính"}


def highlight_evidence(text: str, evidence: str) -> str:
    if not evidence:
        return html.escape(text)
    escaped = html.escape(text)
    parts = [part.strip() for part in evidence.split("|") if part.strip()]
    for part in sorted(parts, key=len, reverse=True):
        escaped_part = html.escape(part)
        if escaped_part in escaped:
            escaped = escaped.replace(
                escaped_part,
                f"<mark style='background:#fef08a;padding:0.1rem 0.2rem;border-radius:0.25rem'>{escaped_part}</mark>",
                1,
            )
    return escaped


def render_absa_card(item: Dict, source_text: str):
    meta = sentiment_meta(item["polarity"])
    aspect = format_aspect_label(item["aspect"])
    highlighted = highlight_evidence(source_text, item.get("evidence", ""))
    st.markdown(
        f"""
        <div style='border:1px solid {meta['border']}; background:{meta['bg']}; border-radius:16px; padding:16px; margin-bottom:12px;'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>
                <div style='font-size:1.1rem; font-weight:700; color:{meta['color']};'>{meta['emoji']} {aspect}</div>
                <div style='font-size:0.9rem; font-weight:600; color:{meta['color']};'>{meta['name']}</div>
            </div>
            <div style='margin-bottom:10px; line-height:1.6;'>{highlighted}</div>
            <div style='display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:8px;'>
                <div style='background:white; border-radius:10px; padding:10px; border:1px solid #e5e7eb;'>
                    <div style='font-size:0.8rem; color:#6b7280;'>Evidence</div>
                    <div style='font-weight:700;'>{html.escape(item.get('evidence', ''))}</div>
                </div>
                <div style='background:white; border-radius:10px; padding:10px; border:1px solid #e5e7eb;'>
                    <div style='font-size:0.8rem; color:#6b7280;'>ATE confidence</div>
                    <div style='font-weight:700;'>{item['ate_score']:.3f}</div>
                </div>
                <div style='background:white; border-radius:10px; padding:10px; border:1px solid #e5e7eb;'>
                    <div style='font-size:0.8rem; color:#6b7280;'>Pair</div>
                    <div style='font-weight:700;'>{item['pair']}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_parallel_sentiment_card(item: Dict):
    meta = sentiment_meta(item["polarity"])
    aspect = format_aspect_label(item["aspect"])
    st.markdown(
        f"""
        <div style='border:1px solid {meta['border']}; background:{meta['bg']}; border-radius:16px; padding:16px; margin-bottom:12px;'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>
                <div style='font-size:1.1rem; font-weight:700; color:{meta['color']};'>{meta['emoji']} {aspect}</div>
                <div style='font-size:0.9rem; font-weight:600; color:{meta['color']};'>{meta['name']}</div>
            </div>
            <div style='font-size:0.92rem; color:#374151; margin-bottom:10px;'>Nhánh ACSC chạy độc lập trên cùng câu đã được tiền xử lý và word-segment.</div>
            <div style='display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:8px;'>
                <div style='background:white; border-radius:10px; padding:10px; border:1px solid #e5e7eb;'>
                    <div style='font-size:0.8rem; color:#6b7280;'>Sentiment score</div>
                    <div style='font-weight:700;'>{item['score']:.3f}</div>
                </div>
                <div style='background:white; border-radius:10px; padding:10px; border:1px solid #e5e7eb;'>
                    <div style='font-size:0.8rem; color:#6b7280;'>Pair</div>
                    <div style='font-weight:700;'>{item['pair']}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_span_badges(spans: List[str]):
    if not spans:
        st.info("ATE chưa phát hiện span nào trong câu hiện tại.")
        return

    badges = "".join(
        f"<span style='display:inline-block;background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;border-radius:999px;padding:0.35rem 0.75rem;margin:0 0.4rem 0.5rem 0;font-weight:600;'>{html.escape(span)}</span>"
        for span in spans
    )
    st.markdown(badges, unsafe_allow_html=True)


def render_one_stage_results(result: Dict):
    outputs = result["outputs"]
    st.subheader("📊 Kết quả phân tích")
    if not outputs:
        st.warning("Pipeline one-stage không phát hiện được aspect nào trong câu hiện tại.")
        return

    summary_cols = st.columns(4)
    summary_cols[0].metric("Số aspect", len(outputs))
    summary_cols[1].metric("Text sau segment", len(result["segmented_text"].split()))
    summary_cols[2].metric("ATE TB", f"{np.mean([item['ate_score'] for item in outputs]):.3f}")
    summary_cols[3].metric("Sentiment TB", f"{np.mean([item['score'] for item in outputs]):.3f}")

    st.markdown("### 🔤 Text sau tiền xử lý + segment")
    st.code(result["segmented_text"] or "(rỗng)", language="text")

    st.markdown("### 📋 Bảng kết quả")
    rows = []
    for item in outputs:
        rows.append(
            {
                "Aspect": format_aspect_label(item["aspect"]),
                "Evidence": item["evidence"],
                "ATE confidence": round(item["ate_score"], 4),
                "Sentiment": sentiment_meta(item["polarity"])["name"],
                "Sentiment score": round(item["score"], 4),
                "Pair": item["pair"],
            }
        )
    st.dataframe(rows, use_container_width=True)

    st.markdown("### 🧩 Thẻ kết quả theo từng aspect")
    for item in outputs:
        render_absa_card(item, result["processed_text"])


def render_parallel_results(result: Dict):
    acsc_outputs = result["acsc_outputs"]
    st.subheader("📊 Kết quả pipeline ATE + ACSC song song")
    if not result["ate_spans"] and not acsc_outputs:
        st.warning("Pipeline song song chưa tạo được đầu ra rõ ràng với câu hiện tại.")
        return

    summary_cols = st.columns(4)
    summary_cols[0].metric("Số aspect", len(acsc_outputs))
    summary_cols[1].metric("Text sau segment", len(result["segmented_text"].split()))
    summary_cols[2].metric("ATE spans", len(result["ate_spans"]))
    avg_score = np.mean([item["score"] for item in acsc_outputs]) if acsc_outputs else 0.0
    summary_cols[3].metric("Sentiment TB", f"{avg_score:.3f}")

    st.markdown("### 🔤 Text sau tiền xử lý + segment")
    st.code(result["segmented_text"] or "(rỗng)", language="text")
    st.caption("Pipeline 2 vẫn giữ bản chất hai nhánh độc lập: ATE sinh spans, ACSC sinh aspect-sentiment, nhưng phần hiển thị được chuẩn hóa cho đồng bộ với Pipeline 1 và 3.")

    st.markdown("### 📋 Bảng kết quả")
    if not acsc_outputs:
        st.info("ACSC chưa dự đoán aspect nào khác lớp None.")
    else:
        rows = []
        for item in acsc_outputs:
            rows.append(
                {
                    "Aspect": format_aspect_label(item["aspect"]),
                    "Evidence": "ATE tách riêng ở phần dưới",
                    "ATE confidence": "-",
                    "Sentiment": sentiment_meta(item["polarity"])["name"],
                    "Sentiment score": round(item["score"], 4),
                    "Pair": item["pair"],
                }
            )
        st.dataframe(rows, use_container_width=True)

    st.markdown("### 🧩 Thẻ kết quả theo từng aspect")
    if acsc_outputs:
        for item in acsc_outputs:
            render_parallel_sentiment_card(item)

    st.markdown("### 🏷️ ATE spans được phát hiện")
    render_span_badges(result["ate_spans"])

    st.markdown("### 📝 Ghi chú về việc gộp output")
    if result["ate_spans"]:
        st.markdown("- **ATE** phát hiện các span: " + ", ".join(f"`{span}`" for span in result["ate_spans"]))
    else:
        st.markdown("- **ATE** chưa phát hiện span nào.")

    if acsc_outputs:
        pairs = ", ".join(f"`{item['pair']}`" for item in acsc_outputs)
        st.markdown(f"- **ACSC** dự đoán các cặp aspect-sentiment: {pairs}")
    else:
        st.markdown("- **ACSC** chưa dự đoán aspect-sentiment nào ngoài lớp None.")


def render_integrated_results(result: Dict):
    outputs = result["outputs"]
    st.subheader("📊 Kết quả pipeline integrated ")
    if not outputs:
        st.warning("Pipeline integrated chưa phát hiện được aspect nào trong câu hiện tại.")
        return

    summary_cols = st.columns(4)
    summary_cols[0].metric("Số aspect", len(outputs))
    summary_cols[1].metric("Evidence threshold", result["threshold_label"])
    neutral_label = f"{float(result['neutral_threshold']):.2f}" if result["neutral_threshold"] is not None else "argmax"
    summary_cols[2].metric("Neutral threshold", neutral_label)
    summary_cols[3].metric("Sentiment TB", f"{np.mean([item['score'] for item in outputs]):.3f}")

    st.markdown("### 🔤 Text đưa vào integrated pipeline")
    st.code(result["model_input_text"] or "(rỗng)", language="text")

    st.markdown("### 📋 Bảng kết quả")
    rows = []
    for item in outputs:
        rows.append(
            {
                "Aspect": format_aspect_label(item["aspect"]),
                "Evidence": item["evidence"],
                "ATE confidence": round(item["ate_score"], 4),
                "Sentiment": sentiment_meta(item["polarity"])["name"],
                "Sentiment score": round(item["score"], 4),
                "Pair": item["pair"],
            }
        )
    st.dataframe(rows, use_container_width=True)

    st.markdown("### 🧩 Thẻ kết quả theo từng aspect")
    for item in outputs:
        render_absa_card(item, result["model_input_text"])


def main():
    st.set_page_config(page_title="ABSA Demo", page_icon="📱", layout="wide")
    st.markdown(
        """
        <style>
        .small-note {color:#6b7280;font-size:0.92rem;}
        .hero {
            padding: 1.25rem 1.4rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #eff6ff, #f5f3ff);
            border: 1px solid #dbeafe;
            margin-bottom: 1rem;
        }
        .hero h1 {margin:0;font-size:2rem;color:#111827;}
        .hero p {margin:0.4rem 0 0 0;color:#374151;line-height:1.6;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class='hero'>
            <h1>📱 Demo ABSA cho review smartphone tiếng Việt</h1>
            <p>Ứng dụng hiện hỗ trợ 3 pipeline: <b>PhoBERT one-stage</b>, <b>ATE + ACSC song song</b> và <b>PhoBERT + BiLSTM + TC-LSTM</b>.<br>
            Mỗi pipeline thể hiện một cách tiếp cận khác nhau cho bài toán ABSA: dự đoán trực tiếp, hai nhánh song song và pipeline integrated end-to-end dựa trên evidence mask.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("🧠 Chọn pipeline")
        pipeline_name = st.radio("Chọn phiên bản mô hình để dự đoán", list(PIPELINE_OPTIONS.keys()))

    artifacts = load_artifacts(pipeline_name)

    with st.sidebar:
        st.subheader("⚙️ Cấu hình mô hình")
        st.write(f"**Pipeline:** {artifacts['pipeline_name']}")
        st.write(f"**Model name:** {artifacts['model_name']}")
        st.write(f"**Checkpoint source:** {artifacts['checkpoint_source']}")
        st.write(f"**Max length:** {artifacts['max_len']}")
        if artifacts["kind"] == "integrated_two_stage":
            st.write(f"**Evidence threshold:** {integrated_threshold_label(artifacts)}")
            neutral_label = f"{float(artifacts['best_neutral_threshold']):.2f}" if artifacts.get("best_neutral_threshold") is not None else "argmax"
            st.write(f"**Neutral threshold:** {neutral_label}")
        st.markdown("---")
        st.subheader("🧪 Ví dụ nhanh")
        examples = [
            "Màn hình đẹp nhưng pin tụt nhanh.",
            "Camera chụp nét, giá hợp lý, đáng tiền.",
            "Máy nóng, lag, chơi game không ổn.",
            "Nhân viên tư vấn nhiệt tình, pin trâu, máy khá mượt.",
        ]
        selected = st.selectbox("Chọn câu mẫu", [""] + examples)
        st.markdown("---")
        st.subheader("🔁 Luồng suy diễn")
        if artifacts["kind"] == "one_stage":
            st.markdown(
                "1. Tiền xử lý + word segmentation\n"
                "2. PhoBERT token classification dự đoán trực tiếp nhãn `aspect#sentiment`\n"
                "3. Gom span và hiển thị kết quả"
            )
        elif artifacts["kind"] == "parallel_two_branch":
            st.markdown(
                "1. Tiền xử lý chung\n"
                "2. Tạo `text segmented`\n"
                "3. Nhánh ATE chạy độc lập để lấy spans\n"
                "4. Nhánh ACSC chạy độc lập để lấy `(aspect, sentiment)`\n"
                "5. Gộp output trong giao diện"
            )
        else:
            st.markdown(
                "1. Chuẩn hóa text đầu vào\n"
                "2. Stage 1 dự đoán evidence mask theo từng aspect\n"
                "3. Stage 2 dùng `evidence_mask + aspect_id` để dự đoán sentiment\n"
                "4. Gom thành đầu ra end-to-end `aspect#polarity`"
            )

    default_text = selected or "Màn hình đẹp nhưng pin tụt nhanh, nhân viên tư vấn nhiệt tình."
    text = st.text_area(
        "Nhập câu review smartphone tiếng Việt",
        value=default_text,
        height=140,
        placeholder="Ví dụ: Màn hình đẹp nhưng pin tụt nhanh...",
    )

    if st.button("Phân tích ABSA", type="primary", use_container_width=True):
        if artifacts["kind"] == "one_stage":
            with st.spinner("Đang chạy PhoBERT one-stage..."):
                result = predict_one_stage(text, artifacts)
            render_one_stage_results(result)
        elif artifacts["kind"] == "parallel_two_branch":
            with st.spinner("Đang chạy ATE và ACSC song song..."):
                result = predict_parallel_pipeline(text, artifacts)
            render_parallel_results(result)
        else:
            with st.spinner("Đang chạy pipeline  integrated..."):
                result = predict_integrated_pipeline(text, artifacts)
            render_integrated_results(result)


if __name__ == "__main__":
    main()
