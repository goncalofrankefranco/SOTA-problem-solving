"""Few-shot candidate for NOAI China 2026 Task 1.

Uses the task-provided local bert-base-chinese checkpoint plus authored Chinese
class anchors. No network calls or external generative model are used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import zipfile

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import torch
from transformers import AutoModel, AutoTokenizer


INTENTS = [
    "游戏技巧", "游戏角色信息", "周边饭馆", "查找高端酒店", "查找地址",
    "健康知识", "查找医疗信息", "美容化妆技巧", "美食烹饪技巧", "软件开发问题",
    "软件使用问题", "查找小说剧情", "查找小说角色信息", "查找营业时间",
    "法律法条解释", "法律问题咨询",
]

# These short examples are original synthetic anchors, not organizer examples.
ANCHORS = {
    "游戏技巧": ["这个游戏怎么快速通关", "游戏里怎样打败最后的首领", "这关有什么打法技巧", "怎么提高游戏操作水平"],
    "游戏角色信息": ["这个游戏角色的技能是什么", "介绍一下角色的背景故事", "某个人物在游戏里属于什么职业", "这个角色怎么获得"],
    "周边饭馆": ["附近有什么好吃的餐厅", "帮我找周边的家常菜馆", "这一带哪里可以吃饭", "离我最近的面馆在哪里"],
    "查找高端酒店": ["推荐附近的五星级酒店", "帮我找一家高档度假酒店", "这个城市有哪些豪华酒店", "我要预订高端酒店"],
    "查找地址": ["请问某某大厦的具体地址是什么", "帮我查这家店在哪条路", "某公司的地址在哪里", "给我这个地点的详细地址"],
    "健康知识": ["熬夜对身体健康有什么影响", "如何保持良好的生活习惯", "日常怎样预防感冒", "喝水太少会有什么健康问题"],
    "查找医疗信息": ["附近哪家医院可以做核磁共振", "查一下这个科室的门诊时间", "哪里可以挂皮肤科的号", "帮我找提供疫苗接种的诊所"],
    "美容化妆技巧": ["新手怎样画自然的眼妆", "油性皮肤如何选择底妆", "分享一下日常化妆步骤", "怎么修眉看起来更自然"],
    "美食烹饪技巧": ["红烧肉怎么做才好吃", "煮米饭放多少水比较合适", "如何烤出外酥里嫩的鸡翅", "这道菜需要先放什么调料"],
    "软件开发问题": ["Python 怎样实现快速排序", "这段程序代码为什么会出现空指针错误", "如何用 SQL 连接两张表", "帮我解释一下这个 API 的实现"],
    "软件使用问题": ["这个软件怎么修改登录密码", "如何在应用里开启深色模式", "Excel 怎样冻结首行", "手机上怎么清除应用缓存"],
    "查找小说剧情": ["这本小说最后的结局是什么", "主角后来有没有找到失散的家人", "某部小说中这段情节发生了什么", "介绍一下故事的主要剧情"],
    "查找小说角色信息": ["小说中的女主角叫什么名字", "某个人物和主角是什么关系", "这本书里有哪些主要角色", "介绍一下小说反派的身份"],
    "查找营业时间": ["这家店今天几点开始营业", "博物馆周末的开放时间是什么", "请问餐厅晚上几点关门", "查一下银行的营业时间"],
    "法律法条解释": ["民法典这一条规定的意思是什么", "解释一下合同法中的相关条文", "某法律第十条具体怎样理解", "这项法律规定的适用范围是什么"],
    "法律问题咨询": ["这种情况我可以起诉对方吗", "遇到劳动纠纷应该怎么维权", "房东不退押金可以怎么处理", "我想咨询一下离婚财产分割问题"],
}

_to_simplified = str.maketrans({
    "這": "这", "遊": "游", "戲": "戏", "關": "关", "麼": "么", "敗": "败",
    "領": "领", "裡": "里", "屬": "属", "獲": "获", "週": "周", "邊": "边",
    "館": "馆", "麵": "面", "飯": "饭", "廳": "厅", "幫": "帮", "檔": "档",
    "級": "级", "華": "华", "預": "预", "訂": "订", "請": "请", "問": "问",
    "條": "条", "體": "体", "醫": "医", "療": "疗", "資": "资", "訊": "讯",
    "個": "个", "廈": "厦", "對": "对", "樣": "样", "紅": "红", "燒": "烧",
    "兩": "两", "張": "张", "職": "职", "針": "针", "調": "调", "話": "话",
    "書": "书", "實": "实", "學": "学", "會": "会", "風": "风", "氣": "气",
    "從": "从", "與": "与", "專": "专", "長": "长", "國": "国", "於": "于",
    "開": "开", "掛": "挂", "號": "号", "妝": "妆", "選": "选", "擇": "择",
    "驟": "骤", "軟": "软", "為": "为", "現": "现", "錯": "错", "誤": "误",
    "連": "连", "應": "应", "啟": "启", "凍": "冻", "結": "结", "機": "机",
    "取": "取", "後": "后", "來": "来", "係": "系", "紹": "绍", "營": "营",
    "業": "业", "幾": "几", "點": "点", "時": "时", "間": "间", "銀": "银",
    "詳": "详", "影": "影", "響": "响", "習": "习", "門": "门", "診": "诊",
    "式": "式", "碼": "码", "說": "说", "發": "发", "項": "项", "適": "适",
    "範": "范", "圍": "围", "訴": "诉", "勞": "劳", "動": "动", "糾": "纠",
    "紛": "纷", "該": "该", "維": "维", "權": "权", "東": "东", "處": "处",
    "諮": "咨", "詢": "询", "離": "离", "財": "财", "產": "产", "這": "这",
})


def canonicalize(text: str) -> str:
    converted = text.translate(_to_simplified)
    # Keep explicit coverage checks easy to repeat while editing the anchors.
    return converted


def last_user_utterance(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    turns = [line.strip()[4:].strip() for line in text.splitlines() if line.strip().startswith("usr:")]
    return turns[-1] if turns else text


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def training_anchors() -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    for idx, label in enumerate(INTENTS):
        prompts = [label, f"查询{label}", f"我想知道{label}相关信息"]
        prompts.extend(canonicalize(query) for query in ANCHORS[label])
        for query in prompts:
            texts.append(canonicalize(query))
            labels.append(idx)
            # Light surface variation yields extra lexical contexts without
            # changing the intent category.
            texts.append(canonicalize("请问" + query))
            labels.append(idx)
    return texts, labels


@torch.inference_mode()
def encode_texts(texts: list[str], tokenizer, model, device: torch.device, batch_size: int = 64) -> np.ndarray:
    chunks = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start:start + batch_size], padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        ).to(device)
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        # Mean pooling is less brittle than relying on a sentence-level CLS head
        # that this checkpoint was not trained with.
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        chunks.append(pooled.cpu().numpy())
    return np.vstack(chunks)


def train_models(train_records: list[dict], tokenizer, encoder, device: torch.device):
    label_to_id = {label: i for i, label in enumerate(INTENTS)}
    seed_texts, seed_labels = [], []
    for row in train_records:
        label = str(row.get("label", "")).strip()
        text = last_user_utterance(row.get("input"))
        if label in label_to_id and text:
            seed_texts.append(canonicalize(text))
            seed_labels.append(label_to_id[label])
    if len(seed_texts) != len(INTENTS) or set(seed_labels) != set(range(len(INTENTS))):
        raise ValueError("Expected exactly one provided labeled example for each of the 16 intent labels")

    synth_texts, synth_labels = training_anchors()
    all_texts = synth_texts + seed_texts
    all_labels = np.asarray(synth_labels + seed_labels, dtype=np.int64)
    embeddings = encode_texts(all_texts, tokenizer, encoder, device)
    weights = np.ones(len(all_labels), dtype=np.float32)
    weights[-len(seed_labels):] = 4.0
    semantic = LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced", random_state=42)
    semantic.fit(embeddings, all_labels, sample_weight=weights)

    lexical = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1, max_features=30000, sublinear_tf=True)
    sparse = lexical.fit_transform(all_texts)
    lexical_model = LogisticRegression(C=4.0, max_iter=1500, class_weight="balanced", random_state=42)
    lexical_model.fit(sparse, all_labels, sample_weight=weights)
    return semantic, lexical, lexical_model


def predict_records(records: list[dict], tokenizer, encoder, device, semantic, lexical, lexical_model) -> list[dict]:
    texts = [canonicalize(last_user_utterance(row.get("input"))) or " " for row in records]
    embedded = encode_texts(texts, tokenizer, encoder, device)
    semantic_p = semantic.predict_proba(embedded)
    lexical_p = lexical_model.predict_proba(lexical.transform(texts))
    # Semantic embeddings cover paraphrases; character n-grams catch explicit
    # domain vocabulary. The blend is a starting point, not a validated weight.
    probs = 0.72 * semantic_p + 0.28 * lexical_p
    labels = [INTENTS[i] for i in probs.argmax(axis=1)]
    return [{"label": label} for label in labels]


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default="/bohr/train-a3ld/v1")
    parser.add_argument("--data-dir", default=os.environ.get("DATA_PATH", "/bohr"))
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    train_dir = Path(args.train_dir)
    model_dir = args.model_dir or str(train_dir / "bert-base-chinese")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)

    train_records = read_jsonl(train_dir / "train.jsonl")
    semantic, lexical, lexical_model = train_models(train_records, tokenizer, encoder, device)
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        records = read_jsonl(data_dir / f"{split}.jsonl")
        predictions = predict_records(records, tokenizer, encoder, device, semantic, lexical, lexical_model)
        if len(predictions) != len(records):
            raise RuntimeError(f"Prediction count mismatch for {split}")
        write_jsonl(predictions, output_dir / f"submission_{split}.jsonl")
    with zipfile.ZipFile(output_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(output_dir / "submission_val.jsonl", "submission_val.jsonl")
        archive.write(output_dir / "submission_test.jsonl", "submission_test.jsonl")
    print(f"Wrote {output_dir / 'submission.zip'} on {device}; validation/test scoring is not performed locally")


if __name__ == "__main__":
    main()
