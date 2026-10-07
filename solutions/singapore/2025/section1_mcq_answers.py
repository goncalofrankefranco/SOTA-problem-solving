"""NOAI Singapore 2025 Final, Section 1 answer key.

The answer choices below are transcribed from the official answer PDF. The
short rationales are independently written summaries, not copies of question
stems. See the corresponding report for source and scoring limitations.
"""

ANSWER_KEY = {
    1: "B",
    2: "A",
    3: "B",
    4: "C",
    5: "A",
    6: "A",
    7: "B",
    8: "B",
    9: "B",
    10: "B",
    11: "B",
    12: "C",
    13: "C",
    14: "B",
    15: "B",
    16: "B",
    17: "B",
    18: "B",
    19: "B",
    20: "A",
}

RATIONALES = {
    1: "PCA finds structure without target labels.",
    2: "Bootstrap aggregation chiefly lowers estimator variance.",
    3: "Reinforcement learning learns from environment feedback.",
    4: "Underfit models do poorly on both training and validation data.",
    5: "Boosting combines weak learners in a dependent sequence.",
    6: "Adam adapts parameter-wise step sizes from gradient history.",
    7: "Vanishing gradients impede optimization in deep networks.",
    8: "Early stopping uses validation performance as regularization.",
    9: "Skip paths improve gradient propagation through deep stacks.",
    10: "A learning-rate schedule changes the step size over training.",
    11: "Pooling downsamples feature-map spatial dimensions.",
    12: "YOLO predicts detections in one forward-stage detector.",
    13: "The RPN proposes candidate regions for a detector.",
    14: "Depth-wise separable convolution reduces parameters and compute.",
    15: "Mask R-CNN predicts object instances and their masks.",
    16: "Attention weights input elements by their relevance to a query.",
    17: "BERT is a pretrained transformer used for NLP transfer learning.",
    18: "Beam search explores several candidate continuations at each step.",
    19: "BERT's CLS representation is commonly used for sequence classification.",
    20: "Causal masking blocks attention to later tokens.",
}


def score_answers(submitted: dict[int, str]) -> int:
    """Count exact matches to the published key (maximum 20)."""
    return sum(
        submitted.get(question, "").strip().upper() == answer
        for question, answer in ANSWER_KEY.items()
    )


if __name__ == "__main__":
    print(" ".join(ANSWER_KEY[i] for i in sorted(ANSWER_KEY)))
    print(f"Official-key consistency: {score_answers(ANSWER_KEY)}/20")
