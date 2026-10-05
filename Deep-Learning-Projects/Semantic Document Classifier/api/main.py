import torch
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer

from api.schemas import (
    BatchClassifyRequest,
    BatchClassifyResponse,
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    LabelConfidence,
    LabelsResponse,
)
from src.model import CustomClassifier

CHECKPOINT = "distilbert-base-uncased"
MAX_LENGTH = 256
MAX_BATCH_SIZE = 32
MODEL_PATH = "models/portfolio_classifier.pth"

# fixed order from sklearn.datasets.fetch_20newsgroups(...).target_names
CLASS_NAMES = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
]

app = FastAPI(title="Semantic Document Classifier")

# loaded once at import time, not inside a route handler — avoids reloading
# the 250MB+ checkpoint on every request
device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = CustomClassifier(CHECKPOINT, num_classes=len(CLASS_NAMES), dropout_rate=0.5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


def classify_text(text: str) -> ClassifyResponse:
    encoded = tokenizer(text, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        logits = model(encoded["input_ids"], encoded["attention_mask"])
        probs = torch.softmax(logits, dim=1)[0]

    top3_probs, top3_idx = torch.topk(probs, k=3)
    top3 = [
        LabelConfidence(label=CLASS_NAMES[idx], confidence=round(prob, 4))
        for idx, prob in zip(top3_idx.tolist(), top3_probs.tolist())
    ]

    return ClassifyResponse(
        predicted_label=top3[0].label,
        confidence=top3[0].confidence,
        top3=top3,
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model="distilbert-custom-head-20newsgroups", num_classes=len(CLASS_NAMES))


@app.get("/labels", response_model=LabelsResponse)
def labels():
    return LabelsResponse(labels=CLASS_NAMES, num_classes=len(CLASS_NAMES))


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    return classify_text(request.text)


@app.post("/classify/batch", response_model=BatchClassifyResponse)
def classify_batch(request: BatchClassifyRequest):
    if len(request.texts) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"batch size exceeds max of {MAX_BATCH_SIZE}")

    # one forward pass per text rather than a single batched tensor — simpler and fast
    # enough at this scale; worth batching properly if this ever needs higher throughput
    predictions = [classify_text(text) for text in request.texts]
    return BatchClassifyResponse(predictions=predictions)
