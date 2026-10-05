from pydantic import BaseModel


class ClassifyRequest(BaseModel):
    text: str


class BatchClassifyRequest(BaseModel):
    texts: list[str]


class LabelConfidence(BaseModel):
    label: str
    confidence: float


class ClassifyResponse(BaseModel):
    predicted_label: str
    confidence: float
    top3: list[LabelConfidence]


class BatchClassifyResponse(BaseModel):
    predictions: list[ClassifyResponse]


class HealthResponse(BaseModel):
    status: str
    model: str
    num_classes: int


class LabelsResponse(BaseModel):
    labels: list[str]
    num_classes: int



# Copy project into the right category folder
cp -r "/Users/prasanthreddy/Documents/AI Job Prep/Phase2_Deep_Learning/Projects/Semantic Document Classifier" ~/Documents/AI-and-Data-Science/Deep-Learning-Projects/

# Commit and push
cd ~/Documents/AI-and-Data-Science
git add .
git commit -m "Add Semantic Document Classifier"
git push
