import torch.nn as nn
from transformers import AutoModel


class CustomClassifier(nn.Module):
    """DistilBERT backbone with a hand-rolled classification head on the [CLS] token.

    Backbone starts frozen — see unfreeze_last_n_layers() for the two-phase
    training strategy this is built around.
    """

    def __init__(self, checkpoint, num_classes, dropout_rate=0.5):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(checkpoint)
        hidden_size = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_n_layers(self, n=2):
        # only the last n layers — full unfreeze overfits fast on a dataset this size
        for layer in self.backbone.transformer.layer[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token aggregates sequence-level meaning
        return self.classifier(cls_output)
