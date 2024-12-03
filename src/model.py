import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

class SentimentModel(pl.LightningModule):
    def __init__(self, vocab_size, embed_dim, output_dim, pretrained_embeddings=None, freeze_embeddings=True):
        super().__init__()
        # Check if pretrained embeddings are provided
        if pretrained_embeddings is not None:
            # Initialize embedding layer with pretrained embeddings
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,  # Pretrained weights
                freeze=freeze_embeddings  # Freeze or unfreeze the embeddings
            )
        else:
            # Initialize random embeddings if no pretrained embeddings are provided
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.pool = nn.AdaptiveMaxPool1d(1)  # Global max pooling
        self.fc = nn.Linear(embed_dim, output_dim)  # Linear layer for binary classification
        self.loss = nn.CrossEntropyLoss() # Loss function for mulit-class classification

        # Metric for evaluation
        self.accuracy = torchmetrics.Accuracy(task="binary")

    def forward(self, x):
        """
        Forward pass: Embedding -> Pooling -> Fully Connected Layer.
        """
        embedded = self.embedding(x)
        # Rearrange dimensions for pooling (pooling expects input with shape [N, C, L])
        embedded = embedded.permute(0, 2, 1)  # 64, 707, 300 --> 64, 300, 707 # Shape: (batch_size, embed_dim, seq_len)
        pool = self.pool(embedded).squeeze(-1) # squeeze(2)
        fc = self.fc(pool)

        return fc

    def training_step(self, batch, batch_idx):
        """
        Single training step: Compute loss and log it.
        """
        texts, labels = batch
        logits = self(texts).squeeze(1)
        loss = self.loss(logits, torch.tensor(labels, dtype=torch.float))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Single validation step: Compute loss and accuracy, log metrics.
        """
        texts, labels = batch
        logits = self(texts).squeeze(1)
        loss = self.loss(logits, torch.tensor(labels, dtype=torch.float))
        preds = torch.sigmoid(logits) > 0.5
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", self.accuracy(preds, labels), prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Single test step: Compute loss and accuracy, log metrics.
        """
        texts, labels = batch
        logits = self(texts).squeeze(1)
        loss = self.loss(logits, torch.tensor(labels, dtype=torch.float))
        preds = torch.sigmoid(logits) > 0.5
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy", self.accuracy(preds, labels), prog_bar=True)
        # accuracy = Accuracy(task="multiclass", num_classes=3, top_k=2)
        # self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=3, top_k=1)
        # --> preds = logits

    def configure_optimizers(self):
        """
        Define optimizer for training.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-3)
