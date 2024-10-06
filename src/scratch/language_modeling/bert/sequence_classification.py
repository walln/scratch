"""Using BERT for sequence classification."""

import jax.numpy as jnp
from flax import nnx

from scratch.datasets.sequence_classification_dataset import (
    dummy_sequence_classification_dataset,
)
from scratch.datasets.utils import patch_datasets_warning
from scratch.language_modeling.bert.model import BertConfig, BertModel
from scratch.language_modeling.trainers.sequence_classification import (
    SequenceClassificationTrainer,
    SequenceClassificationTrainerConfig,
)


class BertForSequenceClassification(nnx.Module):
    """BERT model for sequence classification tasks.

    This module implements a BERT model for sequence classification tasks. The model
    consists of a BERT model followed by a linear layer for classification.
    """

    def __init__(self, config: BertConfig, num_labels: int, *, rngs: nnx.Rngs):
        """Initializes the sequence classification model.

        Args:
            config: Configuration for the BERT model.
            num_labels: Number of labels for classification.
            rngs: Random number generators.
        """
        self.bert = BertModel(config, rngs=rngs)
        self.classifier = nnx.Linear(config.hidden_size, num_labels, rngs=rngs)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
        token_type_ids: jnp.ndarray | None = None,
        position_ids: jnp.ndarray | None = None,
        train=False,
    ):
        """Forward pass of the sequence classification model.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            token_type_ids: Token type IDs.
            position_ids: Position IDs.
            train: Whether the model is in training mode. Defaults to False.
        """
        _, pooled_output = self.bert(
            input_ids, attention_mask, token_type_ids, position_ids, train
        )
        logits = self.classifier(pooled_output)
        return logits


if __name__ == "__main__":
    patch_datasets_warning()
    dataset = dummy_sequence_classification_dataset()
    config = BertConfig()
    model = BertForSequenceClassification(
        config, num_labels=dataset.metadata.num_classes, rngs=nnx.Rngs(0)
    )

    trainer_config = SequenceClassificationTrainerConfig(
        batch_size=2, num_labels=dataset.metadata.num_classes
    )
    trainer = SequenceClassificationTrainer(model, trainer_config=trainer_config)

    trainer.train_and_evaluate(dataset.train, dataset.test)
