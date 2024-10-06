"""Using BERT for token classification."""

import jax.numpy as jnp
from flax import nnx

from scratch.datasets.token_classification_dataset import (
    dummy_token_classification_dataset,
)
from scratch.datasets.utils import patch_datasets_warning
from scratch.language_modeling.bert.model import BertConfig, BertModel
from scratch.language_modeling.trainers.token_classification import (
    TokenClassificationTrainer,
    TokenClassificationTrainerConfig,
)


class BertForTokenClassification(nnx.Module):
    """BERT model for token classification tasks.

    This module implements a BERT model for token classification tasks. The model
    consists of a BERT model followed by a linear layer for classification.
    """

    def __init__(self, config: BertConfig, num_labels: int, *, rngs: nnx.Rngs):
        """Initializes the token classification model.

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
        """Forward pass of the token classification model.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            token_type_ids: Token type IDs.
            position_ids: Position IDs.
            train: Whether the model is in training mode. Defaults to False.
        """
        sequence_outpout, pooled_output = self.bert(
            input_ids, attention_mask, token_type_ids, position_ids, train
        )
        logits = self.classifier(sequence_outpout)
        return logits


if __name__ == "__main__":
    patch_datasets_warning()
    dataset = dummy_token_classification_dataset(batch_size=16, num_labels=10)
    config = BertConfig()
    model = BertForTokenClassification(
        config, num_labels=dataset.metadata.num_labels, rngs=nnx.Rngs(0)
    )

    trainer_config = TokenClassificationTrainerConfig(
        batch_size=dataset.batch_size, num_labels=dataset.metadata.num_labels
    )
    trainer = TokenClassificationTrainer(model, trainer_config=trainer_config)

    trainer.train_and_evaluate(dataset.train, dataset.test)
