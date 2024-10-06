"""BERT model for question answering tasks."""

import jax.numpy as jnp
from flax import nnx

from scratch.datasets.question_answering_dataset import dummy_question_answering_dataset
from scratch.datasets.utils import patch_datasets_warning
from scratch.language_modeling.bert.model import BertConfig, BertModel
from scratch.language_modeling.trainers.question_answering import (
    QuestionAnsweringTrainer,
    QuestionAnsweringTrainerConfig,
)


class BertForQuestionAnswering(nnx.Module):
    """BERT model for question answering tasks.

    This module implements a BERT model for question answering tasks. The model consists
    of a BERT model followed by a linear layer for predicting the start and end of the
    answer span.
    """

    def __init__(self, config: BertConfig, *, rngs: nnx.Rngs):
        """Initializes the question answering model.

        Args:
            config: Configuration for the BERT model.
            rngs: Random number generators.
        """
        self.bert = BertModel(config, rngs=rngs)
        self.qa_outputs = nnx.Linear(config.hidden_size, 2, rngs=rngs)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
        token_type_ids: jnp.ndarray | None = None,
        position_ids: jnp.ndarray | None = None,
        train=False,
    ):
        """Forward pass of the question answering model.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention Mask.
            token_type_ids: Token type IDs.
            position_ids: Position IDs.
            train: Whether the model is in training mode. Defaults to False.
        """
        sequence_output, _ = self.bert(
            input_ids, attention_mask, token_type_ids, position_ids, train
        )
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = jnp.split(logits, 2, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits


if __name__ == "__main__":
    patch_datasets_warning()
    dataset = dummy_question_answering_dataset()
    config = BertConfig()
    model = BertForQuestionAnswering(config, rngs=nnx.Rngs(0))

    trainer_config = QuestionAnsweringTrainerConfig(batch_size=2)
    trainer = QuestionAnsweringTrainer(model, trainer_config=trainer_config)

    trainer.train_and_evaluate(dataset.train, dataset.test)
