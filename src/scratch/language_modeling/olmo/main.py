"""OLMo: Accelerating the Science of Language Models.

This module contains the implementation of the OLMo model.
https://arxiv.org/abs/2402.00838

This file contains the implementation of the OLMo (Open Language Model). OLMo is a
transformer model family that is completely open sourced by allen-ai. The model is higly
performant, and incredibly well documented.

The implementation is based on the paper and codebase for "OLMo: Accelerating the
Science of Language Models" by the Allen Institute for AI.

Reference:
    Walsh, P., Bhagia, A., Kinney, R., Tafjord, O., Jha, A. H., Ivison, H., Magnusson,
    I., Wang, Y., Arora, S., Atkinson, D., Authur, R., Chandu, K. R., Cohan, A., Dumas,
    J., Elazar, Y., Gu, Y., Hessel, J., Khot, T., Merrill, W., Morrison, J.,
    Muennighoff, N., Naik, A., Nam, C., Peters, M. E., Pyatkin, V., Ravichander, A.,
    Schwenk, D., Shah, S., Smith, W., Strubell, E., Subramani, N., Wortsman, M.,
    Dasigi, P., Lambert, N., Richardson, K., Zettlemoyer, L., Dodge, J., Lo, K.,
    Soldaini, L., Smith, N. A., & Hajishirzi, H. (2024).
    OLMo: Accelerating the Science of Language Models.
    arXiv preprint arXiv:2402.00838. https://arxiv.org/abs/2402.00838

To actually train the model, we can use a CLMTrainer that can be imported from the
trainers module. I do not have the resources to run this locally so for testing I
just perform a forward pass and do not backpropgate.

The training has been validated in a colab notebook on TPUs.
"""

from flax import nnx

from scratch.datasets.causal_langauge_modeling_dataset import dummy_clm_dataset
from scratch.language_modeling.olmo.modeling.config import OLMoConfig
from scratch.language_modeling.olmo.modeling.model import OLMo

if __name__ == "__main__":
    config = OLMoConfig()
    model = OLMo(config=config, rngs=nnx.Rngs(0))

    dataset = dummy_clm_dataset(batch_size=4)

    for batch in dataset.train:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        input_ids = input_ids.numpy()
        attention_mask = attention_mask.numpy()

        model(input_ids)
        break

    print(model)
