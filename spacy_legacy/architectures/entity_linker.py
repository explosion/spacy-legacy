from typing import Optional, Iterable, Callable, Dict, Union, List, Any
from itertools import islice
from thinc.api import Optimizer
from thinc.api import set_dropout_rate

import warnings

from spacy.kb import KnowledgeBase
from spacy.language import Language
from spacy.vocab import Vocab
from spacy.training import Example, validate_examples, validate_get_examples
from spacy.errors import Errors, Warnings
from spacy.util import registry

def EntityLinker_v1(
    tok2vec: Model, nO: Optional[int] = None
) -> Model[List[Doc], Floats2d]:
    chain = registry.get("layers", "chain.v1")
    clone = registry.get("layers", "clone.v1")
    with Model.define_operators({">>": chain, "**": clone}):
        token_width = tok2vec.maybe_get_dim("nO")

        Linear = registry.get("layers", "Linear.v1")
        output_layer = Linear(nO=nO, nI=token_width)

        list2ragged = registry.get("layers", "list2ragged.v1")
        reduce_mean = registry.get("layers", "reduce_mean.v1")
        residual = registry.get("layers", "residual.v1")
        Maxout = registry.get("layers", "Maxout.v1")

        model = (
            tok2vec
            >> list2ragged()
            >> reduce_mean()
            >> residual(Maxout(nO=token_width, nI=token_width, nP=2, dropout=0.0))  # type: ignore[arg-type]
            >> output_layer
        )
        model.set_ref("output_layer", output_layer)
        model.set_ref("tok2vec", tok2vec)
    return model
