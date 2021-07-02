from typing import Optional, List
from thinc.types import Floats2d
from thinc.api import Model, zero_init, use_ops

from spacy.tokens import Doc
from spacy.compat import Literal
from spacy.errors import Errors
from spacy.util import registry

# TODO: replace with registered layers after spacy is released with the update
from spacy.ml._precomputable_affine import PrecomputableAffine
from spacy.ml.tb_framework import TransitionModel


def TransitionBasedParser_v1(
    tok2vec: Model[List[Doc], List[Floats2d]],
    state_type: Literal["parser", "ner"],
    extra_state_tokens: bool,
    hidden_width: int,
    maxout_pieces: int,
    use_upper: bool = True,
    nO: Optional[int] = None,
) -> Model:

    chain = registry.get("layers", "chain.v1")
    list2array = registry.get("layers", "list2array.v1")
    Linear = registry.get("layers", "Linear.v1")

    if state_type == "parser":
        nr_feature_tokens = 13 if extra_state_tokens else 8
    elif state_type == "ner":
        nr_feature_tokens = 6 if extra_state_tokens else 3
    else:
        raise ValueError(Errors.E917.format(value=state_type))
    t2v_width = tok2vec.get_dim("nO") if tok2vec.has_dim("nO") else None
    tok2vec = chain(tok2vec, list2array(), Linear(hidden_width, t2v_width))
    tok2vec.set_dim("nO", hidden_width)
    lower = PrecomputableAffine(
        nO=hidden_width if use_upper else nO,
        nF=nr_feature_tokens,
        nI=tok2vec.get_dim("nO"),
        nP=maxout_pieces,
    )
    if use_upper:
        with use_ops("numpy"):
            # Initialize weights at zero, as it's a classification layer.
            upper = Linear(nO=nO, init_W=zero_init)
    else:
        upper = None
    return TransitionModel(tok2vec, lower, upper)
