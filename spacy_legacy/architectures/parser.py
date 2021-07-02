from typing import Optional, List
from thinc.types import Floats2d
from spacy.util import registry
from thinc.api import Model
from spacy.tokens import Doc
from spacy.compat import Literal

# TODO: replace with registered layer after spacy is released with the update
from spacy.ml.models.parser import build_tb_parser_model


@registry.architectures("spacy.TransitionBasedParser.v1")
def transition_parser_v1(
    tok2vec: Model[List[Doc], List[Floats2d]],
    state_type: Literal["parser", "ner"],
    extra_state_tokens: bool,
    hidden_width: int,
    maxout_pieces: int,
    use_upper: bool = True,
    nO: Optional[int] = None,
) -> Model:
    return build_tb_parser_model(
        tok2vec,
        state_type,
        extra_state_tokens,
        hidden_width,
        maxout_pieces,
        use_upper,
        nO,
    )
