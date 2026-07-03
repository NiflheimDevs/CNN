"""
nngraph/layer_catalogue.py

Small shared constant(s) describing the DSL's layer catalogue that
more than one phase needs to agree on. Kept deliberately tiny and
dependency-free -- same leaf-module reasoning as ast_nodes.py and
diagnostics.py.

This is NOT an attempt to consolidate EXPECTED_PARAM_TYPES
(semantic_analyzer.py) and PYTORCH_CLASS_MAP (codegen.py) into one
shared table -- that would be a bigger refactor of already-tested code
than the current feature needs, and was flagged as a possible future
cleanup, not done here. This module exists because STRUCTURAL_OPS
specifically is about to be needed independently by both
semantic_analyzer.py (to know which layer types are EXEMPT from the
new implicit-sum-on-multi-fan-in warning) and codegen.py (to know
which layer types have no backing nn.Module). Defining it twice would
mean the two files could silently drift out of sync the next time
someone adds a new structural op.
"""

# Layer types with no backing nn.Module -- evaluated as tensor
# expressions directly in forward(), never registered as self.<id>.
STRUCTURAL_OPS = {"Add", "Concat", "Residual", "Split"}

# Layer type -> the DSL parameter name whose value is derivable from
# the incoming tensor's shape, if the DSL author chose to omit it.
# Only "how many features/channels flow IN" is ever inferable this
# way -- "how many features/channels flow OUT" (out_features, out_ch,
# hidden_size, embedding_dim, ...) is always the DSL author's own
# design choice and can never be derived from upstream shape, so it
# deliberately has no entry here.
INFERABLE_PARAM = {
    "Linear": "in_features",
    "Conv2d": "in_ch",
    "Conv1d": "in_ch",
    "BatchNorm2d": "num_features",
    "LayerNorm": "normalized_shape",
    "MultiHeadAttn": "embed_dim",
    "LSTM": "input_size",
    "GRU": "input_size",
}