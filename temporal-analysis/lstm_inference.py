"""
lstm_inference.py
=================
The inference-only surface of the sequence branch: turn events into the model's
input sequences and score them with an already-trained checkpoint. Nothing here
trains or updates a model.

The functions live in train_lstm_transformer.py because training and scoring
must build sequences identically; importing them from here keeps consumers such
as pipeline.py from depending on the training script by name. Importing that
module does not train (training only runs under its `__main__` guard).
"""
from train_lstm_transformer import build_event_sequences, prepare_score_frame, score_seqs  # noqa: F401

__all__ = ["prepare_score_frame", "build_event_sequences", "score_seqs"]
