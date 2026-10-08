import numpy as np
import torch

from app.retrieval.visual_search import search_frames


class _Processor:
    def __call__(self, *, text, padding, return_tensors):
        assert text == ["person entering"]
        return {"input_ids": torch.tensor([[1]])}


class _Embedder:
    def get_text_features(self, **inputs):
        assert inputs["input_ids"].item() == 1
        return torch.tensor([[1.0, 0.0]])


def test_free_text_query_ranks_the_closest_frame_embedding():
    analysis = {
        "frames": [
            {"frame_index": 0, "timestamp": 0.0, "embedding": np.array([0.0, 1.0])},
            {"frame_index": 4, "timestamp": 2.0, "embedding": np.array([1.0, 0.0])},
        ]
    }

    matches = search_frames(analysis, "person entering", _Processor(), _Embedder(), "cpu")

    assert [match["frame_index"] for match in matches] == [4, 0]
    assert matches[0]["score"] > matches[1]["score"]
