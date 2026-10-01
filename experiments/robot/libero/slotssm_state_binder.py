from collections import deque

import numpy as np
import torch


class OnlineSubgoalStateBinder:
    def __init__(self, horizon, object_token_num, object_loss, text_encoder, embedding_table):
        self.horizon = horizon
        self.object_token_num = object_token_num
        self.object_loss = object_loss
        self.text_encoder = text_encoder
        self.embedding_table = embedding_table
        self.object_keys = []
        self.snapshots = deque(maxlen=horizon)
        self.last_states = {}
        self.unseen_state_labels = set()

    def update(self, recorder):
        if not recorder.agentview_boxes:
            return

        frame = recorder.agentview_boxes[-1]
        for key in frame:
            if key not in self.object_keys:
                self.object_keys.append(key)

        snapshot = {}
        for key in self.object_keys:
            if key not in frame:
                snapshot[key] = (np.zeros(5, dtype=np.float32), self.last_states.get(key, "0"))
                continue

            datum = frame[key]
            x, y, width, height = np.asarray(datum[1], dtype=np.float32)
            bbox = np.array(
                [x + width / 2, y + height / 2, width, height, 1.0],
                dtype=np.float32,
            )
            state_text = str(datum[2]) if len(datum) > 2 and datum[2] else "0"
            self.last_states[key] = state_text
            snapshot[key] = (bbox, state_text)

        self.snapshots.append(snapshot)

    def _embedding(self, state_text, device):
        if state_text not in self.embedding_table:
            embedding = self.text_encoder.encode_simple([state_text], device).detach().reshape(-1)
            self.embedding_table[state_text] = embedding
            self.unseen_state_labels.add(state_text)
        return self.embedding_table[state_text].reshape(-1).to(device)

    def bind(self, object_outputs):
        visual_tokens = object_outputs["visual_tokens"]
        device = visual_tokens.device
        batch_size, horizon = visual_tokens.shape[:2]
        if batch_size != 1 or horizon != self.horizon:
            raise ValueError(
                f"Expected one {self.horizon}-frame window, got batch={batch_size}, horizon={horizon}"
            )

        zero_embedding = self._embedding("0", device)
        if not self.object_keys or not self.snapshots:
            return zero_embedding.view(1, 1, 1, -1).expand(
                batch_size, horizon, self.object_token_num, -1
            )

        snapshots = list(self.snapshots)
        snapshots = [snapshots[0]] * (horizon - len(snapshots)) + snapshots
        target_bboxes = []
        target_states = []
        for object_key in self.object_keys:
            bbox_sequence = []
            state_sequence = []
            for snapshot in snapshots:
                bbox, state_text = snapshot.get(
                    object_key, (np.zeros(5, dtype=np.float32), "0")
                )
                bbox_sequence.append(bbox)
                state_sequence.append(self._embedding(state_text, device))
            target_bboxes.append(torch.as_tensor(np.stack(bbox_sequence), device=device))
            target_states.append(torch.stack(state_sequence))

        object_gts = {"bboxes": [torch.stack(target_bboxes)]}
        object_preds = {"bboxes": object_outputs["bboxes"].permute(0, 2, 1, 3)}
        indices = self.object_loss.get_match_indices(object_preds, object_gts)

        slot_states = [
            zero_embedding.unsqueeze(0).expand(horizon, -1)
            for _ in range(self.object_token_num)
        ]
        for predicted_slot, tracked_object in zip(*indices[0]):
            slot_states[predicted_slot.item()] = target_states[tracked_object.item()]

        return torch.stack(slot_states).permute(1, 0, 2).unsqueeze(0)