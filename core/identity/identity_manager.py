from typing import List, Any
from . import ReIdEmbeddingModel
from dataclasses import dataclass
import numpy as np
from utils import hungarian_cosine_match

PLAYER_CLASS_ID = 1

@dataclass
class Player:
    id: int
    latest_xyxy: np.ndarray
    tracker_id: int
    embedding: np.ndarray


class IdentityManager:
    def __init__(self, embedding_model: ReIdEmbeddingModel) -> None:
        # stores all 4 players with embeddings
        self.players: List[Player] = []
        self.embedding_model = embedding_model
    
    @property
    def player_embeddings(self):
        return [player.embedding for player in self.players]

    def update(self, frame, detections):
        """
        if len(self.players) == 0:
            if detection contains 4 players:
                compute 4 embeddings and store them in player_embeddings
            else
                return (wait for next frame; need 1 frame with 4 players for initial computation)

        if embeddings already exist:
            generate embeddings of the input detections
            compute cosine similarity to the stored ones
            match the stored ones to the newly computed ones
            return the 4 detections and the corret player_id for them (needed for the pipeline to draw the boxes)
        """
        if len(detections) == 0:
            return # no update
        
        detected_players = detections[detections.class_id == PLAYER_CLASS_ID]
        if len(self.players) == 0:
            if len(detected_players) == 4:
                for idx, (box, track_id) in enumerate(zip(detected_players.xyxy, detected_players.tracker_id)):
                    self.players.append(
                        Player(
                            id=idx+1,
                            latest_xyxy=box,
                            tracker_id=track_id,
                            embedding=self.embedding_model.extract_embedding(frame, box).cpu().numpy()
                        )
                    )
        elif len(self.players) == 4:
            input_embeddings = [self.embedding_model.extract_embedding(frame, box).cpu()
                                for box in detected_players.xyxy]
            matches = hungarian_cosine_match(self.player_embeddings, input_embeddings)
            for match in matches:
                a_index = match["a_index"]
                b_index = match["b_index"]
                input_detection = detected_players[b_index]
                self.players[a_index].latest_xyxy = input_detection.xyxy[0]
                self.players[a_index].tracker_id = input_detection.tracker_id[0]
        else:
            raise Exception("Bad State: must have 0 or 4 players.")

    
    def retrieve_players(self) -> List[Player]:
        return self.players

