from typing import List, Any, Union
from . import ReIdEmbeddingModel, Court
from dataclasses import dataclass
import numpy as np
from utils import hungarian_cosine_match
import logging

PLAYER_CLASS_ID = 1

@dataclass
class Player:
    id: int
    latest_xyxy: np.ndarray
    tracker_id: int
    embedding: np.ndarray
    out_of_frame: bool
    latest_court_position: np.ndarray

    @classmethod
    def get_feet_coords(cls, xyxy):
        x1, _, x2, y2 = xyxy
        feet_x = (x1 + x2) / 2
        feet_y = y2
        return (feet_x, feet_y)
    
    @property
    def viz_color(self):
        if self.out_of_frame:
            return (128, 128, 128) # gray
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        return colors[self.id-1]


class IdentityManager:
    def __init__(self, embedding_model: ReIdEmbeddingModel, court: Court) -> None:
        # stores all 4 players with embeddings
        self.players: List[Player] = []
        self.embedding_model = embedding_model
        self.court = court
    
    @property
    def player_embeddings(self):
        return [player.embedding for player in self.players]

    @property
    def player_xyxy(self):
        return [player.latest_xyxy for player in self.players]

    @property
    def player_tracker_ids(self):
        return [player.tracker_id for player in self.players]

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
                for idx, (box, track_id, court_coords) in enumerate(zip(
                        detected_players.xyxy,
                        detected_players.tracker_id,
                        self.court.get_player_2d_coords(detected_players.xyxy)
                )):
                    self.players.append(
                        Player(
                            id=idx+1,
                            latest_xyxy=box,
                            tracker_id=track_id,
                            embedding=self.embedding_model.extract_embedding(frame, box).cpu().numpy(),
                            out_of_frame=False,
                            latest_court_position=court_coords
                        )
                    )
        elif len(self.players) == 4:
            self.update_eucledian_distance(detected_players)
            # Byte Track Updates
            # bt_updates = [(player, box)
            #               for player, box in zip(self.players, detected_players.xyxy)
            #               if player.tracker_id in detected_players.tracker_id]

            # for (player, box) in bt_updates:
            #     print(f"byte track update for player: {player.id} (bt: {player.tracker_id})")
            #     player.latest_xyxy = box
            #     # could update embedding here as well
            

            # ReID
            # input_embeddings = [self.embedding_model.extract_embedding(frame, box).cpu()
            #                     for box, tracker_id in zip(detected_players.xyxy, detected_players.tracker_id)
            #                     # if tracker_id not in self.player_tracker_ids
            #                     ]
            
            # unmatched_player_embeddings = [player.embedding
            #                                for player in self.players
            #                             #    if player.tracker_id not in detected_players.tracker_id
            #                                ]
            # print("reid update for: ", [player.id
            #                                for player in self.players
            #                             #    if player.tracker_id not in detected_players.tracker_id
            #                                ])
            # matches = hungarian_cosine_match(unmatched_player_embeddings, input_embeddings)
            # for match in matches:
            #     a_index = match["a_index"]
            #     b_index = match["b_index"]
            #     input_detection = detected_players[b_index]
            #     self.players[a_index].latest_xyxy = input_detection.xyxy[0]
            #     self.players[a_index].tracker_id = input_detection.tracker_id[0]
        else:
            raise Exception("Bad State: must have 0 or 4 players.")
    
    def update_eucledian_distance(self, detected_players):
        if len(detected_players) == 0:
            logging.debug("no ")
            return

        detection_court_coords = np.array(self.court.get_player_2d_coords(detected_players.xyxy))
        logging.debug("Detected Coords: \n", detection_court_coords)
        for player, player_court_coords in zip(
                self.players,
                np.array(self.court.get_player_2d_coords(self.player_xyxy))
            ):
            logging.debug(""*50)
            distances = np.linalg.norm(detection_court_coords - player_court_coords, axis=1)
            
            ASSIGNMENT_THRESHOLD = 2 # numbers higher than this represent ID switches as the players cannot "teleport"
            if np.min(distances) < ASSIGNMENT_THRESHOLD:
                idx_closest = np.argmin(distances)
                player.latest_xyxy = detected_players.xyxy[idx_closest] # distances[closest]
                player.latest_court_position = player_court_coords
                player.out_of_frame = False
            else:
                player.out_of_frame = True

            logging.debug(f"Player {player.id} @ {player_court_coords} - distances: {distances})")


    def retrieve_players(self) -> List[Player]:
        return self.players

