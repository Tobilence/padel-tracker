from typing import List, Union
import cv2
import numpy as np

PADEL_COURT_LENGTH = 20.0
PADEL_COURT_WIDTH = 10.0

class Court:

    def __init__(
            self,
            # source_grounding_points: List[List[Union[float, int]]]   <- this will later be detected by model
            flip=True # flips the court by x axis to align with typical video
    ) -> None:
        # Source is hardcoded for now
        LEFT_LOWER_NET = [90, 220]
        RIGHT_LOWER_NET = [540, 225]
        BOTTOM_T_LINE = [315, 340]
        TOP_T_LINE = [317, 190]
        self.src_grounding_points = np.array([
            LEFT_LOWER_NET,
            RIGHT_LOWER_NET,
            BOTTOM_T_LINE,
            TOP_T_LINE
        ])

        self.M = self._get_birdseye_matrix()
    
    def get_player_2d_coords(self, player_xyxy: List[List[int]]):
        """returns the players' 2d representation given the bounding box.
        Returns: Real world coordinates of the players
        """
        result = []
        for bbox in player_xyxy:
            x1, y1, x2, y2 = bbox
            
            # Calculate the "Foot Point" (Bottom Center of the box)
            feet_x = (x1 + x2) / 2
            feet_y = y2 

            point = np.array([[[feet_x, feet_y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point, self.M)
            m_x, m_y = transformed[0][0] # Real world meters

            result.append([m_x, m_y])

        return result
        

    def _get_birdseye_matrix(self, flip_court=True):
        """
        src_points: [left_net (lower end), right_net (lower end), near_T, far_T]
        flip_court: flip court on x axis
        returns: transformation matrix
        """

        dst_pts = np.array([
            [0, 10],        # left net
            [10, 10],       # right net
            [5, 3.05],      # near service T
            [5, 16.95]      # far service T
        ], dtype=np.float32)

        if flip_court:
            court_length = PADEL_COURT_LENGTH
            dst_pts[:,1] = court_length - dst_pts[:,1]

        matrix, _ = cv2.findHomography(self.src_grounding_points, dst_pts)
        return matrix
    