from typing import List, Union
from core.logic import Player
import cv2
import numpy as np

PADEL_COURT_LENGTH = 20.0
PADEL_COURT_WIDTH = 10.0

class CourtVizualizer:

    def __init__(
            self,
            # source_grounding_points: List[List[Union[float, int]]]   <- this will later be detected by model
    ) -> None:
        pass

    def _draw_2d_court(self, scale=20):
        width = int(PADEL_COURT_WIDTH * scale)
        height = int(PADEL_COURT_LENGTH * scale)

        court = np.zeros((height, width, 3), dtype=np.uint8)

        white = (255, 255, 255)
        gray = (200, 200, 200)

        # Outer boundary
        cv2.rectangle(court, (0, 0), (width-1, height-1), white, 2)

        # Net (10m)
        net_y = int(10 * scale)
        cv2.line(court, (0, net_y), (width, net_y), white, 2)

        # Service lines (3.05m from baselines)
        service_near = int(3.05 * scale)
        service_far = int((20 - 3.05) * scale)

        cv2.line(court, (0, service_near), (width, service_near), gray, 2)
        cv2.line(court, (0, service_far), (width, service_far), gray, 2)

        # Center service line
        center_x = width // 2
        cv2.line(court, (center_x, service_near),
                (center_x, service_far), gray, 2)

        return court

    def vizualize_court(self, players: List[Player]):
        """
        player_coords: List of 4 bounding boxes [[x1, y1, x2, y2], ...]
        matrix: The 3x3 Homography matrix calculated earlier
        """
        # 1. Create the 2D canvas
        scale = 20  # defines how many pixels will represent 1 meter: eg 20 -> 1 meter=20 pixels
        view_2d = self._draw_2d_court() 
        
        for player in players:
            m_x, m_y = player.latest_court_position

            # Map to 2D Canvas Pixels
            draw_x = int(m_x * scale)
            draw_y = int(m_y * scale)
            
            # Boundary Check: Ensure coordinates stay within the drawing canvas
            draw_x = np.clip(draw_x, 0, view_2d.shape[1] - 1)
            draw_y = np.clip(draw_y, 0, view_2d.shape[0] - 1)
            
            # 4. Draw on 2D map
            cv2.circle(view_2d,
                       (draw_x, draw_y),
                       10,
                       player.viz_color,
                       -1
            )
            cv2.putText(view_2d,
                        f"P{player.id}",
                        (draw_x + 12, draw_y), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        player.viz_color,
                        1)

        return view_2d