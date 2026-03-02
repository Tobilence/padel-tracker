from typing import List, Union
import cv2
import numpy as np

PADEL_COURT_LENGTH = 20.0
PADEL_COURT_WIDTH = 10.0

class CourtVizualizer:

    def __init__(
            self,
            # source_grounding_points: List[List[Union[float, int]]]   <- this will later be detected by model
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

    def vizualize_players_on_court(self, player_bboxes, flip_court=True):
        M = self._get_birdseye_matrix(flip_court=flip_court)
        return self._vizualize_court(player_bboxes, M)

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
        
    def _draw_2d_court(self, scale=40):
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

    def _vizualize_court(self, player_coords, matrix):
        """
        player_coords: List of 4 bounding boxes [[x1, y1, x2, y2], ...]
        matrix: The 3x3 Homography matrix calculated earlier
        """
        # 1. Create the 2D canvas (using the previous draw function)
        scale = 40 # 1 meter = 40 pixels
        view_2d = self._draw_2d_court() 
        
        # Define colors for the 4 players to tell them apart
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        
        for i, bbox in enumerate(player_coords):
            x1, y1, x2, y2 = bbox
            
            # Calculate the "Foot Point" (Bottom Center of the box)
            feet_x = (x1 + x2) / 2
            feet_y = y2 
            
            # 2. Perspective Transformation
            # We wrap the point in the specific shape OpenCV expects: (1, 1, 2)
            point = np.array([[[feet_x, feet_y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point, matrix)
            m_x, m_y = transformed[0][0] # Real world meters
            
            # 3. Map to 2D Canvas Pixels
            # If your court is 10m wide, draw_x will be between 0 and 400
            draw_x = int(m_x * scale)
            draw_y = int(m_y * scale)
            
            # Boundary Check: Ensure coordinates stay within the drawing canvas
            draw_x = np.clip(draw_x, 0, view_2d.shape[1] - 1)
            draw_y = np.clip(draw_y, 0, view_2d.shape[0] - 1)
            
            # 4. Draw on 2D map
            cv2.circle(view_2d, (draw_x, draw_y), 10, colors[i], -1)
            cv2.putText(view_2d, f"P{i+1}", (draw_x + 12, draw_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return view_2d