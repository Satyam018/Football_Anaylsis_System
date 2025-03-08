import numpy as np
from ultralytics import YOLO
import supervision as sv
import pickle as pk
import os
from utils import get_width, get_center
import cv2


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        pass

    def get_object_tracks(self, frames, read_from=False, stub_path=None):

        if read_from and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pk.load(f)
        detections = self.detect_frames(frames)
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }
        for frame_number, detection in enumerate(detections):
            cls_name = detection.names
            inverse_clas_name = {cls: num for num, cls in cls_name.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)

            for obj_ind, clss_id in enumerate(detection_supervision.class_id):  # replacing goalkeeper
                if (cls_name[clss_id] == 'goalkeeper'):
                    detection_supervision.class_id[obj_ind] = inverse_clas_name['player']

            # Track object
            detection_tracks = self.tracker.update_with_detections(detection_supervision)
            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})
            for frame_detection in detection_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == inverse_clas_name['player']:
                    tracks['players'][frame_number][track_id] = {"bbox": bbox}
                if cls_id == inverse_clas_name['referee']:
                    tracks['referees'][frame_number][track_id] = {"bbox": bbox}

            for frame_detection in detection_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == inverse_clas_name['ball']:
                    tracks['ball'][frame_number][1] = {"bbox": bbox}
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pk.dump(tracks, f)
        return tracks

    def detect_frames(self, frames):
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detection_batch
        return detections

    def annotate(self, video_frames, tracks):
        output_frames = []
        for frame_number, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_number]
            referee_dict = tracks['referees'][frame_number]
            ball_dict = tracks['ball'][frame_number]

            # draw playes
            for track_id, player_data in player_dict.items():
                bbox = player_data['bbox']
                frame = self.draw_ellipse(frame, bbox, (0, 255, 0), track_id)

            for _, referee_data in referee_dict.items():
                bbox = referee_data['bbox']
                frame = self.draw_ellipse(frame, bbox, (120, 110, 255))

            for track_id,ball in ball_dict.items():
                bbox = ball['bbox']
                print(bbox)
                frame = self.draw_triangle(frame, bbox, (0, 0, 255))

            output_frames.append(frame)

        return output_frames

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center(bbox)
        width = get_width(bbox)

        cv2.ellipse(frame, center=(x_center, y2),
                    axes=(int(width), int(0.40 * width)), angle=0,
                    startAngle=-45, endAngle=235,
                    color=color, thickness=2, lineType=cv2.LINE_4)

        rec_width = 40
        rec_height = 20
        x1_rec = x_center - rec_width // 2
        x2_rec = x_center + rec_width // 2
        y1_rec = y2 - rec_height // 2 + 17
        y2_rec = y2 + rec_height // 2 + 17

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rec), int(y1_rec)), (int(x2_rec), int(y2_rec)),
                          color, cv2.FILLED)

            x1_text = x1_rec + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(frame, str(track_id), (int(x1_text), int(y1_rec + 12)), cv2.FONT_HERSHEY_COMPLEX,
                        0.6, (0, 0, 0), 2)
        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_center(bbox)
        traingle_points = np.array([
            [x, y],  # First point
            [x - 10, y - 10],  # Second point
            [x + 10, y - 10]  # Third point
        ],np.int32
        )
        traingle_points = traingle_points.reshape((-1, 1, 2))
        # cv2.circle(frame, (x, y), 5, (0, 0, 0), cv2.FILLED)

        cv2.fillPoly(frame, [traingle_points],  color)
        cv2.drawContours(frame, [traingle_points], 0, (0,0,0), 2)

        return frame
