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
            # print(detections)
            # print("""""""""""""""""""""""""""""""""""")
            # print(detection_tracks)
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
                output_frames.append(frame)
        return output_frames

    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])
        x_center, _ = get_center(bbox)
        width = get_width(bbox)


        cv2.ellipse(frame, center=(x_center, y2),
                    axes=(int(width), int(0.40 * width)),angle=0,
                    startAngle=45, endAngle=135,
                   color=color,thickness=2,lineType=cv2.LINE_4 )

        return frame
