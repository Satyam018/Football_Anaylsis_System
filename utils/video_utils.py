import cv2


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames=[]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames



def save_video(video_frames,video_path):
    format=cv2.VideoWriter_fourcc(*'XVID')
    video_writer=cv2.VideoWriter(video_path,format,24,(video_frames[0].shape[1],video_frames[0].shape[0]))
    for frame in video_frames:
        video_writer.write(frame)

    return video_writer.release()