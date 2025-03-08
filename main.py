
from utils import read_video,save_video
from trackers import Tracker
import os
import sys


def main():
    video_frames=read_video('video/football_video.mp4') #read video
    model_path=os.getcwd()+'\models\\best.pt'
    stub_path=os.getcwd()+'\stubs\\track_stubs.pkl'
    trackers=Tracker(model_path)
    trackers.get_object_tracks(video_frames,True,stub_path)

    save_video(video_frames,'result_video/result_video.avi') #save video



if __name__ == "__main__":
    main()

