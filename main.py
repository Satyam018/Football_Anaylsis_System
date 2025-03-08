
from utils import read_video,save_video
from trackers import Tracker
import os
import sys


def main():
    video_frames=read_video('video/football_video.mp4') #read video
    model_path=os.getcwd()+'\models\\best.pt'
    stub_path=os.getcwd()+'\stubs\\track_stubs.pkl'
    trackers=Tracker(model_path)
    print('track')
    tracks =trackers.get_object_tracks(video_frames,True,stub_path)
    print('annotate')
    final_frames=trackers.annotate(video_frames,tracks)
    print('save')
    save_video(final_frames,'result_video/result_video.avi') #save video
    print('done')



if __name__ == "__main__":
    main()

