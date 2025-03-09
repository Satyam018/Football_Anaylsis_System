import cv2

from utils import read_video,save_video
from trackers import Tracker
import os
import sys
from team_clustering import TeamAssigner


def main():
    video_frames=read_video('video/football_video.mp4') #read video


    model_path=os.getcwd()+'\models\\best.pt'
    stub_path=os.getcwd()+'\stubs\\track_stubs.pkl'
    trackers=Tracker(model_path)
    print('track')
    tracks =trackers.get_object_tracks(video_frames,True,stub_path)

    team_assigner=TeamAssigner()
    team_assigner.assign_team_colors(video_frames[0],tracks['players'][0])

    for frame_number,player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
             team = team_assigner.get_player_team(video_frames[frame_number],
                                                  track['bbox'],
                                                  player_id)
             tracks['players'][frame_number][player_id]['team']=team
             tracks['players'][frame_number][player_id]['team_color']=team_assigner.team_color[team]


    print('annotate')
    final_frames=trackers.annotate(video_frames,tracks)
    print('save')
    save_video(final_frames,'result_video/result_video.avi') #save video
    print('done')


def cropped_image(tracks,video_frames):
    for track_id, player in tracks['players'][0].items():
        bbox = player['bbox']
        frame = video_frames[0]

        croppedimage = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        cv2.imwrite('result_video/cropped.jpg', croppedimage)
        print('cropped')
        break

if __name__ == "__main__":
    main()

