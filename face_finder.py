#!/usr/bin/env python3

import face_recognition
import cv2
import dlib
import glob
import time
import os
import json_tricks
import click
from progress.bar import IncrementalBar
from moviepy import config
from moviepy import tools


def or_lists(list1, list2):
    if len(list1) > len(list2):
        longer_list = list1
        shorter_list = list2
    else:
        longer_list = list2
        shorter_list = list1

    or_list = longer_list
    for idx, e in enumerate(shorter_list):
        or_list[idx] = or_list[idx] or e


def flatten(list1):
    return [x for xs in list1 for x in xs]


def ffmpeg_extract_subclip(filename, t1, t2, targetname=None):
    """ Makes a new video file playing video file ``filename`` between
        the times ``t1`` and ``t2``. """
    name, ext = os.path.splitext(filename)
    if not targetname:
        T1, T2 = [int(1000 * t) for t in [t1, t2]]
        targetname = "%sSUB%d_%d.%s" % (name, T1, T2, ext)

    cmd = [config.get_setting("FFMPEG_BINARY"),
           "-y",
           "-ss", "%0.2f" % t1,
           "-i", filename,
           "-t", "%0.2f" % (t2 - t1),
           "-map", "0", "-vcodec", "copy", "-acodec", "copy", "-sn", targetname]

    tools.subprocess_call(cmd, logger=None)


def calculate_training_data(training_file, faces_folder) -> list:
    known_images = glob.glob(f"{faces_folder.rstrip('/')}/*")

    print("Training...")
    known_encodings = flatten(list(
        map(lambda file: face_recognition.face_encodings(face_recognition.load_image_file(file), num_jitters=10,
                                                         model="large"), known_images)))

    print("Saving training data")
    with open(training_file, 'w') as f:
        f.write(json_tricks.dumps(known_encodings))

    return known_encodings


@click.command()
@click.option('--video', '-v', 'videos', required=True, multiple=True, help="Video(s) to process.")
@click.option('--training-file', '-t', default="./face_training_data.json", help="File to store training data.")
@click.option('--faces-folder', '-f', default="./faces", help="Folder of images to train faces.")
@click.option('--scene-folder', '-s', default="./scenes", help="Folder to output scenes to.")
@click.option('--recalculate-training', default=None, type=bool,
              help="Re-calculate training data if it already exists.")
@click.option('--show-video', default=True, help="Show current video frame that is being processed.")
@click.option('--export-type', default="mp4", help="File type for scene exports. mov, mp4 etc.")
def main(videos, training_file, faces_folder, scene_folder, recalculate_training, show_video, export_type):
    if not dlib.DLIB_USE_CUDA:
        print("Warning: not using CUDA, expect facial recognition to be very slow!")

    if show_video:
        print("Warning: showing video slows down processing substantially!")

    if os.path.exists(training_file):
        if recalculate_training is None:
            recalculate_training = input("Would you like to re-calculate training data? (y/N): ").lower() \
                                   in ['y', 'yes', 'true']

        if recalculate_training:
            known_encodings = calculate_training_data(training_file, faces_folder)
        else:
            with open(training_file, 'r') as f:
                known_encodings = json_tricks.loads(f.read())
    else:
        known_encodings = calculate_training_data(training_file, faces_folder)

    print("Press 'q' to stop the current video and skip to the next one.")

    # only process so often to reduce lag
    process_skips = 10

    for video_file in videos:
        print()
        print("Working on", "'" + video_file + "'")

        video_file_name = os.path.basename(video_file)
        video_capture = cv2.VideoCapture(video_file)
        video_fps = video_capture.get(cv2.CAP_PROP_FPS)

        success = True

        # to calculate fps and store match frames
        frame_count = 0
        frame_count_last_second = 0
        total_frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        last_second = time.time()
        stats = ""

        # dict of encodings -> match
        face_matches = {}
        face_locations = []

        # frames where a match was present
        present_frames = []

        video_progress_bar = IncrementalBar('Processing Video', max=total_frame_count, suffix='%(percent)d%%')
        while success:
            # Grab a single frame of video
            success, frame = video_capture.read()

            if not success:
                video_progress_bar.next()
                continue

            # Only process so many times
            if frame_count % process_skips == 0:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn",
                                                                 number_of_times_to_upsample=1)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, model="large",
                                                                 num_jitters=1)

                for idx, face_encoding in enumerate(face_encodings):
                    # See if the face is a match for the known face(s)
                    face_matches[idx] = any(
                        face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5))

            # only process every other frame, for speed
            # process_this_frame = not process_this_frame

            if show_video:
                # Display the results
                for idx, (top, right, bottom, left) in enumerate(face_locations):
                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    if face_matches.get(idx, False):
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                # Display the resulting image
                cv2.imshow(video_file_name, frame)

            if any(face_matches.values()):
                present_frames.append(frame_count)

            frame_count += 1
            video_progress_bar.next()

            # every second print out some stats
            if int(time.time()) > last_second:
                current_fps = round((frame_count - frame_count_last_second) / (int(time.time()) - last_second))
                speed = current_fps / video_fps

                stats = f"FPS: {current_fps}/{round(video_fps)} {'({:.2f}x)'.format(speed)}"
                video_progress_bar.suffix = f'%(percent)d%% - {stats}'

                last_second = time.time()
                frame_count_last_second = frame_count

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()
        video_progress_bar.finish()

        print("Finished processing video.")

        false_positive_threshold = 120
        false_negative_threshold = 30
        grouped_list = [[]]
        for idx in range(len(present_frames)):
            current_frame = present_frames[idx]

            # if trying to get one before the first frame, just use the first
            if idx > 0:
                previous_frame = present_frames[idx - 1]
            else:
                previous_frame = current_frame

            # get distance between previous - current
            distance_behind = current_frame - previous_frame

            # if the distance is less than the threshold and more than 1
            # add in the missing frames that didn't get detected
            if distance_behind < false_negative_threshold:
                if distance_behind > 1:
                    for i in range(distance_behind - 1):
                        grouped_list[-1].append(previous_frame + i + 1)

            # if the distance was more than the threshold, start a new list, it's a different scene
            if distance_behind >= false_positive_threshold:
                grouped_list.append([])

            grouped_list[-1].append(current_frame)

        grouped_list = list(filter(lambda group: len(group) > 48, grouped_list))

        scene_folder_mod = scene_folder
        if len(videos) > 1:
            scene_folder_mod += f"/{os.path.splitext(video_file_name)[0]}"
        print(f"Exporting scenes to '{scene_folder_mod}'")

        if not os.path.exists(scene_folder_mod):
            os.makedirs(scene_folder_mod)

        export_loading_bar = IncrementalBar('Exporting Scenes', max=len(grouped_list))
        for idx, scene in enumerate(grouped_list):
            scene_file = f"{scene_folder_mod}/Scene{idx}.{export_type}"
            while os.path.exists(scene_file):
                scene_file = f"{scene_file[:-len(export_type) - 1]}-duplicate.{export_type}"
            ffmpeg_extract_subclip(video_file, scene[0] / video_fps, scene[-1] / video_fps, targetname=scene_file)
            export_loading_bar.next()
        export_loading_bar.finish()


if __name__ == "__main__":
    main()
