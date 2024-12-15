'''
@author: Muhittin Gokmen
@date: 2024-02-22
'''
import os, sys, argparse, json, shutil

import evc
import numpy as np
import pandas as pd
import sqlite3
from contextlib import closing

import numpy as np
from hma_kines_detection import Kines  # import the Kines class
from hma_letter_detection import Letters  # import the Letters class
from hma_letter_histograms import LetterHistograms  # import the LetterHistograms class
from hma_kinemes_detection import Kinemes_Class  # import the Kinemes class

import csv

def save_histograms_as_columns_to_csv(histograms, filename):
    """
    Save multiple histograms as columns into a single CSV file for SVM training.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        max_length = max(len(histogram) for histogram in histograms)
        transposed = []
        for i in range(max_length):
            transposed.append([histogram[i] if i < len(histogram) else None for histogram in histograms])
        writer.writerows(transposed)

def extract_histograms_from_output(output):
    """
    Extract histograms from the provided output string.
    """
    histograms = []
    lines = output.split("\n")
    for line in lines:
        if line.startswith("histogram:"):
            histogram_values = line.replace("histogram:", "").strip()
            histograms.append([float(value) for value in histogram_values.strip("[]").split(", ")])
    return histograms

class PoseKinemeHistogram():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.opts = {"part_length": 1800,
                     "sigma": 1.6,
                     "num_intervals": 3,
                     "assumed_blur": 0.5,
                     "signal_border_width": 5,
                     "contrast_threshold": 0.01,
                     "min_size": 32.0,
                     'face_number': 1,
                     'speaking_id': 1,
                     'conf_part_id': 1,
                     "fps": 25,
                     "min_silence": 5,
                     "silent_block_size": 10,
                     "histogram_segment_size": -1,
                     "histogram_segment_number": 8,
                     "histogram_segment_type": "segment_size",
                     "selected_levels": [1,4,7],
                     "letter_dictionary": ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o',
                                           'p','q','r','s','t','u','v','w','x','y','z','_'],
                     "hist_value": "magnitude",
                     "histogram_type": 'kineme_levels_and_combined',
                     "kineme_type": 'single_letters',
                     "speaking_listening": 'separated_and_combined',
                     "video_length": 0,
                     "show_histogram_plots": False,
                     "save_histogram_plots": False,
                     'save_kines_to_file': False,
                     'save_kines_plots_to_file': False,
                     'save_letters_to_file': True,
                     'save_summary_to_file': True,
                     'save_letters_to_db': False,
                     'save_histograms_to_file': True,
                     'save_augmented_videos_to_file': False,
                     'save_kineme_clips_to_file': False,
                     }

        kines_option_keys = ["part_length", "sigma", "num_intervals", "assumed_blur", "signal_border_width",
                             "contrast_threshold", "min_size",
                             'face_number', 'speaking_id', 'conf_part_id', "fps", "min_silence", "silent_block_size",
                             "save_kines_to_file", "save_kines_plots_to_file", "video_length"]
        self.kines_opts = {k: self.opts[k] for k in kines_option_keys}
        letter_option_keys = ["min_silence", "silent_block_size", 'speaking_listening', 'video_length', 'face_number', 'speaking_id',
                              'conf_part_id', "fps", "save_letters_to_file", "save_letters_to_db",
                              "save_summary_to_file"]
        self.letter_opts = {k: self.opts[k] for k in letter_option_keys}
        kineme_option_keys = ["kineme_type", 'speaking_listening', "selected_levels", "fps", "hist_value",
                              "histogram_type", "video_length",
                              "save_kineme_clips_to_file", "save_summary_to_file", "save_augmented_videos_to_file"]
        self.kineme_opts = {k: self.opts[k] for k in kineme_option_keys}
        hist_option_keys = ["fps", "histogram_segment_size", "histogram_segment_number", "histogram_segment_type",
                            'speaking_listening', "selected_levels", "letter_dictionary", "hist_value",
                            "histogram_type", "kineme_type", "video_length", "show_histogram_plots",
                            "save_histogram_plots", "save_histograms_to_file"]
        self.hist_opts = {k: self.opts[k] for k in hist_option_keys}

    def compute_(self, x, s):
        x[:, [0, 1]] = x[:, [1, 0]]
        x = np.rad2deg(x)

        self.video_length = len(x[:, 0].tolist())
        self.kines = Kines(self.kines_opts)
        self.kines_pitch, self.kines_yaw, self.kines_roll = self.kines.get_kines(x)

        self.letterObj = Letters(self.letter_opts)
        self.all_letters = self.letterObj.get_letters(self.kines_yaw, self.kines_pitch, self.kines_roll, x)
        self.all_letters, self.speaker_letters, self.listener_letters = self.letterObj.get_speaker_listener_letters(
            self.all_letters, s)

        self.kinemesObj = Kinemes_Class(self.kineme_opts)
        self.detected_kinemes_nonseparated = self.kinemesObj.get_kinemes(self.all_letters)
        self.detected_kinemes_speaker = self.kinemesObj.get_kinemes(self.speaker_letters)
        self.detected_kinemes_listener = self.kinemesObj.get_kinemes(self.listener_letters)

        self.hist_kineme = LetterHistograms(self.hist_opts, self.kwargs)
        self.kineme_histogram_nonseparated = self.hist_kineme.generate_histograms(self.detected_kinemes_nonseparated)
        self.kineme_histogram_speaker = self.hist_kineme.generate_histograms(self.detected_kinemes_speaker)
        self.kineme_histogram_listener = self.hist_kineme.generate_histograms(self.detected_kinemes_listener)
        self.histogram = self.hist_kineme.select_histogram(self.kineme_histogram_nonseparated,
                                                           self.kineme_histogram_speaker,
                                                           self.kineme_histogram_listener)

        return self.histogram


def save_results(x, pk, angles_filename, kwargs):
    x[:, [0, 1]] = x[:, [1, 0]]
    x = np.rad2deg(x)
    if pk.opts["save_kines_to_file"]:
        kine_obj = Kines(pk.kines_opts)
        if not os.path.exists(kwargs['keypoints_yaw_folder']):
            os.makedirs(kwargs['keypoints_yaw_folder'])
        if not os.path.exists(kwargs['keypoints_pitch_folder']):
            os.makedirs(kwargs['keypoints_pitch_folder'])
        if not os.path.exists(kwargs['keypoints_roll_folder']):
            os.makedirs(kwargs['keypoints_roll_folder'])
        output_yaw = f"{kwargs['keypoints_yaw_folder']}{kwargs['filename_base']}_yaw_kines.csv"
        output_pitch = f"{kwargs['keypoints_pitch_folder']}{kwargs['filename_base']}_pitch_kines.csv"
        output_roll = f"{kwargs['keypoints_roll_folder']}{kwargs['filename_base']}_roll_kines.csv"
        kine_obj.write_kines_to_csv(pk.kines_yaw, pk.kines_pitch, pk.kines_roll, output_yaw, output_pitch, output_roll)

    if pk.opts["save_kines_plots_to_file"]:
        kine_obj = Kines(pk.kines_opts)
        pitch = x[:, 0].tolist()
        yaw = x[:, 1].tolist()
        roll = x[:, 2].tolist()
        if not os.path.exists(kwargs['keypoint_plots_folder']):
            os.makedirs(kwargs['keypoint_plots_folder'])
        output_yaw = f"{kwargs['keypoint_plots_folder']}{kwargs['filename_base']}_yaw_kines.png"
        output_pitch = f"{kwargs['keypoint_plots_folder']}{kwargs['filename_base']}_pitch_kines.png"
        output_roll = f"{kwargs['keypoint_plots_folder']}{kwargs['filename_base']}_roll_kines.png"
        kine_obj.plot_kines_to_file(pk.kines_yaw, pk.kines_pitch, pk.kines_roll, yaw, pitch, roll, kwargs['plotFolder'],
                                    output_yaw, output_pitch, output_roll)

    if pk.opts["save_letters_to_db"]:
        if not os.path.exists(kwargs['databaseFolder']):
            os.makedirs(kwargs['databaseFolder'])
        db_path = kwargs['databaseFolder'] + 'letters.db'
        ltr_obj = Letters(pk.letter_opts)
        ltr_obj.save_letters_to_db(pk.all_letters, angles_filename, db_path)

    if pk.opts["save_letters_to_file"]:
        ltr_obj = Letters(pk.letter_opts)
        lettersFolder = kwargs['lettersFolder'] + "nonseparated/"
        lettersFolderSpeaker = kwargs['lettersFolder'] + "speaker/"
        lettersFolderListener = kwargs['lettersFolder'] + "listener/"
        if not os.path.exists(lettersFolder):
            os.makedirs(lettersFolder)
        if not os.path.exists(lettersFolderSpeaker):
            os.makedirs(lettersFolderSpeaker)
        if not os.path.exists(lettersFolderListener):
            os.makedirs(lettersFolderListener)
        letters_filename = f"{kwargs['filename_base']}_letters.csv"
        letters_filename_speaker = f"{kwargs['filename_base']}_speaker_letters.csv"
        letters_filename_listener = f"{kwargs['filename_base']}_listener_letters.csv"

        if ltr_obj.letter_opts['speaking_listening'] == 'nonseparated':
            print("saving letters to file to: ", letters_filename)
            ltr_obj.save_letters_to_file(pk.all_letters, lettersFolder, letters_filename)
        elif ltr_obj.letter_opts['speaking_listening'] == 'speaking':
            print("saving Speaker letters to file to: ", letters_filename_speaker)
            ltr_obj.save_letters_to_file(pk.speaker_letters, lettersFolderSpeaker, letters_filename_speaker)
        elif ltr_obj.letter_opts['speaking_listening'] == 'listening':
            print("saving Listening letters to file to: ", letters_filename_listener)
            ltr_obj.save_letters_to_file(pk.listener_letters, lettersFolderListener, letters_filename_listener)
        elif ltr_obj.letter_opts['speaking_listening'] == 'separated_and_combined':
            print("saving letters to file to: ", letters_filename)
            print("saving Speaker letters to file to: ", letters_filename_speaker)
            print("saving Listening letters to file to: ", letters_filename_listener)
            ltr_obj.save_letters_to_file(pk.all_letters, lettersFolder, letters_filename)
            ltr_obj.save_letters_to_file(pk.speaker_letters, lettersFolderSpeaker, letters_filename_speaker)
            ltr_obj.save_letters_to_file(pk.listener_letters, lettersFolderListener, letters_filename_listener)
        else:
            print("please select a valid speaking_listening option")

    if pk.opts["save_summary_to_file"]:
        ltr_obj = Letters(pk.letter_opts)
        summaryFolder = kwargs['summaryFolder'] + "nonseparated/"
        summaryFolderSpeaker = kwargs['summaryFolder'] + "speaker/"
        summaryFolderListener = kwargs['summaryFolder'] + "listener/"
        if not os.path.exists(summaryFolder):
            os.makedirs(summaryFolder)
        if not os.path.exists(summaryFolderSpeaker):
            os.makedirs(summaryFolderSpeaker)
        if not os.path.exists(summaryFolderListener):
            os.makedirs(summaryFolderListener)

        summary_filename = f"{kwargs['filename_base']}_nonseparated_summary.txt"
        summary_filename_speaker = f"{kwargs['filename_base']}_speaker_summary.txt"
        summary_filename_listener = f"{kwargs['filename_base']}_listener_summary.txt"

        if ltr_obj.letter_opts['speaking_listening'] == 'nonseparated':
            print("saving letters summary to file to: ", summary_filename)
            ltr_obj.get_letter_summary(pk.all_letters, summaryFolder, summary_filename)
        elif ltr_obj.letter_opts['speaking_listening'] == 'speaking':
            print("saving Speaker letters summary to file to: ", summary_filename_speaker)
            ltr_obj.get_letter_summary(pk.speaker_letters, summaryFolderSpeaker, summary_filename_speaker)
        elif ltr_obj.letter_opts['speaking_listening'] == 'listening':
            print("saving Speaker letters summary to file to: ", summary_filename_listener)
            ltr_obj.get_letter_summary(pk.listener_letters, summaryFolderListener, summary_filename_listener)
        elif ltr_obj.letter_opts['speaking_listening'] == 'separated_and_combined':
            print("saving letters summary to file to: ", summary_filename)
            print("saving Speaker letters summary to file to: ", summary_filename_speaker)
            print("saving Listening letters summary to file to: ", summary_filename_listener)
            ltr_obj.get_letter_summary(pk.all_letters, summaryFolder, summary_filename)
            ltr_obj.get_letter_summary(pk.speaker_letters, summaryFolderSpeaker, summary_filename_speaker)
            ltr_obj.get_letter_summary(pk.listener_letters, summaryFolderListener, summary_filename_listener)
        else:
            print("please select a valid speaking_listening option")

    if pk.opts["save_kineme_clips_to_file"]:
        kinemeObj = Kinemes_Class(pk.kineme_opts)
        detected_kinemes = kinemeObj.get_kinemes(pk.all_letters)
        kin = kinemeObj.kineme(0)
        videoFile = f"{kwargs['videoFolder']}{kwargs['filename_base']}.mp4"
        if not os.path.exists(kwargs['kineme_clips_folder']):
            os.makedirs(kwargs['kineme_clips_folder'])
        level = detected_kinemes['level']
        name = detected_kinemes['letter']
        first_frame = detected_kinemes['first_frame']
        last_frame = detected_kinemes['last_frame']
        selected_kinemes = kinemeObj.kineme_types_mapping.get(kinemeObj.kineme_opts['kineme_type'])
        if not os.path.exists(f"{kwargs['kineme_clips_folder']}{kinemeObj.kineme_opts['kineme_type']}"):
            os.makedirs(f"{kwargs['kineme_clips_folder']}{kinemeObj.kineme_opts['kineme_type']}")

        for i in range(len(level)):
            if name[i] in selected_kinemes:
                output_filename = f"{kwargs['kineme_clips_folder']}{kinemeObj.kineme_opts['kineme_type']}/{kwargs['filename_base']}level{level[i]}_{name[i]}.mp4"
                evc.extract_frames(videoFile, first_frame[i], last_frame[i], output_filename)


def main():
    Data_in_Folder = "./data_in/"
    Data_out_Folder = "./data_out/"
    parser = argparse.ArgumentParser(description="Video demo script.")
    parser.add_argument('--Data_in_Folder', type=str, default=f"{Data_in_Folder}",
                        help='Path to root directory containing all data_in')
    parser.add_argument('--videoFolder', type=str, default=f"{Data_in_Folder}videos/", help='Path to video folder')
    parser.add_argument('--anglesFolder', type=str, default=f"{Data_in_Folder}", help='Path to angles folder')
    parser.add_argument('--speechFolder', type=str, default=f"{Data_in_Folder}", help='Path to speech folder')
    parser.add_argument('--keypoint_folder', type=str, default=f"{Data_out_Folder}keypoints/",
                        help='Path to keypoint folder')
    parser.add_argument('--OutputFolder', type=str, default=f"{Data_out_Folder}augmented_videos/",
                        help='Path to output folder')
    parser.add_argument('--keypoints_yaw_folder', type=str, default=f"{Data_out_Folder}keypoints/yaw/",
                        help='Path to yaw keypoints folder')
    parser.add_argument('--keypoints_pitch_folder', type=str, default=f"{Data_out_Folder}keypoints/pitch/",
                        help='Path to pitch keypoints folder')
    parser.add_argument('--keypoints_roll_folder', type=str, default=f"{Data_out_Folder}keypoints/roll/",
                        help='Path to roll keypoints folder')
    parser.add_argument('--keypoint_plots_folder', type=str, default=f"{Data_out_Folder}keypoints/plots/",
                        help='Path to letter folder')
    parser.add_argument('--lettersFolder', type=str, default=f"{Data_out_Folder}letters/", help='Path to letter folder')
    parser.add_argument('--histFolder', type=str, default=f"{Data_out_Folder}letter_histograms/",
                        help='Path to histogram folder')
    parser.add_argument('--summaryFolder', type=str, default=f"{Data_out_Folder}letter_summaries/",
                        help='Path to summary folder')
    parser.add_argument('--plotFolder', type=str, default=f"{Data_out_Folder}letter_histograms/plots/",
                        help='Path to plot folder')
    parser.add_argument('--histFolderLevels', type=str, default=f"{Data_out_Folder}letter_histograms/" + "levels/",
                        help='Path to histogram levels folder')
    parser.add_argument('--histFolderCombined', type=str, default=f"{Data_out_Folder}letter_histograms/" + "combined/",
                        help='Path to histogram combined folder')
    parser.add_argument('--histFolderBoth', type=str,
                        default=f"{Data_out_Folder}letter_histograms/" + "levels_and_combined/",
                        help='Path to histogram levels and combined folder')
    parser.add_argument('--speaking_labels_file', type=str,
                        default=f"{Data_out_Folder}speaking_labels/agokmen1_speaking_labels.csv",
                        help='Speaking labels of each frame for each video')
    parser.add_argument('--databaseFolder', type=str, default=f"{Data_out_Folder}databases/", help='Path to databases')
    parser.add_argument('--kineme_clips_folder', type=str, default=f"{Data_out_Folder}kineme_clips/",
                        help='Path to letter folder')
    #parser.add_argument('--filename_base', type=str, default=f"{filename_base}", help='Filename base')
    args = parser.parse_args()
    kwargs = vars(args)
    return kwargs

if __name__ == "__main__":
    kwargs = main()
    filename_bases = [
        302, 303, 304, 305, 307, 310, 312, 313, 315, 316, 317, 318, 319, 320, 321, 322,
        324, 325, 326, 327, 328, 330, 331, 333, 335, 336, 338, 339, 340, 341, 343, 344,
        345, 346, 347, 348, 350, 351, 352, 353, 355, 356, 357, 358, 360, 362, 363, 364,
        366, 367, 368, 369, 370, 371, 372, 374, 375, 376, 377, 379, 380, 381, 382, 383,
        385, 386, 388, 389, 390, 391, 392, 393, 395, 397, 400, 401, 402, 403, 404, 406,
        409, 412, 413, 414, 415, 416, 417, 418, 419, 420, 422, 423, 425, 426, 427, 428,
        429, 430, 433, 434, 436, 437, 439, 440, 441, 443, 444, 445, 446, 447, 448, 449,
        451, 454, 455, 456, 457, 458, 459, 463, 464, 468, 471, 472, 473, 474, 475, 476,
        477, 478, 479, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492
    ]

    for filename_base in filename_bases:
        kwargs['filename_base'] = str(filename_base)

        angles_filename = kwargs['anglesFolder'] + kwargs['filename_base'] + ".poses_rad"
        df = pd.read_csv(angles_filename)
        df = df.fillna(-1)
        x = df[['yaw', 'pitch', 'roll']].to_numpy()
        speech_labels_file = f"{kwargs['speechFolder']}{kwargs['filename_base']}.speech_labels"
        if os.path.exists(speech_labels_file):
            ds = pd.read_csv(speech_labels_file)
            ds = ds.fillna(-1)
            s = ds['speaking'].to_numpy()
        else:
            s = np.ones(len(df['yaw']))

        combination_file = "results.txt"
        with open(combination_file, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")

        blocks = []
        current_block = []
        for line in lines:
            if line.strip() == '----------------------------------------':
                if current_block:
                    blocks.append(current_block)
                    current_block = []
            else:
                current_block.append(line)
        if current_block:
            blocks.append(current_block)

        for block in blocks:
            current_index = None
            config_params = {}
            result_line = None
            config_section = False

            for bl_line in block:
                bl_line = bl_line.strip()
                if bl_line.startswith("Index:"):
                    parts = bl_line.split(":")
                    if len(parts) > 1:
                        current_index = parts[1].strip()
                elif bl_line.startswith("Config:"):
                    config_section = True
                elif bl_line.startswith("Result:"):
                    result_line = bl_line
                elif config_section and bl_line:
                    if ":" in bl_line:
                        param_line = bl_line.split(":")
                        key = param_line[0].strip()
                        value = ":".join(param_line[1:]).strip()
                        if value.startswith("[") and value.endswith("]"):
                            inner = value.strip("[] ")
                            if inner:
                                vals = [int(v.strip()) for v in inner.split(",")]
                            else:
                                vals = []
                            config_params[key] = vals
                        elif value.isdigit():
                            config_params[key] = int(value)
                        else:
                            config_params[key] = value

            pk = PoseKinemeHistogram(**kwargs)

            for ckey, cval in config_params.items():
                if ckey in pk.opts:
                    pk.opts[ckey] = cval

            histogram = pk.compute_(x, s)

            print(f"Index: {current_index}")
            if result_line:
                print(result_line)
            print("Histogram: ", histogram)

            output = f"histogram: {histogram}"
            histograms = extract_histograms_from_output(output)
            output_file = f"{kwargs['filename_base']}_Index{current_index}.allhistograms_columns.csv"
            save_histograms_as_columns_to_csv(histograms, output_file)
            print(f"Histograms saved as columns to {output_file}\n")

    sys.exit()
