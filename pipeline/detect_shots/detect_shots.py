__author__      = 'Ernesto Coto'
__copyright__   = 'April 2018'

import os
import sys
import argparse
import platform
import numpy
from scipy.spatial import distance as df
from multiprocessing import freeze_support

# add the web service folder to the sys path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join( dir_path, '..', '..', 'service'))
import imutils

# Program constants
MIN_SHOT_LENGTH = 1
MIN_SHOT_SCORE = 100000
HIST_BINS = 32
HIST_THRESH = 0.2
FRAMES_PER_SECOND = 25
CONVERT_TO_SECONDS = False

def hist_col(im_data, bins):
    """
    Computes a histogram per image channel (assuming RGB pixels in the image data)
    Arguments:
       im_data: 3D image data, the last dimension should hold the pixel values in a linear array of length 3.
       bins: Number of bins in the histogram
    Returns:
       a 1D array containing the histogram counts for each bin, for each pixel channel
    """
    all_counts = []
    for channel_idx in range(3):
        counts, bin_edges = numpy.histogram(im_data[:,:,channel_idx], bins=bins, range=(0.0,255.0))
        all_counts.extend(counts)
    return all_counts

if __name__ == '__main__':
    if 'Windows' in platform.system():
        freeze_support() # a requirement for windows execution

    # check arguments before doing anything
    parser = argparse.ArgumentParser(description='shot boundary detector')
    parser.add_argument('input_frames_dir', metavar='input_frames_dir', type=str, help='Full path to folder with input frames. The file name of each input frame must follow the pattern "ddddd.jpg", where "d" is a digit (0-9)')
    parser.add_argument('output_file', metavar='output_file', type=str, help='Full path to the output shot boundaries file')
    parser.add_argument('-m', dest='min_shot_length', default=MIN_SHOT_LENGTH, type=int, help='Minimum shot length (default: 1)')
    parser.add_argument('-b', dest='hist_bins', default=HIST_BINS, type=int, help='Number of histograms bins (default: 32)')
    parser.add_argument('-t', dest='hist_thresh', default=HIST_THRESH, type=float, help='Histograms threshold (default: 0.2)')
    parser.add_argument('-f', dest='frames_per_second', default=FRAMES_PER_SECOND, type=int, help='Frames per second in source video (default: 25)')
    parser.add_argument('-r', dest='min_shot_score', default=MIN_SHOT_SCORE, type=int, help='Minimum qualifying shot score (default: 100000)')
    parser.add_argument('-s', dest='convert_to_seconds', action='store_true', help='If used, the shot boundary indexes will be converted to seconds (default: shot boundaries indexes are not converted)')
    args = parser.parse_args()

    # check input frames path
    if not os.path.exists(args.input_frames_dir) or not os.path.isdir(args.input_frames_dir):
        print 'ERROR: %s does not exist or is not a valid directory. Aborting !' % args.input_frames_dir
        sys.exit(1)

    # acquire list of images
    video_frames_list = os.listdir(args.input_frames_dir)
    video_frames_list.sort()
    if len(video_frames_list)==0:
        print 'ERROR: There are no frames in %s . Aborting !.' % args.input_frames_dir
        sys.exit(1)

    # init useful vars
    hist_cur = None
    hist_prev =  None
    hist_thresh_adjusted = -1
    im_width = -1
    im_height = -1
    last_shot_integer_start_num = 0
    last_shot_begin_in_real_world = 0
    last_shot_string_start_num = ''

    # go through images
    all_shots = []
    for img_name in video_frames_list:

        full_path = os.path.join( args.input_frames_dir, img_name )
        string_frame_number = img_name.split('.')[0]
        integer_frame_number = int(string_frame_number)

        # read image
        img = imutils.acquire_image(full_path)
        if numpy.all(img==None):
            print 'ERROR: Could not read %s. Aborting !' % full_path
            sys.exit(1)

        if im_width<0 and im_height<0:
            # use first frame info to init some vars
            im_width = img.shape[1]
            im_height = img.shape[0]
            hist_thresh_adjusted =  args.hist_thresh * im_width *im_height * 3
            last_shot_begin_in_real_world = integer_frame_number
            last_shot_integer_start_num = integer_frame_number
            last_shot_string_start_num = string_frame_number

        shot_score = 0
        newshot = False

        # calculate histogram of current image
        hist_cur = hist_col(img, bins=args.hist_bins)

        if hist_prev != None:
            # compute shot score (l1-distance between histograms of previous and current images) ...
            shot_score = df.cityblock(hist_cur, hist_prev)
            # ... and compare against threshold
            if shot_score > hist_thresh_adjusted:
                newshot = True

        # if we have found a new shot boundary and it qualifies for saving ...
        if newshot and ( integer_frame_number - last_shot_integer_start_num >= args.min_shot_length) and shot_score >= args.min_shot_score:

            #print "newshot ! significant change between %d and %d" % (integer_frame_number-1, integer_frame_number)
            #print "shot_score ", shot_score

            # check whether we have to convert to seconds
            if args.convert_to_seconds and args.frames_per_second>1:

                    # do the conversion to seconds, taking into account the frames per second
                    converted_shot_string_start_num = "%05d" % last_shot_begin_in_real_world

                    shot_end_in_real_world = max(integer_frame_number-1-1,0)/float(args.frames_per_second)
                    timing_within_second = float(str(shot_end_in_real_world-int(shot_end_in_real_world)))
                    shot_end_in_real_world_corrected = 0
                    if timing_within_second >= 0.9:
                        shot_end_in_real_world_corrected =  int(shot_end_in_real_world + 0.5)
                    else:
                        shot_end_in_real_world_corrected =  int(shot_end_in_real_world)

                    # if the shot is too short after the conversion to seconds,
                    # the condition below can be false. In such a case, skip the
                    # boundary and wait for the next one
                    if (shot_end_in_real_world_corrected>=last_shot_begin_in_real_world):

                        # ... otherwise convert boundaries to string, save them, and move on
                        converted_shot_string_end_num =  "%05d" % shot_end_in_real_world_corrected
                        all_shots.append( [converted_shot_string_start_num, converted_shot_string_end_num])
                        last_shot_begin_in_real_world = shot_end_in_real_world_corrected + 1
                        last_shot_integer_start_num = integer_frame_number
                        last_shot_string_start_num = string_frame_number

            else:
                #save the boundary in plain format, and move on
                shot_end_string_frame_number = "%05d" % (integer_frame_number -1)
                all_shots.append( [last_shot_string_start_num, shot_end_string_frame_number])
                last_shot_integer_start_num = integer_frame_number
                last_shot_string_start_num = string_frame_number

        # make the current histogram the previous one
        hist_prev = hist_cur

    # add the last shot, if necessary
    num_files = len(video_frames_list)
    if last_shot_integer_start_num <= num_files:

        # check whether we have to convert to seconds
        if args.convert_to_seconds and args.frames_per_second>1:
            converted_shot_string_start_num = "%05d" % last_shot_begin_in_real_world
            converted_shot_integer_end_num = max(num_files-1,0)/float(args.frames_per_second)
            converted_shot_string_end_num = "%05d" % int(converted_shot_integer_end_num)
            all_shots.append( [converted_shot_string_start_num, converted_shot_string_end_num])
        else:
            # save the last shot in plain format
            shot_end_string_frame_number = "%05d" % (num_files-1)
            all_shots.append( [last_shot_string_start_num, shot_end_string_frame_number])

    # save shots to output file
    with open(args.output_file, "w+") as shots_out:
        for shot in all_shots:
            shots_out.write(shot[0] + ' ' + shot[1] + '\n')
