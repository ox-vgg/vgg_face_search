#include <algorithm>
#include <vector>
#include <string>
#include <fstream>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/range/iterator_range.hpp>

#include "jp_jpeg.hpp"
#include "hist_sift.hpp"
#include "hist_col.hpp"

// Program constants
int MIN_SHOT_LENGTH = 1;
int MIN_SHOT_SCORE = 100000;
int HIST_BINS = 32;
double HIST_THRESH = 0.2;
int FRAMES_PER_SECOND = 25;
bool CONVERT_TO_SECONDS = FALSE;

/**
 * Simple template to compute the l1-distance
 */
template <typename T>
size_t
dist_l1(const T* a, const T* b, size_t ndims)
{
    size_t ret = 0;
    for (size_t d = 0; d < ndims; ++d)
        if (a[d] >= b[d])
            ret += a[d] - b[d];
        else
            ret += b[d] - a[d];
    return ret;
}


/**
 * Usage print out
 */
void usage()
{
    std::cout << "Usage: detect_shots INPUT_FRAMES_DIR OUTPUT_FILE [options]" << std::endl;
    std::cout << "  * Mandatory parameters:" << std::endl;
    std::cout << "  INPUT_FRAMES_DIR: Full path to folder with input frames" << std::endl;
    std::cout << "  OUTPUT_FILE: Full path to the output shot boundaries file" << std::endl;
    std::cout << "  * Optional parameters:" << std::endl;
    std::cout << "  -m [MIN_SHOT_LENGTH]: minimum shot length (default: 1)" << std::endl;
    std::cout << "  -b [HIST_BINS]: number of histograms bins (default: 32)" << std::endl;
    std::cout << "  -t [HIST_THRESH]: histograms threshold (default: 0.2)" << std::endl;
    std::cout << "  -f [FRAMES_PER_SECOND]: frames per second in source video (default: 25)" << std::endl;
    std::cout << "  -r [MIN_SHOT_SCORE]: minimum qualifying shot score (default: 100000)" << std::endl;
    std::cout << "  -s : if used, the shot boundary indexes will be converted to seconds (default: shot boundaries indexes are not converted)" << std::endl;
}

/**
 * Main
 */
int main(int argc, char** argv)
{
    // check command-line parameters
    if (argc < 3) {
        usage();
        return 1;
    }

    // mandatory parameters
    boost::filesystem::path frame_folder(argv[1]);
    std::string output_file = argv[2];
    int idx(3);

    // optional parameters
    for( ; idx < argc; ) {
        std::string par = argv[idx];
        if (par=="-m") { idx++; MIN_SHOT_LENGTH = std::max(1,atoi(argv[idx])); }
        else if (par=="-b") { idx++; HIST_BINS = atoi(argv[idx]); }
        else if (par=="-t") { idx++; HIST_THRESH = atof(argv[idx]); }
        else if (par=="-f") { idx++; FRAMES_PER_SECOND = std::max(1,atoi(argv[idx])); }
        else if (par=="-r") { idx++; MIN_SHOT_SCORE = atoi(argv[idx]); }
        else if (par=="-s") { CONVERT_TO_SECONDS = TRUE; }
        else {
            usage();
            return 1;
        }
        idx++;
    }

    // check input folder
    if(! boost::filesystem::is_directory(frame_folder)) {
        std::cout << frame_folder << " is not a directory. Aborting !"  << std::endl;
        return 1;
    }

    // check and create output file
    std::ofstream output_file_fobj;
    if (output_file != "") {
        output_file_fobj.open(output_file.c_str());
    }
    else {
        std::cout << " invalid output filename. Aborting !" << std::endl;
        return 1;
    }

    // scan input directory
    boost::filesystem::directory_iterator end_itr;
    std::vector<boost::filesystem::path> vec_files;
    std::copy(boost::filesystem::directory_iterator(frame_folder), boost::filesystem::directory_iterator(), std::back_inserter(vec_files));
    std::sort(vec_files.begin(), vec_files.end()); // sort, since directory iteration
                                                   // is not ordered on some file systems

    // init useful vars
    unsigned int hist_buf[2 * 3 * HIST_BINS];
    unsigned int* hist_prev = hist_buf;
    unsigned int* hist_cur = hist_buf + 3 * HIST_BINS;
    size_t hist_thresh = -1;
    int global_width = -1, global_height = -1;
    int last_shot_integer_start_num = 0;
    int last_shot_begin_in_real_world = 0;
    std::string last_shot_string_start_num;

    // go through the list of acquired files
    for (std::vector<boost::filesystem::path>::const_iterator it(vec_files.begin()), it_end(vec_files.end()); it != it_end; ++it) {

        int width, height;
        std::string full_file_path = (*it).string();
        std::string file_name = (*it).filename().string();
        size_t lastindex = file_name.find_last_of(".");
        std::string string_frame_number = file_name.substr(0, lastindex);
        int integer_frame_number = boost::lexical_cast<int>(string_frame_number.c_str());

        //std::cout << file_name <<  std::endl;
        //std::cout << integer_frame_number <<  std::endl;

        // read image
        unsigned char* img = jp_jpeg_read(full_file_path, width, height);
        if (!img) {
          std::cout << "Error while loading " << full_file_path << std::endl;
          return 1;
        }

        // set some variables for the first time
        if (global_width<0 && global_height<0) {
          // The setting of these "global" variables means that all images are
          // assumed to have the same dimensions
          global_width = width;
          global_height = height;
          hist_thresh = (size_t)(HIST_THRESH * global_width *global_height * 3);
        }

        double shot_score = 0;
        bool newshot = false;

        // calculate histogram of current image
        hist_col(img, global_width, global_height, HIST_BINS, hist_cur);

        if (integer_frame_number > 0) {
            // compute shot score (l1-distance between histograms of previous and current images) ...
            shot_score = dist_l1(hist_cur, hist_prev, 3 * HIST_BINS);
            // ... and compare against threshold
            if (shot_score > hist_thresh) {
                newshot = true;
                //std::cout << "shot_score " << shot_score << std::endl;
            }
        }
        else {
            // init with first frame info
            last_shot_begin_in_real_world = integer_frame_number;
            last_shot_integer_start_num = integer_frame_number;
            last_shot_string_start_num = string_frame_number;
        }

        // if we have found a new shot boundary and it qualifies for saving ...
        if (newshot && ( integer_frame_number - last_shot_integer_start_num >= MIN_SHOT_LENGTH) && shot_score>=MIN_SHOT_SCORE) {

                //std::cout << "newshot ! significant change between " << integer_frame_number-1 << " and " << integer_frame_number << std::endl;

                // check whether we have to convert to seconds
                if (CONVERT_TO_SECONDS && FRAMES_PER_SECOND>1) {

                    // do the conversion to seconds, taking into account the frames per second
                    std::string converted_shot_string_start_num = (boost::format("%05d") % last_shot_begin_in_real_world ).str();

                    float shot_end_in_real_world = std::max(integer_frame_number-1-1,0)/float(FRAMES_PER_SECOND);
                    float timing_within_second = float(shot_end_in_real_world-int(shot_end_in_real_world));
                    int shot_end_in_real_world_corrected = 0;
                    if (timing_within_second >= 0.9)  {
                        shot_end_in_real_world_corrected =  int(shot_end_in_real_world + 0.5);
                    }
                    else {
                        shot_end_in_real_world_corrected =  int(shot_end_in_real_world);
                    }

                    // if the shot is too short after the conversion to seconds,the condition below can be false.
                    // In such a case, skip the boundary and wait for the next one
                    if (shot_end_in_real_world_corrected>=last_shot_begin_in_real_world) {

                        // ... otherwise convert boundaries to string, save them, and move on
                        std::string converted_shot_string_end_num = (boost::format("%05d") % shot_end_in_real_world_corrected ).str();
                        output_file_fobj << converted_shot_string_start_num << " " << converted_shot_string_end_num << std::endl;

                        last_shot_begin_in_real_world = shot_end_in_real_world_corrected + 1;
                        last_shot_integer_start_num = integer_frame_number;
                        last_shot_string_start_num = string_frame_number;
                    }
                }
                else {
                    // save the boundary in plain format, and move on
                    std::string shot_end_string_frame_number = (boost::format("%05d") % (integer_frame_number -1)).str();
                    output_file_fobj << last_shot_string_start_num << " " << shot_end_string_frame_number << std::endl;
                    last_shot_integer_start_num = integer_frame_number;
                    last_shot_string_start_num = string_frame_number;
                }
        }

        // release image buffer
        delete img;

        // make the current histogram the previous one
        std::swap(hist_cur, hist_prev);

    }

    // add the last shot, if necessary
    int num_files = int(vec_files.size());
    if (last_shot_integer_start_num <= num_files) {

        // check whether we have to convert to seconds
        if (CONVERT_TO_SECONDS && FRAMES_PER_SECOND>1) {
            std::string converted_shot_string_start_num = (boost::format("%05d") % last_shot_begin_in_real_world ).str();
            float converted_shot_integer_end_num = std::max(num_files-1,0)/float(FRAMES_PER_SECOND);
            std::string converted_shot_string_end_num = (boost::format("%05d") % int(converted_shot_integer_end_num) ).str();
            output_file_fobj << converted_shot_string_start_num << " " << converted_shot_string_end_num << std::endl;
        }
        else {
            // save the last shot in plain format
            std::string shot_end_string_frame_number = (boost::format("%05d") % (vec_files.size()-1)).str();
            output_file_fobj << last_shot_string_start_num << " " << shot_end_string_frame_number << std::endl;
        }
    }

    // close output file for good
    output_file_fobj.close();
}
