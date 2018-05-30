#!/bin/bash
# Parameters:
# $1 -> Type of input: "video" or "images"
# $2 -> If input type is "video": Full path to video. If input type is "images": Full path to base folder holding the images referenced by the search service
# $3 -> If input type is "video": Full path to base folder holding the images referenced by the search service. If input type is "images": Full path to text file containing list of images to ingest
# $4 -> Full path to output features file (optional)

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<additional paths>
BASEDIR=$(dirname "$0")
cd "$BASEDIR"
source ../../bin/activate
if [ "$1" = "video" ]; then
    # get video name
    VIDEONAME=$(basename "$2")
    # check variable is not empty
    if  [ ! -z "$VIDEONAME" ]; then
        # remove previous temporary folder/file, if present
        rm -rf "/tmp/${VIDEONAME}"
        rm -f "/tmp/${VIDEONAME}_shots.txt"
        # make new temporary folder
        mkdir -p "/tmp/${VIDEONAME}"
        # extract video fps value
        FPS="$('ffmpeg'  -i "${2}" 2>&1 | sed -n 's/.*, \(.*\) fp.*/\1/p')"
        # extract all video frames and run shot detection
        "ffmpeg"  -i "${2}" -vsync vfr -q:v 1 -start_number 0 -vf scale=iw:ih*\(1/sar\) -loglevel panic  "/tmp/${VIDEONAME}/%05d.jpg"
        ./build/detect_shots "/tmp/${VIDEONAME}/" "/tmp/${VIDEONAME}_shots.txt" -f "${FPS}" -s
        # remove frames and re-extract them, but only 1 frame per second
        rm -rf "/tmp/$VIDEONAME/"
        mkdir -p "/tmp/${VIDEONAME}"
        for i in {0..5400} #0 to 90 minutes
        do
        fname=$(printf "%05d" $i)
        "ffmpeg" -ss $i -i "${2}" -vframes 1 -q:v 1 -vf scale=iw:ih*\(1/sar\) -loglevel panic "/tmp/${VIDEONAME}/${fname}.jpg"
        if [ ! -f "/tmp/${VIDEONAME}/${fname}.jpg" ];  then
          # echo "*********VIDEO FINISHED. BREAKING*********";
          break;
        fi
        done
        # run face tracking and feature extraction with the frames and shots computed before
        if [ "$#" -ne 4 ]; then
            python compute_pos_features_video.py "/tmp/${VIDEONAME}/" "/tmp/${VIDEONAME}_shots.txt" "${3}"
        else
            python compute_pos_features_video.py "/tmp/${VIDEONAME}/" "/tmp/${VIDEONAME}_shots.txt" "${3}" -o "${4}"
        fi
        rm -rf "/tmp/${VIDEONAME}"
        rm -f "/tmp/${VIDEONAME}_shots.txt"
    fi
else
    if [ "$#" -ne 4 ]; then
        python compute_pos_features.py "${2}" "${3}"
    else
        python compute_pos_features.py "${2}" "${3}" -o "${4}"
    fi
fi
