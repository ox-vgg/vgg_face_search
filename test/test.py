import os
import sys
import argparse
import socket
import json

# get access the the backend service settings
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(DIR_PATH, '..', 'service'))
import settings

# Network communication constants
BUFFER_SIZE = 1024
TCP_TERMINATOR = '$$$'
TCP_TIMEOUT = 86400.00

# Test script default settings
DEFAULT_TXT_OUTPUT_FILE = None
DEFAULT_MAX_NUM_RESULTS = 36
DEFAULT_VISUALISE_RESULTS_FLAG = False
DEFAULT_IMAGES_PATH = "." + os.path.sep

def roi_str_to_list(roi_str):
    """
        Converts a ROI encoded as a string into a list of 4 coordinates.
        Arguments:
            roi_str: ROI encoded as a string
        Returns:
            A list of coordinates [x1,y1,x2,y2].
            (x1,y1) is the upper-right corner of the ROI rectangle.
            (x2,y2) is the lower-left corner of the ROI rectangle.
    """
    coords = roi_str.split('_')
    x1 = float(coords[0])
    y1 = float(coords[1])
    x2 = float(coords[2])
    y2 = float(coords[5])
    return [ x1, y1, x2, y2 ]

def custom_request(request, append_end=True):
    """
        Sends a request to the host and port specified in the backend settings
        Arguments:
            request: JSON object to be sent
            append_end: Boolean to indicate whether or not to append
                        a TCP_TERMINATOR value at the end of the request.
        Returns:
            JSON containing the response from the host
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((settings.HOST, settings.PORT))
    except socket.error, msg:
        print 'Connect failed', msg
        return False

    sock.settimeout(TCP_TIMEOUT)

    print 'Request to backend at port %s: %s' % (str(settings.PORT), request)

    if append_end:
        request += TCP_TERMINATOR
    total_sent = 0
    while total_sent < len(request):
        sent = sock.send(request[total_sent:])
        if sent == 0:
            raise RuntimeError("Socket connection broken at port " + str(settings.PORT))
        total_sent = total_sent + sent

    term_idx = -1
    response = ''
    while term_idx < 0:
        try:
            rep_chunk = sock.recv(BUFFER_SIZE)
            if not rep_chunk:
                print 'Connection closed! at port ' + str(settings.PORT)
                sock.close()
                return False
            response = response + rep_chunk
            term_idx = response.find(TCP_TERMINATOR)
        except socket.timeout:
            print 'Socket timeout at port ' + str(self.port)
            sock.close()
            return False

    excess_sz = term_idx + len(TCP_TERMINATOR)
    response = response[0:term_idx]
    sock.close()

    return response


if __name__ == "__main__":
    """ Main method """

    # Parse arguments
    parser = argparse.ArgumentParser(description='vgg_face_search backend service test script')
    parser.add_argument('path_to_sample_images', metavar='path_to_sample_images', type=str,
        help='Full path to input training sample images')
    parser.add_argument('-o', dest='output_file',
        type=str, default=DEFAULT_TXT_OUTPUT_FILE,
        help='Full path to text output file. It ignores the -m parameter and saves all retrieved results. Note that the number of \
              results can also be restricted in the backend service. Default: %s' % str(DEFAULT_TXT_OUTPUT_FILE))
    parser.add_argument('-m', dest='max_num_results',
        default=DEFAULT_MAX_NUM_RESULTS, type=int,
        help='Maximum number of results to be printed (or visualized) to stdout. Default: %i' % DEFAULT_MAX_NUM_RESULTS)
    parser.add_argument('-p', dest='images_path',
        type=str, default=DEFAULT_IMAGES_PATH,
        help='Full path to the folder containing the images of the dataset. Only needed if the -v flag is used and the images \
              are in a location not relative to this script. Default: %s' % str(DEFAULT_TXT_OUTPUT_FILE))
    parser.add_argument('-v', dest='visualize_results',
        default=DEFAULT_VISUALISE_RESULTS_FLAG, action= 'store_true',
        help='If used, the final results will be displayed in a GUI using matplotlib. Default: Disable')
    args = parser.parse_args()


    # 1) Connect to backend and test connection

    print '** Sending selfTest'
    req_obj = {'func': 'selfTest'}
    request = json.dumps(req_obj)
    response = custom_request(request)
    func_out = json.loads(response)
    print 'Received response:'
    print func_out


    # 2) Start a query by getting a query ID

    print '** Sending getQueryId'
    req_obj = {'func': 'getQueryId', 'dataset': 'dummy'}
    request = json.dumps(req_obj)
    response = custom_request(request)
    func_out = json.loads(response)
    print 'Received response:'
    print func_out
    query_id = func_out['query_id']

    # 3) Add training samples from the folder specified in the parameters

    pos_trs_dir = args.path_to_sample_images
    pos_trs_paths = [os.path.join(pos_trs_dir, pos_trs_fname) for
                        pos_trs_fname in os.listdir(pos_trs_dir)]

    for pos_trs_path in pos_trs_paths:
        print '** Sending addPosTrs ', pos_trs_path
        req_obj = {'func': 'addPosTrs',
                   'query_id': query_id,
                   'impath': pos_trs_path}
        request = json.dumps(req_obj)
        response = custom_request(request)
        func_out = json.loads(response)
        print 'Received response:'
        print func_out
        if not func_out['success']: raise RuntimeError("addPosTrs")

    # 4) Train the classifier

    print '** Sending train'
    req_obj = {'func': 'train',
               'query_id': query_id}
    request = json.dumps(req_obj)
    response = custom_request(request)
    func_out = json.loads(response)
    print 'Received response:'
    print func_out
    if not func_out['success']: raise RuntimeError("train")

    # 5) Rank results

    print '** Sending rank'
    req_obj = {'func': 'rank',
               'query_id': query_id}
    request = json.dumps(req_obj)
    response = custom_request(request)
    func_out = json.loads(response)
    print 'Received response:'
    print func_out
    if not func_out['success']: raise RuntimeError("rank")

    # 6) Get ranked results

    print '** Sending getRanking'
    req_obj = {'func': 'getRanking',
               'query_id': query_id}
    request = json.dumps(req_obj)
    response = custom_request(request)
    func_out = json.loads(response)
    print 'Received response:'
    #print func_out # avoid printing a very long output
    if not func_out['success']: raise RuntimeError("getRanking")
    rank_result = func_out['ranklist']
    print 'Retrieved %d results' % len(rank_result)

    # 7) Free the query in the backend to save memory

    print '** Sending releaseQueryId'
    req_obj = {'func': 'releaseQueryId',
               'query_id': query_id}
    request = json.dumps(req_obj)
    response = custom_request(request)
    func_out = json.loads(response)
    print 'Received response:'
    print func_out
    if not func_out['success']: raise RuntimeError("releaseQueryId")

    # 9) (Optional) Save results to text file

    if args.output_file:
        print '** Saving %d results to %s' %(len(rank_result), args.output_file)
        with open(args.output_file, 'w+') as outfile:
            json.dump(rank_result, outfile, indent=2)

    # 10) Print the top-ranking results, up to a limit

    # clamp maximum number of results by the length of the results page
    max_display_results = min(args.max_num_results, len(rank_result))
    print '** Only showing the first %d results. The columns below correspond to: #result, path, roi[x1,y1,x2,y2], score' % max_display_results
    ctr = 1;
    for ritem in rank_result:
        print '%d %s %s %f' % (ctr, ritem['path'], roi_str_to_list(ritem['roi']), ritem['score'])
        ctr = ctr + 1
        if ctr > max_display_results:
            break

    # 11) (Optional) Visualise ranked results in a GUI. 18 images per window.


    if args.visualize_results:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        num_pages_for_visualization = max_display_results/(3*6)
        for i in range(num_pages_for_visualization):

            rlist_plt = rank_result[(i*3*6):(i*3*6)+(3*6)]

            fig, axes = plt.subplots(3, 6, figsize=(12, 6),
                                     subplot_kw={'xticks': [], 'yticks': []})
            fig.subplots_adjust(hspace=0.3, wspace=0.05)

            for ax, ritem in zip(axes.flat, rlist_plt):
                im_path = os.path.join(args.images_path, ritem['path'])
                im = mpimg.imread(im_path)
                ax.imshow(im)
                ax.set_title(ritem['score'])

            plt.show()
            # plt.show(block=(i>=num_pages_for_visualization-1)) # this will only block the last page

        if max_display_results % (3*6) > 0 :

            max_display_results = max_display_results % (3*6)
            i = max(0, num_pages_for_visualization-1)
            rlist_plt = rank_result[(i*3*6):(i*3*6)+max_display_results]

            fig, axes = plt.subplots(3, 6, figsize=(12, 6),
                                     subplot_kw={'xticks': [], 'yticks': []})
            fig.subplots_adjust(hspace=0.3, wspace=0.05)

            for ax, ritem in zip(axes.flat, rlist_plt):
                im_path = os.path.join(args.images_path, ritem['path'])
                im = mpimg.imread(im_path)
                ax.imshow(im)
                ax.set_title(ritem['score'])

            plt.show()
