__author__      = 'Ernesto Coto'
__copyright__   = 'April 2018'

import os
import fileinput
import multiprocessing
import numpy
import simplejson as json
import time
from scipy.spatial import distance as df
import pickle
import traceback

import imutils
import settings
# import face detector
import face_detection_retinaface
# import face feature extractor
import face_features

if settings.KDTREES_RANKING_ENABLED:
    import kdutils

def group_feature_extractor(image_list):
    """
        Body of the thread that runs the face feature extraction for
        a list of images
        Arguments:
            image_list: List of images to be processed. Each item in the list corresponds to a dictionary with
                        at least two keys: "path" and "roi". The "path" should contain the full path to the image
                        file to be processed and "roi" the coordinates of the bounding-box of a face detected on
                        the image.
    """
    list_of_feats = []
    if len(image_list) > 0:
        try:
            # init feature extractor
            feature_extractor = face_features.FaceFeatureExtractor()
            for image in image_list:
                # read image
                theim = imutils.acquire_image(image["path"])
                det = image["roi"]
                # crop image to face detection bounding-box
                crop_img = theim[det[1]:det[3], det[0]:det[2], :]
                # extract features
                feat = feature_extractor.feature_compute(crop_img)
                # reshape for compatibility with ranking function
                feat = numpy.reshape(feat, (1, settings.FEATURES_VECTOR_SIZE))
                # add to list of features to be returned
                list_of_feats.append(feat)
        except Exception as e:
            print ('Exception in group_feature_extractor: ' + str(e))
            list_of_feats = []
            pass
    return list_of_feats



class FaceRetrieval(object):
    """
        Class implementing the face-search engine.
    """

    def __init__(self):
        """
            Initializes the engine.
            Loads into memory the database of features, which should have been computed beforehand.
            Instantiates the face detector, the thread pool and other useful members.
        """
        self.query_id = 0
        self.query_id_lock = multiprocessing.Lock()
        self.worker_pool = multiprocessing.Pool(processes=settings.NUMBER_OF_HELPER_WORKERS)
        self.database = {'paths': [], 'rois': [], 'feats': []}
        self.kdtrees = []
        self.query_data = dict()
        self.face_detector = face_detection_retinaface.FaceDetectorRetinaFace()

        if settings.KDTREES_RANKING_ENABLED:
            print ('Ranking with kdtrees is enabled')
            if os.path.exists(settings.KDTREES_FILE):
                print ('Found precomputed kdtrees...')
                self.kdtrees = kdutils.load_kdtrees(settings.KDTREES_FILE)
            else:
                print ('DID NOT find precomputed kdtrees. The dataset features will not be accessible via kd-trees.')

        print ('Loading dataset...')
        # acquire dataset information the old-fashion way
        with open(settings.DATASET_FEATS_FILE, 'rb') as fin:
            database_content = pickle.load(fin)
            if isinstance(database_content, dict):
                if len(self.kdtrees)==0:
                    if 'feats' in database_content.keys():
                        self.database['feats'].extend(database_content['feats'])
                    else:
                        raise Exception('The features cannot be found. Please check your settings.')
                self.database['paths'].extend(database_content['paths'])
                self.database['rois'].extend(database_content['rois'])
            elif isinstance(database_content, list):
                for entry in database_content:
                    print ('Loading sub-dataset ' +  entry )
                    if os.path.sep not in entry:
                        # in this case, assume it is in the same directory as the DATASET_FEATS_FILE
                        sub_database = os.path.join(os.path.dirname(settings.DATASET_FEATS_FILE), entry)
                    else:
                        sub_database = entry
                    with open(sub_database, 'rb') as fin_chunk:
                        database_chunk_content = pickle.load(fin_chunk)
                    if len(self.kdtrees)==0:
                        if 'feats' in database_chunk_content.keys():
                             self.database['feats'].extend(database_chunk_content['feats'])
                        else:
                            raise Exception('The features cannot be found. Please check your settings.')
                    self.database['paths'].extend(database_chunk_content['paths'])
                    self.database['rois'].extend(database_chunk_content['rois'])
            else:
                raise Exception('Cannot initialize FaceRetrieval. File %s contains corrupted information.' % settings.DATASET_FEATS_FILE)

        print ('Loaded database for %d tracks' % len(self.database['paths']))

        # do one load of this to get things into the cache
        feature_extractor = face_features.FaceFeatureExtractor()

        print ('FaceRetrieval successfully initialized')


    def prepare_success_json_str_(self, success):
        """
            Creates JSON with ONLY a 'success' field
            Parameters:
                success: Boolean value for the 'success' field
            Returns:
                JSON formatted string
        """
        retfail = {}
        retfail['success'] = success
        return json.dumps(retfail)


    def selfTest(self, req_params):
        """
            Simple test function that will return the same JSON object as in the parameter
            Parameters:
                req_params: JSON object
            Returns:
                JSON formatted string
        """
        print ('Server is running')
        return self.prepare_success_json_str_(True)


    def getQueryId(self, req_params):
        """
            Generates a new query ID and returns it.
            Parameters:
                req_params: JSON object with at least the field:
                           - dataset: a short string indicating the name of the dataset being used
            Returns:
                JSON formatted string with the query id and the 'success' field.
        """
        if 'dataset' in req_params:
            dataset = req_params['dataset']
        else:
            return False

        self.query_id_lock.acquire()
        self.query_id = self.query_id+1
        query_id = self.query_id
        self.query_id_lock.release()

        self.query_data[str(query_id)] = dict()
        self.query_data[str(query_id)]["training_started"] = False
        self.query_data[str(query_id)]["dataset"] = dataset
        self.query_data[str(query_id)]["images"] = list()
        self.query_data[str(query_id)]["img_list_lock"] = multiprocessing.Lock()
        return json.dumps({'success': True, 'query_id':query_id})


    def releaseQueryId(self, req_params):
        """
            Deletes any data associated with a query ID.
            Parameters:
                req_params: JSON object with at least the field:
                            - query_id: the id of the query.
            Returns:
                JSON formatted string with 'success' field set to 'False'
                in case of any problems. The 'success' field set to 'True'
                otherwise.
        """
        if 'query_id' in req_params:
            query_id = req_params['query_id']
        else:
            return self.prepare_success_json_str_(False)

        query_id = str(query_id)
        if query_id in self.query_data:
            try:
                del self.query_data[query_id]
                return self.prepare_success_json_str_(True)
            except:
                return self.prepare_success_json_str_(False)
        else:
            return self.prepare_success_json_str_(False)


    def addTrs(self, req_params, pos=True):
        """
            Adds a training image for the classification process.
            Parameters:
                req_params: JSON object with at least the fields:
                            - query_id: the id of the query
                            - impath: full path to the training image
                            Other fields include:
                            - featpath: Full path to the feature file associated to the query
                            - training_started: boolean indicated that the training step has already started and
                                                therefore the image can be discarded
                            - extra_params: Another dictionary with the fields:
                                * from_dataset: Boolean indicating whether the training image
                                                is part of the dataset or not.
                                * uri: unique resource identifier
                                * roi: coordinates of a bounding-box defined on the image
                        the image.
            Returns:
                JSON formatted string with 'success' field set to 'False'
                in case of any problems. The 'success' field set to 'True'
                otherwise.
        """
        # check query id is present
        if 'query_id' in req_params:
            query_id = req_params['query_id']
        else:
            return self.prepare_success_json_str_(False)

        # check whether the training step has already started
        if self.query_data[str(query_id)]["training_started"]:
            # discard the image and return as if nothing happened
            if 'impath' in req_params:
                print ('Training already started. Skipping ' + os.path.basename(req_params['impath']))
            else:
                print ('Training already started. Skipping image.')
            return self.prepare_success_json_str_(True)

        # check image path is present
        if 'impath' in req_params:
            impath = req_params['impath']
        else:
            return self.prepare_success_json_str_(False)

        # get the path to the feature file. Not used at the moment.
        if 'featpath' in req_params:
            featpath = req_params['featpath']

        # check for extra parameters
        if 'extra_params' in req_params:

            if 'from_dataset' in req_params['extra_params']:
                from_dataset = req_params['extra_params']['from_dataset']
            else:
                from_dataset = False

            if 'uri' in req_params['extra_params']:
                uri = req_params['extra_params']['uri']
            else:
                uri = -1

            if 'roi' in req_params['extra_params']:
                # if request specifies a ROI ...
                roi = req_params['extra_params']['roi']
                roi = numpy.array([int(x) for x in roi]).reshape(len(roi)/2, 2)
                xl, yl = roi.min(axis=0)
                xu, yu = roi.max(axis=0)
                roi = [xl, yl, xu, yu]
                print ('Request specifies ROI ' + str(roi))
                # ... check there is a face on the roi
                theim = imutils.acquire_image(impath)
                crop_img = theim[yl:yu, xl:xu, :]
                det = self.face_detector.detect_faces(crop_img, return_best=True)
                if numpy.all(det == None):
                    print ('No detection found in specified ROI')
                    return self.prepare_success_json_str_(False)
                else:
                    # If found, replace the previous with a more accurate one
                    det = det[0]
                    # The coordinates should be already integers, but some basic
                    # conversion is need for compatibility with all face detectors.
                    # Plus we have to get rid of the detection score det[4]
                    det = [int(det[0]), int(det[1]), int(det[2]), int(det[3])]
                    roi = [det[0]+xl, det[1]+yl, det[2]+xl, det[3]+yl]
                    print ('Automatically adjusting ROI to more accurate region ' + str(roi))
            else:
                roi = None
        else:
            from_dataset = False
            roi = None
            uri = -1

        # create empty dictionary for the image information
        img = dict()

        # if the image is brand new ..
        if uri == -1:

            # and no roi was specified ...
            if roi == None:

                # read image
                theim = imutils.acquire_image(impath)
                # run face detector, but only get the best detection.
                # multiple detections are not supported for on-the-fly training images
                det = self.face_detector.detect_faces(theim, return_best=True)

                if numpy.all(det != None):

                    # if a face is found, save it
                    print ('Single ROI detected')
                    # The coordinates should be already integers, but some basic
                    # conversion is need for compatibility with all face detectors.
                    # Plus we have to get rid of the detection score det[4]
                    det = [int(det[0][0]), int(det[0][1]), int(det[0][2]), int(det[0][3])]
                    print ('final det ' + str(det))

                    img["path"] = impath
                    img["roi"] = det
                    if pos == True:
                        img["anno"] = 1
                    else:
                        img["anno"] = -1
                else:
                    print ('No detection found')
                    return self.prepare_success_json_str_(True)

            else:
                # just the save the image along with the specified roi
                img["path"] = impath
                img["roi"] = roi
                if pos == True:
                    img["anno"] = 1
                else:
                    img["anno"] = -1
        else:
            # just the save the image with default values
            # for the roi (if not specified) and the annotation type
            img["path"] = impath
            if roi == None:
                img["roi"] = [0, 0, 0, 0]
            else:
                img["roi"] = roi
            if pos == True:
                img["anno"] = 1
            else:
                img["anno"] = -1

        # save unique identifier (even if it is -1)
        img["uri"] = uri

        # save the image information, if we are still accepting training images
        if str(query_id) in self.query_data.keys():
            if self.query_data[str(query_id)]["training_started"] == False:
                self.query_data[str(query_id)]["images"].append(img)
            else:
                print ('Training already started. Skipping ' + os.path.basename(impath))
        else:
            print ('Query already finished. Skipping ' + os.path.basename(impath))

        # return with success==True
        return self.prepare_success_json_str_(True)


    def addPosTrs(self, req_params):
        """
            Adds a positive training image for the classification process.
            Parameters:
                req_params: JSON object with the fields specified in addTrs()
            Returns:
                JSON formatted string with 'success' field set to 'False'
                in case of any problems. The 'success' field set to 'True'
                otherwise.
        """
        return self.addTrs(req_params, True)


    def addNegTrs(self, req_params):
        """
            Adds a negative training image for the classification process.
            NOTE: Not used at the moment because the frontend does not send negative images.
            Parameters:
                req_params: JSON object with the fields specified in addTrs()
            Returns:
                JSON formatted string with 'success' field set to 'False'
                in case of any problems. The 'success' field set to 'True'
                otherwise.
        """
        return self.addTrs(req_params, False)


    def train(self, req_params):
        """
            Performs the training of the face classifier.
            Divides the list of training images in multiple parts and
            processes the parts in separate threads.
            Parameters:
                req_params: JSON object with at least the field:
                            - query_id: the id of the query.
            Returns:
                JSON formatted string with 'success' field set to 'False'
                in case of any problems. The 'success' field set to 'True'
                otherwise.
        """
        if 'query_id' in req_params:
            query_id = req_params['query_id']
        else:
            return self.prepare_success_json_str_(False)

        query_id = str(query_id)
        if len(self.query_data[query_id]["images"]) == 0:
            print ('No training images found')
            return self.prepare_success_json_str_(False)

        results = None
        t = time.time()
        dataset = self.query_data[query_id]["dataset"]
        self.query_data[query_id]["features"] = []

        try:
            # distribute list of images among helper workers
            print ('Dividing training images among workers')
            num_images = len(self.query_data[query_id]["images"])
            num_images_per_worker = int(round(num_images/(1.0*settings.NUMBER_OF_HELPER_WORKERS)))
            query_data_groups = []
            if num_images_per_worker > 0:
                for idx in range(settings.NUMBER_OF_HELPER_WORKERS-1):
                    lower = idx*num_images_per_worker
                    if ((idx+1)*num_images_per_worker)-1 < num_images:
                        upper = ((idx+1)*num_images_per_worker)-1
                        if lower == upper:
                            query_data_groups.append([self.query_data[query_id]["images"][lower]])
                        elif lower < upper:
                            query_data_groups.append(self.query_data[query_id]["images"][lower:upper+1])

                if upper+1 == num_images-1:
                    query_data_groups.append([self.query_data[query_id]["images"][num_images-1]])
                else:
                    query_data_groups.append(self.query_data[query_id]["images"][upper+1:num_images])

            else:
                for idx in range(settings.NUMBER_OF_HELPER_WORKERS):
                    if idx < num_images:
                        query_data_groups.append([self.query_data[query_id]["images"][idx]])
                    else:
                        query_data_groups.append([])

            # execute parallel feature extraction by groups
            print ('Computing features')
            results = self.worker_pool.map_async(group_feature_extractor, query_data_groups).get(settings.FEATURES_EXTRACTION_TIMEOUT)
        except Exception as e:
            print ('Exception while computing features: ' +  str(e))
            print (traceback.format_exc())
            pass

        if results:
            # accumulate feature values
            feats_accumulator = numpy.zeros((1, settings.FEATURES_VECTOR_SIZE))
            for i in range(len(results)):
                for feat in results[i]:
                    feats_accumulator = feats_accumulator + feat

            # average and normalize
            feats_average = feats_accumulator/len(self.query_data[query_id]["images"])
            feats_average_norm = numpy.linalg.norm(feats_average)
            feats_average_norm = feats_average/max(feats_average_norm, 0.00001)

            # done
            self.query_data[query_id]["features"] = feats_average_norm
            print ('Done computing features ' + str(time.time() - t))
            return self.prepare_success_json_str_(True)
        else:
            # Something went wrong with the feature computation
            return self.prepare_success_json_str_(False)


    def loadClassifier(self, req_params):
        """
            Loads a face classifier from a filepath
            NOTE: Not used at the moment because this engine does not need to
                  save/load a classifier to/from a file
            Parameters:
                req_params: JSON object with at least the field:
                            - query_id: the id of the query
                            - filepath: Full path to the classifier file
            Returns:
                JSON formatted string with 'success' field set to 'False'
                in case of any problems. The 'success' field set to 'True'
                otherwise.
        """
        if 'query_id' in req_params:
            query_id = req_params['query_id']
        else:
            return self.prepare_success_json_str_(False)

        if 'filepath' in req_params:
            filepath = req_params['filepath']
        else:
            return self.prepare_success_json_str_(False)

        # Not used so far. Do nothing.
        return self.prepare_success_json_str_(True)


    def saveClassifier(self, req_params):
        """
            Save a face classifier to a file
            NOTE: Not used at the moment because this engine does not need to
                  save/load a classifier to/from a file
            Parameters:
                req_params: JSON object with at least the field:
                            - query_id: the id of the query
                            - filepath: Full path to the classifier file
            Returns:
                JSON formatted string with 'success' field set to 'False'
                in case of any problems. The 'success' field set to 'True'
                otherwise.
        """
        if 'query_id' in req_params:
            query_id = req_params['query_id']
        else:
            return self.prepare_success_json_str_(False)

        if 'filepath' in req_params:
            filepath = req_params['filepath']
        else:
            return self.prepare_success_json_str_(False)

        # Not used so far. Do nothing.
        return self.prepare_success_json_str_(True)


    def getAnnotations(self, req_params):
        """
            Loads annotations from a file
            Parameters:
                req_params: JSON object with at least the field:
                            - filepath: Full path to the annotations file
            Returns:
                JSON formatted string with 'success' field set to 'False'
                in case of any problems.

                Otherwise, the JSON  will contain the list of annotations along
                with the 'success' field set to 'True'. Each entry in the list
                of annotations will contain: the path to a training image, the
                type of training image (positive/negative/ignore), the bounding-box
                of the face detected in the image, the unique identifier of the image
                and its score in the after the ranking process.
        """
        if 'filepath' in req_params:
            annofile = req_params['filepath']
        else:
            return self.prepare_success_json_str_(False)

        ann_file = fileinput.input(annofile)
        img_list = []
        for line in ann_file:
            line = line.strip()
            arr = line.split('\t')
            x1 = float(arr[1])
            y1 = float(arr[2])
            x2 = x1+float(arr[3])
            y2 = y1+float(arr[4])
            img_list.append({'image':arr[0], 'anno':arr[5], 'roi':[x1, y1, x2, y1, x2, y2, x1, y2, x1, y1],
                            'uri':arr[6], 'score':float(arr[7])})

        return json.dumps({'success':True, 'annos':img_list})


    def saveAnnotations(self, req_params):
        """
            Saves annotations to a file.
            For each training image the following annotations are saved:
                - the path to the training image
                - type of the training image (positive/negative/ignore)
                - the bounding-box of the face detected in the image
                - the unique identifier of the image
                - its score initialized in zero
            Parameters:
                req_params: JSON object with at least the field:
                            - query_id: the id of the query
                            - filepath: Full path to the annotations file
            Returns:
                JSON formatted string with 'success' field set to 'False'
                in case of any problems. The 'success' field set to 'True'
                otherwise.
        """
        if 'query_id' in req_params:
            query_id = req_params['query_id']
        else:
            return self.prepare_success_json_str_(False)

        if 'filepath' in req_params:
            annofile = req_params['filepath']
        else:
            return self.prepare_success_json_str_(False)

        self.query_data[str(query_id)]["training_started"] = True
        try:
            query_id = str(query_id)
            print (self.query_data[query_id]["images"])
            with open(annofile, 'w') as out_file:
                for img in self.query_data[query_id]["images"]:
                    det = img["roi"]
                    anno = str(img["anno"])
                    uri = str(img["uri"])
                    score = '0.00'
                    if 'score' in img.keys():
                        score = str(img["score"])
                    out_str = img["path"]+"\t"+str(det[0])+"\t"+str(det[1])+"\t"+str(det[2])+"\t"+str(det[3])+"\t"+anno+"\t"+uri+"\t"+score+"\n"
                    out_file.write(out_str)

            self.query_data[query_id]["anno_path"] = str(annofile)
            print (self.query_data[query_id]["anno_path"])

        except Exception as e:

            print ('Could not save annotations: ' + str(e))
            return self.prepare_success_json_str_(False)

        return self.prepare_success_json_str_(True)


    def rank(self, req_params):
        """
            Ranks the images in the dataset with respect to the features extracted
            from the training images
            Parameters:
                req_params: JSON object with at least the field:
                            - query_id: the id of the query
            Returns:
                JSON formatted string with 'success' field set to 'False'
                in case of any problems. The 'success' field set to 'True'
                otherwise.
        """
        if 'query_id' in req_params:
            query_id = req_params['query_id']
        else:
            return self.prepare_success_json_str_(False)

        query_id = str(query_id)
        print ('Ranking Data')

        if settings.KDTREES_RANKING_ENABLED:
            bests = []
            accum_len = 0
            # split number of results among kdtrees to create smaller sublists of results ...
            max_results_num_splitted = (settings.MAX_RESULTS_RETURN*1.0)/len(self.kdtrees)
            # ... but always get one more element to avoid rounding errors when converting
            # to int, and this does not harm the sorting process.
            max_results_num_splitted = int(max_results_num_splitted + 1)
            for idx in range(len(self.kdtrees)):
                dd, ii = self.kdtrees[idx].query(self.query_data[query_id]["features"], k=max_results_num_splitted)
                for idx2 in range(max_results_num_splitted):
                    bests.append( ( dd[0][idx2], ii[0][idx2] + accum_len ) )
                accum_len = accum_len + self.kdtrees[idx].n
            bests_sorted = numpy.array(bests, dtype=[('dist', float), ('idx', int)])
            bests_sorted.sort(order='dist')
            ranking_indexes = [ item[1] for item in bests_sorted[0:settings.MAX_RESULTS_RETURN] ]
            dst = [ item[0] for item in bests_sorted[0:settings.MAX_RESULTS_RETURN] ]
        else:
            dst = df.cdist(self.database['feats'], self.query_data[query_id]["features"])
            ranking_indexes = numpy.argsort(dst, axis=None)
            ranking_indexes = ranking_indexes[0:settings.MAX_RESULTS_RETURN]

        print ('Done computing distances')

        ranking_list = []
        for i in range(len(ranking_indexes)):
            idx = ranking_indexes[i]
            ranking_dict = {}
            ranking_dict['path'] = self.database['paths'][idx]
            det = self.database['rois'][idx]
            roi_str = '%0.2f_%0.2f_%0.2f_%0.2f_%0.2f_%0.2f_%0.2f_%0.2f_%0.2f_%0.2f' % (
                    # x1  , y1   ,  x2  ,  y1   ,x2    ,y2    ,x1    ,y2    ,x1    ,y1
                    det[0], det[1], det[2], det[1], det[2], det[3], det[0], det[3], det[0], det[1])
            ranking_dict['roi'] = roi_str
            if settings.KDTREES_RANKING_ENABLED:
                ranking_dict['score'] = dst[i]
            else:
                ranking_dict['score'] = dst[idx][0]
            # change the score of all "bad" results according to the MAX_RESULTS_SCORE settings,
            # but only if it is enable (i.e. MAX_RESULTS_SCORE > 0 )
            if settings.MAX_RESULTS_SCORE > 0 and ranking_dict['score'] > settings.MAX_RESULTS_SCORE:
                ranking_dict['score'] = -1
            # check underlying type of results and
            # remove one dimension if necessary
            # for compatibility with json.dumps
            if isinstance(ranking_dict['path'], numpy.ndarray):
                ranking_dict['path'] = ranking_dict['path'][0]
            ranking_list.append(ranking_dict)

        self.query_data[query_id]["rankings"] = ranking_list

        print ('Ranking Done')
        return self.prepare_success_json_str_(True)


    def getRanking(self, req_params):
        """
            Retrieves the ranked list of results of a face search
            Parameters:
                req_params: JSON object with at least the field:
                            - query_id: the id of the query
            Returns:
                JSON formatted string with 'success' field set to 'False'
                in case of any problems.

                Otherwise, the JSON  will contain the ranked list along
                with the 'success' field set to 'True'. Each entry in the
                list will contain: the path to an image in the dataset
                and the bounding-box of the face detected in the image. The
                bounding-box is returned in string form with the template
                'x1_y1_x2_y1_x2_y2_x1_y2_x1_y1' where (x1,y1) and (x2,y2)
                are the top-left and bottom-right coordinates of the box,
                respectively.
        """
        if 'query_id' in req_params:
            query_id = req_params['query_id']
        else:
            return self.prepare_success_json_str_(False)

        query_id = str(query_id)

        if query_id in self.query_data:
            if "rankings" in self.query_data[query_id]:
                return json.dumps({'success': True, 'ranklist':self.query_data[query_id]["rankings"]})
            else:
                return self.prepare_success_json_str_(False)
        else:
            return self.prepare_success_json_str_(False)


    def testFunc(self, req_params):
        """
            Simple test function that will return the same JSON object as in the parameter
            It can be used to test serve_request().
            Parameters:
                req_params: JSON object
            Returns:
                JSON formatted string
        """
        return json.dumps(req_params)


    def serve_request(self, request, pid):
        """
            Redirects a request to the correspondent function
            Parameters:
                request: JSON object with at least the fields:
                        - pid: A simple ID for the current process
                        - func: Name of the function to be invoked
            Returns:
                JSON containing the response of the invoked function,
                or JSON with the 'success' field set to 'False' if the
                function is not supported.
        """
        try:
            req_params = json.loads(request)
            req_params['pid'] = pid
            if 'func' in req_params:
                method = getattr(self, req_params['func'])
                if method:
                    rval = method(req_params)
                    return rval
                else:
                    return self.prepare_success_json_str_(False)

            else:
                return self.prepare_success_json_str_(False)
        except Exception as e:
            print ('Exception in serve_request: ' + str(e))
            return self.prepare_success_json_str_(False)
