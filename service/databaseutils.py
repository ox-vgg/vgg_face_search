__author__      = 'Ernesto Coto'
__copyright__   = 'August 2018'

import settings
import kdutils
import os
import pickle # used for saving the lists and dictionaries
import dill   # used for saving the kd-trees
import time
import multiprocessing


def build_database_features_kdtrees():
    """
        IMPORTANT: This method must be executed before remove_features_from_database().

        For large datasets of images, the features can be stored in multiple kd-trees
        to speed up the distance computation between feature vectors.

        This method will extract the features from a previously generated database file
        and convert it into kd-trees. If the database file points to a list of sub-databases,
        one kd-tree file will be generated for each sub-database, and the main kd-tree file
        will just contain the list of kd-tree files corresponding to each sub-database.

        The KDTREES_DATASET_SPLIT_SIZE variable is used to split the database in pieces.
        Each piece will be converted into a kd-tree. Each kd-tree file can containing multiple
        kd-trees.

        Building and querying one kd-tree can take a long time, so you can adjust the
        KDTREES_DATASET_SPLIT_SIZE variable to reduce the complexity of these operations.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html
    """
    try:
        if os.path.exists(settings.KDTREES_FILE):
            print ('Found kd-trees file. Nothing to be done')
        else:
            worker_pool = multiprocessing.Pool(processes=settings.NUMBER_OF_HELPER_WORKERS)
            kdtrees = []
            print ('Building kd-trees for ' + settings.DATASET_FEATS_FILE)
            with open(settings.DATASET_FEATS_FILE, 'rb') as fin:
                database_content = pickle.load(fin)
                if isinstance(database_content, dict):
                    kdutils.build_kdtrees(database_content['feats'], settings.KDTREES_DATASET_SPLIT_SIZE, worker_pool, settings.KDTREES_FILE)
                elif isinstance(database_content, list):
                    for entry in database_content:
                        database_chunk_content = None
                        splitted_dataset = []
                        if os.path.sep not in entry:
                            # in this case, assume it is in the same directory as the DATASET_FEATS_FILE
                            sub_database = os.path.join(os.path.dirname(settings.DATASET_FEATS_FILE), entry)
                        else:
                            sub_database = entry
                        print ('Loading sub-database ' + entry)
                        sub_kdtree_fname = "kdtree_" + entry
                        with open(sub_database, 'rb') as fin_chunk:
                            database_chunk_content = pickle.load(fin_chunk)
                        kdutils.build_kdtrees(database_chunk_content['feats'], settings.KDTREES_DATASET_SPLIT_SIZE,
                                              worker_pool, os.path.join(os.path.dirname(settings.DATASET_FEATS_FILE), sub_kdtree_fname))
                        # delete after saving to disk to release memory
                        del database_chunk_content['feats']
                        kdtrees.append(sub_kdtree_fname)

                    # save kd-tree list
                    print ('Saving kd-tree list ' + str(kdtrees))
                    with open(settings.KDTREES_FILE, 'wb') as fout:
                        dill.dump(kdtrees, fout)

    except Exception as e:
        print ('Failed building kd-trees. Reason: ' + str(e))
        pass


def remove_features_from_database():
    """
        IMPORTANT: This method must be executed after build_database_features_kdtrees().

        If you have saved the feature vector into kd-trees, then it is convenient to
        remove them from the original database file, to avoid loading the features twice
        in memory. This is specially useful for large datasets.

        Use this method to produce new dataset files that will not include the feature
        vectors. Once you have done it, change the DATASET_FEATS_FILE variable in the
        settings to point to the new main database file.

        Note that if the original database file included multiple sub-databases, new versions
        of the sub-databases will be produced as well.
    """
    try:
        print ('Loading database ' + settings.DATASET_FEATS_FILE)
        with open(settings.DATASET_FEATS_FILE, 'rb') as fin:
            database_content = pickle.load(fin)
            if isinstance(database_content, dict):
                database = {'paths': [], 'rois': []}
                database['paths'].extend(database_content['paths'])
                database['rois'].extend(database_content['rois'])
                # save to new database file
                NEW_DATASET_FILE = settings.DATASET_FEATS_FILE.replace('.pkl', '_nofeats.pkl')
                print ('Generating new database ' + NEW_DATASET_FILE)
                with open(NEW_DATASET_FILE, 'wb') as fout:
                    pickle.dump(database, fout, pickle.HIGHEST_PROTOCOL)
            elif isinstance(database_content, list):
                database_list = []
                for entry in database_content:
                    if os.path.sep not in entry:
                        # in this case, assume it is in the same directory as the DATASET_FEATS_FILE
                        sub_database = os.path.join(os.path.dirname(settings.DATASET_FEATS_FILE), entry)
                    else:
                        sub_database = entry
                    print ('Loading sub-database ' + sub_database)
                    with open(sub_database, 'rb') as fin_chunk:
                        database_chunk_content = pickle.load(fin_chunk)
                    database = {'paths': [], 'rois': []}
                    database['paths'].extend(database_chunk_content['paths'])
                    database['rois'].extend(database_chunk_content['rois'])
                    # save to new sub-database file
                    NEW_SUB_DATASET_FILE = sub_database.replace('.pkl', '_nofeats.pkl')
                    print ('Generating new sub-database ' + NEW_SUB_DATASET_FILE)
                    with open(NEW_SUB_DATASET_FILE, 'wb') as fout:
                        pickle.dump(database, fout, pickle.HIGHEST_PROTOCOL)
                    database_list.append(entry.replace('.pkl', '_nofeats.pkl'))
                # save to new database file
                NEW_DATASET_FILE = settings.DATASET_FEATS_FILE.replace('.pkl', '_nofeats.pkl')
                print ('Generating new database ' + NEW_DATASET_FILE)
                with open(NEW_DATASET_FILE, 'wb') as fout:
                    pickle.dump(database_list, fout, pickle.HIGHEST_PROTOCOL)
            else:
                raise Exception('File %s contains corrupted information.' % settings.DATASET_FEATS_FILE)
    except Exception as e:
        print ('Failed building new database. Reason: ' + str(e))
        pass
