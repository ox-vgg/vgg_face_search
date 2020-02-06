__author__      = 'Ernesto Coto'
__copyright__   = 'August 2018'

from scipy.spatial import cKDTree
import dill # used for saving the kd-trees
import time
import os


def build_kdtrees(feats, split_size, thead_pool, filename):
    """
        Builds a file containing the specified feature vectors
        in the form of a list of kd-trees
        Parameters:
            feats: List of feature vectors to be saved
            split_size: Number of vectors per kd-tree. The feats list will
                        be split in pieces of this size
            thead_pool: Pool of threads to be used for parallel computation
                        of the kd-trees
            filename: Full path and name to the resulting file storing
                      the kd-trees
    """
    try:
        print ('Building kd-trees for ' + filename)
        num_feats = len(feats)
        splitted_feats = []
        if num_feats > split_size:
            NUM_DATASET_SPLITS = num_feats/split_size
            for idx in range(NUM_DATASET_SPLITS):
                if idx < NUM_DATASET_SPLITS-1:
                    splitted_feats.append(feats[idx*split_size:(idx+1)*split_size])
                else:
                    splitted_feats.append(feats[idx*split_size:] )
        else:
            splitted_feats = [ feats ]
        t = time.time()
        kdtrees = thead_pool.map( cKDTree, splitted_feats )
        print ('Done building %d kd-trees in t=%f' %( len(kdtrees) , time.time() - t))
        t = time.time()
        with open(filename, 'wb') as f:
            dill.dump(kdtrees, f)
        print ('Done saving kd-trees in t=%f' % (time.time() - t))
    except Exception as e:
        print ('Failed building kd-trees. Reason: ' + str(e))
        pass


def load_kdtrees(filename):
    """
        Load a list of kd-trees from the specified file.
        Parameters:
            filename: Full path and name to the resulting file storing
                      the kd-trees
        Returns:
            A list of kd-tree objects
    """
    try:
        kdtrees = []
        if os.path.exists(filename):
            t = time.time()
            with open(filename, 'rb') as fin:
                kdtrees_file_content = dill.load(fin)
                if len(kdtrees_file_content)>0:
                    if isinstance(kdtrees_file_content[0], str):
                        for entry in kdtrees_file_content:
                            if os.path.sep not in entry:
                                # in this case, assume it is in the same directory as the file specified in the parameter
                                sub_kdtree = os.path.join(os.path.dirname(filename), entry)
                            else:
                                sub_kdtree = entry
                            print ('Loading sub-kdtree file ' + sub_kdtree)
                            with open(sub_kdtree, 'rb') as fin_sub_kdtree:
                                kdtrees.extend( dill.load(fin_sub_kdtree) )
                    else:
                        kdtrees.extend( kdtrees_file_content )
                else:
                    kdtrees = []
            print ('Done loading %d kd-trees in t=%f' % ( len(kdtrees), time.time() - t))
    except Exception as e:
        print ('Failed loading kd-trees. Reason: ' + str(e))
        kdtrees = []
        pass

    return kdtrees
