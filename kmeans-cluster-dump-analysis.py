import numpy as np
from numpy import linalg as la

# cosine distance between 2 tf-idf vectors
def cosine_distance(x, y) :
    xs, ys = set(x.keys()), set(y.keys())
    
    # inner product between x and y
    prod = sum( [ x[key]*y[key] for key in xs.intersection(ys) ] )
    
    # norms of x and y needed for cosine calculation
    xnorm, ynorm = la.norm( list(x.values()) ), la.norm( list(y.values()) )
      
    return 1.0 - ( prod / (xnorm*ynorm) )


# squared_euclidean distance between 2 tf-idf vectors
def squared_euclidean_distance(x, y) :
    xs, ys = set(x.keys()), set(y.keys())
    
    # sum squared error for commond words
    sse = sum( [ (x[key]-y[key])**2.0 for key in xs.intersection(ys) ] )
    # sse for those words that only appear in x
    sse += sum( [ x[key]**2.0 for key in xs.difference(ys) ] )
    # sse for those words that only appear in y
    sse += sum( [ y[key]**2.0 for key in ys.difference(xs) ] )
    
    return sse # squared norm 2  of x - y


# create a mapping to map from distance type to its associated caculation
distance_calculations = {'Cosine': cosine_distance, 'Squared_Euclidean': euclidean_distance}


# coverting array of (word, TF-IDF) tuples into Python dictionary format
def array_to_dict(arr) :
    x = {}
    
    # iterating through a list of key-value pairs, i.e. (word, TF-IDF)
    for kv_pair in arr : 
        word = list(kv_pair.keys())[0] # key -> word
        x[word] = kv_pair[word] # value -> TF-IDF
        
    return x # sparsed dictionary format


# computing mean sum squared error (MSSE) given a Mahout's cluster dump txt file and distance measure
# distance can be either cosine, euclidean or squared eucliean
def mean_squared_error(fpath, distance) :
    with open(fpath, 'r') as f : # reading a txt file where the cluster dump is contained
        
        # sum squared error or intra-cluster distance
        sse = 0.0
        
        # total number of data points
        npoints = 0
        
        for text in f.readlines() :        

            # distance between a data point and its cluster centre (centroid) is labelled with 'distance' key        
            if '[distance=' in text :
                # extract the distance, square it and then add it to SSE (sum squared error)
                sse = sse + float( text.split(': ')[1].replace('[distance=', '').replace(']', '') )**(2.0 if distance != 'squared_euclidean' else 1.0)
                
                # adding up the total number of data points
                npoints = npoints + 1
           

    return sse / npoints # mean sse 


# computing mean sum squared error (MSSE) given a Mahout's cluster dump txt file and distance measure
# distance can be either cosine, euclidean or squared eucliean
def mean_squared_between_distance(fpath, fpath_k1, distance) :
    
    # overall center of mass
    cm = None
    
    # to extract centre of mass from k=1 cluster dump
    with open(fpath_k1, 'r') as f : # reading a txt file where the cluster dump is contained
        # cluster metadata are embedded into the line labelled with 'identifier' key
        if "\"identifier\":\"" in text :
            # cluster label
            label = text.split(',')[0].split("\":\"")[-1].replace("\"", '')
                
            # cluster's centroid is labelled with 'c' key
            cn_info = text.split("],\"c\":")[-1] # centroid info + the other info also with [-1] slicing
                
            # trim out the end of the text that are not part of the cluster information 
            # convert centroid's text informaton into array and then into a Python dictionary format
            # add centroid to the cluster metadata
            cm = array_to_dict( eval( cn_info[:(cn_info.find("}],\"")+2)] ) )
            
            
    
    # cluster centres
    centroids = {}
    
    # extract centroids from the cluster dump where k is arbitary
    with open(fpath, 'r') as f : # reading a txt file where the cluster dump is contained
        
        for text in f.readlines() :
            # cluster metadata are embedded into the line labelled with 'identifier' key
            if "\"identifier\":\"" in text :
                # cluster label
                label = text.split(',')[0].split("\":\"")[-1].replace("\"", '')
                
                # cluster's centroid is labelled with 'c' key
                cn_info = text.split("],\"c\":")[-1] # centroid info + the other info also with [-1] slicing
                
                # initialise cluster metadata 
                centroids[label] = {}
                
                # trim out the end of the text that are not part of the cluster information 
                # convert centroid's text informaton into array and then into a Python dictionary format
                # add centroid to the cluster metadata
                centroids[label]['centroid'] = array_to_dict( eval( cn_info[:(cn_info.find("}],\"")+2)] ) )
                                
                # number of data points within a cluster is labelled using 'n' key.
                # extract number of data points, i.e. part (at the end) of cn_info 
                # add npoints to the cluster metadata
                centroids[label]['npoints'] = int( cn_info[(cn_info.find("}],\"")+2):].split('\"n\":')[-1].replace('}','') )
                
    
    
    # between cluster sum squared distance and total number of data points
    bss, n = 0.0, 0    
    
    # iterating through all the cluster centres
    for label in centroids.keys() :
        centroid = centroids[label]['centroid']
        npoints = centroids[label]['npoints']
        
        # given a cluster, compute between cluster distance, i.e. bss, for that cluster 
        # and adding up to form an overall bss
        bss = bss + npoints * distance_calculations[distance](cm,centroid)**(2.0 if distance != 'squared_euclidean' else 1.0)
              
        n = n + npoints
        
    return bss / npoints # mean squared between distance
