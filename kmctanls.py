#
# Source: https://github.com/axl-knight/mahout/
# License: MIT (Free for Commercial Use)
# Author: Axl Knight
#

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
distance_calculations = {'Cosine': cosine_distance, 'SquaredEuclidean': squared_euclidean_distance}


# coverting array of (word, TF-IDF) tuples into Python dictionary format
def array_to_dict(arr) :
    x = {}
    
    # iterating through a list of key-value pairs, i.e. (word, TF-IDF)
    for kv_pair in arr : 
        word = list(kv_pair.keys())[0] # key -> word
        x[word] = kv_pair[word] # value -> TF-IDF
        
    return x # sparsed dictionary format


# multiply tfidf vector by a scalar
def dict_mul(x, n) :
    return { key: n*x[key] for key in x.keys() }

# add two tfidef vectors
def dict_add(x, y) :
    # words appear in tf-idf vectors, initialise result
    xs, ys, z = set(x.keys()), set(y.keys()), {}
    
    # commond words
    for key in xs.intersection(ys) :
        z[key] = x[key] + y[key]
        
    # words only appear in x
    for key in xs.difference(ys) :
        z[key] = x[key]
        
    # words only appear in y
    for key in ys.difference(xs) :
        z[key] = y[key]
        
    return z
  
    
# computing mean sum squared error (MSSE) given a Mahout's cluster dump txt file and distance measure
# distance can be either cosine, euclidean or squared eucliean
def cluster_evaluation(fpath, measure) :
    
    # output clusters centres
    clusters = {}
    
    with open(fpath, 'r') as f : # reading a txt file where the cluster dump is contained
        
        for text in f.readlines() :        
            # cluster metadata are embedded into the line labelled with 'identifier' key
            if "\"identifier\":\"" in text :
                # cluster label
                label = text.split(',')[0].split("\":\"")[-1].replace("\"", '')
                
                # cluster's centroid is labelled with 'c' key
                cn_info = text.split("],\"c\":")[-1] # centroid info + the other info also with [-1] slicing
                
                # initialise cluster metadata 
                clusters[label] = {}
                
                # trim out the end of the text that are not part of the cluster information 
                # convert centroid's text informaton into array and then into a Python dictionary format
                # add centroid to the cluster metadata
                clusters[label]['centroid'] = array_to_dict( eval( cn_info[:(cn_info.find("}],\"")+2)] ) )
                                
                # number of data points within a cluster is labelled using 'n' key.
                # extract number of data points, i.e. part (at the end) of cn_info 
                # add npoints to the cluster metadata
                clusters[label]['npoints'] = int( cn_info[(cn_info.find("}],\"")+2):].split('\"n\":')[-1].replace('}','') )
                
                # initialise SSE
                clusters[label]['sse'] = 0.0
                
                # intialise min and max distances
                clusters[label]['min'], clusters[label]['max'] = 1.0, 0.0
                

            # distance between a data point and its cluster centre (centroid) is labelled with 'distance' key        
            if '[distance=' in text :
                # extract the distance, square it and then add it to SSE (sum squared error)
                distance = float( text.split(': ')[1].replace('[distance=', '').replace(']', '') )**(2.0 if measure != 'SquaredEuclidean' else 1.0)
                clusters[label]['sse'] += distance
                
                # minimum distance until now
                if clusters[label]['min'] > distance :
                    clusters[label]['min'] = distance
                
                # maximum distance until now
                if clusters[label]['max'] < distance :
                    clusters[label]['max'] = distance
           
 
    # iterating through all the cluster centres and computing grand centre of mass
    gcm, N = {}, 0
    for label in clusters.keys() :
        
        npoints = clusters[label]['npoints']
        
        if len(gcm) > 0 :
            gcm = dict_add( gcm, dict_mul(clusters[label]['centroid'], npoints) )
        else :
            npoints = clusters[label]['npoints']
            gcm = dict_mul( clusters[label]['centroid'], npoints )
            
        N = N + npoints    
            
    # grand centre of mass
    gcm = dict_mul(gcm, 1.0/N)
    
    
    # between cluster sum squared distance, within cluster sum squared distance, number of data points
    bss, wss, N = 0.0, 0.0, 0    
    
    # iterating through all the cluster centres
    for label in clusters.keys() :
        centroid = clusters[label]['centroid']
        npoints = clusters[label]['npoints']
        
        # given a cluster, compute between cluster distance, i.e. bss, for that cluster
        clusters[label]['bss'] = npoints * distance_calculations[measure](gcm,centroid)**(2.0 if measure != 'SquaredEuclidean' else 1.0)    
        # overall bss = sum of individual's bss  
        bss = bss + clusters[label]['bss']
        
        # within-cluster sum square distance
        wss = wss + clusters[label]['sse']
        
        # total number of data points
        N = N + npoints
        
      
    return { 'Mean Between Sum Square (MBSS)': bss / N,
             'Mean Within Sum Square (MWSS)': wss / N }