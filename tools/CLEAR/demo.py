from multiprocessing import Pool
import itertools
import numpy as np
import sys
import time

# Decide between singlecore and multicore execution
multicore = np.size(sys.argv) > 1 and sys.argv[1] == '--multicore'
provider = Pool() if multicore else itertools # Pool() utilises all cores

'''
Copyleft Oct 30, 2016 Arya Iranmehr, PhD Student, Bafna Lab, UC San Diego,  Email: airanmehr@gmail.com
'''
from CLEAR import *

start_time = time.time()

data=loadSync()
Ts=precomputeTransitions(data)# Ts.to_pickle(path+'T.df');
E,CD=precomputeCDandEmissions(data)
variantScores=HMM(CD, E, Ts)
regionScores=scanGenome(variantScores.alt-variantScores.null,{'CLEAR':lambda x: x.mean()},winSize=200000,step=100000,minVariants=2)
#Manhattan(regionScores,std_th=2)
#print variantScores
variantScores.to_csv('output_variant_scores.csv', sep='\t')
#regionScores.to_csv('output_region_scores.csv', sep='\t')
#np.savetxt('output_variant_scores_vs_region_scores.csv', variantScores, delimiter='\t')
#print regionScores

end_time = time.time()
print end_time - start_time
print 'done'
