## Author : Sanchit Singhal
## Date : 04/23/2019

# import required libaries
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.community import label_propagation_communities
from networkx.algorithms.community import girvan_newman, modularity

# functions needed
# girman-newman method, optimized with modularity
def girvan_newman_opt(G, verbose=False):
    runningMaxMod = 0
    commIndSetFull = girvan_newman(G)
    for iNumComm in range(2,len(G)):
        if verbose:
            print('Commnity detection iteration : %d' % iNumComm)
        iPartition = next(commIndSetFull)  # partition with iNumComm communities
        Q = modularity(G, iPartition)  # modularity
        if Q>runningMaxMod:  # saving the optimum partition and associated info
            runningMaxMod = Q
            OptPartition = iPartition
    return OptPartition

# import data
G_netsci = nx.read_gml('netscience.gml')  # network science co-authorship

print("Total Network science co-authorship network nodes, n:",
      len(G_netsci.nodes()), sep='')
print("Total Network science co-authorship network edges, m:",
      len(G_netsci.edges()), sep='')

# Extract the giant component of the network
GC_netsci = max(nx.connected_component_subgraphs(G_netsci), key=len)
print("Giant Component of Network science co-authorship network nodes, n:",
      len(GC_netsci.nodes()), sep='')
print("Giant Component of Network science co-authorship network edges, m:",
      len(GC_netsci.edges()), sep='')
# calculate number of modules
commInd_netsci_gn = girvan_newman_opt(GC_netsci)
print("Number of modules in Giant Component of Network science co-authorship network:",
      len(commInd_netsci_gn), sep='')
# calculate modularity
Q = modularity(GC_netsci, commInd_netsci_gn)
print("Modularity of Giant Component of Network science co-authorship network:",
      Q, sep='')

#extract largest module as a sub-network
index=[]
for listitem in commInd_netsci_gn:
    x=0
    for item in listitem:
        x+=1
    index.append(x)
GC_netsci=G_netsci.subgraph(commInd_netsci_gn[np.argmax(index)])

# Extract the giant component of the sub-network
GC_netsci_sub = max(nx.connected_component_subgraphs(GC_netsci), key=len)
print("Giant Component of Sub-Network science co-authorship network nodes, n:",
      len(GC_netsci_sub.nodes()), sep='')
print("Giant Component of Sub-Network science co-authorship network edges, m:",
      len(GC_netsci_sub.edges()), sep='')
# calculate number of modules
commInd_netsci_gn = girvan_newman_opt(GC_netsci_sub)
print("Number of modules in Giant Component of Sub-Network science co-authorship network:",
      len(commInd_netsci_gn), sep='')
# calculate modularity
Q = modularity(GC_netsci_sub, commInd_netsci_gn)
print("Modularity of Giant Component of Sub-Network science co-authorship network:",
      Q, sep='')
