## Author : Sanchit Singhal
## Date : 04/23/2019

# import required libaries
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.community import label_propagation_communities
from networkx.algorithms.community import girvan_newman, modularity
from sklearn.metrics import adjusted_rand_score
import json

g = nx.read_edgelist("email-Eu-core.txt",create_using=nx.Graph())

print("Email network nodes, n:",
      len(g.nodes()), sep='')
print("Email network edges, m:",
      len(g.edges()), sep='')

# degree distribution, against ranks, log-log
k = [d for n, d in g.degree()]
sk = sorted(k, reverse=True)
rank = np.arange(len(sk)) + 1
plt.plot(sk,rank,'b-')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Degree')
plt.ylabel('Rank')
plt.title('Degree distribution')
plt.show()

# closeness centrality
Cclo = nx.closeness_centrality(g)

# sorting nodes by closeness centrality
Cclo_node = Cclo.keys()
Cclo_k = Cclo.values()
sortedNodes = sorted(zip(Cclo_node, Cclo_k),
                     key=lambda x: x[1], reverse=True)
sCclo_node, sCclo_k = zip(*sortedNodes)

print('Node             \tCloseness centrality')
for iNode in range(10):
    print('%-16s\t' % str(sCclo_node[iNode]), end='')
    print('%6.4f' % sCclo_k[iNode])
print()

#read the txt file
G_dept_true = nx.read_edgelist("email-Eu-core-department-labels.txt",create_using=nx.Graph())
dept_lp_true = label_propagation_communities(G_dept_true)
y_true = [list(x) for x in iter(dept_lp_true)]

#modularity
Q= modularity(G_dept_true,y_true)
print('Modularity of True Department Network',Q)
print("Number of Modules in True Department Network:",len(y_true))

#read json file
G_dept_pred = nx.read_edgelist("EmailPartitionLouvain.json",create_using=nx.Graph())
dept_lp_pred = label_propagation_communities(G_dept_pred)
y_pred = [list(x) for x in iter(dept_lp_pred)]

#modularity
Q= modularity(G_dept_pred,y_pred)
print('Modularity of Predicted Department Network',Q)
print("Number of Modules in Predicted Department Network:",len(y_pred))

# generating the detected community labels & adj rand index

y_true_gn = [n for n in G_dept_true.nodes()]
# make a node list. Then replace the node name with the true
# communitiy assignment.
for j,jComm in enumerate(dept_lp_true):
    for k in jComm:
        y_true_gn[y_true_gn.index(k)] = j

y_pred_gn = [n for n in G_dept_pred.nodes()]
# make a node list. Then replace the node name with the predicted
# communitiy assignment.
for j,jComm in enumerate(dept_lp_pred):
    for k in jComm:
        y_pred_gn[y_pred_gn.index(k)] = j

# ARI (for some reason the lengths don't match - it seems like the json file has extra nodes)
if len(y_true_gn) == len(y_pred_gn):
    rand_gn = adjusted_rand_score(y_true_gn,y_pred_gn)
    print('ARI of Department Network',rand_gn)


