from graphviz import Digraph

f = Digraph('Decision tree')


f.attr('node', shape='ellipse', style='filled', color='cyan')
f.node('R',label='Root node')


f.attr('node', shape='ellipse', style='filled', color='lightblue')
f.node('d1', label='Decision node')
f.node('d2', label='Decision node')
f.node('d3', label='Decision node')


f.attr('node', shape='ellipse', style='filled', color='orange')
f.node('L1', label='Leaf')
f.node('L2', label='Leaf')
f.node('L3', label='Leaf')
f.node('L4', label='Leaf')
f.node('L5', label='Leaf')

f.edge('R', 'd1', label='True')
f.edge('R', 'd2', label='False')
f.edge('d1', 'L1')
f.edge('d1', 'L2')
f.edge('d2', 'd3')
f.edge('d2', 'L3')
f.edge('d3', 'L4')
f.edge('d3', 'L5')


f.save('SampleDecisionTree.dot')
