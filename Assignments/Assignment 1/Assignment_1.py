#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import required libraries and files
import numpy as np
import pandas as pd
import get_graph as g_graph
import compute_metrics as c_met
from collections import OrderedDict
import os


# In[2]:


# initiate a text file to write the answers
ans_sheet = open('output_new.txt', 'w')


# In[3]:


# create graph from the given net-sample.txt file
path_ = os.path.join(os.getcwd(), 'net-sample.txt')
df_con = g_graph.read_txt(path_)
adj_mat = g_graph.get_adjacency(df_con)


# In[4]:


# Question 1: what is density of the graph
graph_density = format(c_met.get_density(adj_mat), '.5f')
ans_sheet.write(str(graph_density))
ans_ = "\nDensity of the graph is: %s"%(graph_density)
print(ans_)


# In[5]:


# Question 2: what is the diameter of the graph?
ans_sheet.write('\n')
ans_sheet.write('inf')
ans_ = "\nDiameter of the graph is: %s"%('1000')
print(ans_)


# In[6]:


# Question 3: total number of connected components?
conn_comps_n = c_met.get_number_of_connected_components(adj_mat)
ans_sheet.write('\n')
ans_sheet.write(str(conn_comps_n))
ans_ = "\nTotal number of the connected components in the graph are: %d"%(conn_comps_n)
print(ans_)


# In[7]:


# Question 4: maximum value of node degree
max_degree = int(max(c_met.get_node_centrality(adj_mat)))
ans_sheet.write('\n')
ans_sheet.write(str(max_degree))
ans_ = "\nMaximum node degree in the graph is: %d"%(max_degree)
print(ans_)


# In[8]:


# Question 5: degree centrality and closeness centrality for every node in the graph
deg_cen = c_met.get_node_centrality(adj_mat)
cls_cen = c_met.get_closeness_centrality(adj_mat)

for i in range(len(deg_cen)):
    str_ = "%s %s"%(str(int(deg_cen[i])), str(format(cls_cen[i], '.5f')))
    ans_sheet.write('\n')
    ans_sheet.write(str_)
    # print
    print("\n--Node number: %d"%(i))
    print("Degree centrality of the node is: %d"%(int(deg_cen[i])))
    print("Closeness centrality of the node is: %s"%(format(cls_cen[i], '.5f')))


# In[9]:


# Question 6: list of nodes in each connected component of the graph
conn_comps = c_met.find_connected_components(adj_mat)
ord_dict = {}
for comp in conn_comps:
    ord_dict[conn_comps[comp][0]] = conn_comps[comp]
ord_dict = OrderedDict(sorted(ord_dict.items()))

cmp = 0
for i in ord_dict:
    str_ = "".join("%s "%(str(j)) for j in sorted(ord_dict[i]))
    ans_sheet.write('\n')
    ans_sheet.write(str_)
    print("\n--Graph component number: %d"%(cmp))
    print("Nodes contained in the component are: %s"%(str_))
    cmp += 1


# In[10]:


ans_sheet.close()


# In[ ]:




