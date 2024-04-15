### Format of CVRP datatset.
We define the formation of a feasible solution for CVRP. 
Rather than treating a visit to the depot as a separate step, 
we use binary variables to indicate whether a customer node is reached 
the depot or another customer node. 
Specifically, in a feasible solution, a node is assigned a value of 1 
if it is reached via the depot and a value of 0 
if it is reached through another customer node. 

For example, a feasible CVRP10 solution is:
```bash
[0,1,2,3,0,4,5,6,0,7,8,9,10,0]
```
where 0 represents the depot, can be denoted as follows:

```bash
[[1,2,3,4,5,6,7,8,9,10]
 [1,0,0,1,0,0,1,0,0,0 ]]
```
In this notation, the first row represents the sequence of customer nodes 
in the solution, and the second row indicates whether each node is reached via the depot or another customer node

We provide an example to show how to perform such transformation and create the dataset in this file: 
```bash
'LEHD_main/CVRP/Transform_data/cvrp_example_transform_data.py'
```