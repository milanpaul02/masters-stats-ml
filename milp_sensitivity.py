#Made with AI assistance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings as wr
wr.filterwarnings('ignore')
from ucimlrepo import fetch_ucirepo 

from sklearn.ensemble import RandomForestRegressor
import gurobipy as gp
from gurobipy import *

# fetch dataset and convert to pandas dataframe
auto_mpg = fetch_ucirepo(id=9).data.original 

#remove the car_name and acceleration columns

auto_mpg = auto_mpg.drop(columns=['car_name', 'acceleration'])

#remove the rows with missing values for horsepower
auto_mpg.dropna(subset=['horsepower'], inplace=True, ignore_index=True)

#fit a random forest of 25 trees each of maximum depth 4 to the data
rf = RandomForestRegressor(n_estimators=25, max_depth=4, random_state=42)
rf.fit(auto_mpg.drop(columns=['mpg']), auto_mpg['mpg'])

#define the function to find the largest output gap p where the model is (p, F)-sensitive for a given F.
def sensitivity(rf, F):

    #create a gurobi model
    model = gp.Model("sensitivity_regression")

    #create the arrays with all the guards occurring in the random forest for each feature

    #guards_list is a list of 6 empty lists, one for each feature
    guards_list = [[] for i in range(6)]

    for tree in rf.estimators_:
        tree_ = tree.tree_
        for i in range(tree_.node_count):
            if tree_.feature[i] >= 0: #if the node is not a leaf
                feature_index = tree_.feature[i]
                guard_value = tree_.threshold[i]
                guards_list[feature_index].append(guard_value)

    # Convert lists to numpy arrays for further processing if needed
    guards_list = [np.array(guards) for guards in guards_list]

    #drop duplicates and sort the elements in guards_list for each feature
    guards_list = [np.unique(guards) for guards in guards_list]

    #create gurobipy variables for both x and x' to encode the position with respect to the guards for each feature
    x_vars = [[] for i in range(6)]
    x_prime_vars = [[] for i in range(6)]
    for i in range(6):
        for j in range(len(guards_list[i])):
            x_var = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            x_prime_var = model.addVar(vtype=GRB.BINARY, name=f"x_prime_{i}_{j}")
            x_vars[i].append(x_var)
            x_prime_vars[i].append(x_prime_var)

    #add the constraints regarding the variables encoding the positions of x and x'
    for i in range(6):
        for j in range(len(guards_list[i])):
            if j != len(guards_list[i]) - 1: #if it's not the last guard, add the constraint that if x is less than or equal to the guard, then x is also less than or equal to the next guard
                model.addConstr(x_vars[i][j] <= x_vars[i][j+1], name=f"constraint_x_{i}_{j}")
                model.addConstr(x_prime_vars[i][j] <= x_prime_vars[i][j+1], name=f"constraint_x_prime_{i}_{j}")
        
            #add the constraint that x and x' are equal for all features not in F
            if i not in F:
                model.addConstr(x_vars[i][j] == x_prime_vars[i][j], name=f"constraint_F_{i}_{j}")

    #create binary variables for each node of each tree of the tree to encode whether x has visited the node or not
    node_vars = [[] for i in range(25)]
    for i in range(25):
        tree_ = rf.estimators_[i].tree_
        for j in range(tree_.node_count):
            node_var = model.addVar(vtype=GRB.BINARY, name=f"node_{i}_{j}")
            node_vars[i].append(node_var)

    #create binary variables for each node of each tree of the tree to encode whether x' has visited the node or not
    node_vars_prime = [[] for i in range(25)]
    for i in range(25):
        tree_ = rf.estimators_[i].tree_
        for j in range(tree_.node_count):
            node_var_prime = model.addVar(vtype=GRB.BINARY, name=f"node_prime_{i}_{j}")
            node_vars_prime[i].append(node_var_prime)

    #add the constraints to encode the condition that the root nodes of all trees are visited by both x and x'
    for i in range(25):
        model.addConstr(node_vars[i][0] == 1, name=f"root_node_{i}")
        model.addConstr(node_vars_prime[i][0] == 1, name=f"root_node_prime_{i}")

    #add the constraint to encode the condition that if x has visited a node and if it satisfies the guard, then it has also visited the left child of the node
    #and if it does not satisfy the guard, then it has also visited the right child
    for i in range(25):
        tree_ = rf.estimators_[i].tree_
        for j in range(tree_.node_count):
            if tree_.feature[j] >= 0: #if the node is not a leaf
                feature_index = tree_.feature[j]
                guard_value = tree_.threshold[j]
                #find the index of the guard value in guards_list[i] for the threshold tree_.threshold[j]
                guard_index = guards_list[feature_index].tolist().index(guard_value)

                #add the constraint that if x has visited node v whose feature is feature_index and if x is lesser than or equal to the guard value, then x has also visited the left child of v
                model.addConstr(node_vars[i][j] + x_vars[feature_index][guard_index] - 1 <= node_vars[i][tree_.children_left[j]], name=f"left_child_{i}_{j}")
                #add the constraint that if x has visited node v whose feature is feature_index and if x is greater than the guard value, then x has also visited the right child of v
                model.addConstr(node_vars[i][j] + (1 - x_vars[feature_index][guard_index]) - 1 <= node_vars[i][tree_.children_right[j]], name=f"right_child_{i}_{j}")

                #add the constraint that if x' has visited node v whose feature is feature_index and if x' is lesser than or equal to the guard value, then x' has also visited the left child of v
                model.addConstr(node_vars_prime[i][j] + x_prime_vars[feature_index][guard_index] - 1 <= node_vars_prime[i][tree_.children_left[j]], name=f"left_child_prime_{i}_{j}")
                #add the constraint that if x' has visited node v whose feature is feature_index and if x' is greater than the guard value, then x' has also visited the right child of v
                model.addConstr(node_vars_prime[i][j] + (1 - x_prime_vars[feature_index][guard_index]) - 1 <= node_vars_prime[i][tree_.children_right[j]], name=f"right_child_prime_{i}_{j}")

        #add the constraint that exactly one of the leaves of each tree is visited by x
        leaf_indices = [k for k in range(tree_.node_count) if tree_.feature[k] < 0] #the indices of the leaves of the tree
        model.addConstr(gp.quicksum(node_vars[i][k] for k in leaf_indices) == 1, name=f"leaf_x_{i}")
        #add the constraint that exactly one of the leaves of each tree is visited by x'
        model.addConstr(gp.quicksum(node_vars_prime[i][k] for k in leaf_indices) == 1, name=f"leaf_x_prime_{i}")

    #initialize the variable to encode the output of the forest for x, which is a weighted sum of the binary variables for the leaves where the weights are the values at the leaves
    output_x = gp.quicksum(node_vars[i][k] * rf.estimators_[i].tree_.value[k] for k in leaf_indices for i in range(25))/25

    #initialize the variable to encode the output of the forest for x' which is a weighted sum of the binary variables for the leaves where the weights are the values at the leaves
    output_x_prime = gp.quicksum(node_vars_prime[i][k] * rf.estimators_[i].tree_.value[k] for k in leaf_indices for i in range(25))/25

    #define the objective function to maximize the output gap between x and x' for the random forest
    model.setObjective(output_x - output_x_prime, GRB.MAXIMIZE)

    #return the optimal value of the objective function, which is the largest output gap p where the model is (p, F)-sensitive
    model.optimize()
    if model.status == GRB.OPTIMAL:
        return model.objVal
    else:
        return None

output_gap = []

#Check the worst-case function sensitivity for different feature sets F. The features are cylinders, displacement, horsepower, weight, model_year and origin.
for F in [[0], [1], [2], [3], [4], [5], [1,2,3,4,5],[0,2,3,4,5],[0,1,3,4,5],[0,1,2,4,5],[0,1,2,3,5],[0,1,2,3,4]]:
    output_gap.append(sensitivity(rf, F))
        
print(auto_mpg)
print(output_gap)

