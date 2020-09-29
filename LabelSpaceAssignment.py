import torch
import torch.nn as nn
from itertools import chain, combinations
import numpy as np
import warnings
import time
import pickle
import cplex as cpl

class LabelSpaceAssignment():

    def __init__(self, classes):
        ###########################################
        ## Initialize the class
        ## Input:
        ##      classes: number of classes
        ## Returns:
        ##      self.classes: number of classes (D)
        ##      self.max_combination: Number of maximum combination of joint labels
        ##      self.mutual_exclude: mutual pair of labels which cant exist and needs to be excluded
        ##      self.include_negative: True if negative class is possible
        ##      self.use_detection: True if detection scores are used 
        ###########################################

        self.classes = classes
        self.max_combination = 4
        # mutual_exclude = pickle.load(open('', 'rb'))#.nonzero()
        # self.mutual_exclude = mutual_exclude
        self.include_negative = False
        self.use_detection = True

    def __powerset_generator(self, i):
        j = range(i)
        for subset in chain.from_iterable(combinations(j, r) for r in range(len(j)+1)):
            yield subset

    def labelpowerset(self, bag_label, only_single_data_bag=False):
        ###########################################
        ## Generate powerset of labels from a bag label
        ## Input:
        ##      bag_label: (1 x D)
        ##      only_single_data_bag: False (if true return the bag label as labelpowerset)
        ## Returns:
        ##      labels: subset of labels (N_o)
        ##      set_indices: location indices of positive class
        ###########################################
        if only_single_data_bag:
            max_combination = max(self.max_combination,bag_label.sum().int().item())
        else:
            max_combination = self.max_combination
        indices = (bag_label == 1).nonzero(as_tuple=True)[1]
        labels = []
        set_indices = [[] for i in range(indices.shape[0])]
        counter = 0
        
        for i in self.__powerset_generator(indices.shape[0]):
            if len(i) == 0 or len(i) > max_combination:
                continue
            plabel = torch.zeros_like(bag_label)
            tempind = indices[i,]
            # if self.mutual_exclude != [] and sum([len(set(tempind).intersection(set(t)))>1 for t in zip(self.mutual_exclude[0],self.mutual_exclude[1])]) > 0:
            #     continue
            for j in range(tempind.shape[0]):
                plabel[0,tempind[j]] = 1
                set_indices[i[j]].append(counter)
            labels.append(plabel)
            counter = counter + 1
        labels = torch.cat(labels,dim=0)
        return labels, set_indices

    def getNormalizedScore(self, bag_label, inst_data, dets = None):

        ###########################################
        ## Getting the normalized score for a batch
        ## Input:
        ##      bag_label: labels of each bag (1 x D)
        ##      inst_data: all instances in the batch (N_p x D)
        ##      dets: detection score for each instance (N_p x 1)
        ## Important variables
        ##      $\omega$ is the powerset of a bag
        ## Returns:
        ##      scores: normalized detection score (N_o x N_p)
        ##      omega_small: $\omega$ (N_o x D)
        ###########################################

        num_data = inst_data.shape[0]  #### Total number of instances in a bag (N_p)
        num_class = bag_label.shape[1] #### Total number of class (D)
        num_sets = 0 #### Cardinatlity of $\Omega$ (N_o)


        if bag_label[0,:].sum(-1) > 0: #### If the bag label is negative
            omega_small, set_indices = self.labelpowerset(bag_label,(num_data==1))
        else:
            omega_small = torch.zeros(1,num_class,requires_grad=False).type_as(bag_label)
            set_indices = [[0]]
            
        cardinality_powerset = omega_small.shape[0] #### cardinatlity of $\omega$

        if self.include_negative and bag_label[0,:].sum(-1) > 0: #### if only negative label then that is the only solution
            cardinality_powerset = cardinality_powerset + 1
            omega_small_temp = torch.zeros((cardinality_powerset, num_class)).type_as(inst_data)
            omega_small_temp[1:, :] = omega_small
            omega_small = omega_small_temp
            # # set_indices.insert(0,torch.empty(0).type_as(bag_label))

            
        num_sets += cardinality_powerset

        ### Assert of the size
        assert (omega_small.shape[0] == num_sets)
        
        tempdata = inst_data.unsqueeze(0).repeat(num_sets, 1,1) ### Extend it to 3D
        templabel = omega_small.unsqueeze(1).repeat(1,num_data,1) ### Extend it to 3D


        ### code to incorporate  all zero label if any
        neg_class_indicator = (templabel.sum(-1,True)==0).type_as(templabel)
        templabel = torch.cat((templabel,neg_class_indicator),dim=-1)
        neg_class_logit = -tempdata.max(-1,True)[0]
        tempdata = torch.cat((tempdata,neg_class_logit),dim=-1)


        relData= (templabel * tempdata).sum(-1)
        
        relData = torch.exp(relData.clamp(min=-50.0,max=50.0))
        
        if self.use_detection:
            scores = (relData/(relData.sum(dim=0,keepdims=True))+0.000001) * dets.transpose(1,0) #* np.expand_dims(dets[:,0],axis=0).repeat(label_space.shape[0], axis=0)
        else:
            scores = (relData/(relData.sum(dim=0,keepdims=True))+0.000001)

        assert (torch.isnan(scores).any() == False)

        return scores, omega_small

    def assignmentSingle(self, data, bag_label, dets = None):
    
        ###########################################
        ## Get assignment for a batch
        ## Input:
        ##      data: all instances in the batch (N_p x D)
        ##      bag_label: labels of each bag (N x D)
        ##      dets: detection score for each instance (N_p x 1)
        ## Returns:
        ##      scores is the score matrix for optimization (N_o x N_p)
        ##      omega is the powerset (N_o x D)
        ##      inst_labels is the linear programming solved instance labels (N_p x D)
        ###########################################
        
        if dets == None:
            self.use_detection = False

        num_inst = int(data.shape[0]) ### N_p
        num_class = bag_label.shape[1] ### D
        num_bags = bag_label.shape[0]  ### N

        scores, omega = self.getNormalizedScore(bag_label, data, dets)
        num_sets = omega.shape[0] ### N_o
        inst_labels, omega_indx = self.solve_lp_cplex(bag_label,scores,omega)

        return inst_labels, scores, omega_indx

    def locate_subset_indices(self,bag_label,omega):
        ###########################################
        ## returns the indices for the subsets in omega given the bag_label (code in numpy)
        ## Input:
        ##      bag_label: labels of each bag (1 x D)
        ##      omega: powerset (N_o x D)
        ## Returns:
        ##      indix: list of indices for each class in bag_label
        ###########################################

        num_class = omega.shape[1] ### D
        num_sets = omega.shape[0] ### N_o

        indix = []

        ### number of non-zero indices
        num_indices = bag_label.sum().astype(int) + 1 ## +1 for negative class (ie all zeros)
        
        for j in range(num_indices):
            if j > 0:
                class_indices = bag_label.nonzero()[0][j-1]
                indix.append((omega[:,class_indices] == 1).nonzero()[0])
            else:
                ### If negative class in the omega
                if (omega.sum(-1) == 0).nonzero()[0].shape[0] > 0 and bag_label.sum() == 0:
                    indix.append((omega.sum(-1) == 0).nonzero()[0])

        return indix

    def solve_lp_cplex(self,bag_label,scores,omega):
        ###########################################
        ## Solves the linear assignment
        ##      Convertion of torch to numpy for Linear Programming solving
        ## Input:
        ##      bag_label: labels of each bag (1 x D)
        ##      scores: score matrix for optimization (N_o x N_p)
        ##      omega: powerset (N_o x D)
        ## Returns:
        ##      inst_label: instant labels (N_p x D)
        ##      omega_indx: indices of omega (N_p)
        ###########################################

        num_sets = int(scores.shape[0]) ### N_o
        num_inst = int(scores.shape[1]) ### N_p
        num_class = int(bag_label.shape[1]) ### D

        ### Initialize space for return value        
        inst_label = torch.zeros((num_inst,num_class),requires_grad=False).type_as(scores)
        omega_indx = torch.zeros((num_inst),requires_grad=False).type_as(scores)

        ### Convert into numpy for linear programming
        scores_numpy = (scores.clone().detach().cpu().numpy())#.astype(np.int)
        bag_label_numpy = bag_label.clone().cpu().numpy()
        omega_numpy = omega.clone().cpu().numpy()

        #### One single shot optimization

        #### define the problem
        assignment_model = cpl.Cplex()

        ### Add variable
        assignment_model.variables.add(names= ["x"+str(i*num_inst+j) for i in range(num_sets) for j in range(num_inst)])
       
        ### define type of variable
        for i in range(num_sets):
            for j in range(num_inst):
                assignment_model.variables.set_types("x"+str(i*num_inst + j), assignment_model.variables.type.binary)
                assignment_model.variables.set_lower_bounds("x"+str(i*num_inst + j), 0.0)
                assignment_model.variables.set_upper_bounds("x"+str(i*num_inst + j), 1.0)
        constraints = [] 

        ###########################################
        #### Constraint for linear programming (Eq. 5)
        ##  variable == 1
        ###########################################
        for j in range(num_inst):
            assignment_model.linear_constraints.add(
                lin_expr= [cpl.SparsePair(ind= ["x"+str(i*num_inst + j) for i in range(num_sets)],
                    val= [1.0 for i in range(num_sets)])],
                rhs= [1.0],
                names = ["c1"+str(j)],
                senses = ['E']
                )

        ###########################################
        #### Constraint for linear programming (Eq. 6)
        ##  variable >= 1
        ###########################################
        indix = self.locate_subset_indices(bag_label_numpy[0,:],omega_numpy)
        for l,indxx in enumerate(indix):
            assignment_model.linear_constraints.add(
                lin_expr= [cpl.SparsePair(ind= ["x"+str(i*num_inst + j) for i in indxx for j in range(num_inst)],
                    val= [1.0 for i in indxx for j in range(num_inst)])],
                rhs= [1.0],
                names = ["c2"+str(l)],
                senses = ['G']
                )

        ###########################################
        #### Objective function for linear programming 
        ##  maximize (scores * variable)
        ###########################################
        for i in range(num_sets): 
            for j in range(num_inst):
                assignment_model.objective.set_linear("x"+str(i*num_inst + j), float(scores_numpy[i,j]))
        assignment_model.objective.set_sense(assignment_model.objective.sense.maximize)
        

        ### Solve assignment
        assignment_model.set_log_stream(None)
        assignment_model.set_error_stream(None)
        assignment_model.set_warning_stream(None)
        assignment_model.set_results_stream(None)
        assignment_model.solve()

        #### Check if assignment fails
        if "infeasible" in assignment_model.solution.get_status_string(assignment_model.solution.get_status()) or \
                "unbounded"  in assignment_model.solution.get_status_string(assignment_model.solution.get_status()):
            print ("Unsolved - Max per instance = ", assignment_model.solution.get_status_string(assignment_model.solution.get_status()))
            _, omega_indx = scores.max(0)
            # inst_label = torch.index_select(omega, 0, omega_indx.long())
            inst_label = bag_label.repeat(num_inst, 1)

        else:
            solution = assignment_model.solution.get_values()

            if sum(solution) != num_inst:
                print ("Unsolved - Max per instance = ",
                       assignment_model.solution.get_status_string(assignment_model.solution.get_status()))
                _, omega_indx = scores.max(0)
                # inst_label = torch.index_select(omega, 0, omega_indx.long())
                inst_label = bag_label.repeat(num_inst, 1)

            #### Retrieve the assignment if successful
            for i in range(num_sets):
                for j in range(num_inst):
                    if solution[assignment_model.variables.get_indices("x"+str(i*num_inst + j))] == 1.0:
                        inst_label[j,:] = omega[i,:]
                        omega_indx[j] = i

        return inst_label, omega_indx
