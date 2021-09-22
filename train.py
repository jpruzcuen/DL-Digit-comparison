import torch
from torch import nn
from torch.nn import functional as F
 
# ==== Train Joint Network ==== #

'''
Parameters
    -------
    model: neural network (WSN or WOSN)
    train_input: training data
    train_target: labels of digit comparison for test data 
    train_classes: hot labels of digit classification for test data 
    mini_batch_size: size of mini batches
    eta: learning rate
    aux_loss: if True, the model uses an auxiliary loss during training 

Returns
-------
    acc_loss: accumulated loss 

'''

def train_model_joint(model,train_input, train_target, train_classes, mini_batch_size = 100, eta = 0.55, aux_loss = True):
    criterion_digit = nn.CrossEntropyLoss()
    criterion_comps = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = eta)
    nb_epochs = 25
    
    # Train the model
    for e in range(nb_epochs):
        acc_loss = 0
        # Mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):
            
            digits,compare = model(train_input.narrow(0, b, mini_batch_size)) 

            loss_d1 = 0 # loss digit 1
            loss_d2 = 0 # loss digit 2
            if ((aux_loss == True) and (digits != None)):
                digit1 = digits[:,0:10]
                digit2 = digits[:,10:20]
             
                #Classification loss
                loss_d1 = criterion_digit(digit1, train_classes[:,0].narrow(0, b, mini_batch_size)) 
                loss_d2 = criterion_digit(digit2, train_classes[:,1].narrow(0, b, mini_batch_size))


            #Comparison loss
            comp_loss = criterion_comps(compare.squeeze(), train_target.narrow(0, b, mini_batch_size).float()) 

            #Total loss
            loss = 0.5*loss_d1 + 0.5*loss_d2 + comp_loss
            
            acc_loss = acc_loss + loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    return acc_loss
    

# ==== Train Split Network ==== #

'''
Parameters
    -------
    ClassModel: network for classification (ClassifyNet)
    CompModel:  network for comparison (CompareNet)
    train_input: training data
    train_target: labels of digit comparison for test data 
    train_classes1: hot labels of digit classification for digit1
    train_classes2: hot labels of digit classification for digit2
    mini_batch_size: size of mini batches
    ClassEta: learning rate for classification stage
    CompEta: learning rate for comparison stage

Returns
-------
    accumulated loss 

'''

def train_model_split(ClassModel,CompModel,train_input, train_target, train_classes1, train_classes2, mini_batch_size = 100, ClassEta = 0.1, CompEta = 0.3):
    
    ClassCriterion = nn.CrossEntropyLoss()
    ClassOptimizer = torch.optim.SGD(ClassModel.parameters(), lr = ClassEta)

    CompCriterion = nn.CrossEntropyLoss()
    CompOptimizer = torch.optim.SGD(CompModel.parameters(), lr = CompEta)
    
    nb_epochs = 25

    # Train the model
    for e in range(nb_epochs):
        
        accClassLoss = 0
        accCompLoss = 0
        
        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):

            ## CLASIFICATION
            class_output, class_output_reshaped = ClassModel(train_input.narrow(0, b, mini_batch_size)) # N x 20
            
            #Classification loss
            
            loss1 = ClassCriterion(class_output[0:mini_batch_size], train_classes1.narrow(0, b, mini_batch_size)) # batch size x c
            loss2 = ClassCriterion(class_output[mini_batch_size:2*mini_batch_size], train_classes2.narrow(0, b, mini_batch_size))
            
            ClassLoss = loss1 + loss2
            
            ## COMPARISON
            comp_output = CompModel(class_output_reshaped.detach()) # Comparison output

            #Comparison loss
            CompLoss = CompCriterion(comp_output, train_target.narrow(0, b, mini_batch_size)) #Nx2

            
            accClassLoss = accClassLoss + ClassLoss.item()
            accCompLoss = accCompLoss + CompLoss.item()

            CompModel.zero_grad()
            ClassModel.zero_grad()

            ClassLoss.backward() 
            CompLoss.backward()

            CompOptimizer.step()
            ClassOptimizer.step()

    return accClassLoss + accCompLoss