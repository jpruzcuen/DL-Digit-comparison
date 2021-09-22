import torch
from torch import nn

# ==== Calculate Error for Joint Net ==== # 

'''
Parameters
    -------
    model: neural network (WSN or WOSN)
    test_input: testing data
    test_target: labels of digit comparison for test data 
    test_classes: hot labels of digit classification for test data 
    mini_batch_size: size of mini batches

Returns
-------
    percentage of error during testing

'''

def compute_errors_joint(model, test_input, test_target, test_classes,mini_batch_size=100):
    errors1 = 0
    errors2 = 0
    comp_errors = 0
    
    for b in range(0, test_input.size(0), mini_batch_size):
        
        digits,compare = model(test_input.narrow(0, b, mini_batch_size)) 
        for n in range(mini_batch_size):
            if digits != None:
                # Check if digits are correct
                if(torch.argmax(digits[n,0:10]) != test_classes[b+n,0]):
                    errors1 += 1
                if(torch.argmax(digits[n,10:20]) != test_classes[b+n,1]):
                    errors2 += 1

                class_errors = errors1+errors2

            # Check if compare is correct
            if compare[n] > 0.5 :
                compare[n] = 1
            else:
                compare[n] = 0

            if compare[n] != test_target[b+n]:
                comp_errors += 1

    if digits != None:
        return class_errors/(test_input.size(0)*2)*100, comp_errors/(test_input.size(0))*100
    else:
        return comp_errors/(test_input.size(0))*100
    
    
# ==== Calculate Error for Split Network ==== # 

'''
Parameters
    -------
    ClassModel: network for classification (ClassifyNet)
    CompModel:  network for comparison (CompareNet)
    test_input: testing data
    test_target: labels of digit comparison for test data 
    test_classes1: hot labels of digit classification for digit1 
    test_classes2: hot labels of digit classification for digit2
    mini_batch_size: size of mini batches

Returns
-------
    
    error percentage from classification stage , error percentage from comparison stage

'''

def compute_errors_split(ClassModel, CompModel, test_input, test_target, test_classes1, test_classes2, mini_batch_size=100):
    
    errors1 = 0
    errors2 = 0
    comp_errors = 0
    
    for b in range(0, test_input.size(0), mini_batch_size):
        
        
        class_output, class_output_reshaped = ClassModel(test_input.narrow(0, b, mini_batch_size)) #2Nx10
        comp_output = CompModel(class_output_reshaped)
        
        for sample in range(mini_batch_size):
            
            # Check digit1 
            if(torch.argmax(class_output[sample]) != test_classes1[b+sample]):
                errors1 += 1
                
            # Check digi2
            if(torch.argmax(class_output[mini_batch_size+sample]) != test_classes2[b+sample]):
                errors2 += 1

            # Check if comparison is correct
            if(torch.argmax(comp_output[sample]) != test_target[b+sample]):
                comp_errors += 1
                
        class_errors = errors1+errors2
    
    return class_errors/(test_input.size(0)*2)*100, comp_errors/(test_input.size(0))*100
