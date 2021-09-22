import torch
from torch import nn
from torch.nn import functional as F
import dlc_practical_prologue as prologue
from train import *
from architectures import *
from errors import *

##### ======================== MAIN ======================== #####
    
def main():
    
    # ==== Load the data ==== #
    print("Loading data...")
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(1000)
    mean, std = train_input.mean(), train_input.std()
    train_input = train_input.sub(mean).div(std)
    
    
    # Declare all models
    model_wsn = WSN(batch_norm = True, dropout = False)
    model_wsn_aux = WSN(batch_norm = True, dropout = False)
    model_wosn = WOSN(batch_norm = True, dropout = False)
    model_wosn_aux = WOSN(batch_norm = True, dropout = False)
    
    # Split model with batchnorm and dropout
    classify_ftft = ClassifyNet(dropout = False, batch_normalization = True)
    compare_ftft = CompareNet(dropout = False, batch_normalization = True)
    
    classify_ttft = ClassifyNet(dropout = True, batch_normalization = True)
    compare_ttft = CompareNet(dropout = False, batch_normalization = True)
    
    classify_ftff = ClassifyNet(dropout = False, batch_normalization = True)
    compare_ftff = CompareNet(dropout = False, batch_normalization = False)
    
    
    # Train the networks
    print("Training joint model without weight sharing and without aux loss ... (1/7)")
    loss_model_wosn = train_model_joint(model_wosn, train_input, train_target, train_classes, aux_loss = False)
    print("Training joint model with weight sharing and without aux loss... (2/7)")
    loss_model_wsn = train_model_joint(model_wsn, train_input, train_target, train_classes, aux_loss = False)
    print("Training joint model without weight sharing and with aux loss ... (3/7)")
    loss_model_wosn_aux = train_model_joint(model_wosn_aux, train_input, train_target, train_classes, aux_loss = True)
    print("Training joint model with weight sharing and with aux loss... (4/7) ")
    loss_model_wsn_aux = train_model_joint(model_wsn_aux, train_input, train_target, train_classes, aux_loss = True)
    

    print("\nTraining split model classify(dropout = true, batchnorm = true), compare(dropout = false, batchnorm = true)... (5/7)")
    loss_split_ttft = train_model_split(classify_ttft, compare_ttft, train_input, train_target, train_classes[:,0], train_classes[:,1])
    print("Training split model classify(dropout = false, batchnorm = true), compare(dropout = false, batchnorm = false)... (6/7)")
    loss_split_ftff = train_model_split(classify_ftff, compare_ftff, train_input, train_target, train_classes[:,0], train_classes[:,1])
    print("Training split model classify(dropout = false, batchnorm = true), compare(dropout = false, batchnorm = true)... (7/7)")
    loss_split_ftft = train_model_split(classify_ftft, compare_ftft, train_input, train_target, train_classes[:,0], train_classes[:,1])


    
    
    # Compute all the errors
    print("Calculating errors..")
    _, error_wsn = compute_errors_joint(model_wsn, test_input, test_target, test_classes,mini_batch_size=100)
    _, error_wsn_aux = compute_errors_joint(model_wsn_aux, test_input, test_target, test_classes,mini_batch_size=100)
    error_wosn = compute_errors_joint(model_wosn, test_input, test_target, test_classes,mini_batch_size=100)
    error_wosn_aux = compute_errors_joint(model_wosn_aux, test_input, test_target, test_classes,mini_batch_size=100)
    
    temp, error_split_ftft = compute_errors_split(classify_ftft, compare_ftft, test_input, test_target, test_classes[:,0], test_classes[:,1])
    _, error_split_ttft = compute_errors_split(classify_ttft, compare_ttft, test_input, test_target, test_classes[:,0], test_classes[:,1])
    _, error_split_ftff = compute_errors_split(classify_ftff, compare_ftff, test_input, test_target, test_classes[:,0], test_classes[:,1])
    
    
    res1 = "Joint Network without Weight Sharing & without Aux Loss: Training loss: {:.2f}, Test Error {:.2f} %".format(loss_model_wosn,error_wosn)
    res2 = "Joint Network with Weight Sharing & without Aux Loss: Training loss: {:.2f}, Test Error {:.2f} %".format(loss_model_wsn,error_wsn)
    res3 = "Joint Network without Weight Sharing & with Aux Loss: Training loss: {:.2f}, Test Error {:.2f} %".format(loss_model_wosn_aux,error_wosn_aux)
    res4 = "Joint Network without Weight Sharing & with Aux Loss: Training loss: {:.2f}, Test Error {:.2f} %".format(loss_model_wsn_aux,error_wsn_aux)

    res5 = "Split Network[TT][FT]: Training loss: {:.2f}, Test Error {:.2f} %".format(loss_split_ttft, error_split_ttft)
    res6 = "Split Network[FT][FF]: Training loss: {:.2f}, Test Error {:.2f} %".format(loss_split_ftff, error_split_ftff)
    res7 = "Split Network[FT][FT]: Training loss: {:.2f}, Test Error {:.2f} %".format(loss_split_ftft, error_split_ftft)

    
    # Print Results
    print("\n\n========= Final Results =========")
    print(res1)
    print(res2)
    print(res3)
    print(res4)
    print("\nThe networks below are in format: CompareNet = [Dropout, Batchnorm], ClassifyNet = [Dropout, Batchnorm], where [T/F = True/False]")
    print(res5)
    print(res6)
    print(res7)
        

if __name__ == "__main__":
    main()

    

