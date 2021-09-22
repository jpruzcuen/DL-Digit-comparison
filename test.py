import torch
from torch import nn
from torch.nn import functional as F
import dlc_practical_prologue as prologue


# ==== Joint Network with Weight Sharing ==== #

class WSN(nn.Module):
    def __init__(self, batch_norm = False, dropout = False):
        super().__init__()

        # Digit Classification
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 49, kernel_size=2)
        self.fc1 = nn.Linear(196, 100)
        self.fc2 = nn.Linear(100, 10)

        # Digit comparison  
        self.fc3 = nn.Linear(20,1)

        # Extra features: batch normalization and dropout
        self.batch_norm = batch_norm
        self.batch_norm1 = nn.BatchNorm1d(100)
        self.batch_norm2 = nn.BatchNorm1d(10) 
         
        self.dropout = dropout
        self.drop = nn.Dropout(0.5)


    def forward(self, inp):

        N = inp.shape[0]
        out = torch.zeros(N,20)

        # Digit classification 
        for ii in range(inp.shape[1]): 
            x = inp[:,ii,:,:].unsqueeze_(1)  # Extract digit, size Nx1x14x14
            x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
            x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
            x = self.fc1(x.view(-1, 196))
            if self.dropout: x = self.drop(x)
            if self.batch_norm: x = self.batch_norm1(x)
            x = F.relu(x)                 
            x = self.fc2(x) #Nx10
            if self.dropout: x = self.drop(x)
            if self.batch_norm: x = self.batch_norm2(x)

            # out is Nx20, first 10 cols have digit 1 
            # and last 10 cols have digit 2
            out[:,torch.arange(10)+ii*10] = x    

        # Comparison 
        x = self.fc3(out.view(-1,20))
        if self.dropout: x = self.drop(x)
        x = F.relu(x) #Nx1
        x = torch.sigmoid(x) # Bound output between [0,1] for BCELoss
        
        return out,x


# ==== Joint Network without Weight Sharing ==== #

class WOSN(nn.Module):
    def __init__(self, batch_norm = False, dropout = False):
        super().__init__()

        # Digit Classification
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 49, kernel_size=2)
        self.fc1 = nn.Linear(196, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10,1)

        # Extra features: batch normalization and dropout
        self.batch_norm = batch_norm
        self.batch_norm1 = nn.BatchNorm1d(100)
        self.batch_norm2 = nn.BatchNorm1d(10) 
         
        self.dropout = dropout
        self.drop = nn.Dropout(0.2)


    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = self.fc1(x.view(-1, 196))
        if self.dropout: x = self.drop(x)
        if self.batch_norm: x = self.batch_norm1(x)
        x = F.relu(x)                 
        x = self.fc2(x) #Nx10
        if self.dropout: x = self.drop(x)
        if self.batch_norm: x = self.batch_norm2(x)

        # Comparison
        x = self.fc3(x.view(-1,10))
        if self.dropout: x = self.drop(x)
        x = F.relu(x) #Nx1
        x = torch.sigmoid(x) # Bound output between [0,1] for BCELoss
        
        return None,x

    
# ==== Train Joint Network ==== #

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
    

# ==== Calculate Error for Joint Net ==== # 

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
    
    

# ==== Split Network ==== #

class ClassifyNet(nn.Module):
    def __init__(self, dropout = False, batch_normalization = False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 49, kernel_size=2)

        self.fc1 = nn.Linear(196, 100)
        self.fc2 = nn.Linear(100, 10) #end of classification

        self.bn1 = nn.BatchNorm2d(32) # C from an expected input of size (N, C, H, W)
        self.bn2 = nn.BatchNorm2d(49)

        self.drop1 = nn.Dropout(0.5) # Have dropout ratio p = 0.5

        self.dropout = dropout 
        self.batch_normalization = batch_normalization


    def forward(self, input):

        N = input.shape[0]

        out = torch.zeros(2*N,10)
        reshaped_out = torch.zeros(N,20)

        for ii in range(input.shape[1]): # Loop over the channels
            x = input[:,ii,:,:].unsqueeze_(1) # Nx1x14x14
            x = self.conv1(x) 

            if self.batch_normalization: x = self.bn1(x)
            x = F.relu(F.max_pool2d(x, kernel_size=2))
            x = self.conv2(x)

            if self.batch_normalization: x = self.bn2(x)
            x = F.relu(F.max_pool2d(x, kernel_size=2))
            x = x.view(-1, 196) # Get to correct size

            if self.dropout: x = self.drop1(x)
            x = F.relu(self.fc1(x)) #Nx10
            
            if self.dropout: x = self.drop1(x)
            x = F.relu(self.fc2(x))

            out[N*ii:N*(ii+1),:] = x #output is 2Nx10
        
        reshaped_out[:, 0:10] = out[0:N,:]
        reshaped_out[:,10:20] = out[N:2*N,:] # Nx20
        
        return (out,reshaped_out)



# ==== Train Split Network ==== #

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
    

    
# ==== Calculate Error for Split Network ==== # 

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

    
    
##### ======================== MAIN ======================== #####
    
def main():
    
    # ==== Load the data ==== #
    print("Loading data..")
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

    

