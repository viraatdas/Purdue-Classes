from mycnnlib import *

data_loaders, dataset_sizes, class_names = load_data(data_dir='./dogs', batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Gamma is the pre-trained model that we will make into a feature extractor
#  This will download the pre-trained model and assign it to Gamma

Gamma = models.resnet18(pretrained=True) # ResNet-18 pre-trained over ImageNet 1000 
                                         # => 1000 different classes

# Replace last feedforward layer with an indentity matrix
#  The last feedforward of this model is "fc"
out_size = Gamma.fc.in_features
Gamma.fc = nn.Identity(out_size, out_size)
Gamma = Gamma.to(device)
# The above just copies the input of the last layer to the output,
#    this way we don't need to rewrite the "forward()" function of the model

# Create a single feedforward layer with for classification
#  - Input: out_size neurons
#  - Output: number of classes
g = nn.Linear(out_size, 120, bias=True).to(device)

# Negative log-likelihood loss. It will perform the softmax automatically
criterion = nn.CrossEntropyLoss()

# Observe that all parameters in g and Gamma are optimized
#  We want to optimize only g, not Gamma
params = list(g.parameters())
optimizer_ft = optim.SGD(params, lr=0.001)


import threading
import queue
# Producer function that places data on the Queue and pushes it to the GPU memory asynchronously 

def producer(id,device,my_queue,loader):
    # Place our data on the Queue
    for batchIdx, batchTrainData in enumerate(loader):
        # non_blocking=True makes the operation asynchronous in Pytorch 
        # Any computation using batchTrainData[0], batchTrainData[1]
        batchTrainData[0], batchTrainData[1] = batchTrainData[0].to(device, non_blocking=True), batchTrainData[1].to(device, non_blocking=True)
        # Here, the producer queues the data for the consumer
        #   Note that queue is blocking when full
        my_queue.put((batchIdx, batchTrainData[0], batchTrainData[1]))

   # (MISSING STOP CRITERIA. HOW WILL CONSUMER KNOW WHEN TO STOP?)
    my_queue.put((batchIdx, None, None))


from torch.cuda.amp import GradScaler 
from torch.cuda.amp import autocast 


# Function that trains the model
def train_model(Gamma, g, criterion, optimizer, num_epochs=25):
    since = time.time()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training phase
        ## (CHANGE HERE)
        Gamma.eval() # Set Gamma to test mode
        g.train()  # Set g to training mode

        running_loss = 0.0
        running_corrects = 0
        total_examples = 0

        # Start producer at each epoch
        my_queue = queue.Queue(10) # Careful with GPU memory. This Queue(10) requires 10x more GPU memory than the traditional approach
        dataproducer = threading.Thread(target=producer,args=(0,device,my_queue,data_loaders['train']))
        dataproducer.start()


        # Decide if we are using mixed precision
        mixed_precision = True

        if mixed_precision:
            # Creates a GradScaler once at the beginning of training.
            # - init_scale defined the maximum representable number
            # - after growth_interval, multiply scaling factor by growth_factor
            # - If got overflow, divide scalling factor by backoff_factor
            scaler = GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5,
            growth_interval=100, enabled=True)

        if mixed_precision:
            print("Using GPU mixed precision arithmetic: FP16 & FP32")
        else:
            print("Using FP32 arithmetic")



        # Iterate over data as a consumer.
        while True:

            batchIdx, inputs, labels = my_queue.get()

            if inputs == None:
                break

            # zero all gradients
            optimizer.zero_grad()

            if mixed_precision:
                # Runs the forward pass with autocasting.
                with autocast():
                    #  Model output 
                    # (CHANGE HERE TO MAKE SURE GAMMA IS NOT OPTIMIZED WITH g)
                    outputs = g(Gamma(inputs).detach()) 

                    loss = criterion(outputs, labels) # loss negative log-likelihood loss using the output "probabilities" 
                                                # actually, the CrossEntropyLoss() does
                                                # the softmax for us
            else:
                #  Model output
                # (CHANGE HERE)
                outputs = g(Gamma(inputs).detach()) 

                loss = criterion(outputs, labels) # loss negative log-likelihood loss using the output "probabilities" 
                                            # actually, the CrossEntropyLoss() does
                                            # the softmax for us

            # Predictions
            _, preds = torch.max(outputs, 1) # predicted label (we will use later)


            # Backpropagation and optimize
            if mixed_precision:
                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if mixed_precision:
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)
            else:
                optimizer.step()

            ## Missing step in the mixed-precision optimization
            ##  Which step is missing?
            if mixed_precision:
                # Updates the scale for next iteration.
                scaler.update()
            
            else:
                # Does nothing... FP32 optimization does generally need to scale gradients
                pass 

            # compute statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_examples += inputs.size(0)

            if batchIdx % 100 == 99:
                batch_loss = running_loss / total_examples
                batch_acc = running_corrects.double() / total_examples
                running_loss = 0.0
                running_corrects = 0
                total_examples = 0

                print(f'Batch indices [{batchIdx-99:3}-{batchIdx:3}], Avg Loss: {batch_loss:.4f} Avg Acc: {batch_acc:.4f}')

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    return g


if __name__ == '__main__':
    model_ft = train_model(Gamma=Gamma, g=g, criterion=criterion, optimizer=optimizer_ft, num_epochs=10)

