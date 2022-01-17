import seaborn as sns
import torch
import numpy as np
from torchvision.datasets import ImageFolder
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm


def train_network(encoder, decoder, criterion, optimizer,n_epochs, 
                  dataloader_train, lr_scheduler=None, epsilon=.001,
                  writer=None, save_checkpoint_at=None, debug_run=False):
    """
    Trains a neural Network with given inputs and parameters.
    
    params:
    --------------------
    encoder: torch.Network
        Image Encoder Network to extract image features from images.
        
    decoder: torch.Network
        Decoder Network for Generating Image Captions
        
    criterion: 
        Cost-Function used for the network optimizatio
        
    optimizer: torch.Optimizer
        Optmizer for the network
        
    n_epochs: int
        Defines how many times the whole dateset should be fed through the network
        
    dataloader_train: torch.Dataloader 
        Dataloader with the batched dataset
        
    dataloader_val: torch.Dataloader
        Dataloader with validation set to calculate validation loss
        if None: No validation Loop will be performed during training
        
    lr_scheduler: float
        Learning Rate Scheduler, to adapt LR during training
        Will be multiplied at n-steps with a given Gamma Hyperparam
    
    writer:
        Tensorboard writer from from torchsummary SummaryWriter Instance
        
    epsilon: float
        Stopping Criterion regarding to change in cost-function between two epochs.
        
    save_checkpoint_at: str
        If this parameter is not None it will save the model at epoch.

    debug_run:
        If true than only one batch will be put through network.
        
    returns:
    ---------------------
    encoder:
        Trained Torch Encoder Model

    decoder:
        Trained Torch Decoder Model
        
    losses: dict
        dictionary of losses of all batches and Epochs.
        
    """
    print(20*'=', 'Start Training', 20*'=')
    if save_checkpoint_at:
        try:
            os.mkdir(save_checkpoint_at)
        except FileExistsError as ae:
            print(ae)
    batch_losses, epoch_losses = [], []
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    encoder.to(dev), decoder.to(dev)
    criterion.to(dev)

    encoder.train(), decoder.train()
    overall_length = len(dataloader_train)
    try: # To interrupt training whenever necessary wihtout loosing progress
        with tqdm(total=n_epochs*overall_length, disable=debug_run) as pbar:
            for epoch in range(n_epochs):  # loop over the dataset multiple times
                running_loss = 0.0
                for i, data in enumerate(dataloader_train):
                    # get the inputs
                    images, captions, lengths = data
                    images, captions = images.to(dev), captions.to(dev)
                    # zero the parameter gradients
                    encoder.zero_grad(), decoder.zero_grad()
                    
                    # forward + backward + optimize
                    features = encoder(images)
                    out = decoder(features, captions, lengths)
                    #print('Predicted for first batch', out[:len(captions[0])].argmax(1))
                    #print('True for first batch', captions[0])
                    
                    loss = criterion(out, captions.reshape(-1)) # Targets can be labels as documented for CE-Loss
                    loss.backward()
                    optimizer.step()

                    # calc and print stats
                    batch_losses.append(loss.item())
                    if writer:
                        writer.add_scalar('Loss/batch', loss.item(), i)
                    running_loss += loss.item()                
                    pbar.set_description(f'Epoch: {epoch+1}/{n_epochs} // Running Loss: {np.round(running_loss, 3)} ')
                    pbar.update(1)
                    if debug_run:
                        print('- Training Iteration passed. -')
                        break
                    
                if debug_run:
                    # Breaks loop 
                    print('Finished Debug Run')
                    break
                
                if save_checkpoint_at and not debug_run:
                    torch.save(encoder.state_dict(), save_checkpoint_at + f'checkpoint_encoder_ep{epoch}.ckpt')
                    torch.save(decoder.state_dict(), save_checkpoint_at + f'checkpoint_decoder_ep{epoch}.ckpt')

                if lr_scheduler:
                    lr_scheduler.step()

                print(f'Epoch {epoch+1} // Train Loss: {round(running_loss, 2)}')
                epoch_losses.append(running_loss)
                if writer:
                    writer.add_scalar('Loss/epoch', running_loss, epoch)
                if epoch > 0:
                    diff = np.abs(epoch_losses[-2] - running_loss)
                    if diff < epsilon:
                        print('- Network Converged. Stopping Training. -')
                        break
                        
    except KeyboardInterrupt as ke:
        # Handles Keyboardinterrupt and returns trained network and results
        print(ke)
        
    print(20*'=', 'Finished Training', 20*'=')
                                                 
    return encoder, decoder, dict(batch=batch_losses, 
                                    epoch=epoch_losses)



def collate_fn(data):
    """Function to preprocess Batches after loading it with the Dataloader"""
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    images = torch.stack(images, 0)

    # Extract length per caption
    lengths = [len(caption) for caption in captions]
    # Init zero tensor with amx length for each target 
    targets = torch.zeros(len(captions), max(lengths)).long() + 3
    for i, caption in enumerate(captions):
        end = lengths[i]
        # Replace part with target values and fill remaining with zeros
        targets[i, :end] = caption[:end]  
        
    return images, targets, lengths