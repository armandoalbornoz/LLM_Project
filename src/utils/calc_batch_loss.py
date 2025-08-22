import torch 

def calc_batch_loss(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss


def calculate_loss(data_loader, model, device):
    total_loss=0
    if len(data_loader) == 0:
        return float("nan")
    
    num_batches =  len(data_loader)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_batch_loss(input_batch=input_batch, target_batch=target_batch, model=model, device=device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches # average loss per batch
        
    