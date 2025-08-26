import torch

def calc_batch_loss(input_batch, target_batch, model, device):
    """
    Calculates how the model performs on a specific batch 
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

def calculate_loss(data_loader, model, device, num_batches=None):
    """
    Calculates how the model performs on average on a set of batches
    """
    total_loss=0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches =  min(len(data_loader), num_batches)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_batch_loss(input_batch=input_batch, target_batch=target_batch, model=model, device=device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches # average loss per batch
        
    

def evaluate_model(model, train_dataloader, test_dataloader, device, eval_iter):
    """
    Calculates how the model has performed so far 
    """
    model.eval()
    with torch.no_grad():
        train_loss = calculate_loss(train_dataloader, model, device, eval_iter)
        test_loss = calculate_loss(test_dataloader, model, device, eval_iter)

    model.train()
    return train_loss, test_loss

def train_model(model, train_dataloader, test_dataloader, optimizer, device, 
                num_epochs, eval_freq, eval_iter):
    train_losses, test_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_dataloader:
            optimizer.zero_grad()  # Resets loss gradients from previous batch
            loss = calc_batch_loss(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            # Evaluation logic
            if global_step % eval_freq == 0:
                train_loss, test_loss = evaluate_model(model, train_dataloader, test_dataloader, device, eval_iter)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "f"Train loss {train_loss:.3f}, "f"Val loss {test_loss:.3f}")
                
    return train_losses, test_losses, track_tokens_seen