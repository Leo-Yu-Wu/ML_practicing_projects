def train_model(model, device, train_loader, optimizer, criterion, num_epoch=2000):
    model.to(device)
    model.train()
    for epoch in range(num_epoch):
        running_loss = 0.0
        for input, label in train_loader:
            input = input.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            input = input.view(input.size(0), -1)
            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"epoch {epoch+1}/{num_epoch}, loss: {running_loss / len(train_loader)}")


