from datetime import timedelta
from utils import accuracy_score
import torch
import time


def train(model, config, train_iter):
    epoch_start_time = time.time()
    model.to(config.device)
    total_batch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learn_rate)
    loss = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(config.num_epochs):
        for i, data in enumerate(train_iter):
            x, y = data[0].to(config.device), data[1].to(config.device)
            y_pred = model(x)
            batch_loss = loss(y_pred, y)
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                train_acc = accuracy_score(y, y_pred)
                train_loss = batch_loss.item()
                msg = 'Time: {},  Train Loss: {},  Train Acc: {}'
                time_diff = timedelta(seconds=int(round(time.time()-epoch_start_time)))
                print(msg.format(time_diff, train_loss, train_acc))
            total_batch += 1

    model.eval()
    if config.target_type == 'multi':
        torch.save(model.state_dict(), './saved_dict/model_xjl_multi.pkl')
    else:
        torch.save(model.state_dict(), './saved_dict/model_binary_multi.pkl')