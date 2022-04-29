import numpy as np
import torch
from tqdm.auto import tqdm

class NN():
    def __init__(self, model, criterion, optimizer, scheduler=None, score_func=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.score_func = score_func

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(self.device)

    def fit(self, loaders, num_epochs, output_dir=None):

        torch.manual_seed(0)
        np.random.seed(0)
        best_loss = np.inf

        for epoch in range(1, num_epochs + 1):

                running_train_loss = 0
                running_val_loss = 0
                running_score = 0

                with tqdm(loaders['train'], unit='batch') as tepoch:
                    for i, (batch, label) in enumerate(tepoch):
                        tepoch.set_description(f'Epoch {epoch} training')
                        inputs = batch.to(self.device)
                        labels = label.unsqueeze(1).to(self.device)

                        self.optimizer.zero_grad()

                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        loss.backward()
                        self.optimizer.step()

                        batch_loss = loss.item()
                        running_train_loss += batch_loss
                        avg_train_loss = running_train_loss / (i + 1)

                        tepoch.set_postfix(loss=avg_train_loss)

                with torch.no_grad():
                    with tqdm(loaders['valid'], unit='batch') as tepoch:
                        for i, (batch, label) in enumerate(tepoch):
                            tepoch.set_description(f'Epoch {epoch} validation')
                            inputs = batch.to(self.device)
                            labels = label.unsqueeze(1).to(self.device)

                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, labels)

                            batch_loss = loss.item()
                            running_val_loss += batch_loss
                            avg_val_loss = running_val_loss / (i + 1)
                            
                            if self.score_func:
                                running_score += self.score_func(labels.to('cpu'), outputs.to('cpu'))
                                avg_score = running_score / (i + 1)
                                tepoch.set_postfix(loss=avg_val_loss, auc=avg_score)
                            else:
                                tepoch.set_postfix(loss=avg_val_loss)

                if self.scheduler:
                    self.scheduler.step(avg_val_loss)

                if output_dir:
                    if avg_val_loss < best_loss:
                        best_loss = avg_val_loss
                        torch.save(self.model.state_dict(), output_dir + '/best.pt')
                        best_epoch = epoch

        if output_dir:
            print(f'Epoch {best_epoch} model with val loss {best_loss} saved at {output_dir}/best.pt')

    def predict(self, loaders):
        
        self.model.eval()
        predictions = torch.Tensor()
        
        with torch.no_grad():
            with tqdm(loaders['test'], unit='batch') as tpass:
                for batch in tpass:
                    tpass.set_description('Inference')
                    inputs = batch.to(self.device)
                    outputs = self.model(inputs)
                    
                    predictions = torch.cat((predictions, outputs.to('cpu')), dim=0)
                    
        return predictions.squeeze(1)