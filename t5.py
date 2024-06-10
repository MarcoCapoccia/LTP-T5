import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
import json
from tqdm import tqdm

#dataset class for loading and preprocessing the data
class QADataset(Dataset):
    def __init__(self, source_file, target_file, tokenizer, max_length=128):
        self.sources = open(source_file, 'r').readlines()
        self.targets = open(target_file, 'r').readlines()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        source_text = self.sources[idx].strip()
        target_text = self.targets[idx].strip()

        source = self.tokenizer(source_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        target = self.tokenizer(target_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")

        return {
            'input_ids': source['input_ids'].squeeze(),
            'attention_mask': source['attention_mask'].squeeze(),
            'labels': target['input_ids'].squeeze(),
            'decoder_attention_mask': target['attention_mask'].squeeze()
        }

#tokenizer and model init
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

#set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#create datasets
train_dataset = QADataset('PATH TO src-train.txt', 'PATH TO tgt-train.txt', tokenizer)
val_dataset = QADataset('PATH TO src-dev.txt', 'PATH TO tgt-dev.txt', tokenizer)
test_dataset = QADataset('PATH TO src-test.txt', 'PATH TO tgt-test.txt', tokenizer)

#create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

#training loop
def train(model, train_loader, val_loader, epochs, learning_rate=1e-4):
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    
    for epoch in tqdm(range(epochs)):
        model.train()
        total_train_loss = 0

        #training step
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            decoder_attention_mask = batch['decoder_attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, decoder_attention_mask=decoder_attention_mask)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        
        #validation step
        with torch.no_grad():
            for batch in tqdm(val_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                decoder_attention_mask = batch['decoder_attention_mask'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, decoder_attention_mask=decoder_attention_mask)
                loss = outputs.loss

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    return train_losses, val_losses

#train the model
epochs = 10
train_losses, val_losses = train(model, train_loader, val_loader, epochs)

#save the model and tokenizer
model.save_pretrained('t5_question_generation_model')
tokenizer.save_pretrained('t5_question_generation_tokenizer')

#save the training and validation losses to a file
losses = {
    'train_losses': train_losses,
    'val_losses': val_losses
}

with open('training_losses.json', 'w') as f:
    json.dump(losses, f)

#evaluation on the test set
model.eval()
predictions = []
references = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128, num_beams=4)
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        predictions.extend(decoded_preds)
        references.extend(decoded_labels)

#format references for metrics
references = [[ref] for ref in references]

#save the predictions and references
results = {
    'predictions': predictions,
    'references': references
}

with open('predictions_and_references.json', 'w') as f:
    json.dump(results, f)
