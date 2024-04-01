import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForMultipleChoice, BertTokenizer, AdamW  # Changed from XLMRoberta to Bert
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Set the computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for computation.")

class MultipleChoiceDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        choices_inputs = self.tokenizer(
            text=[record['startphrase']] * 2,
            text_pair=[record['ending1'], record['ending2']],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = choices_inputs['input_ids'].squeeze(0)
        attention_mask = choices_inputs['attention_mask'].squeeze(0)
        labels = torch.tensor(record['labels'], dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def evaluate(model, dataloader):
    model.eval()
    predictions = []
    true_labels = []
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits
        batch_predictions = torch.argmax(logits, dim=1).tolist()
        predictions.extend(batch_predictions)
        true_labels.extend(batch['labels'].tolist())
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    return predictions, metrics

def main(df_train, df_test, model_name, lang, k, num_epochs=30, save_model_flag=True):
    model_path = "base.pkl"
    tokenizer = BertTokenizer.from_pretrained(model_name)  # Changed from XLMRobertaTokenizer

    # Check if the model file exists
    if os.path.exists(model_path):
        print("Loading model from saved file.")
        model = torch.load(model_path)
        model.to(device)
    else:
        print("Model file not found. Training new model.")
        model = BertForMultipleChoice.from_pretrained(model_name)  # Changed from XLMRobertaForMultipleChoice
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

        train_dataset = MultipleChoiceDataset(df_train, tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f"Training loss: {total_loss / len(train_dataloader)}")

        if save_model_flag:
            torch.save(model, model_path)
            print(f"Model saved to {model_path}")

    # Evaluation on the test set
    test_dataset = MultipleChoiceDataset(df_test, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=64)
    test_predictions, metrics = evaluate(model, test_dataloader)

    # Save metrics to file
    os.makedirs("outputs", exist_ok=True)
    metrics_path = f"outputs/mbert_metrics_{lang}_{k}.txt"
    with open(metrics_path, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key.capitalize()}: {value}\n")
    print(f"Metrics saved to {metrics_path}")

    # Append predictions to test dataframe and save
    df_test['predictions'] = test_predictions
    df_test.to_csv(f"outputs/mbert_with_predictions_{lang}_{k}.txt", index=False)
    print(f"Test dataframe with predictions saved to outputs/mbert_with_predictions_{lang}_{k}.txt")

if __name__ == "__main__":
    # Placeholder paths for your dataset files
    langs_list = [ "hi", "id", "jv", "kn", "su", "sw", "yo"]
    k_val = [2, 4, 6, 8, 10]
    for lang in langs_list:
      for i in k_val:
         df_train = pd.read_csv("test_data/en_train.csv") 
         path = "few_shot_train/" + lang + "/" + lang + "_" + str(i) + ".csv"
         df_test = pd.read_csv(path)
         # df_test = pd.read_csv("few_shot_train/kn/kn_2.csv")
         lang = lang
         k = str(i)
         model_name = "bert-base-multilingual-cased"  # Changed to m-BERT
         main(df_train, df_test, model_name, lang, k)
