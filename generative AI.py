import os
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from datasets import Dataset

# File path
file_path = 'C:\\Users\\vchan\\Desktop\\cf\\CF_PP_2025.csv'

chunk_size = 10000

# Initialize dataframes list
dataframes = []

try:
    # Read the CSV file in smaller chunks
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
        dataframes.append(chunk)

    # Concatenate the dataframes
    df = pd.concat(dataframes, ignore_index=True)

except FileNotFoundError:
    print(f"File not found: {file_path}")
    df = pd.DataFrame()  # Create an empty dataframe to avoid issues
    # You can choose to continue or exit gracefully without terminating the script.

# Check if the dataframe is empty
if df.empty:
    print("No data loaded from the CSV file.")
    # Handle the case when no data is loaded, for example, exit gracefully or take other actions.

# Continue with the rest of the code if data is loaded

# Check dataset size
dataset_size = len(df)
print(f"Dataset size: {dataset_size}")

# Check for missing values
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print("Missing values in the dataset:")
    print(missing_values)


# Define columns with integer values
integer_columns = ['ELECTION', 'FILING', 'DATE', 'ZIP', 'AMNT', 'PREVAMNT', 'PAY_METHOD']

# Convert specified integer columns to text
df[integer_columns] = df[integer_columns].astype(str)

# Convert all values in the specified columns to strings
text_columns = ['RECIPNAME', 'NAME', 'CITY', 'STATE', 'OCCUPATION', 'EMPNAME', 'EMPCITY'] + integer_columns
df[text_columns] = df[text_columns].astype(str)

# Concatenate text columns including integer columns into a single text column
df['text'] = df[text_columns].agg(' '.join, axis=1)

# Initialize GPT-2 tokenizer and model
model_name = 'gpt2-medium'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the padding token to the eos_token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize your dataset
tokenized_data = tokenizer.batch_encode_plus(
    df['text'].tolist(),
    add_special_tokens=True,
    max_length=280,
    padding='max_length',
    truncation=True,
    return_tensors='pt',
)

# Create a dataset from the tokenized data
dataset = Dataset.from_dict(tokenized_data)

# Initialize a data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./fine_tuned_gpt2_model',
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize a Trainer instance for training
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Update the optimizer to use torch.optim.AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

num_warmup_steps = 5000

scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / num_warmup_steps))

# Fine-tuning loop
num_epochs = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.train()

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in dataloader:
        input_ids = torch.stack(batch['input_ids']).to(device)
        attention_mask = torch.stack(batch['attention_mask']).to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Save the fine-tuned model and tokenizer
save_directory = "fine_tuned_campaign_finance_model"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

try:
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print("Fine-tuned model and tokenizer saved successfully.")
except Exception as e:
    print(f"Error saving the model: {e}")

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained(save_directory)
tokenizer = GPT2Tokenizer.from_pretrained(save_directory)

model.eval()

# User input
input_prompt = "Find the average contribution made to the political parties in the provided dataset?"
input_ids = tokenizer.encode(input_prompt, return_tensors="pt", padding=True, truncation=True)

generated_output = model.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)

generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
print(generated_text)
