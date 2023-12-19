import torch
import pandas as pd
import sys
import numpy as np
import re
import evaluate
import copy
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

# Hyperparameter
TRAIN_ON_WHOLE_DATASET = True
SAMPLE_NO = 20
BALANCE_DATA = True
CHECK_FOR_PHASE = 'Passed_III'
EPOCHS = 10
SEED = 42
OUTPUT_DIR = "NSF_eligibilities_Bio"
#FILE_NAME = "datasets/NSF_allIndications_MM_11_october_2021.csv"
FILE_NAME = 'data/NSF_eligibilities.csv'
LOAD_ADDITIONAL_DATA = True
ADDITIONAL_FILES = ["data/brief_dataset.csv"]
EXTRACT_COLUMNS = [['nct_id', 'brief']]
CHOOSE_CUSTOM_SUBSET = True
FEATURE_SUBSET = ['DrugIndicationID', 'Disease_Group', 'Indication', 'Drug_Class', 'Pivotal', 'Primary_Endpoint_Count', 'Secondary_Endpoint_Count', 
                  'allocation', 'masking', 'investigator_masked', 'has_dmc', 'number_of_arms', 'enrollment', 'has_expanded_access', 'NumberCountries', 
                  'Lead_Agency', 'Trial_start_date', 'healthy_volunteers']

### FUNCTIONS ###
def Count(df, column_name, value):
    result = (df[column_name] == value).sum()
    print(f"Number of rows in df where {column_name} is {value}: {result}")

def BalanceData(df):
    Count(df, CHECK_FOR_PHASE, 'Yes')
    Count(df, CHECK_FOR_PHASE, 'No')
    pos_val = df.loc[df[CHECK_FOR_PHASE] == 'Yes']
    neg_val = df.loc[df[CHECK_FOR_PHASE] == 'No']

    # Find abundant and deficient set
    if len(pos_val) > len(neg_val):
        abundant_set = pos_val
        deficient_set = neg_val
        state_1 = "Passed"
        state_2 = "Failed"
    elif len(pos_val) < len(neg_val):
        abundant_set = neg_val
        deficient_set = pos_val
        state_1 = "Failed"
        state_2 = "Passed"
    else:
        print(f"Data is already balanced.")
        return df

    sampled_abundant_set = abundant_set.sample(n=len(deficient_set), random_state=42)
    df_balanced = pd.concat([sampled_abundant_set, deficient_set])
    df_balanced = df_balanced.sort_index()
    print(f"Balanced dataset to:\n - {state_1} = {len(sampled_abundant_set)}\n - {state_2} = {len(deficient_set)}")
    return df_balanced 

def LoadAdditionalData(df, file_name, columns, merge_column):
    print(columns)
    df_additional = pd.read_csv(file_name)
    print(df_additional)
    df_merged = pd.merge(df, df_additional[columns], on=merge_column, how='inner')
    df_merged = df_merged.drop_duplicates()
    return df_merged

### CODE ###
# Load dataset
df = pd.read_csv(FILE_NAME)
df = df.rename(columns={'NCTID' : 'nct_id'})

if LOAD_ADDITIONAL_DATA:
    for index in range(len(ADDITIONAL_FILES)):
        print(EXTRACT_COLUMNS[index])
        print(EXTRACT_COLUMNS[index][0])
        df = LoadAdditionalData(df, ADDITIONAL_FILES[index], EXTRACT_COLUMNS[index], EXTRACT_COLUMNS[index][0])

# Load whole dataset or samples
if TRAIN_ON_WHOLE_DATASET:
    df_samples = df
else:
    df_samples = df.sample(n=SAMPLE_NO, random_state=SEED)

print(df_samples)

# balance the dataset
if BALANCE_DATA:
    df_samples = BalanceData(df_samples)
    
phases = ['Passed_I','Passed_II','Passed_III']
if CHECK_FOR_PHASE in phases:
    phases.remove(CHECK_FOR_PHASE)

# Drop unnecessary columns and print column names
df_samples_cleaned = df_samples.drop([
    #'DrugIndicationID',
    #'Pivotal',
    #'Primary_Endpoint_Count',
    #'Secondary_Endpoint_Count',
    #'LeadIndication',
    #'nct_id',
    #'inclusion_count', 
    #'exclusion_count',
    #'sampling_method', 
    #'population', 
    #'minimum_age', 
    #'maximum_age',
    #'healthy_volunteers',
    'Trial_Phase',
    'Unnamed: 0.1',
    'Unnamed: 0',
    'Phase_III_complete_date_drug',
    'Phase_III_reach_date_drug',
    'Phase_II_reach_date_drug',
    'Phase_III_complete_date_drugIndication',
    'Phase_III_reach_date_drugIndication',
    'Phase_II_reach_date_drugIndication',
    'Phase_III_date_flag_drug',
    'Phase_III_date_flag_drugIndication'], axis=1)
df_samples_cleaned = df_samples_cleaned.drop([phases[0], phases[1]], axis=1)
df_samples_cleaned = df_samples_cleaned.rename(columns={CHECK_FOR_PHASE : 'label'})
df_samples_cleaned['label'] = df_samples_cleaned['label'].replace({'Yes': 1, 'No': 0})

column_names = df_samples_cleaned.columns.tolist()
column_names.remove('label')

if CHOOSE_CUSTOM_SUBSET:
    column_names = FEATURE_SUBSET

print(f'Using features:\n{column_names}\n')

text = []
labels = []

# Create text datasets
for index, row in df_samples_cleaned.iterrows():
    combined_text = ""

    # Automatic text creation from all columns
    for column in column_names:
        combined_text += f" {column} = {row[column]},"
    combined_text = combined_text.rstrip(",")  # Remove the trailing comma

    # Replace multiple spaces with a single space
    combined_text = re.sub(' +', ' ', combined_text)
    text.append(combined_text)
    labels.append(row['label'])

text_dataset = pd.DataFrame({'text' : text, 'label' : labels})
text.clear()
labels.clear()

# Check dataset
pd.set_option('display.max_colwidth',6000)

print("Data Example:")
print(text_dataset.iloc[0])

# Evluation of Feature Importance
acc_results = []
f1_results = []
recall_results = []
precision_results = []
#roc_auc_results = []

accuracy = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
recall_metric = evaluate.load('recall')
precision_metric = evaluate.load("precision")
#roc_auc_score = evaluate.load("roc_auc")

# Function that passes your predictions and labels to compute to calculate the accuracy
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1  = f1_metric.compute(predictions=predictions, references=labels)
    recall = recall_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels) 
  
    acc_results.append(acc)
    f1_results.append(f1)
    recall_results.append(recall)
    precision_results.append(precision)
    return f1

# Create map of expected labels
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# Initilalize the model
model = AutoModelForSequenceClassification.from_pretrained(
"emilyalsentzer/Bio_ClinicalBERT", num_labels=2, id2label=id2label, label2id=label2id
)

# Set training parameters
training_args = TrainingArguments(
  output_dir=OUTPUT_DIR,
  learning_rate=2e-5,
  per_device_train_batch_size=16,
  per_device_eval_batch_size=16,
  num_train_epochs=EPOCHS,
  weight_decay=0.01,
  evaluation_strategy="epoch",
  save_strategy="epoch",
  load_best_model_at_end=True,
  push_to_hub=False,
)

# Preprocess the data and train the model
ds_pandas = Dataset.from_pandas(text_dataset)
ds = ds_pandas.train_test_split(test_size=0.2, seed=SEED)
ds["test"][1]

# Clinical Trial
#tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

# Apply preprocessing function over the entire dataset
tokenized_ds = ds.map(preprocess_function, batched=True)

# Create a batch of examples
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Print model Evaluation
accuracies = [entry['accuracy'] for entry in acc_results]
recalls = [entry['recall'] for entry in recall_results]
precisions = [entry['precision'] for entry in precision_results]
f1s = [entry['f1'] for entry in f1_results]

data = {'precision' : precisions, 'recall' : recalls, 'f1' : f1s,'accuracy': accuracies}
table = pd.DataFrame(data)
print(table)


