import json
import matplotlib.pyplot as plt
from datasets import load_metric

#load training and validation losses
with open('PATH TO training_losses.json', 'r') as f:
    losses = json.load(f)

train_losses = losses['train_losses']
val_losses = losses['val_losses']

#plot the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid(True)
plt.show()

#load predictions and references
with open('PATH TO predictions_and_references.json', 'r') as f:
    results = json.load(f)

predictions = results['predictions']
references = results['references']

#ensure predictions and references are strings
if isinstance(predictions[0], list):
    predictions = [' '.join(pred) for pred in predictions]
if isinstance(references[0], list):
    references = [' '.join(ref) for ref in references]

#initialize the metrics
rouge = load_metric('rouge')
meteor = load_metric('meteor')
bleu = load_metric('bleu')

#compute the metrics
rouge_result = rouge.compute(predictions=predictions, references=references, rouge_types=['rougeL'])
meteor_result = meteor.compute(predictions=predictions, references=references)
bleu_result = bleu.compute(predictions=[pred.split() for pred in predictions], references=[[ref.split()] for ref in references])

#print results
print("ROUGE-L:", rouge_result['rougeL'].mid)
print("METEOR:", meteor_result['meteor'])
print("BLEU-1:", bleu_result['precisions'][0])
print("BLEU-2:", bleu_result['precisions'][1])
print("BLEU-3:", bleu_result['precisions'][2])
print("BLEU-4:", bleu_result['precisions'][3])
