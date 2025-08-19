


#------------------------------------------------------------------------------------#
          ## Prediction function for protein sequence location in cell ##   
#------------------------------------------------------------------------------------#


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# List your types and corresponding model directories - Only 2 trained models for now
TYPES = [
    "extracellular",
    "reticulum",
    "golgi",
    "membrane", 
    "mitochondria",
    "nuclear",
    "peroxisome",
    "ribosome",
]

MODEL_DIR_TEMPLATE = ('models/model_{}')

# Load models and tokenizers

models = {}
tokenizers = {}


for t in TYPES:
    model_dir = MODEL_DIR_TEMPLATE.format(t)
    models[t] = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizers[t] = AutoTokenizer.from_pretrained(model_dir)

def predict_localization(sequence):
    sequence = sequence.upper()

    # Check input size 
    
    if len(sequence) >= 64 and len(sequence) <= 1218:

        scores = {}
        for t in TYPES:
            tokenizer = tokenizers[t]
            model = models[t]
            inputs = tokenizer(sequence, return_tensors='pt', padding=True, truncation=True)
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                prob = torch.softmax(logits, dim=1)[0, 1].item()  # probability for class 1
                scores[t] = prob

        # Get the type with highest probability and the highest probability the numeric value
        ## Maybe if more than one has a prob > 0.5 it needs to print both?

        predicted_prob = max(scores.values())
        predicted_type = max(scores, key=scores.get)

        
        if  predicted_prob <= 0.5:
            predicted_type = 'Not Matched'

        return predicted_type, scores
    
    elif len(sequence) < 64 or len(sequence) > 1218: 
        print('Invalid input: Sequence length is too short or too long')

    elif not sequence:
        print('Invalid input: Please provide a sequence')


# Example usage:

if __name__ == "__main__":
    test_seq = "MAAETLLSSLLGLLLLGLLLPASLTGGVGSLNLEELSEMRYGIEILPLPVMGGQSQSSDVVIVSSKYKQRYECRLPAGAIHFQREREEETPAYQGPGIPELLSPMRDAPCLLKTKDWWTYEFCYGRHIQQYHMEDSEIKGEVLYLGYYQSAFDWDDETAKASKQHRLKRYHSQTYGNGSKCDLNGRPREAEVRFLCDEGAGISGDYIDRVDEPLSCSYVLTIRTPRLCPHPLLRPPPSAAPQAILCHPSLQPEEYMAYVQRQADSKQYGDKIIEELQDLGPQVWSETKSGVAPQKMAGASPTKDDSKDSDFWKMLNEPEDQAPGGEEVPAEEQDPSPEAADSASGAPNDFQNNVQVKVIRSPADLIRFIEELKGGTKKGKPNIGQEQPVDDAAEVPQREPEKERGDPERQREMEEEEDEDEDEDEDEDERQLLGEFEKELEGILLPSDRDRLRSEVKAGMERELENIIQETEKELDPDGLKKESERDRAMLALTSTLNKLIKRLEEKQSPELVKKHKKKRVVPKKPPPSPQPTGKIEIKIVRPWAEGTEEGARWLTDEDTRNLKEIFFNILVPGAEEAQKERQRQKELESNYRRVWGSPGGEGTGDLDEFDF"
    pred_type, all_scores = predict_localization(test_seq)
    print(f"Predicted type: {pred_type}")
    print("All model scores:", all_scores) 

