
import pandas as pd


def model_ribosome(df_ribosome):

    

    ## Import libraries

    import sys 

    if sys.argv[1].lower() == 'true':
        acc = True
    else:
        acc = False

    # Load basic modules
    import os
    import sys
    import time
    #import flash_attn   #cannont install and import for some reason
    from os import path
    import gc
    import torch
    import json
    

    # Load data modules
    import numpy as np
    import pandas as pd
    from random import randrange
    from progressbar import ProgressBar

    # Load machine learning modules
    #import triton
    import transformers
    from torch.utils.data import TensorDataset, DataLoader
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
    from transformers import AutoTokenizer, AutoModel, EarlyStoppingCallback, set_seed, BitsAndBytesConfig
    from accelerate import FullyShardedDataParallelPlugin, Accelerator
    from peft import (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        prepare_model_for_kbit_training,
    )


    model_name="RaphaelMourad/Mistral-Peptide-v1-15M"

    # BnB configuration

    print('Initialazing bnb configuration.../n')
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    #print(bnb_config)
    print('Done!')

    # Load FSPD configuration

    print('Initialazing FSDP configuration.../n')
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    #print(fsdp_plugin, '/n')
    print('Done!')


    # Initialize Lora configuration

    print('Initialazing Parameter-Efficient-Fine-Tuning.../n')
    peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.5, #prevents overfitting
            bias="none",
            task_type="SEQ_CLS",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
        )
    #print(peft_config, '/n')
    print('Done!')


    # Initialize training arguments

    print('Initialazing Training Arguments.../n')
    training_args = transformers.TrainingArguments(
        output_dir="./ribosome_results",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=2e-5,  # Adjusted learning rate for binary classification
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.1,
        bf16=True, #FALSE IN GPU PERFORMS BETTER???
        report_to="none",
        load_best_model_at_end = True,
    )

    import os
    os.environ["WANDB_DISABLED"] = "true"

    #print(training_args, '/n')
    print('Done!')


    # Load tokenizer

    print('Load tokenizer.../n')
    tokenizer_ribosome=transformers.AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=1000,
        padding_side="left",
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer_ribosome.eos_token = '[EOS]'
    tokenizer_ribosome.pad_token = '[PAD]'
    #print(tokenizer, '/n')
    print('Done!')

    num_labels = 2

    # Import the build in functions module

    sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

    from functions import SupervisedDataset, DataCollatorForSupervisedDataset, compute_metrics

    from sklearn.model_selection import train_test_split

    ## Defining all these so that I can use the Supervised dataset to pass the
    # split data through a tokenizer

    # Split the dataframe into training, validation, and test sets --- 80%/20%
    train_df, temp_df = train_test_split(df_ribosome, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Make the paths -- NEED TO CHANGE THE FOLDER --
    temp_train_path = "data/splits/ribosome_temp_train.csv"
    temp_val_path = "data/splits/ribosome_temp_val.csv"
    temp_test_path = "data/splits/ribosome_temp_test.csv"

    # Save the dfs into csvs
    train_df.to_csv(temp_train_path, index=False)
    val_df.to_csv(temp_val_path, index=False)
    test_df.to_csv(temp_test_path, index=False)


    print(f"Training data saved to: {temp_train_path}")
    print(f"Validation data saved to: {temp_val_path}")
    print(f"Test data saved to: {temp_test_path}")

    # Define datasets using the temporary file paths
    train_dataset = SupervisedDataset(tokenizer=tokenizer_ribosome, data_path=temp_train_path, kmer=-1)
    val_dataset = SupervisedDataset(tokenizer=tokenizer_ribosome, data_path=temp_val_path, kmer=-1)
    test_dataset = SupervisedDataset(tokenizer=tokenizer_ribosome, data_path=temp_test_path, kmer=-1)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer_ribosome)

    print("Training dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))



    # Load model

    print('Model loading.../n')
    model_01=transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        output_hidden_states=False,
        #quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
    )
    model_01.config.pad_token_id = tokenizer_ribosome.pad_token_id
    #print(model, '/n')
    print('Done!')
    

    # Start training 

    print('Initialize trainer.../n')
    trainer = transformers.Trainer(
        model=model_01,  
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    print('Training started/n')
    trainer.local_rank=training_args.local_rank
    trainer.train()
    print('Done')


    results_path = os.path.join(training_args.output_dir, "metrics")
    os.makedirs(results_path, exist_ok=True)

    results = trainer.evaluate(eval_dataset=test_dataset)
    with open(os.path.join(results_path, "test_results.json"), "w") as f:
        json.dump(results, f)

    # Load metrics back into pandas
    file_metric = os.path.join(results_path, "test_results.json")
    data_expe = pd.read_json(file_metric, typ='series')

    model_dir = "models/model_ribosome"
    model_01.save_pretrained(model_dir)
    tokenizer_ribosome.save_pretrained(model_dir)

    ## Need to check if the metrics are good
    return model_01, tokenizer_ribosome, data_expe


import pandas as pd
df = pd.read_csv('data/model_data/model_ribosome_data.csv')
model, tokenizer, metrics = model_ribosome(df)





