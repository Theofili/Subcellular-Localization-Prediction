



#------------------------------------------------------------------------------------#
    ## Fetched protein sequences from different cellular compartments
    ## Made binary classification dataframes for each model
#------------------------------------------------------------------------------------#


import pandas as pd
from  get_data import GetData
import pandas as pd


# Download the data -- 9 subcellular locations 
## Maybe this can be a function? 

nuclear_df = GetData.get_data(db="protein",title='nuclear' , size=10000, output='nuclear')
membrane_df = GetData.get_data(db="protein", title='membrane', size=10000, output='membrane')
mitochondrial_df = GetData.get_data(db="protein", title='mitochondrial', size=10000, output='mitochondrial')
extracellular_df = GetData.get_data(db="protein", title='extracellular', size=10000, output='extracellular') 
golgi_df = GetData.get_data(db="protein", title='golgi', size=10000, output='golgi')
reticulum_df = GetData.get_data(db="protein", title='reticulum', size=10000, output='reticulum')
ribosome_df = GetData.get_data(db="protein", title='ribosome', size=10000, output='ribosome')
lysosome_df = GetData.get_data(db="protein", title='lysosome', size=10000, output='lysosome')
peroxisome_df = GetData.get_data(db="protein", title='peroxisome', size=10000, output='peroxisome')


# Combine all dataframes 

full_df = pd.concat([nuclear_df, membrane_df, mitochondrial_df, extracellular_df, golgi_df, reticulum_df, ribosome_df, lysosome_df, peroxisome_df], ignore_index=True) 

full_df['length'] = full_df['sequence'].str.len()

# Save to csv

full_df.to_csv('data/proteins/protein_data.csv') # all data with duplicates

df = full_df.copy() # Make a copy to use if needed 


# Get important information about the dataset used

full_df.drop_duplicates(inplace=True)
print(f'The structure of the dataset \n: {df}')

longest_seq = full_df['sequence'].str.len().max()
shortest_seq = full_df['sequence'].str.len().min()

ratio = full_df.value_counts('type')

print(f'The ratio between labels is:\n{ratio}')
print(f'The longest sequence in the dataframe is: {longest_seq}')
print(f'The shortest sequence in the dataframe is: {shortest_seq}')

lower = full_df['length'].quantile(0.05)  
upper = full_df['length'].quantile(0.95)

print(f'The lower bound of the sequence lengths is: {lower}')
print(f'The upper bound of the sequence lengths is: {upper}')



# Filter the sequences based on upper and lower bounds

filtered_df = full_df[full_df['sequence'].str.len().between(76, 1218)]

filtered_df.to_csv('data/proteins/filtered_protein_data.csv', index=False)

print(filtered_df.value_counts('type'))


def make_binary_type_df(full_df, target_type, sample_n=None, out_csv=None):
    
    # Filter for the target type and length

    df_target = filtered_df[filtered_df['type'] == target_type].copy()
    df_target = df_target.drop_duplicates('sequence')

    # Sample from full_df for negatives

    if sample_n is not None:
        df_neg = full_df.sample(n=sample_n, random_state=42)
    else:
        df_neg = full_df

    # Concatenate and drop duplicates -- Searching from the complete dataframe will have duplicates 

    df_combined = pd.concat([df_target, df_neg])
    df_combined = df_combined.drop_duplicates('sequence')
    if 'length' in df_combined.columns:
        df_combined = df_combined.drop(columns=['length'])

    # Map types to binary

    df_combined['type'] = df_combined['type'].apply(lambda x: 1 if x == target_type else 0)

    # Save to CSV

    if out_csv is not None:
        df_combined.to_csv(out_csv, index=False)

    return df_combined



os.chdir("data/model_data")

# Get 50:50 binary classification dataframes for each type -- Adjust sample sizes to number of target sequences
### Maybe sample number should be len(df)+len(df)/4

model_extracellular_df = make_binary_type_df(filtered_df, target_type='extracellular', sample_n =200, out_csv='model_extracellular_data.csv')
model_membrane_df = make_binary_type_df(filtered_df, target_type='transmembrane', sample_n=5000, out_csv='model_membrane_data.csv')
model_nuclear_df = make_binary_type_df(filtered_df, target_type='nuclear', sample_n=3000, out_csv='model_nuclear_data.csv')
model_mitochondrial_df = make_binary_type_df(filtered_df, target_type='mitochondrial', sample_n=3000, out_csv='model_mitochondrial_data.csv')
model_golgi_df = make_binary_type_df(filtered_df, target_type='golgi', sample_n=500, out_csv='model_golgi_data.csv')
model_reticulum_df = make_binary_type_df(filtered_df, target_type='reticulum', sample_n=500, out_csv='model_reticulum_data.csv')
model_ribosome_df = make_binary_type_df(filtered_df, target_type='ribosome', sample_n=200, out_csv='model_ribosome_data.csv')
model_lysosome_df = make_binary_type_df(filtered_df, target_type='lysosome', sample_n=5000, out_csv='model_lysosome_data.csv')
model_peroxisome_df = make_binary_type_df(filtered_df, target_type='peroxisome', sample_n=300, out_csv='model_peroxisome_data.csv')











