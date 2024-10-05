import pandas as pd 
from utils import *

def filter_crows(filepath):
    df = pd.read_csv(filepath)
    print(f'Total number of examples: {len(df)}') # 1508 
    # filter out non-gender examples 
    df = df[df['bias_type'] == 'gender']
    print(f'# of examples with gender label: {len(df)}')
    num_row = len(df)
    # filter out count(gend_annot) < 3
    print('===== Filtering out misannotated examples...')
    df = df.assign(anno_count = df['annotations'].apply(lambda x: count_label(x, 'gender')))
    drop_df = df[df['anno_count'] < 3]
    gend_df = df[df['anno_count'] >= 3]
    print(f'# of examples dropped: {num_row-len(gend_df)}')
    num_row = len(gend_df)
    # filter out examples containing non-binary gender word
    print('===== Filtering out non-binary examples...')
    gend_df = gend_df.assign(non_binary = gend_df.apply(check_non_binary, axis=1))
    drop_df = pd.concat([drop_df, gend_df[gend_df['non_binary']!=0]], ignore_index=True)   
    drop_df.to_csv('./results/dropped_examples.csv', index=False)               
    gend_df = gend_df[gend_df['non_binary']==0]
    print(f'# of examples dropped: {num_row-len(gend_df)}')
    # reset index 
    gend_df = gend_df[['sent_more',	'sent_less']].reset_index(drop=True)
    # correct typos
    print('===== Correting typos...')
    gend_df = correct_typos(gend_df)
    # check capitalized & period
    print('===== Proofreading...')
    filtered_crows = gend_df.apply(check_sentence, axis=1)
    filtered_crows.to_csv('./results/filtered_crows.csv', index=False)
    print(f'===== Filtered crowS-pair file saved!')
    return filtered_crows

def prepare_sentences(df):       
    # sentence pair length check 
    df['pair_length'] = df.apply(lambda row: check_length(row['sent_more'], row['sent_less']), axis=1)
    print(f"# of different length examples: {len(df[df['pair_length'] != 'same'])}")
    df[['diff_more', 'diff_less']] = df.apply(get_unique_words, axis=1, result_type='expand')

    # rearrange gender sentence, nouns, unique words  
    df['gender_swap'] = df.apply(check_gender, axis=1)
    df = df.rename(columns={'sent_more': 'male_sent', 'sent_less': 'female_sent',\
                            'diff_more':'male_uniques', 'diff_less':'female_uniques'})
    df = df.apply(swap_columns, axis=1)
    df[['male_noun', 'female_noun', 'male_diff', 'female_diff']] = df.apply(group_words, axis=1, result_type='expand')
    print('===== Verifying examples with gender identity issue...')
    df = df.apply(clean_gender_noun, axis=1)

    # filter out gender-related error examples 
    num_row = len(df)
    df[df['gender_swap'] == 'Error'].to_csv('./results/error_rows.csv')
    df = df[df['gender_swap'] != 'Error'] 
    df = df.drop('gender_swap', axis=1)
    print(f'# of gender-related error examples: {num_row-len(df)}')
    # duplicate the pairs with different pair nuance
    df['pair_nuance'] = 'same'
    df['pair_nuance'] = df.apply(check_nuance, axis=1)
    same_rows = df[df['pair_nuance'] == 'same']
    diff_rows = df[df['pair_nuance'] == 'diff']
    print(f'# of examples to copy: {len(df)-len(same_rows)}')
    dup_male_rows = diff_rows.apply(dup_diff_male, axis=1)
    dup_female_rows = diff_rows.apply(dup_diff_female, axis=1)
    total_df = pd.concat([same_rows, dup_male_rows, dup_female_rows], ignore_index=True)
    print(f'Verifying final # of examples: {len(total_df) == len(same_rows) + 2 * len(diff_rows)}')

    # final pair's sentences are equal in length
    num_row = len(total_df)
    total_df['pair_length'] = total_df.apply(lambda row: \
                                            check_length(row['male_sent'], row['female_sent']), axis=1)
    total_df = total_df[total_df['pair_length'] == 'same']
    assert num_row - len(total_df) == 0

    print(f'# of total sentence pairs: {len(total_df)}')
    total_df.to_csv('./results/processed_sentences.csv')
    print('CSV file with sentence pairs saved!')

file_path = 'crows_pairs_anonymized.csv'
filtered_df = filter_crows(file_path)
prepare_sentences(filtered_df)







