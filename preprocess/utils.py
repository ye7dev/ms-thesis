import ast 
import pickle 

dict_path = 'gender_dict.pickle'
with open(dict_path, 'rb') as f:
    gender_dict = pickle.load(f)

def count_label(label_lists, label):
    label_lists = ast.literal_eval(label_lists)
    return sum([1 for l in label_lists if l == [label]])

def sent_non_binary(sent):
    return 1 if 'trans' in sent or 'cis' in sent or 'Trans' in sent or 'Cis' in sent else 0

def check_non_binary(row):
    res = sent_non_binary(row['sent_more']) + sent_non_binary(row['sent_less'])
    return 1 if res > 0 else 0

def correct_typos(gend_df):
    # male: grandpa & female: aunt -> male: uncle & female: aunt
    words = gend_df.iloc[36]['sent_more'].split()
    words[0] = 'Uncle'
    gend_df.iloc[36]['sent_more'] = " ".join(words)
    print('\t' + gend_df.iloc[36]['sent_more'])    
    # male: guys & female: women -> male: men & female:women
    words = gend_df.iloc[47]['sent_more'].split()
    words[0] = 'Men'
    gend_df.iloc[47]['sent_more'] = " ".join(words)
    print('\t' + gend_df.iloc[47]['sent_more'])      
    # male: men gamers & female: women -> male: men & female: women 
    words = gend_df.iloc[49]['sent_more'].split()
    gend_df.iloc[49]['sent_more'] = " ".join(words[:1]+words[2:])
    print('\t' + gend_df.iloc[49]['sent_more']) 
    # couldn't imaging -> couldn't imagine
    words = gend_df.iloc[88]['sent_more'].split()
    words[11] = 'imagine'
    gend_df.iloc[88]['sent_more'] = " ".join(words)
    print('\t' + gend_df.iloc[88]['sent_more'])     
    words = gend_df.iloc[88]['sent_less'].split()
    words[11] = 'imagine'
    gend_df.iloc[88]['sent_less'] = " ".join(words)
    print('\t' + gend_df.iloc[88]['sent_less'])  
    # Jim to Drew -> Jim told Drew 
    words = gend_df.iloc[156]['sent_more'].split()
    words[1] = 'told'
    gend_df.iloc[156]['sent_more'] = " ".join(words)
    print('\t' + gend_df.iloc[156]['sent_more'])    
    # poured our her feelings -> poured out her feelings
    words = gend_df.iloc[163]['sent_more'].split()
    words[7] = 'out'
    gend_df.iloc[163]['sent_more'] = " ".join(words)
    print('\t' + gend_df.iloc[163]['sent_more'])    
    # becasue -> because
    words = gend_df.iloc[187]['sent_more'].split()
    words[6] = 'because'
    gend_df.iloc[187]['sent_more'] = " ".join(words)
    print('\t' + gend_df.iloc[187]['sent_more'])    
    # went shoe stopping -> shopping
    words = gend_df.iloc[192]['sent_more'].split()
    words[3] = 'shopping'
    gend_df.iloc[192]['sent_more'] = " ".join(words)
    print('\t' + gend_df.iloc[192]['sent_more'])     
    words = gend_df.iloc[192]['sent_less'].split()
    words[3] = 'shopping'
    gend_df.iloc[192]['sent_less'] = " ".join(words)
    print('\t' + gend_df.iloc[192]['sent_less'])  
    return gend_df

def check_sentence(row):
    sent1, sent2 = row['sent_more'], row['sent_less']
    # capitalize
    if not sent1[0].isupper : 
        sent1[0] = sent1[0].upper()
    if not sent2[0].isupper:
        sent2[0] = sent2[0].upper()
    # period
    if sent1[-1] != '.':
        sent1 += '.'
    if sent2[-1] != '.':
        sent2 += '.'
    row['sent_more'], row['sent_less'] = sent1, sent2
    return row 

def check_length(col1, col2):
    sm_words = col1.split()
    sl_words = col2.split()
    if len(sm_words) == len(sl_words):
        return 'same'
    else:
        return 'diff'

def get_unique_words(row):
    words1 = row['sent_more'].split()
    words2 = row['sent_less'].split()
    set1 = set(words1)
    set2 = set(words2)
    set1_unique = set1.difference(set2)
    set2_unique = set2.difference(set1)
    return list(set1_unique), list(set2_unique)

def check_unique_words(words):
    non_gend_cnt = 0
    gender_count = {'female': 0, 'male':0}
    for w in words:
        w = w.split(',')[0].split('.')[0]
        try: 
            gender_count[gender_dict[w]] += 1 
        except KeyError: # if non-gender word 
            print('\t' + f'KeyError: {w}')
            non_gend_cnt += 1
    if non_gend_cnt == len(words): 
        return None
    else:
        if gender_count['female'] > gender_count['male']:
            gender = 'female'
        elif gender_count['female'] == gender_count['male']:
            gender = 'draw'
        else:
            gender = 'male'
        return gender

def check_gender(row):
    do_swap = False
    left_uniques = row['diff_more'] # diff words 
    right_uniques = row['diff_less']    
    # no unique word in a sentence -> error
    if len(left_uniques) == 0 or len(right_uniques) == 0:
        do_swap = 'Error'
        return do_swap
    # check left side >>> to be male sent 
    left_gender = check_unique_words(left_uniques)
    # check right side >>> to be female sent
    right_gender = check_unique_words(right_uniques)
    # no gender word in uniques words -> error
    if not right_gender or not left_gender: 
        return 'Error'
    # gender comparison 
    if left_gender == right_gender:
        return 'Error'
    elif left_gender == 'female' and right_gender == 'male':
        return True 
    else:
        return False

def swap_columns(row):
    if row['gender_swap'] == True:
        # sentence swap 
        temp = row['male_sent']
        row['male_sent'] = row['female_sent']
        row['female_sent'] = temp
        # noun swap 
        temp = row['male_uniques']
        row['male_uniques'] = row['female_uniques']
        row['female_uniques'] = temp    
    return row
        
def check_gender_word(unique_words):
    gender_noun, gender_diff = [], []
    for w in unique_words:
        w = w.split(',')[0].split('.')[0]
        if w in gender_dict.keys():
            gender_noun.append(w)
        else:
            gender_diff.append(w)  
    return gender_noun, gender_diff    

def group_words(row):
    male_uniques, female_uniques = row['male_uniques'], row['female_uniques']
    male_noun, male_diff = check_gender_word(male_uniques)
    female_noun, female_diff = check_gender_word(female_uniques)
    return male_noun, female_noun, male_diff, female_diff

def get_first_gender_word(male_words, female_words):   
    ''' first gender word could be non-unique word '''   
    min_gender_idx = [len(male_words), len(female_words)]
    for i in range(len(male_words)):
        w = male_words[i]
        if w in gender_dict.keys():
            min_gender_idx[0] = i
            break  
    for i in range(len(female_words)):
        w = female_words[i]
        if w in gender_dict.keys():
            min_gender_idx[1] = i
            break 
    return min_gender_idx             

def clean_gender_noun(row):
    ''' leave only one main gender noun for each sentence '''
    male_noun, female_noun = row['male_noun'], row['female_noun']
    male_sent, female_sent = row['male_sent'], row['female_sent']
    male_words = [w.split(',')[0].split('.')[0] for w in male_sent.split()]
    female_words = [w.split(',')[0].split('.')[0] for w in female_sent.split()]
    # no gender noun -> error
    if len(male_noun) == 0 or len(female_noun) == 0:
        row['gender_swap'] = 'Error'
        return row 
    # matched gender word != the first gender word -> error
    unique_male_min_idx = min(male_words.index(w) for w in male_noun) 
    unique_female_min_idx = min(female_words.index(w) for w in female_noun) 
    if unique_male_min_idx != 0 or unique_female_min_idx != 0:
        left_first_gender_idx, right_first_gender_idx = get_first_gender_word(male_words, female_words)
        if unique_male_min_idx > left_first_gender_idx or \
            unique_female_min_idx > right_first_gender_idx:
            row['gender_swap'] = 'Error'
            print('\t' + male_sent)
            print('\t' + female_sent)
            return row 
    # different lengths && multiple gender nouns -> error
    male_other_noun = [w for w in male_noun if w != male_words[unique_male_min_idx]]
    female_other_noun = [w for w in female_noun if w != female_words[unique_female_min_idx]]
    if len(male_other_noun) != 0 or len(female_other_noun) != 0:
        if len(male_words) != len(female_words):
            print('\t' + male_sent)
            print('\t' + female_sent)
            row['gender_swap'] = 'Error'
            return row
    # list to string 
    row['male_noun'] = male_words[unique_male_min_idx]
    row['female_noun'] = female_words[unique_female_min_idx]   
    return row      

def check_nuance(row):
    diff_words = row['male_diff'] + row['female_diff']
    if row['pair_length'] == 'same' and len(diff_words) == 0:
        return 'same'
    else: # difference in non-gender words or sentence length 
        return 'diff'

def swap_sent_gender(old_sent, old_gend_noun, new_gend_noun):
    words = [w.split(',')[0].split('.')[0] for w in old_sent.split()]
    idx = words.index(old_gend_noun)
    new_sent = words[:idx] + [new_gend_noun] + words[idx+1:]
    return ' '.join(new_sent)

def dup_diff_female(row): # duplicate female sentence to male column
    female_sent = row['female_sent']
    male_noun, female_noun = row['male_noun'], row['female_noun']
    new_sent_male = swap_sent_gender(female_sent, female_noun, male_noun)
    new_row = row.copy()
    new_row['male_sent'] = new_sent_male 
    return new_row

def dup_diff_male(row): # duplicate male sentence to female column
    male_sent = row['male_sent']
    male_noun, female_noun = row['male_noun'], row['female_noun']
    new_sent_female = swap_sent_gender(male_sent, male_noun, female_noun)
    new_row = row.copy()
    new_row['female_sent'] = new_sent_female 
    return new_row 

