import pandas as pd 
import yake
import spacy 

def get_extractor():
    language = "en"
    max_ngram_size = 1
    deduplication_threshold = 0.5
    num_keywords = 4
    kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, \
        dedupLim=deduplication_threshold, top=num_keywords, features=None)
    return kw_extractor

def get_index(words_list, keyword):
    try: # exact matching   
        idx = words_list.index(keyword)
    except: # partial matching
        for i in range(len(words_list)):
            mw = words_list[i]
            if keyword in mw:
                idx = i 
                break 
    return idx 

def split_sentence(row):
    male_sent, female_sent = row['male_sent'], row['female_sent']
    male_noun, female_noun = row['male_noun'], row['female_noun']
    male_words = [w for w in male_sent.split()]
    female_words = [w for w in female_sent.split()]
    # split at keyword            
    male_keywords = extractor.extract_keywords(male_sent)
    idx = None 
    for kw, _ in male_keywords:
        if male_noun in kw: continue 
        idx = get_index(male_words, kw)
        if idx < 2: continue
        # include previous word if connected with 'and'/'or'
        if male_words[idx-1] in ['and', 'or']:
            idx = idx-2   
        break
    if idx == None:
        row['keyword'] = 'Error'
        return row
    # split at as + articles
    if 'as' in male_words[:idx]:
        temp = male_words[:idx]
        as_idx = temp.index('as')
        if temp[as_idx+1] in ['a', 'an', 'the']:
            if as_idx+2 != idx:
                idx = as_idx+2
                kw = temp[idx]
    # get pos tag of original next word 
    docu = nlp(male_sent)
    for w in docu:
        if w.text == kw:
            row['target_pos'] = 'NOUN' if w.pos_ == 'PROPN' else w.pos_
    row['male_sent'] = " ".join(male_words[:idx])
    row['female_sent'] = " ".join(female_words[:idx])
    row['keyword'] = kw
    return row

def check_error(row):
    sent = row['male_sent']
    words = sent.split()
    gender_noun = row['male_noun']
    if gender_noun == 'Mr': gender_noun += '.'
    target_pos = row['target_pos']
    pos_tags = [w.pos_ for w in nlp(sent)]
    if row['keyword'] == 'Error': 
        print('\t' + 'No keyword: '+ sent)
        return 1
    if target_pos == 0: 
        print('\t' + 'No target pos: ' + sent)
        return 1
    if gender_noun not in words: 
        if gender_noun + ',' not in words:
            print('\t' + 'No gender noun: ' + sent)
            return 1 
    if 'VERB' not in pos_tags and 'AUX' not in pos_tags:
        if 'shirk' in words: # corner case
            return 0
        print('\t' + 'No verb: ' + sent)
        return 1
    return 0

def correct_pos_tag(row):
    if row['keyword'] == 'tomorrow':
        old_tag, new_tag = row['target_pos'], 'ADV'
    elif row['keyword'] == 'coding':
        old_tag, new_tag = row['target_pos'], 'VERB'
    elif row['keyword'] == 'Avon':
        old_tag, new_tag = row['target_pos'], 'NOUN'
    else:
        old_tag, new_tag = row['target_pos'], row['target_pos']
    if old_tag != new_tag:
        print('\t' + 'prompt: ' + row['male_sent'] + ' | ' + row['keyword'])
        print('\t' + f'old tag: {old_tag} ---> new tag: {new_tag}')
        row['target_pos'] = new_tag 
    return row

def dup_article(row):
    male_prompt, female_prompt = row['male_prompt'], row['female_prompt']
    if male_prompt.split()[-1] == 'a':
        new_words = male_prompt.split()[:-1] + ['an']
        row['male_prompt'] = " ".join(new_words)
        new_words = female_prompt.split()[:-1] + ['an']
        row['female_prompt'] = " ".join(new_words)
    elif male_prompt.split()[-1] == 'an':
        new_words = male_prompt.split()[:-1] + ['a']
        row['male_prompt'] = " ".join(new_words)
        new_words = female_prompt.split()[:-1] + ['a']
        row['female_prompt'] = " ".join(new_words)
    return row

extractor = get_extractor()
nlp = spacy.load('en_core_web_sm')
file_path = './results/processed_sentences.csv'
df = pd.read_csv(file_path)
df['keyword'], df['target_pos'], df['error'] = 0, 0, 0

# get keyword and its pos tag / split sentence
df = df.apply(split_sentence, axis=1)

# filter out error sentences
print('===== Filtering out error sentences...')
df['error'] = df.apply(check_error, axis=1)
df[df['error'] == 1].to_csv('./results/error_prompts.csv')
print(f"Decrease in # of examples: {sum(df['error'])}")
df = df[df['error'] == 0]

# manual correction of pos tag 
print('===== Correcting wrong pos tags...')
df = df.apply(correct_pos_tag, axis=1)

df = df.rename(columns={'male_sent': 'male_prompt', 'female_sent': 'female_prompt'})
col_list = ['male_prompt', 'female_prompt', 'keyword', 'target_pos',\
            'male_noun', 'female_noun',]
df = df[col_list]
df[['keyword', 'target_pos']].to_csv('./results/next_word_pos.csv', index=False)
print('# of keywords grouped by pos tag')
print(df['target_pos'].value_counts())

# a -> an, vice versa
indef_articles = df[df['male_prompt'].str.split().str[-1].isin(['a', 'an'])]
print(f'# of examples to copy: {len(indef_articles)}')
indef_articles = indef_articles.apply(dup_article, axis=1)
df = pd.concat([df, indef_articles], ignore_index=True)
print(f'# of final prompts: {len(df)}')
df.to_csv('./results/automated_prompt.csv', index=False)
print('CSV file with prompts saved!')