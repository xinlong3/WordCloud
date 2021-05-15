from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import math
import nltk
from nltk import pos_tag
import helpers

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def keyword_extraction_text_rank(file_str, d=0.85, window_size=3, threshold=0.0001, T=10, custom_T=False):
    # extract sentences and tag and lower-case each word
    file_str = file_str.lower()
    filter_tags = set(['JJ','NN', 'NNS', 'NNP', 'NNPS'])
    file_tokenized_tagged = pos_tag(word_tokenize(file_str))
    print("file_tokenized_tagged here: ", file_tokenized_tagged)

    # filter the sentences so that only words with desired labels are in the sentences
    file_tokenized_tagged_filtered = []
    for word_tag_pair in file_tokenized_tagged:
        if word_tag_pair[1] in filter_tags:
            file_tokenized_tagged_filtered.append(word_tag_pair)
    print("filtered file_tokenized_tagged here: ", file_tokenized_tagged_filtered)




    # dictionary that records the coocurrence of words and is regarded as the representation of the graph
    coocc_dict = defaultdict(set)
    for i in range(len(file_tokenized_tagged_filtered)):
        if file_tokenized_tagged_filtered[i][1] not in filter_tags:
            continue
        for j in range(max(0, i - window_size), min(len(file_tokenized_tagged_filtered), i + window_size + 1)):
            if file_tokenized_tagged_filtered[i][0] != file_tokenized_tagged_filtered[j][0] and file_tokenized_tagged_filtered[j][1] in filter_tags:
                coocc_dict[file_tokenized_tagged_filtered[i][0]].add(file_tokenized_tagged_filtered[j][0])
                coocc_dict[file_tokenized_tagged_filtered[j][0]].add(file_tokenized_tagged_filtered[i][0])



    print("coocc_dict here: ", coocc_dict)
    print("coocc_dict['numbers'] here: ", coocc_dict['numbers'])


    # calculate scores iteratively until convergence
    score_prev_iter = {}
    score_this_iter = {}
    for key in coocc_dict:
        score_this_iter[key] = 1
    flag = True
    while flag or (not helpers.has_converged(score_this_iter, score_prev_iter, threshold)):
        flag = False
        for vi, neighbors in coocc_dict.items():
            score_prev_iter = score_this_iter.copy()
            score_this_iter[vi] = (1 - d) + d * sum([score_prev_iter[vj] / len(coocc_dict[vj]) for vj in neighbors])
    sorted_vertices = sorted(score_this_iter, key=score_this_iter.get, reverse=True)
    print("score_this_iter here: ", score_this_iter)
    print("sorted_vertices here: ", sorted_vertices)
    print("size of sorted_vertices: ", len(sorted_vertices))
    for v in sorted_vertices:
        print(v, score_this_iter[v])
        print(v, score_prev_iter[v])
        print("iter difference: ", score_this_iter[v] - score_prev_iter[v])

    # record the index of the keywords
    print("sanity check of ftt: ", file_tokenized_tagged)
    list_of_kw_index = []
    for i, word in enumerate(file_tokenized_tagged):
        if word[0] in sorted_vertices:
            list_of_kw_index.append(i)

    # sort the recorded indices
    list_of_kw_index = sorted(list_of_kw_index)

    # post-processing; score each word/phrase and use T as a upper bound to return keywords
    single_word_set = set()  # The set of single word to avoid standing alone by themselves
    key_words_to_return = set()
    map_keyword_to_score = {}
    frequency_dict = defaultdict(int)
    i = 0
    while i < len(list_of_kw_index):
        curr_mutiple = [file_tokenized_tagged[list_of_kw_index[i]][0]]
        curr_mutiple_score = 0
        curr_mutiple_score += score_this_iter[file_tokenized_tagged[list_of_kw_index[i]][0]]
        if i == len(list_of_kw_index) - 1:
            map_keyword_to_score[curr_mutiple[0]] = curr_mutiple_score
            frequency_dict[curr_mutiple[0]] += 1
            break
        while list_of_kw_index[i] == list_of_kw_index[i + 1] - 1:
            i = i + 1
            curr_mutiple = curr_mutiple + [file_tokenized_tagged[list_of_kw_index[i]][0]]
            curr_mutiple_score += score_this_iter[file_tokenized_tagged[list_of_kw_index[i]][0]]
            single_word_set.add(file_tokenized_tagged[list_of_kw_index[i]][0])
            if len(curr_mutiple) == 2:
                single_word_set.add(curr_mutiple[0])
            if i + 1 == len(list_of_kw_index):
                break
        curr_mutiple_joined = ' '.join(curr_mutiple)
        map_keyword_to_score[curr_mutiple_joined] = curr_mutiple_score
        frequency_dict[curr_mutiple_joined] += 1
        i = i + 1

    sorted_keyword_to_return = sorted(map_keyword_to_score, key=map_keyword_to_score.get, reverse=True)
    for kw in sorted_keyword_to_return:
        print(kw, ": ", map_keyword_to_score[kw])

    # return the top T keywords
    if not custom_T:
        T = math.ceil(len(sorted_vertices) / 3)
    
    i = 0
    for w in sorted_keyword_to_return:
        if w not in single_word_set:
            key_words_to_return.add(w)
            i += 1
        if i == T:
            break

    return key_words_to_return, frequency_dict

if __name__ == '__main__':
    file1 = open('text.txt')
    file_str = file1.read().replace('\n', '')
    file1.close()
    keywords, frequency_dict = keyword_extraction_text_rank(file_str)
    print("key words extracted from the text: ", keywords)
    print("obtained frequency dict: ", frequency_dict)
