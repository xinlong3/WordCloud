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
    file_sentences = sent_tokenize(file_str)
    filter_tags = set(['JJ','NN', 'NNS', 'NNP', 'NNPS'])
    file_tokenized_tagged = []
    for sen in file_sentences:
        tagged_sentence = pos_tag(word_tokenize(sen))
        for i in range(len(tagged_sentence)):
            tagged_sentence[i] = (tagged_sentence[i][0].lower(), tagged_sentence[i][1])
        file_tokenized_tagged.append(tagged_sentence)

    # dictionary that records the coocurrence of words and is regarded as the representation of the graph
    coocc_dict = defaultdict(set)
    for sen_ls in file_tokenized_tagged:
        for i in range(len(sen_ls)):
            if sen_ls[i][1] not in filter_tags:
                continue
            for j in range(max(0, i - window_size), min(len(sen_ls), i + window_size + 1)):
                if i != j and sen_ls[j][1] in filter_tags:
                    coocc_dict[sen_ls[i][0]].add(sen_ls[j][0])
                else:
                    continue

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
            score_this_iter[vi] = (1 - d) + d * sum([(1 / len(coocc_dict[vj])) * score_prev_iter[vj] for vj in neighbors])
    sorted_vertices = sorted(score_this_iter, key=score_this_iter.get, reverse=True)

    # get the top T vertices
    if not custom_T:
        T = math.floor(len(sorted_vertices) / 3)
    top_T_kw_list = sorted_vertices[:T]
    top_T_kw_set = set(top_T_kw_list)

    # record the index of the keywords
    list_of_kw_index_all_sen = []
    for sen_ls in file_tokenized_tagged:
        list_of_kw_index_this_sen = []
        for i, word in enumerate(sen_ls):
            if word[0] in top_T_kw_set:
                list_of_kw_index_this_sen.append(i)
        list_of_kw_index_all_sen.append(list_of_kw_index_this_sen)

    # sort the recorded indices
    for i in range(len(list_of_kw_index_all_sen)):
        list_of_kw_index_all_sen[i] = sorted(list_of_kw_index_all_sen[i])

    # post-processing
    key_words_to_return = set()
    frequency_dict = defaultdict(int)
    for x in range(len(list_of_kw_index_all_sen)):
        i = 0
        while i < len(list_of_kw_index_all_sen[x]):
            curr_mutiple = [file_tokenized_tagged[x][list_of_kw_index_all_sen[x][i]][0]]
            if i == len(list_of_kw_index_all_sen[x]) - 1:
                key_words_to_return.add(curr_mutiple[0])
                #print("curr_multiple here: ", curr_mutiple)
                #print("freq_dict here: ", frequency_dict)
                frequency_dict[curr_mutiple[0]] += 1
                break
            while list_of_kw_index_all_sen[x][i] == list_of_kw_index_all_sen[x][i + 1] - 1:
                i = i + 1
                curr_mutiple = curr_mutiple + [file_tokenized_tagged[x][list_of_kw_index_all_sen[x][i]][0]]
                if i + 1 == len(list_of_kw_index_all_sen[x]):
                    break
            curr_mutiple_joined = ' '.join(curr_mutiple)
            key_words_to_return.add(curr_mutiple_joined)
            frequency_dict[curr_mutiple_joined] += 1
            i = i + 1
    return key_words_to_return, frequency_dict

if __name__ == '__main__':
    file1 = open('text.txt')
    file_str = file1.read().replace('\n', '')
    file1.close()
    keywords, frequency_dict = keyword_extraction_text_rank(file_str)
    print("key words extracted from the text: ", keywords)
    print("obtained frequency dict: ", frequency_dict)
