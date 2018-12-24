# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
import string
import math



#####################################################################
#                           MAIN CODE                               #
#####################################################################


"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'i like pie' and 'i like cake' in my training set
    Then train_set := [['i','like','pie'], ['i','like','cake']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was spam and second one was ham.
    Then train_labels := [0,1]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter you provided with --laplace (1.0 by default)
    """
    # TODO: Write your code here
    # return predicted labels of development set
    # print("Train Set" + str(train_set))
    # print("Original Set Size:" + str(get_total_words_in(train_set)))

    # PREPROCESS THE TRAIN SET
    train_set = pre_process(train_set)

    # print("Stemmed Train Set" + str(train_set))
    # print("Stemmed Set Size:" + str(get_total_words_in(train_set)))

    # BUILD THE TF DICTIONARIES
    spam_dict, ham_dict = build_word_dictionaries(train_set, train_labels)
    # print("Spam Dict:" + str(spam_dict))
    # print("Ham Dict:" + str(ham_dict))

    total_spam_words_count = sum(spam_dict.values())
    total_ham_words_count = sum(ham_dict.values())
    # print("total_spam_words:" + str(total_spam_words_count))
    # print("total_ham_words_count:" + str(total_ham_words_count))

    # BUILD THE IDF DICTIONARY
    df_dict_training = build_document_frequency_dictionaries(train_set)
    df_dict_dev = build_document_frequency_dictionaries(dev_set)
    # print("Spam DF Dict:" + str(spam_df_dict))
    # print("Ham DF Dict:" + str(ham_df_dict))
    # spam_prob_dict = build_MLE_probabilities(spam_dict, total_spam_words_count, smoothing_parameter)
    # ham_prob_dict = build_MLE_probabilities(ham_dict, total_spam_words_count, smoothing_parameter)
    # print("Spam Prob Dict:" + str(spam_prob_dict))
    # print("Ham Prob Dict:" + str(ham_prob_dict))
    vocabulary_size = get_total_words_in(train_set)
    vocabulary_size_bigrams = get_total_bigrams_in(train_set)

    #PREPROCESS THE DEV SET
    dev_set = pre_process(dev_set)

    # 1. CALCULATE PROBABILITIES USING TF only (Naive MLE)
    # prediction = classify_dev_set(dev_set, spam_dict, ham_dict, total_spam_words_count, total_ham_words_count, smoothing_parameter, vocabulary_size)

    # 2. CALCULATE PROBABILITIES USING TF - IDF
    # prediction = classify_dev_set_with_idf(dev_set, spam_dict, ham_dict, total_spam_words_count, total_ham_words_count, df_dict_training, len(train_set), smoothing_parameter, vocabulary_size)
    # prediction = classify_dev_set_with_idf(dev_set, spam_dict, ham_dict, total_spam_words_count, total_ham_words_count,df_dict_dev, len(dev_set),smoothing_parameter, vocabulary_size)

    #EXTRA CREDIT - Bigram Model
    spam_bigram_dict, ham_bigram_dict = build_bigram_word_dictionaries(train_set, train_labels)
    total_spam_bigrams_count = sum(spam_bigram_dict.values())
    total_ham_bigrams_count = sum(ham_bigram_dict.values())
    # prediction = classify_dev_set_bigrams(dev_set, spam_bigram_dict, ham_bigram_dict, total_spam_bigrams_count, total_ham_bigrams_count, smoothing_parameter, vocabulary_size_bigrams)

    df_dict_training_bigrams =build_document_frequency_dictionaries_bigrams(train_set)
    # prediction = classify_dev_set_with_idf_bigrams(dev_set, spam_bigram_dict, ham_bigram_dict, total_spam_bigrams_count, total_ham_bigrams_count, df_dict_training_bigrams, len(train_set), smoothing_parameter, vocabulary_size_bigrams)


    #EXTRA_CREDIT_MIXTURE_MODEL
    # prediction = classify_dev_set_with_mixture_model(dev_set, spam_dict, ham_dict, total_spam_words_count, total_ham_words_count,
    #                                                  spam_bigram_dict, ham_bigram_dict, total_spam_bigrams_count, total_ham_bigrams_count,
    #                                                  smoothing_parameter, vocabulary_size, 0.04, vocabulary_size_bigrams)

    prediction = classify_dev_set_with_mixture_model_idf(dev_set, spam_dict, ham_dict, total_spam_words_count, total_ham_words_count,
                                                      df_dict_training, len(train_set),
                                                      spam_bigram_dict, ham_bigram_dict, total_spam_bigrams_count, total_ham_bigrams_count,
                                                      df_dict_training_bigrams,
                                                      smoothing_parameter, vocabulary_size, 0.04, vocabulary_size_bigrams)


    return prediction

#####################################################################
#                           CLASSIFIERS                             #
#####################################################################

def classify_dev_set(dev_set, spam_dict, ham_dict, total_spam_words_count, total_ham_words_count, smoothing_parameter, vocabulary_size):
    """
    The most naive unigram MLE (TF) based classifier
    :param dev_set:
    :param spam_dict:
    :param ham_dict:
    :param total_spam_words_count:
    :param total_ham_words_count:
    :param smoothing_parameter:
    :param vocabulary_size:
    :return:  predictions
    """
    prediction = []
    no_of_words = vocabulary_size + 1 #1 indicates the UNK set
    for each_email in dev_set:
        spam_prob = 0
        ham_prob = 0
        for each_word in each_email:
            if (spam_dict.get(each_word, 0) + smoothing_parameter) != 0:
                spam_prob += math.log10((spam_dict.get(each_word, 0) + smoothing_parameter) / (total_spam_words_count + smoothing_parameter * no_of_words))
            if (ham_dict.get(each_word, 0) + smoothing_parameter) != 0:
                ham_prob += math.log10((ham_dict.get(each_word, 0) + smoothing_parameter) / (total_ham_words_count + smoothing_parameter * no_of_words))
        # print("Email:" + str(each_email))
        # print("Spam Prob:" + str(spam_prob))
        # print("Ham Prob:" + str(ham_prob))
        if spam_prob > ham_prob:
            prediction.append(1)
        else:
            prediction.append(0)
    return prediction


def classify_dev_set_with_idf(dev_set, spam_dict, ham_dict, total_spam_words_count, total_ham_words_count, df_dict, no_of_docs, smoothing_parameter, vocabulary_size):
    """
    A smarter TF-IDF style unigram NB classifer
    :param dev_set:
    :param spam_dict:
    :param ham_dict:
    :param total_spam_words_count:
    :param total_ham_words_count:
    :param df_dict:
    :param no_of_docs:
    :param smoothing_parameter:
    :param vocabulary_size:
    :return: predictions
    """
    prediction = []
    no_of_words = vocabulary_size + 1 #1 indicates the UNK set
    for each_email in dev_set:
        spam_prob = 0
        ham_prob = 0
        for each_word in each_email:
            word_df = df_dict.get(each_word, 0)
            idf_value = 1
            if word_df != 0:
                idf_value = math.log10((no_of_docs + 1) / word_df)
            if (spam_dict.get(each_word, 0) + smoothing_parameter) != 0:
                spam_prob += math.log10((spam_dict.get(each_word, 0) * idf_value + smoothing_parameter) / (total_spam_words_count + smoothing_parameter * no_of_words))
            if (ham_dict.get(each_word, 0) + smoothing_parameter) != 0:
                ham_prob += math.log10((ham_dict.get(each_word, 0) * idf_value + smoothing_parameter) / (total_ham_words_count + smoothing_parameter * no_of_words))

        # print("Email:" + str(each_email))
        # print("Spam Prob:" + str(spam_prob))
        # print("Ham Prob:" + str(ham_prob))
        if spam_prob > ham_prob:
            prediction.append(1)
        else:
            prediction.append(0)
    return prediction


def classify_dev_set_bigrams(dev_set, spam_dict, ham_dict, total_spam_bigrams_count, total_ham_bigrams_count, smoothing_parameter, vocabulary_size):
    """
    A Naive Bigram based NB classfier
    :param dev_set:
    :param spam_dict:
    :param ham_dict:
    :param total_spam_bigrams_count:
    :param total_ham_bigrams_count:
    :param smoothing_parameter:
    :param vocabulary_size:
    :return: predictions
    """
    prediction = []
    no_of_words = vocabulary_size + 1 #1 indicates the UNK set
    for each_email in dev_set:
        spam_prob = 0
        ham_prob = 0
        each_email_bigrams = [each_email[i:i + 2] for i in range(len(each_email) - 1)]
        for each_bigram in each_email_bigrams:
            each_bigram_key = str(each_bigram)
            if (spam_dict.get(each_bigram_key, 0) + smoothing_parameter) != 0:
                spam_prob += math.log10((spam_dict.get(each_bigram_key, 0) + smoothing_parameter) / (total_spam_bigrams_count + smoothing_parameter * no_of_words))
            if (ham_dict.get(each_bigram_key, 0) + smoothing_parameter) != 0:
                ham_prob += math.log10((ham_dict.get(each_bigram_key, 0) + smoothing_parameter) / (total_ham_bigrams_count + smoothing_parameter * no_of_words))
        # print("Email:" + str(each_email))
        # print("Spam Prob:" + str(spam_prob))
        # print("Ham Prob:" + str(ham_prob))
        if spam_prob > ham_prob:
            prediction.append(1)
        else:
            prediction.append(0)
    return prediction


def classify_dev_set_with_idf_bigrams(dev_set, spam_dict, ham_dict, total_spam_words_count, total_ham_words_count, df_dict, no_of_docs, smoothing_parameter, vocabulary_size):
    """
    A TF-IDF style bigram classifier
    :param dev_set:
    :param spam_dict:
    :param ham_dict:
    :param total_spam_words_count:
    :param total_ham_words_count:
    :param df_dict:
    :param no_of_docs:
    :param smoothing_parameter:
    :param vocabulary_size:
    :return: predictions
    """
    prediction = []
    no_of_words = vocabulary_size + 1 #1 indicates the UNK set
    for each_email in dev_set:
        spam_prob = 0
        ham_prob = 0
        each_email_bigrams = [each_email[i:i + 2] for i in range(len(each_email) - 1)]
        for each_bigram in each_email_bigrams:
            each_bigram_key = str(each_bigram)
            word_df = df_dict.get(each_bigram_key, 0)
            idf_value = 1
            if word_df != 0:
                idf_value = math.log10((no_of_docs + 1) / word_df)
            if (spam_dict.get(each_bigram_key, 0) + smoothing_parameter) != 0:
                spam_prob += math.log10((spam_dict.get(each_bigram_key, 0) * idf_value + smoothing_parameter) / (total_spam_words_count + smoothing_parameter * no_of_words))
            if (ham_dict.get(each_bigram_key, 0) + smoothing_parameter) != 0:
                ham_prob += math.log10((ham_dict.get(each_bigram_key, 0) * idf_value + smoothing_parameter) / (total_ham_words_count + smoothing_parameter * no_of_words))

        # print("Email:" + str(each_email))
        # print("Spam Prob:" + str(spam_prob))
        # print("Ham Prob:" + str(ham_prob))
        if spam_prob > ham_prob:
            prediction.append(1)
        else:
            prediction.append(0)
    return prediction


def classify_dev_set_with_mixture_model(dev_set, spam_dict, ham_dict, total_spam_words_count, total_ham_words_count, spam_bigram_dict, ham_bigram_dict, total_spam_bigrams_count, total_ham_bigrams_count, smoothing_parameter, vocabulary_size, lambda_value, vocabulary_size_bigrams):
    """
    A naive mixture model
    :param dev_set:
    :param spam_dict:
    :param ham_dict:
    :param total_spam_words_count:
    :param total_ham_words_count:
    :param spam_bigram_dict:
    :param ham_bigram_dict:
    :param total_spam_bigrams_count:
    :param total_ham_bigrams_count:
    :param smoothing_parameter:
    :param vocabulary_size:
    :param lambda_value:
    :param vocabulary_size_bigrams:
    :return: predictions
    """
    prediction = []
    no_of_words = vocabulary_size + 1  # 1 indicates the UNK set
    no_of_bigrams = vocabulary_size_bigrams + 1  # 1 indicates the UNK set
    for each_email in dev_set:
        spam_unigram_prob = 0
        ham_unigram_prob = 0

        #CALCULATE UNIGRAM PROBS
        for each_word in each_email:
            if (spam_dict.get(each_word, 0) + smoothing_parameter) != 0:
                spam_unigram_prob += math.log10((spam_dict.get(each_word, 0) + smoothing_parameter) / (
                total_spam_words_count + smoothing_parameter * no_of_words))
            if (ham_dict.get(each_word, 0) + smoothing_parameter) != 0:
                ham_unigram_prob += math.log10((ham_dict.get(each_word, 0) + smoothing_parameter) / (
                total_ham_words_count + smoothing_parameter * no_of_words))

        #CALCULATE BIGRAM PROBS
        spam_bigram_prob = 0
        ham_bigram_prob = 0
        each_email_bigrams = [each_email[i:i + 2] for i in range(len(each_email) - 1)]
        for each_bigram in each_email_bigrams:
            each_bigram_key = str(each_bigram)
            if (spam_bigram_dict.get(each_bigram_key, 0) + smoothing_parameter) != 0:
                spam_bigram_prob += math.log10((spam_bigram_dict.get(each_bigram_key, 0) + smoothing_parameter) / (
                total_spam_bigrams_count + smoothing_parameter * no_of_words))
            if (ham_bigram_dict.get(each_bigram_key, 0) + smoothing_parameter) != 0:
                ham_bigram_prob += math.log10((ham_bigram_dict.get(each_bigram_key, 0) + smoothing_parameter) / (
                total_ham_bigrams_count + smoothing_parameter * no_of_words))

        spam_prob = (1-lambda_value) * spam_unigram_prob + lambda_value * spam_bigram_prob
        ham_prob = (1-lambda_value) * ham_unigram_prob + lambda_value * ham_bigram_prob
        if spam_prob > ham_prob:
            prediction.append(1)
        else:
            prediction.append(0)
    return prediction


def classify_dev_set_with_mixture_model_idf(dev_set, spam_dict, ham_dict, total_spam_words_count, total_ham_words_count, df_dict, no_of_docs, spam_bigram_dict, ham_bigram_dict, total_spam_bigrams_count, total_ham_bigrams_count, df_dict_bigram, smoothing_parameter, vocabulary_size, lambda_value, vocabulary_size_bigrams):
    """
    The final version - TF -IDF style unigram and bigram calculations
    :param dev_set:
    :param spam_dict:
    :param ham_dict:
    :param total_spam_words_count:
    :param total_ham_words_count:
    :param df_dict:
    :param no_of_docs:
    :param spam_bigram_dict:
    :param ham_bigram_dict:
    :param total_spam_bigrams_count:
    :param total_ham_bigrams_count:
    :param df_dict_bigram:
    :param smoothing_parameter:
    :param vocabulary_size:
    :param lambda_value:
    :param vocabulary_size_bigrams:
    :return:
    """
    prediction = []
    no_of_words = vocabulary_size + 1  # 1 indicates the UNK set
    for each_email in dev_set:
        #CALCULATE UNIGRAM PROBS IDF
        spam_unigram_prob = 0
        ham_unigram_prob = 0
        for each_word in each_email:
            word_df = df_dict.get(each_word, 0)
            idf_value = 1
            if word_df != 0:
                idf_value = math.log10((no_of_docs + 1) / word_df)
            if (spam_dict.get(each_word, 0) + smoothing_parameter) != 0:
                spam_unigram_prob += math.log10((spam_dict.get(each_word, 0) * idf_value + smoothing_parameter) / (
                total_spam_words_count + smoothing_parameter * no_of_words))
            if (ham_dict.get(each_word, 0) + smoothing_parameter) != 0:
                ham_unigram_prob += math.log10((ham_dict.get(each_word, 0) * idf_value + smoothing_parameter) / (
                total_ham_words_count + smoothing_parameter * no_of_words))

        #CALCULATE BIGRAM PROBS IDF
        spam_bigram_prob = 0
        ham_bigram_prob = 0

        each_email_bigrams = [each_email[i:i + 2] for i in range(len(each_email) - 1)]
        for each_bigram in each_email_bigrams:
            each_bigram_key = str(each_bigram)
            word_df = df_dict_bigram.get(each_bigram_key, 0)
            idf_value = 1
            if word_df != 0:
                idf_value = math.log10((no_of_docs + 1) / word_df)
            if (spam_bigram_dict.get(each_bigram_key, 0) + smoothing_parameter) != 0:
                spam_bigram_prob += math.log10((spam_bigram_dict.get(each_bigram_key, 0) * idf_value + smoothing_parameter) / (
                    total_spam_bigrams_count + smoothing_parameter * no_of_words))
            if (ham_bigram_dict.get(each_bigram_key, 0) + smoothing_parameter) != 0:
                ham_bigram_prob += math.log10((ham_bigram_dict.get(each_bigram_key, 0) * idf_value + smoothing_parameter) / (
                    total_ham_bigrams_count + smoothing_parameter * no_of_words))

        spam_prob = (1-lambda_value) * spam_unigram_prob + lambda_value * spam_bigram_prob
        ham_prob = (1-lambda_value) * ham_unigram_prob + lambda_value * ham_bigram_prob

        if spam_prob > ham_prob:
            prediction.append(1)
        else:
            prediction.append(0)
    return prediction


#####################################################################
#                           UTILITY FUNCTIONS                       #
#####################################################################

def pre_process(email_set):
    """
    This does some stemming on the input emails, removes punctuations and \r\n characters
    :param email_set: List of emails
    :return: processed email set
    """
    result = []
    ps = PorterStemmer()
    for each_email in email_set:
        result_nest = []
        for each_word in each_email:
            each_word = each_word.strip()
            each_word = each_word.translate(str.maketrans('', '', string.punctuation))
            each_word = ps.stem(each_word)
            if each_word:
                result_nest.append(each_word)
        if result_nest:
            result.append(result_nest)
    return result


def get_total_words_in(email_set):
    """
    Fetches the total words in the email set
    :param email_set:
    :return:
    """
    flat_list = [item for sublist in email_set for item in sublist]
    return len(set(flat_list))


def get_total_bigrams_in(email_set):
    """
    Fetches the total number of bigrams in an email set
    :param email_set:
    :return:
    """
    answer = 0
    for each_email in email_set:
        each_email_bigrams = [each_email[i:i + 2] for i in range(len(each_email) - 1)]
        answer += len(each_email_bigrams)
    return answer


def build_document_frequency_dictionaries(train_set):
    """
    Helps build the DF dictionaries for use in TF-IDF
    :param train_set:
    :return:
    """
    df_dict = {}
    for each_email_index in range(len(train_set)):
        each_email = set(train_set[each_email_index])
        for each_word in each_email:
            df_dict[each_word] = df_dict.get(each_word, 0) + 1
    return df_dict


def build_document_frequency_dictionaries_bigrams(train_set):
    """
    build_document_frequency_dictionaries() but for bigrams
    :param train_set:
    :return:
    """
    df_dict = {}
    for each_email_index in range(len(train_set)):
        each_email = train_set[each_email_index]
        each_email_bigrams = [each_email[i:i + 2] for i in range(len(each_email) - 1)]
        duplicate_set = set()
        for each_bigram in each_email_bigrams:
            each_bigram_key = str(each_bigram)
            if each_bigram_key not in duplicate_set:
                df_dict[each_bigram_key] = df_dict.get(each_bigram_key, 0) + 1
                duplicate_set.add(each_bigram_key)
    return df_dict


def build_word_dictionaries(train_set, train_labels):
    """
    Build the TF values
    :param train_set:
    :param train_labels:
    :return:
    """
    spam_dict = {}
    ham_dict = {}
    for each_email_index in range(len(train_set)):
        each_email = train_set[each_email_index]
        email_label = train_labels[each_email_index]

        for each_word in each_email:
            if email_label == 1:
                spam_dict[each_word] = spam_dict.get(each_word, 0) + 1
            else:
                ham_dict[each_word] = ham_dict.get(each_word, 0) + 1
    return spam_dict, ham_dict


def build_bigram_word_dictionaries(train_set, train_labels):
    """
    Build the Term Frequency values, but for bigrams
    :param train_set:
    :param train_labels:
    :return:
    """
    spam_bigram_dict = {}
    ham_bigram_dict = {}
    for each_email_index in range(len(train_set)):
        each_email = train_set[each_email_index]
        email_label = train_labels[each_email_index]
        each_email_bigrams = [each_email[i:i + 2] for i in range(len(each_email) - 1)]
        for each_bigram_word in each_email_bigrams:
            each_bigram_key = str(each_bigram_word)
            if email_label == 1:
                spam_bigram_dict[each_bigram_key] = spam_bigram_dict.get(each_bigram_key, 0) + 1
            else:
                ham_bigram_dict[each_bigram_key] = ham_bigram_dict.get(each_bigram_key, 0) + 1
    return spam_bigram_dict, ham_bigram_dict



