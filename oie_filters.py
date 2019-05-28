#!/usr/bin/env python
# coding: utf-8
import nltk
import re
import pprint
import en_coref_md
from nltk import Tree
from stanford_open_ie_python_wrapper import stanford_ie
from stanfordnlp.server import CoreNLPClient
import os

patterns = """
    NP:    {<DT><WP><VBP>*<RB>*<VBN><IN><NN>}
           {<NN|NNS|NNP|NNPS><IN>*<NN|NNS|NNP|NNPS>+}
           {<JJ|JJR|JJS>*<NN|NNS|NNP|NNPS><CC>*<NN|NNS|NNP|NNPS>+}
           {<JJ|JJR|JJS>*<NN|NNS|NNP|NNPS>+}
           
    """
print("Getting NP parser..")
NPChunker = nltk.RegexpParser(patterns)
print("Getting coref parser..")
coref_parser = en_coref_md.load()
stanford_corenlp_path = "/Users/krishna.aruru/stanfordnlp_resources/stanford-corenlp-full-2018-10-05"
os.environ["CORENLP_HOME"] = "/Users/krishna.aruru/stanfordnlp_resources/stanford-corenlp-full-2018-10-05"

def convert_to_tree(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    sentences = [NPChunker.parse(sent) for sent in sentences]
    return sentences

def get_noun_phrases(text):
    sentences = convert_to_tree(text)
    nps = []
    for sent in sentences:
        tree = NPChunker.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label() == "NP":
                nps.append(" ".join([word for word, _ in subtree.leaves()]))
    return nps

def get_corefs(paragraph):
    doc = coref_parser(paragraph)
    refs = {}
    if doc._.has_coref:
        for cluster in doc._.coref_clusters:
            for mention in cluster.mentions:
                refs[mention.start_char] = ( mention.end_char, cluster.main.text)
    return refs

def deref_text(sentence, coref_mapping):
    output = ""
    i = 0
    while i < len(sentence):
        if i in coref_mapping:
            pos, replacement = coref_mapping[i]
            output += replacement
            if pos == i:
                i += 1
            else:
                i = pos
        else:
            output += sentence[i]
            i += 1
    return output


def remove_stopwords(text):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    output = []
    for word in nltk.tokenize.word_tokenize(text):
        if not word.lower() in stopwords:
            output.append(word)
    removed = " ".join(output)
    if not removed:
        return text
    return removed

def get_ngrams(sentence):
    tokens = nltk.word_tokenize(sentence)
    req_n_grams = list()
    if len(tokens) <= 4:
        return [sentence]
    for i in range(4, len(tokens) + 1):
        n_grams = nltk.ngrams(tokens, i)
        for gram in n_grams:
            req_n_grams.append(" ".join(gram))
    return req_n_grams

def preprocess(paragraph, generate_ngrams=False):
    ref_mapping = get_corefs(paragraph)
    paragraph = deref_text(paragraph, ref_mapping)
    # ";" is used to seperate the subject, relation and object by both Stanford and OpenIE. 
    # Better to remove it from out text so that we don't get confused in output.
    paragraph.replace(";", "")
    paragraph = [line for line in nltk.sent_tokenize(paragraph)
                     if len(nltk.word_tokenize(line)) > 3]
    if generate_ngrams:
        paragraph = flatten([get_ngrams(line) for line in paragraph])
    return paragraph

def flatten(list_2d):
    return [item for sublist in list_2d for item in sublist]

def process_batch(batch, generate_ngrams=False):
    batch = [preprocess(line, generate_ngrams) for line in batch]
    batch = flatten(batch)
    nps = [get_noun_phrases(line) for line in batch]
    nps = set(flatten(nps))
    return batch, nps

def filter_relations(relations, nounphrases):
    rels = [(subj, relation, obj) for subj, relation, obj in relations
            if (subj in nounphrases or obj in nounphrases)]
    rels = [(subj, remove_stopwords(relation), obj) 
            for subj, relation, obj in rels]
    return rels

def get_relations(paragraph):
    paragraph = preprocess(paragraph)
    nps = set(get_noun_phrases(paragraph))
    rels = get_oie_relations(nltk.sent_tokenize(paragraph))
    return filter_relations(rels, nps)

def normalize_relation(relation):
    """
    Normalizes a relation similar to Reverb. Removes tokens which have POS
    from the set ignore_pos below. The only exception is the case when there
    are no nouns in the relation. In that case we don't remove the adjective
    """
    ignore_pos = {"MD", "DT", "PDT", "WDT", "JJ", "RB", "PRP$"}
    noun_exists = False
    for _, tag in relation:
        if tag.startswith("N"):
            noun_exists = True
            break
    new_rel = []
    for word, tag in relation:
        if tag.startswith("J") and not noun_exists:
            new_rel.append((word, tag))
            continue
        if not tag in ignore_pos:
            new_rel.append((word, tag))

    return new_rel


def get_oie_relations(sentences, lemmatize=False, normalize=False):
    results = []
    
    with CoreNLPClient(annotators=["openie"], timeout=60000, memory='8G') as client:
        for sentence in sentences:
            annotation = client.annotate(sentence)
            curr_results = []
            for annotated_sentence in annotation.sentence:
                token_map = {}
                for i, e in enumerate(annotated_sentence.token):
                    token_map[i] = {
                        "pos": e.pos,
                        "lemma": e.lemma,
                        "word": e.word
                    }
                index = "word" if not lemmatize else "lemma"
                
                for triple in annotated_sentence.openieTriple:
                    subj = [(token_map[tok.tokenIndex]["word"], token_map[tok.tokenIndex]["pos"])
                            for tok in triple.subjectTokens]
                    obj = [(token_map[tok.tokenIndex]["word"], token_map[tok.tokenIndex]["pos"])
                            for tok in triple.objectTokens]
                    rel = [(token_map[tok.tokenIndex][index], token_map[tok.tokenIndex]["pos"])
                            for tok in triple.relationTokens]
                    if normalize:
                        rel = normalize_relation(rel)
                    
                    subj = " ".join([sub for sub, pos in subj])
                    obj = " ".join([ob for ob, pos in obj])
                    rel = " ".join([r for r, pos in rel])
                    curr_results.append((subj, rel, obj))

                curr_results = list(set(curr_results))
                results.extend(curr_results)

    return set(results)


if __name__ == "__main__":
    sentences = [line.strip() for line in open("data/vogue_non_empty_descriptions.txt").readlines()][0:3]
    print("Preprocessing the text data")
    sentences, nps = process_batch(sentences, True)
    batch_size = 1000 if len(sentences) >= 1000 else len(sentences)
    num_batches = len(sentences) / batch_size
    with open("outputs/stanfordoie_corenlp_outputs_with_ngrams_not_normalized_morhed.txt", "w") as fw:
        for i in range(0, len(sentences) - batch_size + 1, batch_size):
            print("Processing batch: {} of {}".format(i/batch_size, num_batches))
            rels = get_oie_relations(sentences[i: i+batch_size], True, False)
            for relation in rels:
                fw.write("|".join([x for x in relation]))
                fw.write("\n")          

