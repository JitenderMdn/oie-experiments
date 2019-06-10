#!/usr/bin/env python
# coding: utf-8
import nltk
import re
import pprint
import en_coref_md
from nltk import Tree
from stanford_open_ie_python_wrapper import stanford_ie
from stanfordnlp.server import CoreNLPClient
from config import config
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
os.environ["CORENLP_HOME"] = config["CORENLP_HOME"]

def convert_to_tree(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    sentences = [NPChunker.parse(sent) for sent in sentences]
    return sentences

def get_noun_phrases(text):
    trees = convert_to_tree(text)
    nps = []
    for tree in trees:
        for subtree in tree.subtrees():
            if subtree.label() == "NP":
                nps.append(" ".join([word.lower() for word, _ in subtree.leaves()]))
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

def get_ngrams(sentence, nps):
    tokens = nltk.word_tokenize(sentence)
    req_n_grams = list()
    if len(tokens) <= 4:
        return [sentence]
    for i in range(4, len(tokens) + 1):
        n_grams = nltk.ngrams(tokens, i)
        for gram in n_grams:
            temp = " ".join(gram)
            # flag = False
            # for np in nps:
            #     if np in temp:
            #         flag = True
            #         break
            # if flag:
            #     req_n_grams.append(temp)
            req_n_grams.append(temp)
    return req_n_grams

def preprocess(paragraph, generate_ngrams=False, nps=None):
    ref_mapping = get_corefs(paragraph)
    paragraph = deref_text(paragraph, ref_mapping)
    # ";" is used to seperate the subject, relation and object by both Stanford and OpenIE. 
    # Better to remove it from out text so that we don't get confused in output.
    paragraph.replace(";", "")
    paragraph = [line for line in nltk.sent_tokenize(paragraph)
                     if len(nltk.word_tokenize(line)) > 3]
    if generate_ngrams:
        paragraph = flatten([get_ngrams(line, nps) for line in paragraph])
    return paragraph

def flatten(list_2d):
    return [item for sublist in list_2d for item in sublist]

def process_batch(batch, generate_ngrams=False):
    nps = [get_noun_phrases(line) for line in batch]
    nps = set(flatten(nps))
    batch = [preprocess(line, generate_ngrams, nps) for line in batch]
    batch = flatten(batch)
    return batch, nps

def filter_relations(relations, nounphrases):
    rels = [rel for rel in relations
            if _valid_relation_np(rel, nounphrases, True, False)]
    # rels = [(subj, remove_stopwords(relation), obj) 
    #         for subj, relation, obj in rels]
    return rels

def _valid_relation_np(relation, nounphrases, strict=True, strict_merge=True):
    subj, _, obj = relation
    s_flag = False
    o_flag = False
    if strict_merge:
        s_flag = subj in nounphrases
        o_flag = obj in nounphrases
        val = (s_flag and o_flag) if strict else (s_flag or o_flag)
        return val

    for phrase in nounphrases:
        if containedIn(phrase, subj, False):
            s_flag = True
        if containedIn(phrase, obj, False):
            o_flag = True
        val = (s_flag and o_flag) if strict else (s_flag or o_flag)
        if val:
            return val
    return False


def get_relations(paragraph):
    paragraph = preprocess(paragraph)
    nps = [get_noun_phrases(line) for line in paragraph]
    nps = set(flatten(nps))
    rels = get_oie_relations(paragraph, True, True)
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

def containedIn(string1, string2, ignore_stopwords=True):
    words = string1.split(" ")
    req = set(string2.split(" "))

    if ignore_stopwords:
        stopwords = set(nltk.corpus.stopwords.words('english'))
        words = [word for word in words if word.lower() not in stopwords]
        req = set([word for word in req if word.lower() not in stopwords])

    for word in words:
        if word not in req:
            return False
    return True

def merge_strings(strings):
    strings = sorted(strings, key=lambda x: len(x))
    req = set()
    for string in strings:
        if string in req:
            continue
        else:
            for r in strings.copy():
                if containedIn(r, string):
                    req.discard(r)
            req.add(string)
    return req

def merge_peripherals(peripherals):
    i = 0
    j = 1
    flag = False
    while i < len(peripherals):
        j = i + 1
        while j < len(peripherals):
            s1, o1 = peripherals[i]
            s2, o2 = peripherals[j]
            if (containedIn(s1, s2, False) or containedIn(s2, s1, False)) and (containedIn(o1, o2, False) or containedIn(o2, o1, False)):
               s = s1 if len(s1) > len(s2) else s2
               o = o1 if len(o1) > len(o2) else o2
               peripherals[i] = (s, o)
               del peripherals[j]

            else:
                j += 1
        i += 1
    
    return peripherals



def merge_relations(relations):
    print("Merging relations..")
    subject_rel_mapping = {}
    object_rel_mapping = {}
    subject_object_mapping = {}
    rel_mapping = {}
    print("Getting subject relation mapping..")
    for sub, rel, ob in relations:
        subject_rel_mapping[(sub, rel)] = subject_rel_mapping.get((sub, rel), []) + [ob]
    print("Merging objects with common subject, relation..")
    subj_filtered_rels = []
    for key, objs in subject_rel_mapping.items():
        objects = merge_strings(objs)
        subj_filtered_rels.extend([(key[0], key[1], ob) for ob in objects])
    print("Getting relation object mapping..")
    for sub, rel, ob in subj_filtered_rels:
        object_rel_mapping[(rel, ob)] = object_rel_mapping.get((rel, ob), []) + [sub]
    print("Merging subjects with common relation, object..")
    obj_filtered_rels = []
    for key, subjs in object_rel_mapping.items():
        subjects = merge_strings(subjs)
        obj_filtered_rels.extend([(sub, key[0], key[1]) for sub in subjects])
    print("Getting subject, object mapping..")
    for sub, rel, ob in obj_filtered_rels:
        subject_object_mapping[(sub, ob)] = subject_object_mapping.get((sub, ob), []) + [rel]
    print("Merging relations with same subject and object..")
    rel_filtered_rels = []
    for key, rels in subject_object_mapping.items():
        relations = merge_strings(rels)
        rel_filtered_rels.extend([(key[0], rel, key[1]) for rel in relations])
    print("Getting rel mapping")
    for sub, rel, ob in rel_filtered_rels:
        rel_mapping[rel] = rel_mapping.get(rel, []) + [(sub, ob)]
    merged = []
    for rel, values in rel_mapping.items():
        values = merge_peripherals(values)
    print("Relations merged!")
    return rel_filtered_rels

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
                    subj = [(token_map[tok.tokenIndex]["word"].lower(), token_map[tok.tokenIndex]["pos"])
                            for tok in triple.subjectTokens]
                    obj = [(token_map[tok.tokenIndex]["word"].lower(), token_map[tok.tokenIndex]["pos"])
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
    in_file = "data/vogue_non_empty_descriptions.txt"
    out_file = "outputs/vogue_ngrams_lemma_normalize_merge_f5_stopwords_present_fuzzy_np_check_on_both_fuzzy_contained_in.txt"
    merge = True
    sentences = [line.strip() for line in open(in_file).readlines()[:5]]
    print("Preprocessing the text data")
    sentences, nps = process_batch(sentences, True)
    batch_size = 1000 if len(sentences) >= 1000 else len(sentences)
    num_batches = len(sentences) / batch_size
    with open(out_file, "w") as fw:
        for i in range(0, len(sentences) - batch_size + 1, batch_size):
            print("Processing batch: {} of {}".format(i/batch_size, num_batches))
            rels = get_oie_relations(sentences[i: i+batch_size], True, True)
            if merge:
                rels = merge_relations(rels)
            rels = filter_relations(rels, nps)
            for relation in rels:
                fw.write("|".join([x for x in relation]))
                fw.write("\n")          

