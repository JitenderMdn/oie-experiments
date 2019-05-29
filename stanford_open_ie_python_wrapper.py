#!/usr/bin/env python
# coding: utf-8
import os
import uuid
from config import config
from subprocess import Popen
from sys import stderr

JAVA_BIN_PATH = 'java'
STANFORD_IE_FOLDER = config['STANFORD_IE_FOLDER']
tmp_folder = "/tmp/openie"
if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)

def process_entity_relations(entity_relationships_str):
    relations = []
    for relation in entity_relationships_str:
        tuple_start = relation.find("(") + 1
        tuple_end = relation.find(")")
        tup = relation[tuple_start:tuple_end].split(";")
        relations.append(tup)
    return relations


def stanford_ie(sentences):
    if not type(sentences) == list:
        sentences = [sentences]
    id_ = str(uuid.uuid4())
    input_file = os.path.join(tmp_folder, "{}_in.txt".format(id_))
    output_file = os.path.join(tmp_folder, "{}_out.txt".format(id_))
    err_file = os.path.join(tmp_folder, "err.txt")
    
    with open(input_file, "w") as fw:
        for sentence in sentences:
            fw.write(sentence)
            fw.write("\n")
    
    command = 'cd {}; {} -mx4g -cp "stanford-openie.jar:stanford-openie-models.jar:lib/*" '\
              'edu.stanford.nlp.naturalli.OpenIE {} -format ollie > {}'.format(
                STANFORD_IE_FOLDER, JAVA_BIN_PATH, input_file, output_file
              )
    print("Executing command: {}".format(command))
    process = Popen(command, stdout=stderr, shell=True)
    process.wait()
    assert not process.returncode, 'Error: The command returned with a non 0 return code'
    
    os.remove(input_file)
    with open(output_file, "r") as fr:
        raw_results = fr.readlines()
    os.remove(output_file)
    
    results = process_entity_relations(raw_results)
    return results

