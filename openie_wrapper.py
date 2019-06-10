import os
import uuid
from config import config
from subprocess import Popen
from sys import argv, stderr

JAVA_BIN_PATH = 'java'
DOT_BIN_PATH = 'dot'
OIE_FOLDER = '/mnt/vamshi/OpenIE-standalone'

tmp_folder = "/tmp/openie"
if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)

def is_extraction(description):
    possible_confidence = description.split(" ")[0]
    try:
        conf = float(possible_confidence)
        return True
    except ValueError:
        return False

def merge_tuple(tup):
    if len(tup) < 4:
        return tup
    else:
        return (tup[0],tup[1], ";".join(tup[2:]))

def extract_tuple(tuple_str):
    conf = tuple_str.split(" ")[0]
    tuple_str = tuple_str[len(conf):].strip()
    if tuple_str.startswith("Context"):
        tuple_str = ":".join(tuple_str.split(":")[1:])
    tup = [x.strip() for x in tuple_str[1:-1].split(";")]
    return merge_tuple(tup)

def process_entity_relations(entity_relationships_str):
    relations = [relation for relation in entity_relationships_str if is_extraction(relation)]
    relations = [extract_tuple(relation) for relation in relations]
    return relations

def openie_ie(sentences):
    assert type(sentences) == list, 'Error: the input to this method should be a list of strings'
    id_ = str(uuid.uuid4())
    input_file = os.path.join(tmp_folder, "{}_in.txt".format(id_))
    output_file = os.path.join(tmp_folder, "{}_out.txt".format(id_))
    err_file = os.path.join(tmp_folder, "err.txt")
    
    with open(input_file, "w") as fw:
        for sentence in sentences:
            fw.write(sentence)
            fw.write("\n")
    
    command = 'cd {}; {} -Xmx10g -XX:+UseConcMarkSweepGC -jar openie-assembly.jar --ignore-errors {} {}'.format(
                OIE_FOLDER, JAVA_BIN_PATH, input_file, output_file
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

