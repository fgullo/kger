import sys, getopt, os, codecs
import load

def get_dist(v1,v2):
    d = 0.0
    for i in range(0,len(v1)):
        d += (v1[i]-v2[i])*(v1[i]-v2[i])
    return d


def get_avg_dist(obj,embeddings,obj2id):
    dist_sum = 0.0
    count = 0
    for i in range(0,len(obj)-1):
        i_id = obj2id[obj[i]]
        for j in range(i+1,len(obj)):
            j_id = obj2id[obj[j]]
            d = get_dist(embeddings[i_id],embeddings[j_id])
            count += 1
            dist_sum += d
    avg_dist = 0.0 if count == 0 else dist_sum/count
    return avg_dist


if __name__ == '__main__':
    rule_support_file = None
    embedding_file = None
    kg_folder = None
    short_params = "r:e:k:"
    long_params = ["rulesupportfile=","embeddingfile=","kgfolder="]
    try:
        arguments, values = getopt.getopt(sys.argv[1:], short_params, long_params)
    except getopt.error as err:
        # Output error, and return with an error code
        print("embedding_stats.py -r <rule_support_file> -e <embedding_file> -k <kg_folder>")
        #print (str(err))
        sys.exit(2)

    for arg, value in arguments:
        if arg in ("-r", "--rulesupportfile"):
            rule_support_file = value
        elif arg in ("-e", "--embeddingfile"):
            embedding_file = value
        if arg in ("-k", "--kgfolder"):
            kg_folder = value

    (ent_embeddings,rel_embeddings) = load.load_embeddings(embedding_file)
    (entity2id, relation2id) = load.load_openke_dataset(kg_folder)

    if rule_support_file:
        rules = codecs.open(rule_support_file, 'r', encoding='utf-8', errors='ignore')
        line = rules.readline() #skip heading
        line = rules.readline()
        while line:
            tokens = line.split('\t')
            entities = tokens[2].split()
            avg_dist_ent = get_avg_dist(entities,ent_embeddings,entity2id)
            relations = tokens[3].split()
            avg_dist_rel = get_avg_dist(relations,rel_embeddings,relation2id)
            print(str(avg_dist_ent) + '\t' + str(avg_dist_rel))
            line = rules.readline()
        rules.close()
    else:
        print("ERROR: rule-support file not provided")
        sys.exit(2)
