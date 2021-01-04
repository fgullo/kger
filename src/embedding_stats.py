import sys, getopt, os, codecs
import load

#squared Euclidean distance
def get_dist(v1,v2):
    d = 0.0
    for i in range(0,len(v1)):
        d += (v1[i]-v2[i])*(v1[i]-v2[i])
    return d

"""
def get_avg_dist(src_obj,target_obj,embeddings,obj2id):
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
"""

def get_avg_dist(src_obj,target_obj,embeddings,obj2id):
    dist_sum = 0.0
    count = 0
    for x in src_obj:
        dist_sum_x = 0.0
        x_id = obj2id[x]
        count_x = 0
        for y in target_obj:
            y_id = obj2id[y]
            d = get_dist(embeddings[x_id],embeddings[y_id])
            count_x += 1
            dist_sum_x += d
        dist_avg_x = 0.0 if count_x == 0 else dist_sum_x/count_x
        dist_sum += dist_avg_x
        count += 1
    dist_avg = 0.0 if count == 0 else dist_sum/count
    return dist_avg

if __name__ == '__main__':
    rule_support_file = None
    embedding_file = None
    kg_folder = None
    output_file = None
    short_params = "r:e:k:o:"
    long_params = ["rulesupportfile=","embeddingfile=","kgfolder=","outputfile="]
    try:
        arguments, values = getopt.getopt(sys.argv[1:], short_params, long_params)
    except getopt.error as err:
        # Output error, and return with an error code
        print("embedding_stats.py -r <rule_support_file> -e <embedding_file> -k <kg_folder> -o <output_file>")
        #print (str(err))
        sys.exit(2)

    for arg, value in arguments:
        if arg in ("-r", "--rulesupportfile"):
            rule_support_file = value
        elif arg in ("-e", "--embeddingfile"):
            embedding_file = value
        elif arg in ("-k", "--kgfolder"):
            kg_folder = value
        elif arg in ("-o", "--outputfile"):
            output_file = value

    (ent_embeddings,rel_embeddings) = load.load_embeddings(embedding_file)
    (entity2id, relation2id) = load.load_openke_dataset(kg_folder)

    heading = 'RULE' + '\t' + 'EXAMPLE' + '\t' + 'AVG_DIST_E2E' + 't' + 'AVG_DIST_E2RoW' + '\t' + 'AVG_DIST_R2R' + '\t' + 'AVG_DIST_R2RoW'
    if rule_support_file and output_file:
        count = 0
        rules = codecs.open(rule_support_file, 'r', encoding='utf-8', errors='ignore')
        output = open(output_file, 'w')
        output.write(heading)
        line = rules.readline() #skip heading
        line = rules.readline()
        while line:
            tokens = line.split('\t')
            rule = tokens[0]
            example = tokens[1]
            entities = tokens[2].split()
            relations = tokens[3].split()
            avg_dist_ent2ent = get_avg_dist(entities,entities,ent_embeddings,entity2id)
            avg_dist_ent2row = get_avg_dist(entities,list(entity2id.keys()),ent_embeddings,entity2id)
            avg_dist_rel2rel = get_avg_dist(relations,relations,rel_embeddings,relation2id)
            avg_dist_rel2row = get_avg_dist(relations,list(relation2id.keys()),rel_embeddings,relation2id)
            #print(rule + '\t' + example + '\t' + str(avg_dist_ent2ent) + '\t' + str(avg_dist_ent2row) + '\t' + str(avg_dist_rel2rel) + '\t' + str(avg_dist_rel2row))
            output.write('\n' + rule + '\t' + example + '\t' + str(avg_dist_ent2ent) + '\t' + str(avg_dist_ent2row) + '\t' + str(avg_dist_rel2rel) + '\t' + str(avg_dist_rel2row))
            output.flush()
            line = rules.readline()
            count += 1
            print('#processed lines: ' + str(count))
        rules.close()
        output.close()
    else:
        print("ERROR: rule-support file and/or putput file not provided")
        sys.exit(2)
