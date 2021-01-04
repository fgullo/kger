import sys, getopt, os, codecs
import load

#squared Euclidean distance
def get_dist(v1,v2):
    d = 0.0
    for i in range(0,len(v1)):
        d += (v1[i]-v2[i])*(v1[i]-v2[i])
    return d

#area of the minimum bounding hyper-rectangle
def mbr_area(embeddings):
    min_coordinates = []
    max_coordinates = []
    for i in range(0,len(embeddings[0])):
        min_coordinates[i] = embeddings[0][i]
        max_coordinates[i] = embeddings[0][i]
    for e in embeddings:
        for i in range(1,len(e)):
            if e[i] < min_coordinates[i]:
                min_coordinates[i] = e[i]
            if e[i] > max_coordinates[i]:
                max_coordinates[i] = e[i]

    area = 1.0
    for i in range(0,len(min_coordinates)):
        area *= (max_coordinates[i] - min_coordinates[i])

    return area

def pairwise_dist(src_ids,target_ids,src_embeddings,target_embeddings):
    dist = {}
    for x_id in src_ids:
        x_embedding = src_embeddings[x_id]
        for y_id in target_ids:
            y_embedding = target_embeddings[y_id]
            d = get_dist(x_embedding,y_embedding)
            dist[(x_id,y_id)] = d
    return dist

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
"""

def avg_dist(src_ids,target_ids,pairwise_dist):
    dist_sum = 0.0
    count = 0
    for i in src_ids:
        dist_sum_i = 0.0
        count_i = 0
        for j in target_ids:
            dist_sum_i += pairwise_dist[(i,j)]
            count_i += 1
        dist_avg_i = 0.0 if count_i == 0 else dist_sum_i/count_i
        dist_sum += dist_avg_i
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
    print('Embeddings successfully loaded!')
    (entity2id, relation2id) = load.load_openke_dataset(kg_folder)
    print('OpenKE dataset successfully loaded!')

    all_entities = list(entity2id.keys())
    all_entity_ids = list(entity2id.values())
    all_relations = list(relation2id.keys())
    all_relation_ids = list(relation2id.values())
    print('Computing r2r pairwise distances')
    rr_pairwise_dist = pairwise_dist(all_relation_ids,all_relation_ids,rel_embeddings,rel_embeddings)
    print('Computing e2r pairwise distances')
    er_pairwise_dist = pairwise_dist(all_entity_ids,all_relation_ids,ent_embeddings,rel_embeddings)
    print('Computing e2e pairwise distances')
    ee_pairwise_dist = pairwise_dist(all_entity_ids,all_entity_ids,ent_embeddings,ent_embeddings)

    print('Computing entity MBRs')
    alle_mbr_area = mbr_area(ent_embeddings)
    print('Computing entity-relation MBRs')
    aller_mbr_area = mbr_area(ent_embeddings+rel_embeddings)
    print('Computing relation MBRs')
    allr_mbr_area = mbr_area(rel_embeddings)

    heading = 'RULE' + '\t' + 'EXAMPLE' + '\t'\
    'AVG_DIST_E2E' + '\t' + 'AVG_DIST_E2ALLE' + '\t'\
    'AVG_DIST_E2R' + '\t' + 'AVG_DIST_E2ALLR' + '\t'\
    'AVG_DIST_R2R' + '\t' + 'AVG_DIST_R2ALLR' + '\t'\
    'E_MBR' + '\t' + 'ALLE_MBR' + '\t'\
    'ER_MBR' + '\t' + 'ALLER_MBR' + '\t'\
    'R_MBR' + '\t' + 'ALLR_MBR'
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
            entity_ids = [entity2id[e] for e in entities]
            relation_ids = [relation2id[r] for r in entities]

            e2e_avgdist = avg_dist(entity_ids,entity_ids,ee_pairwise_dist)
            e2alle_avgdist = avg_dist(entity_ids,all_entity_ids,ee_pairwise_dist)
            e2r_avgdist = avg_dist(entity_ids,relation_ids,er_pairwise_dist)
            e2allr_avgdist = avg_dist(entity_ids,all_relation_ids,er_pairwise_dist)
            r2r_avgdist = avg_dist(relation_ids,relation_ids,rr_pairwise_dist)
            r2allr_avgdist = avg_dist(relation_ids,all_relation_ids,rr_pairwise_dist)

            e_embeddings = [ent_embeddings[e_id] for e_id in entity_ids]
            r_embeddings = [rel_embeddings[r_id] for r_id in relation_ids]
            e_mbr_area = mbr_area(e_embeddings)
            er_mbr_area = mbr_area(e_embeddings+r_embeddings)
            r_mbr_area = mbr_area(r_ambeddings)

            output_line = '\t'.join([rule,example,e2e_avgdist,e2alle_avgdist,e2r_avgdist,e2allr_avgdist,r2r_avgdist,r2allr_avgdist,e_mbr_area,alle_mbr_area,er_mbr_area,aller_mbr_area,r_mbr_area,allr_mbr_area])
            #print(rule + '\t' + example + '\t' + str(avg_dist_ent2ent) + '\t' + str(avg_dist_ent2row) + '\t' + str(avg_dist_rel2rel) + '\t' + str(avg_dist_rel2row))
            output.write(output_line)
            output.flush()
            line = rules.readline()
            count += 1
            print('#processed lines: ' + str(count))
        rules.close()
        output.close()
    else:
        print("ERROR: rule-support file and/or output file not provided")
        sys.exit(2)
