import sys, getopt, os, codecs
import load


def get_first_order_moment(vectors):
    moments = [float(vectors[0][j]) for j in range(0,len(vectors[0]))]
    for i in range(1,len(vectors)):
        for j in range(0,len(vectors[i])):
            moments[j] += vectors[i][j]
    for j in range(0,len(moments)):
        moments[j] /= len(vectors)
    return moments

def get_second_order_moment(vectors):
    moments = [float(vectors[0][j]*vectors[0][j]) for j in range(0,len(vectors[0]))]
    for i in range(1,len(vectors)):
        for j in range(0,len(vectors[i])):
            moments[j] += vectors[i][j]*vectors[i][j]
    for j in range(0,len(moments)):
        moments[j] /= len(vectors)
    return moments

#fast avg squared Euclidean distance between all pairs of vectors in the two given sets
def fast_avg_euclidean_dist(vectors1,vectors2):
    first_order_moment1 = get_first_order_moment(vectors1)
    first_order_moment2 = get_first_order_moment(vectors2)
    second_order_moment1 = get_second_order_moment(vectors1)
    second_order_moment2 = get_second_order_moment(vectors2)

    dist = 0.0
    for j in range(0,len(first_order_moment1)):
        dist += second_order_moment1[j] - 2*first_order_moment1[j]*first_order_moment2[j] + second_order_moment2[j]
    return dist

def fast_avg_euclidean_dist_momentsgiven(first_order_moment1,first_order_moment2,second_order_moment1,second_order_moment2):
    dist = 0.0
    for j in range(0,len(first_order_moment1)):
        dist += second_order_moment1[j] - 2*first_order_moment1[j]*first_order_moment2[j] + second_order_moment2[j]
    return dist

#area of the minimum bounding hyper-rectangle
def mbr_area(embeddings):
    min_coordinates = [embeddings[0][i] for i in range(0,len(embeddings[0]))]
    max_coordinates = [embeddings[0][i] for i in range(0,len(embeddings[0]))]
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

"""
#squared Euclidean distance
def get_dist(v1,v2):
    d = 0.0
    for i in range(0,len(v1)):
        d += (v1[i]-v2[i])*(v1[i]-v2[i])
    return d

def pairwise_dist(src_ids,target_ids,src_embeddings,target_embeddings):
    dist = {}
    for x_id in src_ids:
        x_embedding = src_embeddings[x_id]
        for y_id in target_ids:
            y_embedding = target_embeddings[y_id]
            d = get_dist(x_embedding,y_embedding)
            dist[(x_id,y_id)] = d
    return dist

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
"""

def debug():
    vectors = [[1,2,3],[3,2,1],[2,2,2]]
    print(get_first_order_moment(vectors))
    print(get_second_order_moment(vectors))
    print(fast_avg_euclidean_dist([[2,2,3],[1,3,4]],[[2,2,3],[1,3,4]]))
    sys.exit(-1)

if __name__ == '__main__':
    #debug()
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

    print('Computing (first-order and second-order) moments of all entities and relations')
    fo_moment_alle = get_first_order_moment(ent_embeddings)
    so_moment_alle = get_second_order_moment(ent_embeddings)
    fo_moment_allr = get_first_order_moment(rel_embeddings)
    so_moment_allr = get_second_order_moment(rel_embeddings)

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

            current_ent_embeddings = [ent_embeddings[entity2id[e]] for e in entities]
            fo_moment_currente = get_first_order_moment(current_ent_embeddings)
            so_moment_currente = get_second_order_moment(current_ent_embeddings)
            current_rel_embeddings = [rel_embeddings[relation2id[r]] for r in relations]
            fo_moment_currentr = get_first_order_moment(current_rel_embeddings)
            so_moment_currentr = get_second_order_moment(current_rel_embeddings)

            e2e_avgdist = fast_avg_euclidean_dist_momentsgiven(fo_moment_currente,fo_moment_currente,so_moment_currente,so_moment_currente)
            e2alle_avgdist = fast_avg_euclidean_dist_momentsgiven(fo_moment_currente,fo_moment_alle,so_moment_currente,so_moment_alle)
            e2r_avgdist = fast_avg_euclidean_dist_momentsgiven(fo_moment_currente,fo_moment_currentr,so_moment_currente,so_moment_currentr)
            e2allr_avgdist = fast_avg_euclidean_dist_momentsgiven(fo_moment_currente,fo_moment_allr,so_moment_currente,so_moment_allr)
            r2r_avgdist = fast_avg_euclidean_dist_momentsgiven(fo_moment_currentr,fo_moment_currentr,so_moment_currentr,so_moment_currentr)
            r2allr_avgdist = fast_avg_euclidean_dist_momentsgiven(fo_moment_currentr,fo_moment_allr,so_moment_currentr,so_moment_allr)

            e_mbr_area = mbr_area(current_ent_embeddings)
            er_mbr_area = mbr_area(current_ent_embeddings+current_rel_embeddings)
            r_mbr_area = mbr_area(current_rel_embeddings)

            output_line = '\t'.join([rule,example,str(e2e_avgdist),str(e2alle_avgdist),str(e2r_avgdist),str(e2allr_avgdist),str(r2r_avgdist),str(r2allr_avgdist),str(e_mbr_area),str(alle_mbr_area),str(er_mbr_area),str(aller_mbr_area),str(r_mbr_area),str(allr_mbr_area)])
            #print(rule + '\t' + example + '\t' + str(avg_dist_ent2ent) + '\t' + str(avg_dist_ent2row) + '\t' + str(avg_dist_rel2rel) + '\t' + str(avg_dist_rel2row))
            output.write('\n' + output_line)
            output.flush()
            line = rules.readline()
            count += 1
            print('#processed lines: ' + str(count))
        rules.close()
        output.close()
    else:
        print("ERROR: rule-support file and/or output file not provided")
        sys.exit(2)
