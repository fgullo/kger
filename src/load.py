import sys, codecs

def load_kg(path):
    kg = {}
    reverse_kg = {}
    kg_file_facts = 0
    kg_file = codecs.open(path, 'r', encoding='utf-8', errors='ignore')
    line = kg_file.readline()
    while line:
        kg_file_facts += 1
        tokens = line.split('\t')
        subject = tokens[0].strip()
        predicate = tokens[1].strip()
        object = tokens[2].strip()[0:-1]

        if predicate not in kg.keys():
            kg[predicate] = {}
        if subject not in kg[predicate]:
            kg[predicate][subject] = set()
        kg[predicate][subject].add(object)

        if predicate not in reverse_kg.keys():
            reverse_kg[predicate] = {}
        if object not in reverse_kg[predicate]:
            reverse_kg[predicate][object] = set()
        reverse_kg[predicate][object].add(subject)

        line = kg_file.readline()
    kg_file.close()
    #print(kg['<dealsWith>']['<Azerbaijan>'])
    #print("KG-file facts: " + str(kg_file_facts))
    return (kg,reverse_kg)

def load_rules(path):
    rules = []
    rule_file = codecs.open(path, 'r', encoding='utf-8', errors='ignore')
    line = rule_file.readline() #skip heading
    line = rule_file.readline()
    while line:
        rule = line.split('\t')[0]
        rule_tokens = rule.split('=>')
        body_str = rule_tokens[0]
        head_str = rule_tokens[1]

        head = head_str.split()
        body_str_tokens = body_str.split()
        body = []
        for i in range(0,len(body_str_tokens))[0::3]:
            #print(body_str_tokens[i:i+3])
            body.append(body_str_tokens[i:i+3])

        rule = [head,body]
        rules.append(rule)

        line = rule_file.readline()

    rule_file.close()
    return rules

if __name__ == '__main__':
    (kg,reverse_kg) = load_kg('../data/kg/yago2core.10kseedsSample.compressed.notypes.tsv')

    kg_facts = 0
    for p in kg.keys():
        for s in kg[p].keys():
            kg_facts += len(kg[p][s])
    print("KG loaded facts: " + str(kg_facts))

    reverse_kg_facts = 0
    for p in reverse_kg.keys():
        for s in reverse_kg[p].keys():
            reverse_kg_facts += len(reverse_kg[p][s])
    print("Reverse-KG loaded facts: " + str(reverse_kg_facts))

    rules = load_rules('../data/rule/amie-rules_yago2-sample.txt')
    print("Loaded rules: " + str(len(rules)))
    print("Example rule: " + str(rules[0]))
