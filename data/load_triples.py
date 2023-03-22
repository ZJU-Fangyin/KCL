import torch
import pdb
from collections import defaultdict as ddict

class Triples:

    def __init__(self, data_dir="../code/triples"):
        self.data = self.load_data(data_dir)
        self.entities, self.entity2id = self.get_entities(self.data)
        self.attributes, self.attribute2id = self.get_attributes(self.data)
        self.relations, self.relation2id = self.get_relations(self.data)
        self.triples = self.read_triple(self.data, self.entity2id, self.relation2id)
        self.h2rt = self.h2rt(self.triples)
        self.t2rh = self.t2rh(self.triples)

    def load_data(self, data_dir):
        with open("%s.txt" % (data_dir), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        relationid = [i for i in range(len(relations))]
        relation2id = dict(zip(relations, relationid))
        return relations, relation2id

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        entityid = [i for i in range(len(entities))]
        entity2id = dict(zip(entities, entityid))
        return entities, entity2id

    def get_attributes(self, data):
        attributes = sorted(list(set([d[0] for d in data])))
        attributeid = [i for i in range(len(attributes))]
        attribute2id = dict(zip(attributes, attributeid))
        return attributes, attribute2id

    def read_triple(self, data, entity2id, relation2id):
        '''
        Read triples and map them into ids.
        '''
        triples = []
        for triple in data:
            h = triple[0]
            r = triple[1]
            t = triple[2]
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
        return triples
        
    def h2rt(self, triples):    # dict: attribute_id  --> list[(rel_id, atom_id)]
        h2rt = ddict(list)
        for tri in triples:
            h, r, t = tri
            h2rt[h].append((r,t))
        return h2rt

    def t2rh(self, triples):    # dict: atom_id  --> list[(rel_id, attribute_id)]
        t2rh = ddict(list)
        for tri in triples:
            h, r, t = tri
            t2rh[t].append((r,h))
        return t2rh

if __name__ == '__main__':
    data = Triples()
    pdb.set_trace()
    print(data.data, '\n')
    print(data.relation2id, '\n')
    print(data.entity2id, '\n')
    print(data.triples)

