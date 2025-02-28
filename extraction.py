from features import *
import pickle
from spacy.tokens import DocBin
from spacy.tokens import Doc


with open('all_speeches.pkl', 'rb') as infile:
    all_speeches = pickle.load(infile)

with open('all_docs.pkl', 'rb') as infile:
    all_speeches_as_docs = pickle.load(infile)


feature_matrices = {"CDU/CSU": [],
                    "SPD": [],
                    "BÜNDNIS 90/DIE GRÜNEN": [],
                    "DIE LINKE": [],
                    "AfD": [],
                    "FDP": []}


nlp = spacy.blank("de")
for key in all_speeches_as_docs.keys():
    print(key)
    data = all_speeches_as_docs[key]
    doc_bin = DocBin().from_bytes(data)
    docs = list(doc_bin.get_docs(nlp.vocab))
    all_speeches_as_docs[key] = docs
    

for key in feature_matrices.keys():
    for doc in all_speeches_as_docs[key]:
        feature_matrices[key].append([
            get_I_count_average(doc),
            get_we_count_average(doc),
            get_Sie_count_average(doc),
            get_exclamation_share(doc),
            get_question_share(doc),
            get_avg_sentence_length(doc),
            get_avg_dependency_length(doc),
            get_negation_share(doc)
            #get_wir_Sie_sentence_share(doc, nlp)
        ])

with open('feature_matrices.pkl', 'wb') as outfile:
    pickle.dump(feature_matrices, outfile)

# for key in all_speeches_as_docs.keys():
#     print(key)
#     print(len(all_speeches_as_docs[key]))
#     ges_doc = Doc.from_docs(all_speeches_as_docs[key], ensure_whitespace=True)
#     print(get_negation_share(ges_doc))
# average_I_count_per_token = 0
# cdu = all_speeches_as_docs["CDU/CSU"]
# for doc in cdu:
#     average_I_count_per_token += get_I_count_average(doc)
# print(average_I_count_per_token/len(cdu))



# now = time.time()
# test_doc = all_speeches_as_docs["SPD"][0]
# print(time.time()-now)
# print(get_I_count_average(test_doc))
# print(time.time()-now)
# print(get_we_count_average(test_doc))
# print(time.time()-now)
# print(get_Sie_count_average(test_doc))
# print(time.time()-now)
# print(get_exclamation_share(test_doc))
# print(time.time()-now)
# print(get_avg_sentence_length(test_doc))
# print(time.time()-now)
#
