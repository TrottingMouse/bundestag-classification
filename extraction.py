from features import *
import pickle
from spacy.tokens import DocBin
from spacy.tokens import Doc
import pandas as pd
import spacy



with open('all_docs.pkl', 'rb') as infile:
    all_speeches_as_docs = pickle.load(infile)


# Initialize an empty DataFrame
feature_matrices = pd.DataFrame(columns=[
    "Party", "I_count_avg", "we_count_avg", "Sie_count_avg",
    "exclamation_share", "question_share", "avg_sentence_length",
    "avg_dependency_length", "negation_share"
])

nlp = spacy.blank("de")
for key in all_speeches_as_docs.keys():
    print(key)
    data = all_speeches_as_docs[key]
    doc_bin = DocBin().from_bytes(data)
    docs = list(doc_bin.get_docs(nlp.vocab))
    all_speeches_as_docs[key] = docs

# Populate DataFrame
rows = []
for key in all_speeches_as_docs.keys():
    for doc in all_speeches_as_docs[key]:
        rows.append({
            "party": key,
            "I_count_avg": get_I_count_average(doc),
            "we_count_avg": get_we_count_average(doc),
            "Sie_count_avg": get_Sie_count_average(doc),
            "noun_share": get_noun_share(doc),
            "modal_verb_share": get_modal_verb_share(doc),
            "exclamation_share": get_exclamation_share(doc),
            "question_share": get_question_share(doc),
            "avg_sentence_length": get_avg_sentence_length(doc),
            "avg_dependency_length": get_avg_dependency_length(doc),
            "negation_share": get_negation_share(doc)
        })

# Convert list of dictionaries to DataFrame
feature_matrices = pd.DataFrame(rows)

# Display the resulting DataFrame
print(feature_matrices.head())


with open('feature_matrices.pkl', 'wb') as outfile:
    pickle.dump(feature_matrices, outfile)





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
