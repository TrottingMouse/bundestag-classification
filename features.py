import spacy
from spacy.matcher import Matcher
import nltk
from textblob_de import TextBlobDE


def get_I_count_average(speech_as_doc):
    """Input: spacy-doc-object.
    Returns the number of the word 'ich' per token."""
    n_tokens = 0
    num_ich = 0
    for token in speech_as_doc:
        n_tokens += 1
        if token.text.lower() == "ich":
            num_ich += 1
    return num_ich/n_tokens


def get_we_count_average(speech_as_doc):
    """Input: spacy-doc-object.
    Returns the number of the word 'wir' per token."""
    n_tokens = 0
    num_wir = 0
    for token in speech_as_doc:
        n_tokens += 1
        if token.text.lower() == "wir":
            num_wir += 1
    return num_wir/n_tokens


def get_Sie_count_average(speech_as_doc):
    """Input: spacy-doc-object.
    Returns the number of the word 'Sie' per token."""
    n_tokens = 0
    num_sie = 0
    for token in speech_as_doc:
        n_tokens += 1
        if token.text == "Sie" and "Fem" not in token.morph.get("Gender"):
            num_sie += 1
    return num_sie/n_tokens


def get_exclamation_share(speech_as_doc):
    """Input: spacy-doc-object.
    Returns the share of sentences ending with an exclamation mark."""
    n_exclamations = 0
    n_sentences = 0
    for sentence in speech_as_doc.sents:
        n_sentences += 1
        if sentence[-1].text == "!":
            n_exclamations += 1
    return n_exclamations/n_sentences


def get_question_share(speech_as_doc):
    """Input: spacy-doc-object.
    Returns the share of sentences ending with a question mark."""
    n_questions = 0
    n_sentences = 0
    for sentence in speech_as_doc.sents:
        n_sentences += 1
        if sentence[-1].text == "?":
            n_questions += 1
    return n_questions/n_sentences


def get_avg_sentence_length(speech_as_doc):
    """Input: spacy-doc-object.
    Returns the average sentence length in tokens."""
    n_sentences = len(list(speech_as_doc.sents))
    return sum(len(sent) for sent in list(speech_as_doc.sents))/n_sentences


def get_avg_dependency_length(speech_as_doc):
    """Input: spacy-doc-object.
    Returns the average dependency length inside a sentence
    averaged over all sentences."""
    sum_dist_averages = 0
    n_sentences = 0
    for sentence in speech_as_doc.sents:
        sum_distances = 0
        for token in sentence:
            head = token.head
            dep_dist = abs(head.i - token.i)
            sum_distances += dep_dist
        sum_dist_averages += sum_distances/len(sentence)
        n_sentences += 1
    return sum_dist_averages/n_sentences


def get_wir_Sie_sentence_share(speech_as_doc, nlp_object):
    """Input: spacy-doc-object, spacy-nlp-object.
    Returns the share of sentences containing a 'wir-Sie'-juxtaposition
    (in any order)."""
    matcher = Matcher(nlp_object.vocab)
    patterns = [[{"LEMMA": "wir"},
                 {"OP": "+"},
                 {"TEXT": {"IN": ["Sie", "Ihnen", "Ihr", "Ihre"]}}],
                [{"TEXT": {"IN": ["Sie", "Ihnen", "Ihr", "Ihre"]}},
                 {"OP": "+"},
                 {"LEMMA": "wir"}]]
    matcher.add("wir-Sie", patterns)
    n_matches = 0
    n_sentences = 0
    for sentence in speech_as_doc.sents:
        n_sentences += 1
        if matcher(sentence) != []:
            n_matches += 1
    return n_matches/n_sentences


def get_negation_share(speech_as_doc):
    """Input: spacy-doc-object.
    Returns the share of sentences containing a negation."""
    n_negations = 0
    n_tokens = 0
    for token in speech_as_doc:
        if token.lemma_ in ["nicht", "kein", "nie", "nichts"]:
            n_negations += 1
        n_tokens += 1
    return n_negations/n_tokens


def get_sentiment(speech_as_doc):
    """Input: spacy-doc-object.
    Returns the sentiment of the speech (on a scale from -1 to 1)."""
    blob = TextBlobDE(speech_as_doc.text)
    return blob.sentiment.polarity
