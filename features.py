from spacy.matcher import Matcher


def get_I_count_average(speech_as_doc):
    """
    Calculates the average occurrence of the word 'ich' per token.

    Args:
        speech_as_doc (spacy.tokens.Doc): The spaCy Doc object representing
            the speech.

    Returns:
        float: The number of times 'ich' appears per token.
    """
    n_tokens = 0
    num_ich = 0
    for token in speech_as_doc:
        n_tokens += 1
        if token.text.lower() == "ich":
            num_ich += 1
    return num_ich/n_tokens


def get_we_count_average(speech_as_doc):
    """
    Calculates the average occurrence of the word 'wir' per token.

    Args:
        speech_as_doc (spacy.tokens.Doc): The spaCy Doc object representing
            the speech.

    Returns:
        float: The number of times 'wir' appears per token.
    """
    n_tokens = 0
    num_wir = 0
    for token in speech_as_doc:
        n_tokens += 1
        if token.text.lower() == "wir":
            num_wir += 1
    return num_wir/n_tokens


def get_Sie_count_average(speech_as_doc):
    """
    Calculates the average occurrence of the word 'Sie' (excluding third
    person singular pronouns) per token.

    Args:
        speech_as_doc (spacy.tokens.Doc): The spaCy Doc object representing
            the speech.

    Returns:
        float: The number of times 'Sie' appears per token.
    """
    n_tokens = 0
    num_sie = 0
    for token in speech_as_doc:
        n_tokens += 1
        # Making sure that the third person singular pronouns are not included
        if token.text == "Sie" and "Fem" not in token.morph.get("Gender"):
            num_sie += 1
    return num_sie/n_tokens


def get_wir_Sie_sentence_share(speech_as_doc, nlp_object):
    """
    Calculates the share of sentences containing a 'wir-Sie' juxtaposition
    (in any order).

    Args:
        speech_as_doc (spacy.tokens.Doc): The spaCy Doc object representing
            the speech.
        nlp_object (spacy.language.Language): The spaCy NLP object.

    Returns:
        float: The share of sentences containing a 'wir-Sie' juxtaposition.
    """
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


def get_noun_share(speech_as_doc):
    """
    Calculates the share of nouns per token.

    Args:
        speech_as_doc (spacy.tokens.Doc): The spaCy Doc object representing
            the speech.

    Returns:
        float: The number of nouns per token.
    """
    n_tokens = 0
    num_noun = 0
    for token in speech_as_doc:
        n_tokens += 1
        if token.pos_ in {"NOUN", "PROPN"}:
            num_noun += 1
    return num_noun/n_tokens


def get_modal_verb_share(speech_as_doc):
    """
    Calculates the share of modal verbs per token.

    Args:
        speech_as_doc (spacy.tokens.Doc): The spaCy Doc object representing
            the speech.

    Returns:
        float: The number of modal verbs per token.
    """
    modals = ["können", "müssen", "dürfen", "sollen", "wollen", "mögen"]
    n_tokens = 0
    num_modv = 0
    for token in speech_as_doc:
        n_tokens += 1
        if token.lemma_ in modals:
            num_modv += 1
    return num_modv/n_tokens


def get_exclamation_share(speech_as_doc):
    """
    Calculates the share of sentences ending with an exclamation mark.

    Args:
        speech_as_doc (spacy.tokens.Doc): The spaCy Doc object representing
            the speech.

    Returns:
        float: The share of sentences ending with an exclamation mark.
    """
    n_exclamations = 0
    n_sentences = 0
    for sentence in speech_as_doc.sents:
        n_sentences += 1
        if sentence[-1].text == "!":
            n_exclamations += 1
    return n_exclamations/n_sentences


def get_question_share(speech_as_doc):
    """
    Calculates the share of sentences ending with a question mark.

    Args:
        speech_as_doc (spacy.tokens.Doc): The spaCy Doc object representing
            the speech.

    Returns:
        float: The share of sentences ending with a question mark.
    """
    n_questions = 0
    n_sentences = 0
    for sentence in speech_as_doc.sents:
        n_sentences += 1
        if sentence[-1].text == "?":
            n_questions += 1
    return n_questions/n_sentences


def get_avg_sentence_length(speech_as_doc):
    """
    Calculates the average sentence length in tokens.

    Args:
        speech_as_doc (spacy.tokens.Doc): The spaCy Doc object representing
            the speech.

    Returns:
        float: The average sentence length in tokens.
    """
    n_sentences = len(list(speech_as_doc.sents))
    return sum(len(sent) for sent in list(speech_as_doc.sents))/n_sentences


def get_avg_dependency_length(speech_as_doc):
    """
    Calculates the average dependency length inside a sentence, averaged
    over all sentences.

    Args:
        speech_as_doc (spacy.tokens.Doc): The spaCy Doc object representing
            the speech.

    Returns:
        float: The average dependency length per sentence.
    """
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


def get_negation_share(speech_as_doc):
    """
    Calculates the share of tokens that are negations.

    Args:
        speech_as_doc (spacy.tokens.Doc): The spaCy Doc object representing
            the speech.

    Returns:
        float: The share of tokens that are negations.
    """
    n_negations = 0
    n_tokens = 0
    for token in speech_as_doc:
        if token.lemma_ in ["nicht", "kein", "nie", "nichts"]:
            n_negations += 1
        n_tokens += 1
    return n_negations/n_tokens
