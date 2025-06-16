"""This script reads all the XML files in a given directory
and stores the content of each speech into a dictionary.
The dictionary is sorted by party group.
It also parses the speeches into a spacy DocBin object."""


import xml.etree.ElementTree as ET
import pickle
import spacy
from spacy.tokens import DocBin


# Define minimum speech length in characters
min_speech_length = 1500

number_of_sessions = 212


all_speeches = {"CDU/CSU": [],
                "SPD": [],
                "BÜNDNIS 90/DIE GRÜNEN": [],
                "DIE LINKE": [],
                "AfD": [],
                "FDP": [],
                "Fraktionslos": [],
                "BSW": [],
                "Die Linke": []}

all_speeches_as_docs = {k: [] for k in all_speeches}


def read_xml_file(xml_inpath):
    tree = ET.parse(xml_inpath)
    root = tree.getroot()
    return root


def correct_fraktion_names(fraktion):
    """correcting some formatting mistakes of the files
    in the party group names"""
    if "\n" in fraktion:
        fraktion = fraktion.replace(" ", "")
        fraktion = fraktion.replace("\n", " ")
        fraktion = fraktion.replace("S", "S ")
    if fraktion == "fraktionslos":
        fraktion = "Fraktionslos"
    return fraktion


def normalize_text(text):
    """Deletes extra spaces, tabs and newlines from the text"""
    text = text.replace("\n", " ").replace("\t", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text


# Read XML files and extract speeches into separate lists for each party
for sitzungsnummer in range(1, number_of_sessions + 1):
    root = read_xml_file(f'Files/20{sitzungsnummer:03}.xml')
    sitzung = root.find("sitzungsverlauf")

    # Integer with three states to check if we are at the beginning,
    # in the middle or outside of a speech
    add_to_speech = 0

    current_speech = ""
    last_speech = ""
    fraktion = ""
    current_speaker = ""
    last_speaker = ""

    # All nodes in the XML file that are related to speeches are named "p"
    for p in sitzung.findall(".//p"):
        if p.attrib["klasse"] == "redner":
            try:
                fraktion = p.find(".//fraktion").text
                fraktion = correct_fraktion_names(fraktion)
                add_to_speech = 2
                current_speaker = p.find(".//redner").attrib["id"]
            except AttributeError:
                print(
                    f"Government/administration speech in session "
                    f"{sitzungsnummer}"
                )
                print()

        # Label for the beginning of a speech
        # Also for the beginning of a phrase of the Bundestagspräsident
        # (marking the end of a speech)
        if p.attrib["klasse"] == "J_1":
            # End of a speech of an MP
            if add_to_speech == 1:
                norm_speech = normalize_text(current_speech)

                # if there was an interjection, append the rest of the speech
                # to the last speech
                if current_speaker == last_speaker:
                    # check if the first parts were already long enough
                    if (
                        all_speeches[fraktion] != []
                        and all_speeches[fraktion][-1] == last_speech
                    ):
                        all_speeches[fraktion][-1] += " " + norm_speech[:-1]
                    else:
                        if len(last_speech + norm_speech) >= min_speech_length:
                            all_speeches[fraktion].append(
                                last_speech + " " + norm_speech[:-1]
                            )
                else:
                    if len(current_speech) >= min_speech_length:
                        # append the speech to the dictionary
                        # without last space character
                        all_speeches[fraktion].append(norm_speech[:-1])
                    last_speech = ""

                last_speaker = current_speaker
                if last_speech != "":
                    last_speech += " "
                last_speech += norm_speech[:-1]
                current_speech = ""
                add_to_speech = 0
            # Beginning of a new speech by an MP
            if add_to_speech == 2:
                current_speech += p.text + " "
                add_to_speech = 1

        # Labels for all the other parts of the speech
        if p.attrib["klasse"] in ["J", "O"]:
            if add_to_speech == 1:
                current_speech += p.text + " "
    print(f"Finished reading session number {sitzungsnummer}.")

nlp = spacy.load('de_core_news_sm')
for key in all_speeches_as_docs.keys():
    doc_bin = DocBin(store_user_data=True)
    speeches_list = all_speeches[key]
    for doc in nlp.pipe(speeches_list):
        doc_bin.add(doc)
    bytes_data = doc_bin.to_bytes()
    print(f"Processed {key}")
    all_speeches_as_docs[key] = bytes_data


with open('all_docs.pkl', 'wb') as f:
    pickle.dump(all_speeches_as_docs, f)


# commented out because not needed, remove comment if desired

# with open('all_speeches.pkl', 'wb') as outfile:
#     pickle.dump(all_speeches, outfile)

for i in all_speeches:
    print(f"For party {i} there are {len(all_speeches[i])} speeches.")
