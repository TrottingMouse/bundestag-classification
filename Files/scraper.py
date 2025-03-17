"""This script downloads the xml-Files from the Bundestag website and makes them readable"""

import requests

# For every debate there is an xml-file and the number is in its Link
#all debates till parliamentary summer break
for sitzungsnummer in range(212, 213):
    # URL of the XML file
    url = f"https://dserver.bundestag.de/btp/20/20{sitzungsnummer:03}.xml"

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the XML content to a file in the current directory
        with open(f"Files/20{sitzungsnummer:03}.xml", "wb") as file:
            file.write(response.content)
        print(f"XML file has been saved as 20{sitzungsnummer:03}.xml")
    else:
        print("Failed to retrieve the XML file. Status code:",
              response.status_code)

# replacing inconsistent whitespace and inconsistencies in XML files
# weird list formatting because of tabs and line breaks in Strings
search_symbols = [" ", "<p>", "<p/>",
                  """<name>
							<vorname>Dirk-UlrichAlexander</vorname>
							<nachname>Mende Föhr</nachname>
							<fraktion>SPDCDU/CSU</fraktion>
						</name>
					</redner>Dirk-Ulrich Mende (SPD):</p>""",
                    """<name>
							<vorname>Dirk-UlrichAlexander</vorname>
							<nachname>Mende Föhr</nachname>
							<fraktion>SPDCDU/CSU</fraktion>
						</name>
					</redner>Alexander Föhr (CDU/CSU):</p>""",
                    """<name><vorname>Dirk-UlrichAlexander</vorname><nachname>Mende
                    Föhr</nachname><fraktion>SPDCDU/CSU</fraktion></name></redner>Dirk-Ulrich Mende (SPD):</p>""",
                    """<name><vorname>Dirk-UlrichAlexander</vorname><nachname>Mende
                    Föhr</nachname><fraktion>SPDCDU/CSU</fraktion></name></redner>Alexander Föhr (CDU/CSU):</p>""",
                    '''</name>
            <p klasse="J">'''
                    ]
                
replace_symbols = [" ", '<p klasse="J">', '',
                   """<name>
							<vorname>Dirk-Ulrich</vorname>
							<nachname>Mende</nachname>
							<fraktion>SPD</fraktion>
						</name>
					</redner>Dirk-Ulrich Mende (SPD):</p>""",
                    """<name>
							<vorname>Alexander</vorname>
							<nachname>Föhr</nachname>
							<fraktion>CDU/CSU</fraktion>
						</name>
					</redner>Alexander Föhr (CDU/CSU):</p>""",
                    """<name><vorname>Dirk-Ulrich</vorname><nachname>Mende
                    </nachname><fraktion>SPD</fraktion></name></redner>Dirk-Ulrich Mende (SPD):</p>""",
                    """<name><vorname>Alexander</vorname><nachname>
                    Föhr</nachname><fraktion>CDU/CSU</fraktion></name></redner>Alexander Föhr (CDU/CSU):</p>""",
                    '''</name>
            <p klasse="J_1">'''
            ]
                   

for sitzungsnummer in range(1, 213):
    with open(f'Files/20{sitzungsnummer:03}.xml', 'r') as file:
        content = file.read()

        # searching and replacing
        for i in range(len(search_symbols)):
            content = content.replace(search_symbols[i], replace_symbols[i])

    # overwriting the file with the replaced content
    with open(f'Files/20{sitzungsnummer:03}.xml', 'w') as file:
        file.write(content)
