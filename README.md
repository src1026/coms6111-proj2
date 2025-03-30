COMS6111 Project 2: Iterative Set Expansion (ISE) Relation Extraction
Team Members:
Riz Chen (sc5144)

Jonathan Cheung (yc4528)

Submitted Files:
- Python script (old_ise.py)
- Logs from runs (log_gemini.txt, log_spanbert.txt)
- README (this file)

Overview:
This project implements Iterative Set Expansion (ISE) for Relation Extraction using two different NLP models, SpanBERT and Google Gemini. Given a seed query, our implementation extracts tuples corresponding to the specified relation (e.g., "Schools_Attended", "Work_For", "Live_In", "Top_Member_Employees") by iteratively querying the Google Custom Search Engine and extracting information from web pages.

Dependencies:
This program must be executed on a Google Cloud VM instance set up according to the class-provided instructions, but configured with sufficient memory and resources. Ensure the following dependencies are installed and correctly configured.

Software Requirements:
- Python 3.9 or higher
- pip (Python package installer)

Installation Instructions:
Clone the SpanBERT repository and setup dependencies:

git clone https://github.com/Shreyas200188/SpanBERT
cd SpanBERT
pip3 install -r requirements.txt
bash download_finetuned.sh

Place the provided Python file (old_ise.py) into the SpanBERT folder, along with requirements.txt, gemini_helper_6111.py and spacy_help_functions.py. Make sure you run old_ise.py within this SpanBERT directory. 

Usage:
Run the Python program with the following command structure:

python3 old_ise.py [-spanbert|-gemini] <r> <t> <q> <k>

Command-line Arguments:
Model Selection (-spanbert or -gemini)

-spanbert: Uses the SpanBERT model. Requires a confidence threshold (t).

-gemini: Uses the Google Gemini API. Confidence threshold is ignored.

Relation (r):

1: Schools_Attended (PERSON → ORGANIZATION)

2: Work_For (PERSON → ORGANIZATION)

3: Live_In (PERSON → LOCATION)

4: Top_Member_Employees (ORGANIZATION → PERSON)

Confidence Threshold (t): (Float, required only for SpanBERT)

Recommended value: 0.7 or higher.

Seed Query (q): (String) Initial search term.

Number of Tuples (k): (Integer) The number of desired tuples.

Example Usage:
SpanBERT example (extracting top 10 Work_For relations):

python3 old_ise.py -spanbert 2 0.7 "Bill Gates Microsoft" 10
Google Gemini example (extracting 5 Schools_Attended relations):

python3 old_ise.py -gemini 1 0 "Mark Zuckerberg Harvard" 5


Internal Design & Project Structure:
The project structure involves several key components:

- Web Query & Retrieval (send_query): Performs a Google Custom Search to retrieve URLs for processing.

- Webpage Parsing (BeautifulSoup): Extracts plain text from webpages, truncating to 10,000 characters.

- Named Entity Recognition (spaCy): Annotates sentences from retrieved text to identify relevant entities.

- Relation Extraction: SpanBERT Pipeline predicts relationships between entity pairs using SpanBERT with provided confidence scores. Google Gemini API extracts relationships through prompts to Gemini API, returning tuples without confidence scores.

- Iterative Expansion Logic maintains a set of extracted tuples (X), iteratively updates search queries based on the highest confidence unqueried tuples and stops upon achieving the desired number of tuples or if no further expansions are possible.

API Keys are all hardcoded in 
API	Provided Key
Google Custom Search API Key	AIzaSyCbvJbYa8AKkBHDd5efd63Ksdd6TfxcojE
Google Custom Search Engine ID	c33c1185e479a47da
(Note: Google Gemini API Key is hardcoded in the provided program; no need to submit separately.)

Implementation of Step 3 (From Project Instructions):
Step 3 involves:

Retrieving and processing web pages (skipping already-visited URLs).

Extracting text content from HTML pages (max 10,000 characters).

Annotating text with spaCy for named entities.

Identifying and processing candidate sentences for relation extraction using either SpanBERT or Gemini, depending on the selected mode.

Filtering entity pairs by specified relation and confidence threshold (SpanBERT only).

Adding new tuples to the relation set.

This iterative approach continues until either the desired number (k) of tuples is collected, or no further tuples can be discovered ("stall" scenario).

Important Notes:
Always ensure you run the program on a Google Cloud VM with sufficient memory (recommended: 16 GB or higher).

You may encounter rate limits with the Google APIs if performing extensive queries; manage API quotas accordingly.

Ensure your Google API keys and the Custom Search Engine are correctly set up before running the code.

