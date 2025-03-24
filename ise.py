import requests
from bs4 import BeautifulSoup
import sys
import spacy
from spanbert import SpanBERT
from spacy_help_functions import get_entities, create_entity_pairs


entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]

def send_query(query):
        GOOGLE_API_KEY = sys.argv[1]
        GOOGLE_CX_ID = sys.argv[2]
        # retrieve the top 10 results from google using default value
        service = build(
        "customsearch", "v1", developerKey=os.getenv("GOOGLE_API_KEY")
        )
        res = (
            service.cse()
            .list(
                q=" ".join(query),
                cx=os.getenv("GOOGLE_CX_ID"),
                num=10
            )
            .execute()
        )
        # pprint.pprint(res)
        return res['items']

def is_valid_entity(subj_type, obj_type, relation_id):
    # 
    if relation_id == "1":  # Schools_Attended: (PERSON, ORGANIZATION)
        return subj_type == "PERSON" and obj_type == "ORGANIZATION"
    if relation_id == "2":  # Work_For: (PERSON, ORGANIZATION)
        return subj_type == "PERSON" and obj_type == "ORGANIZATION"
    if relation_id == "3":  # Live_In: (PERSON, LOCATION-like)
        return subj_type == "PERSON" and obj_type in ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
    if relation_id == "4":  # Top_Member_Employees: (ORGANIZATION, PERSON)
        return subj_type == "ORGANIZATION" and obj_type == "PERSON"
    return False

def main():
    gemini_api_key = sys.argv[3]
    option = sys.argv[4] # spanbert or gemini 

    r = sys.argv[5] # which relation to extract from
    t = sys.argv[6] # confidence threshold, ignored if using gemini

    q = sys.argv[7] # seed query
    k = sys.argv[8] # number of tuples to return

    # Load spacy model
    nlp = spacy.load("en_core_web_lg") 
    # Load pre-trained SpanBERT model
    spanbert = SpanBERT("./pretrained_spanbert")  

    # 1. Initialize X, the set of extracted tuples, as the empty set.
    extracted_tuples = set()

    # 2. Query your Google Custom Search Engine to obtain the URLs for the top-10 webpages for query q; 
    # you can reuse your own code from Project 1 for this part if you so wish
    pages = send_query(query)
    urls = []
    for page in pages:
        urls.append(page.get('link'))
    seen = set()

    # 3. For each URL from the previous step that you have not processed before 
    # (you should skip already-seen URLs, even if this involves processing fewer than 10 webpages in this iteration):

    # 3.a. Retrieve the corresponding webpage; if you cannot retrieve the webpage (e.g., because of a timeout), 
    # just skip it and move on, even if this involves processing fewer than 10 webpages in this iteration.
    for url in urls:
        if url not in seen:
            response = requests.get(url)
            if response.status_code == 200:
                # 3.b. Extract the actual plain text from the webpage using Beautiful Soup.
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text()
                # 3.c. If the resulting plain text is longer than 10,000 characters, truncate the text to its first 10,000 characters (for efficiency) and discard the rest.
                if len(text) > 10000:
                    text = text[:10000]
                seen.add(url)
                # 3.d. Use the spaCy library to split the text into sentences and extract named entities 
                # (e.g., PERSON, ORGANIZATION).
                """
                So to annotate the text, you should implement two steps. 
                Then, you should construct entity pairs and run the expensive SpanBERT model, 
                separately only over each entity pair that contains both required named entities for the relation of interest, as specified above. 
                IMPORTANT: You must not run SpanBERT for any entity pairs that are missing one or two entities of the type required by the relation. 
                If a sentence is missing one or two entities of the type required by the relation, you should skip it and move to the next sentence.
                While running the second step over a sentence, SpanBERT looks for some predefined set of relations in a sentence. 
                We are interested in just the four relations mentioned above. 
                (If you are curious about the other relations available, please check the complete list as well as an article with a detailed description.)
                """
                doc = nlp(text)
                for sentence in doc.sents:
                    print("\n\nProcessing sentence: {}".format(sentence))
                    print("Tokenized sentence: {}".format([token.text for token in sentence]))
                    ents = get_entities(sentence, entities_of_interest)
                    print("spaCy extracted entities: {}".format(ents))

                    # 3.e. If -spanbertis specified, use the sentences and named entity pairs as input to SpanBERT 
                    # to predict the corresponding relations, and extract all instances of the relation specified by input parameter r. 
                    if option == "-spanbert":
                        candidate_pairs = []
                        sentence_entity_pairs = create_entity_pairs(sentence, entities_of_interest)
                        for ep in sentence_entity_pairs:
                            # TODO: keep subject-object pairs of the right type for the target relation (e.g., Person:Organization for the "Work_For" relation)
                            candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})  # e1=Subject, e2=Object
                            candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})  # e1=Object, e2=Subject
                                            # Classify Relations for all Candidate Entity Pairs using SpanBERT
                        candidate_pairs = [p for p in candidate_pairs if not p["subj"][1] in ["DATE", "LOCATION"]]  # ignore subject entities with date/location type
                        print("Candidate entity pairs:")
                        for p in candidate_pairs:
                            print("Subject: {}\tObject: {}".format(p["subj"][0:2], p["obj"][0:2]))
                        print("Applying SpanBERT for each of the {} candidate pairs. This should take some time...".format(len(candidate_pairs)))

                        if len(candidate_pairs) == 0:
                            continue
                        
                        relation_preds = spanbert.predict(candidate_pairs)  # get predictions: list of (relation, confidence) pairs
                        # Print Extracted Relations
                        print("\nExtracted relations:")
                        for ex, pred in list(zip(candidate_pairs, relation_preds)):
                            print("\tSubject: {}\tObject: {}\tRelation: {}\tConfidence: {:.2f}".format(ex["subj"][0], ex["obj"][0], pred[0], pred[1]))

                            # TODO: focus on target relations
                            # '1':"per:schools_attended"
                            # '2':"per:employee_of"
                            # '3':"per:cities_of_residence"
                            # '4':"org:top_members/employees"

                    
                    # Otherwise, if -gemini is specified, use the Google Gemini API for relation extraction. 
                    # See below for details on how to perform this step.
                    elif option == "-gemini":
                        continue

# 3.f. If -spanbert is specified, identify the tuples that have an associated extraction confidence of at least t and add them to set X. Otherwise, if -gemini is specified, identify all the tuples that have been extracted and add them to set X (we do not receive extraction confidence values from the Google Gemini API, so feel free to hard-code in a value of 1.0 for the confidence value for all Gemini-extracted tuples).

    
    # (we do not receive extraction confidence values from the Google Gemini API, 
    # so feel free to hard-code in a value of 1.0 for the confidence value for all Gemini-extracted tuples).

    # Remove exact duplicates from set X: if X contains tuples that are identical to each other, 
    # keep only the copy that has the highest extraction confidence (if -spanbert is specified) and 
    # remove from X the duplicate copies. (You do not need to remove approximate duplicates, for simplicity.)


    # If X contains at least k tuples, return the top-k such tuples and stop. 
    # If -spanbert is specified, your output should have the tuples sorted in decreasing order by extraction confidence, 
    # together with the extraction confidence of each tuple. 

    # If -gemini is specified, your output can have the tuples in any order 
    # (if you have more than k tuples, then you can return an arbitrary subset of k tuples). 
    # (Alternatively, you can return all of the tuples in X, not just the top-k such tuples; 
    # this is what the reference implementation does.)
    # Otherwise, select from X a tuple y such that 
    # (1) y has not been used for querying yet and 
    # (2) if -spanbert is specified, y has an extraction confidence that is highest among the tuples in X that have not yet been used for querying. 
    # (You can break ties arbitrarily.) 
    # Create a query q from tuple y by just concatenating the attribute values together, and go to Step 2. 
    # If no such y tuple exists, then stop. (ISE has "stalled" before retrieving k high-confidence tuples.)

if __name__ == "__main__":
    main()