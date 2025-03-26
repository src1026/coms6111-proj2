import requests
from bs4 import BeautifulSoup
import sys
import spacy
from spanbert import SpanBERT
from spacy_help_functions import get_entities, create_entity_pairs
from gemini_helper_6111 import get_gemini_completion
import json
from googleapiclient.discovery import build

entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
GOOGLE_API_KEY = "AIzaSyCbvJbYa8AKkBHDd5efd63Ksdd6TfxcojE"
GOOGLE_CX_ID = "c33c1185e479a47da"
GEMINI_API_KEY = 'AIzaSyBDkYccGdkn3-z4L_spz7bzsjmBaWToHAw'

def send_query(query):
        # retrieve the top 10 results from google using default value
        service = build(
        "customsearch", "v1", developerKey=GOOGLE_API_KEY
        )
        res = (
            service.cse()
            .list(
                q=" ".join(query),
                cx=GOOGLE_CX_ID,
                num=10
            )
            .execute()
        )
        # pprint.pprint(res)
        return res['items']

def is_valid_entity(subj_type, obj_type, relation_id):
    # filter out entity pairs that don't contain named entities
    # of the right type for the target relation of interest r
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
    # gemini_api_key = sys.argv[3]
    option = sys.argv[4] # spanbert or gemini 

    r = sys.argv[5] # which relation to extract from
    t = float(sys.argv[6]) # confidence threshold, ignored if using gemini
    q = sys.argv[7] # seed query
    k = int(sys.argv[8]) # number of tuples to return

    # Load spacy model
    nlp = spacy.load("en_core_web_lg") 
    # Load pre-trained SpanBERT model
    spanbert = SpanBERT("./pretrained_spanbert")
    relation_desc = {"1": "Schools_Attended", "2": "Work_For", "3": "Live_In", "4": "Top_Member_Employees"}
    relation_map = {
        "1": "per:schools_attended",
        "2": "per:employee_of",
        "3": "per:cities_of_residence",
        "4": "org:top_members/employees"
    }
    target_relation = relation_map[r]
    
    print("Loading pre-trained spanBERT from ./pretrained_spanbert\n")
    print("____")
    print("Parameters:")
    print("Client key\t=", GOOGLE_API_KEY)
    print("Engine key\t=", GOOGLE_CX_ID)
    print("Gemini key\t=", GEMINI_API_KEY)
    print("Method\t\t=", option[1:])  # remove leading '-' for display
    print("Relation\t=", relation_desc[r])
    print("Threshold\t=", t)
    print("Query\t\t=", q)
    print("# of Tuples\t=", k)
    print("Loading necessary libraries; This should take a minute or so ...")

    # 1. Initialize X, the set of extracted tuples, as the empty set.
    X = set()
    used_queries = set()
    iteration = 0

    processed_urls = set()
    while True:
        # 2. Query your Google Custom Search Engine to obtain the URLs for the top-10 webpages for query q; 
        # you can reuse your own code from Project 1 for this part if you so wish
        print(f"\n=========== Iteration: {iteration} - Query: {q} ===========\n")
        iteration += 1
        pages = send_query(q.split())
        urls = []
        for page in pages:
            urls.append(page.get('link'))

        # 3. For each URL from the previous step that you have not processed before 
        # (you should skip already-seen URLs, even if this involves processing fewer than 10 webpages in this iteration):

        # 3.a. Retrieve the corresponding webpage; if you cannot retrieve the webpage (e.g., because of a timeout), 
        # just skip it and move on, even if this involves processing fewer than 10 webpages in this iteration.
        for idx, url in enumerate(urls):
            if url in processed_urls:
                print(f"URL {url} already processed. Skipping.")
                continue
            print(f"\nURL ( {idx+1} / {len(urls)}): {url}")
            print("\tFetching text from url ...")

            if url not in processed_urls:
                response = requests.get(url)
                if response.status_code != 200:
                    print(f"\tUnable to fetch URL. Skipping.")
                    continue
                if response.status_code == 200:
                    # 3.b. Extract the actual plain text from the webpage using Beautiful Soup.
                    soup = BeautifulSoup(response.text, 'html.parser')
                    text = soup.get_text()
                    # 3.c. If the resulting plain text is longer than 10,000 characters, truncate the text to its first 10,000 characters (for efficiency) and discard the rest.
                    if len(text) > 10000:
                        text = text[:10000]
                    processed_urls.add(url)
                    print("\tWebpage length (num characters):", len(text))
                    # 3.d. Use the spaCy library to split the text into sentences and extract named entities 
                    # (e.g., PERSON, ORGANIZATION).
                    doc = nlp(text)
                    print("\tAnnotating the webpage using spacy...")
                    sentences = list(doc.sents)
                    print("\tExtracted {} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...".format(len(sentences)))
                    
                    sentences_with_annotation = 0
                    count_before = len(X)

                    for sent_idx, sentence in enumerate(sentences):
                        if sent_idx % 5 == 0:
                            print(f"\tProcessed {sent_idx} / {len(sentences)} sentences")
                        # print("\n\nProcessing sentence: {}".format(sentence))
                        # print("Tokenized sentence: {}".format([token.text for token in sentence]))
                        # print("spaCy extracted entities: {}".format(ents))

                        # filter out sentences that don't contain named entities of the right type for the target relation of interest r
                        required_types = None
                        if r == "1" or r == "2":
                            required_types = {"PERSON", "ORGANIZATION"}
                        elif r == "3":
                            required_types = {"PERSON"}
                        elif r == "4":
                            required_types = {"ORGANIZATION", "PERSON"}

                        ents = get_entities(sentence, entities_of_interest)
                        entity_types = {etype for _, etype in ents}
                        if required_types and not required_types.issubset(entity_types):
                            print("Sentence does not contain the required entity types. Skipping.")
                            continue

                        # 3.e.i If -spanbertis specified, use the sentences and named entity pairs as input to SpanBERT 
                        # to predict the corresponding relations, and extract all instances of the relation specified by input parameter r. 
                        
                        candidate_pairs = []
                        sentence_entity_pairs = create_entity_pairs(sentence, entities_of_interest)
                        for ep in sentence_entity_pairs:
                            tokens = ep[0]
                            subj = ep[1]
                            obj = ep[2]
                            # TODO: keep subject-object pairs of the right type for the target relation (e.g., Person:Organization for the "Work_For" relation)
                            if is_valid_entity(subj[1], obj[1], r):
                                candidate_pairs.append({"tokens": tokens, "subj": subj, "obj": obj})
                                candidate_pairs.append({"tokens": tokens, "subj": obj, "obj": subj})
                        if not candidate_pairs:
                            # print("No candidate entity pairs found in sentence.")
                            continue
                        sentences_with_annotation += 1
                            
                        if option == "-spanbert":
                            relation_preds = spanbert.predict(candidate_pairs)
                            # 3.f.i If -spanbert is specified, identify the tuples that have an associated extraction confidence of at least t 
                            # and add them to set X.
                            for ex, pred in zip(candidate_pairs, relation_preds):
                                subj_text = ex["subj"][0]
                                obj_text = ex["obj"][0]
                                relation, confidence = pred
                                print("\t=== Extracted Relation ===")
                                print("\tInput tokens:", " ".join(token.strip() for token in ex["tokens"]))
                                print(f"\tOutput Confidence: {confidence:.8f} ; Subject: {subj_text} ; Object: {obj_text} ;")

                                if relation == target_relation and confidence >= t:
                                    print("\tAdding to set of extracted relations")
                                    print("\t==========")
                                    X.add((subj_text, relation, obj_text, confidence))
                                else:
                                    if relation != target_relation:
                                        print("\tCurrent relation: ", relation)
                                        print("\tRelation is not the target relation. Ignoring this.")
                                    if confidence < t:
                                        print("\tConfidence is lower than threshold confidence. Ignoring this.")
                                    print("\t==========")

                        
                        # 3.e.ii Otherwise, if -gemini is specified, use the Google Gemini API for relation extraction. 
                        # 3.f.ii identify all the tuples that have been extracted and add them to set X 
                        # (we do not receive extraction confidence values from the Google Gemini API, 
                        # so feel free to hard-code in a value of 1.0 for the confidence value for all Gemini-extracted tuples).
                        elif option == "-gemini":
                            confidence = 1.0
                            model_name = "gemini-2.0-flash"
                            max_tokens = 100
                            temperature = 0.2
                            top_p = 1
                            top_k = 32

                            prompt_text = "identify the relations and return it as a tuple."
                            # f"Given a sentence, extract all relations for the target relation.\n"
                            
                            if r == "1":
                                prompt_text += "Relation: Schools_Attended\nSubject: PERSON, Object: ORGANIZATION\n"
                            elif r == "2":
                                prompt_text += "Relation: Work_For\nSubject: PERSON, Object: ORGANIZATION\n"
                            elif r == "3":
                                prompt_text += "Relation: Live_In\nSubject: PERSON, Object: LOCATION\n"
                            elif r == "4":
                                prompt_text += "Relation: Top_Member_Employees\nSubject: ORGANIZATION, Object: PERSON\n"
                            """
                            prompt_text += (
                                    "Return the answer as a JSON array of triples [subject, relation, object]. "
                                    "If no relation is found, return an empty JSON array [].\n"
                            )"""
                            prompt_text += (
                                    "return the relation as a tuple.\n for example: (sundar pichai, google)"
                            )
                            prompt_text += f"Sentence: {sentence.text.strip()}"
                            

                            response_text = get_gemini_completion(prompt_text, model_name, max_tokens, temperature, top_p, top_k)
                            print("Gemini API response:", response_text)
                            try:
                                data = json.loads(response_text)
                            except json.decoder.JSONDecodeError as e:
                                print("JSONDecodeError:", e)
                                print("Skipping this sentence due to invalid JSON response.")
                                continue
                            for subj, rel, obj in json.loads(response_text):
                                X.add((subj, rel, obj, 1.0))

                            print(response_text)
            # After processing all sentences in the URL
            relations_from_url = len(X) - count_before
            print(f"\n\tExtracted annotations for {sentences_with_annotation} out of total {len(sentences)} sentences")
            print(f"\tRelations extracted from this website: {relations_from_url} (Overall: {len(X)})")


            # 4. Remove exact duplicates from set X: if X contains tuples that are identical to each other, 
            # keep only the copy that has the highest extraction confidence (if -spanbert is specified) and 
            # remove from X the duplicate copies. (You do not need to remove approximate duplicates, for simplicity.)
            X_dict = dict() # key: triple tuples; value: confidence
            for subj, rel, obj, confidence in X:
                key = (subj, rel, obj)
                if key not in X_dict or confidence > X_dict[key]:
                    X_dict[key] = confidence

            X_deduplicated = [(subj, rel, obj, confidence) for (subj, rel, obj), confidence in X_dict.items()]

            # 5. If X contains at least k tuples, return the top-k such tuples and stop. 
            # If -spanbert is specified, your output should have the tuples sorted in decreasing order by extraction confidence, 
            # together with the extraction confidence of each tuple. 
            if len(X_deduplicated) >= k:
                if option == "-spanbert":
                    X_final = sorted(X_deduplicated, key=lambda x: x[3], reverse=True)[:k]
                elif option == "-gemini":
                    # If -gemini is specified, your output can have the tuples in any order 
                    # (if you have more than k tuples, then you can return an arbitrary subset of k tuples). 
                    # (Alternatively, you can return all of the tuples in X, not just the top-k such tuples; 
                    # # this is what the reference implementation does.)
                    X_final = X_deduplicated
                print("\n================== ALL RELATIONS for", target_relation, f"( {len(X_final)} ) =================")
                for i, (subj, rel, obj, conf) in enumerate(X_final):
                    print(f"Confidence: {conf:.7f} \t\t| Subject: {subj} \t\t| Object: {obj}")
                print("Total # of iterations =", iteration)
                return

            # 6. Otherwise, select from X a tuple y such that 
            # (1) y has not been used for querying yet and 
            # (2) if -spanbert is specified, y has an extraction confidence that is highest among the tuples in X that have not yet been used for querying. 
            # (You can break ties arbitrarily.) 
            # Create a query q from tuple y by just concatenating the attribute values together, and go to Step 2. 
            # If no such y tuple exists, then stop. (ISE has "stalled" before retrieving k high-confidence tuples.)
            else: 
                X_sorted = sorted(X, key=lambda x: x[3], reverse=True)
                next_query_tuple = None
                for y in X_sorted:
                    subj, rel, obj, confidence = y
                    key = (subj, rel, obj)
                    if key not in used_queries:
                        next_query_tuple = y
                        used_queries.add(key)
                        break
                if next_query_tuple is None:
                    print("ISE has stalled. No more new tuples to query.")
                    X_deduplicated = [(subj, rel, obj, conf) for (subj, rel, obj), conf in X_dict.items()]
                    print("\n================== ALL RELATIONS for", target_relation, f"( {len(X_deduplicated)} ) =================")
                    for i, (subj, rel, obj, conf) in enumerate(X_deduplicated):
                        print(f"Confidence: {conf:.7f} \t\t| Subject: {subj} \t\t| Object: {obj}")
                    print("Total # of iterations =", iteration)
                    return

                subj, rel, obj, confidence = next_query_tuple
                q = f"{subj} {obj}"
    
if __name__ == "__main__":
    main()