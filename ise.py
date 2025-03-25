import requests
from bs4 import BeautifulSoup
import sys
import spacy
from SpanBERT.spanbert import SpanBERT
from spacy_help_functions import get_entities, create_entity_pairs
from gemini_helper_6111 import get_gemini_completion
import json
from googleapiclient.discovery import build

entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]

def send_query(query):
        GOOGLE_API_KEY = sys.argv[1]
        GOOGLE_CX_ID = sys.argv[2]
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
    gemini_api_key = sys.argv[3]
    option = sys.argv[4] # spanbert or gemini 

    r = sys.argv[5] # which relation to extract from
    t = float(sys.argv[6]) # confidence threshold, ignored if using gemini
    q = sys.argv[7] # seed query
    k = int(sys.argv[8]) # number of tuples to return

    # Load spacy model
    nlp = spacy.load("en_core_web_lg") 
    # Load pre-trained SpanBERT model
    spanbert = SpanBERT("./pretrained_spanbert")

    relation_map = {
        "1": "per:schools_attended",
        "2": "per:employee_of",
        "3": "per:cities_of_residence",
        "4": "org:top_members/employees"
    }
    target_relation = relation_map[r]

    # 1. Initialize X, the set of extracted tuples, as the empty set.
    X = set()
    used_queries = set()
    iteration = 0

    while True:
        # 2. Query your Google Custom Search Engine to obtain the URLs for the top-10 webpages for query q; 
        # you can reuse your own code from Project 1 for this part if you so wish
        print(f"\n\n=========== Iteration: {iteration} - Query: {q} ===========\n")
        iteration += 1
        pages = send_query(q.split())
        urls = []
        for page in pages:
            urls.append(page.get('link'))
        seen = set()

        # 3. For each URL from the previous step that you have not processed before 
        # (you should skip already-seen URLs, even if this involves processing fewer than 10 webpages in this iteration):

        # 3.a. Retrieve the corresponding webpage; if you cannot retrieve the webpage (e.g., because of a timeout), 
        # just skip it and move on, even if this involves processing fewer than 10 webpages in this iteration.
        for idx, url in enumerate(urls):
            print(f"\nURL ( {idx+1} / {len(urls)}): {url}")
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
                    doc = nlp(text)
                    for sentence in doc.sents:
                        print("\n\nProcessing sentence: {}".format(sentence))
                        print("Tokenized sentence: {}".format([token.text for token in sentence]))
                        ents = get_entities(sentence, entities_of_interest)
                        print("spaCy extracted entities: {}".format(ents))

                        # 3.e.i If -spanbertis specified, use the sentences and named entity pairs as input to SpanBERT 
                        # to predict the corresponding relations, and extract all instances of the relation specified by input parameter r. 
                        if option == "-spanbert":
                            candidate_pairs = []
                            sentence_entity_pairs = create_entity_pairs(sentence, entities_of_interest)
                            for ep in sentence_entity_pairs:
                                tokens = ep[0]
                                subj = ep[1]
                                obj = ep[2]
                                # TODO: keep subject-object pairs of the right type for the target relation (e.g., Person:Organization for the "Work_For" relation)
                                if is_valid_entity(subj[1], obj[1], r):
                                    candidate_pairs.append({"tokens": tokens, "subj": subj, "obj": obj})
                        if not candidate_pairs:
                            print("No candidate entity pairs found in sentence.")
                            continue
                            
                            relation_preds = spanbert.predict(candidate_pairs)
                            # 3.f.i If -spanbert is specified, identify the tuples that have an associated extraction confidence of at least t 
                            # and add them to set X.
                            for ex, pred in zip(candidate_pairs, relation_preds):
                                subj_text = ex["subj"][0]
                                obj_text = ex["obj"][0]
                                relation, confidence = pred

                                if relation == target_relation and confidence >= t:
                                    X.add((subj_text, relation, obj_text, confidence))

                        
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

                            prompt_text = f"Given a sentence, extract all relations for the target relation.\n"
                            if r == "1":
                                prompt_text += "Relation: Schools_Attended\nSubject: PERSON, Object: ORGANIZATION\n"
                            elif r == "2":
                                prompt_text += "Relation: Work_For\nSubject: PERSON, Object: ORGANIZATION\n"
                            elif r == "3":
                                prompt_text += "Relation: Live_In\nSubject: PERSON, Object: LOCATION\n"
                            elif r == "4":
                                prompt_text += "Relation: Top_Member_Employees\nSubject: ORGANIZATION, Object: PERSON\n"
                            prompt_text += f"Sentence: {sentence.text.strip()}"

                            response_text = get_gemini_completion(prompt_text, model_name, max_tokens, temperature, top_p, top_k)

                            for subj, rel, obj in json.loads(response_text):
                                X.add((subj, rel, obj, 1.0))

                            print(response_text)

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
                for i, (subj, rel, obj, conf) in enumerate(X_final):
                    print(f"{i+1}. ({subj}, {rel}, {obj})\tConfidence: {conf:.2f}")
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
                for y in X_deduplicated:
                    subj, rel, obj, confidence = y
                    key = (subj, rel, obj)
                    if key not in used_queries:
                        next_query_tuple = y
                        used_queries.add(key)
                        break
                if next_query_tuple is None:
                    print("ISE has stalled. No more new tuples to query.")
                    return

                subj, rel, obj, confidence = next_query_tuple
                q = f"{subj} {obj}"
    
if __name__ == "__main__":
    main()
