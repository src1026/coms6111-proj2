
"""
Overall, your program should receive as input:

Your Google Custom Search Engine JSON API Key from Project 1
Your Google Engine ID from Project 1
Your Google Gemini API key, generated as explained above
An argument that indicates whether we are using SpanBERT (-spanbert) or Google Gemini (-gemini) for the extraction process
An integer r between 1 and 4, indicating the relation to extract: 1 is for Schools_Attended, 2 is for Work_For, 3 is for Live_In, and 4 is for Top_Member_Employees
A real number t between 0 and 1, indicating the "extraction confidence threshold," which is the minimum extraction confidence that we request for the tuples in the output; t is ignored if we are using -gemini
A "seed query" q, which is a list of words in double quotes corresponding to a plausible tuple for the relation to extract (e.g., "bill gates microsoft" for relation Work_For)
An integer k greater than 0, indicating the number of tuples that we request in the output
Then, your program should perform the following steps:
"""

def main():
    option = "" # spanbert or gemini 
    r = 0 # which relation to extract from
    t = 0 # confidence threshold

    # 1. Initialize X, the set of extracted tuples, as the empty set.
    extracted_tuples = set()

    # 2. Query your Google Custom Search Engine to obtain the URLs for the top-10 webpages for query q; 
    # you can reuse your own code from Project 1 for this part if you so wish

    # 3. For each URL from the previous step that you have not processed before 
    # (you should skip already-seen URLs, even if this involves processing fewer than 10 webpages in this iteration):

    # 4. Retrieve the corresponding webpage; if you cannot retrieve the webpage (e.g., because of a timeout), 
    # just skip it and move on, even if this involves processing fewer than 10 webpages in this iteration.
    # Extract the actual plain text from the webpage using Beautiful Soup.
    # If the resulting plain text is longer than 10,000 characters, truncate the text to its first 10,000 characters (for efficiency) and discard the rest.

    # 5. Use the spaCy library to split the text into sentences and extract named entities 
    # (e.g., PERSON, ORGANIZATION). See below for details on how to perform this step.

    # 6. If -spanbertis specified, use the sentences and named entity pairs as input to SpanBERT to predict the corresponding relations, 
    # and extract all instances of the relation specified by input parameter r. 
    # Otherwise, if -gemini is specified, use the Google Gemini API for relation extraction. See below for details on how to perform this step.

    # If -spanbert is specified, identify the tuples that have an associated extraction confidence of at least t and add them to set X. 
    # Otherwise, if -gemini is specified, identify all the tuples that have been extracted and add them to set X 
    # (we do not receive extraction confidence values from the Google Gemini API, 
    # so feel free to hard-code in a value of 1.0 for the confidence value for all Gemini-extracted tuples).

    # Remove exact duplicates from set X: if X contains tuples that are identical to each other, keep only the copy that has the highest extraction confidence (if -spanbert is specified) and remove from X the duplicate copies. (You do not need to remove approximate duplicates, for simplicity.)
    # If X contains at least k tuples, return the top-k such tuples and stop. 
    # If -spanbert is specified, your output should have the tuples sorted in decreasing order by extraction confidence, together with the extraction confidence of each tuple. 
    # If -gemini is specified, your output can have the tuples in any order (if you have more than k tuples, then you can return an arbitrary subset of k tuples). 
    # (Alternatively, you can return all of the tuples in X, not just the top-k such tuples; this is what the reference implementation does.)
    # Otherwise, select from X a tuple y such that (1) y has not been used for querying yet and 
    # (2) if -spanbert is specified, y has an extraction confidence that is highest among the tuples in X that have not yet been used for querying. (You can break ties arbitrarily.) 
    # Create a query q from tuple y by just concatenating the attribute values together, and go to Step 2. 
    # If no such y tuple exists, then stop. (ISE has "stalled" before retrieving k high-confidence tuples.)

    query = ""
    while True:
        print("Please enter query and target precision. Example: per se 0.8")
        user_input = sys.stdin.readline().strip().split()
        try:
            target_precision = float(user_input[-1])
            if 0 <= target_precision <= 1:
                query = user_input[:-1]
                break
            else:
                print("Invalid target precision. Please enter a numeric value between 0 and 1.")
        except (ValueError, IndexError):
            print("Invalid input. Please try again!")

    while True:

        # 2. retrieve the top 10 results from google using default value 
        # for the various API parameters without modifying these default values
        retrieved_pages = send_query(query)
        html_pages = filter_html(retrieved_pages)

        # 3. present results to user, so that user can mark all webpages that are relevant
        # for each page in the query result, display title, URL, description returned by Google
        num_relevant_pages = 0
        num_html_pages = len(html_pages)
        
        relevant_pages = []
        non_relevant_pages = []
        for i in range(len(html_pages)):
            page = html_pages[i]
            print_divider()
            print("Result", i+1)
            print("URL:", page.get('link'))
            print("Title", page.get('title'))
            print("Summary:", page.get('snippet'))
            print("Is this page relevant? (Y/N)")
            print_divider()
            if input() == "Y":
                num_relevant_pages += 1
                relevant_pages.append(page)
            else:
                non_relevant_pages.append(page)

        # 4. if the precision@10 of the results from step 2 for the relevance judgment
        # of step 3 is greater than or equal to the target value, or prevision@10 is 0, then stop
        curr_precision = num_relevant_pages / num_html_pages
        print("Current precision:", curr_precision)

        if curr_precision >= target_precision:
            print("Desired precision reached, done.")
            print_feedback_summary(query, curr_precision, True, target_precision)
            sys.exit()

        elif curr_precision == 0:
            print("No relevant pages found. Query can't be expanded. Terminating program.")
            sys.exit()
             
        # implement tf-idf, extract top 2 words from the relevant pages
        query = update_query(query, relevant_pages, non_relevant_pages)
        print("Total query", query)
        #query = update_query_word2vec(query, relevant_pages)
        
        # otherwise, use pages marked as relevant to automatically derive new words that are likely to identify more relevant pages
        # can introduce at most 2 new words during each round

if __name__ == "__main__":
    main()