# Imports
import google.generativeai as genai

# Apply Gemini API Key
GEMINI_API_KEY = ''  # Substitute your own key here
genai.configure(api_key=GEMINI_API_KEY)

# Generate response to prompt
def get_gemini_completion(prompt, model_name="gemini-2.0-flash", max_tokens=200, temperature=0.2, top_p=1, top_k=32):
    # Initialize a generative model
    model = genai.GenerativeModel(model_name)

    # Configure the model with your desired parameters
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )

    # Generate a response
    response = model.generate_content(prompt, generation_config=generation_config)

    return response.text.strip() if response.text else "No response received"

def main():
    # Sample Prompt
    prompt_text = """Given a sentence, extract all the Nouns.
sentence: Rob is an engineer at NASA and he lives in California.
extracted:"""

    # Feel free to modify the parameters below.
    # Documentation: https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini
    model_name = "gemini-2.0-flash"
    max_tokens = 100
    temperature = 0.2
    top_p = 1
    top_k = 32

    response_text = get_gemini_completion(prompt_text, model_name, max_tokens, temperature, top_p, top_k)
    print(response_text)

main()