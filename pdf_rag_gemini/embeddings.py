import google.generativeai as genai
import config

# Configure Gemini with API key
genai.configure(api_key=config.GOOGLE_API_KEY)


# Generate embedding
res = genai.embed_content(
    model="models/text-embedding-004",
    content="the Godfather",
    task_type="retrieval_document"
)

# Print the embedding values
print(res["embedding"])

