# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np

# Database connection details
db_params = {
    "host": "localhost",
    "database": "surajdharmapuram",
    "user": "surajdharmapuram",
    "password": "surajdharmapuram",
    "port": "5432"
}

# Connect to PostgreSQL
conn = psycopg2.connect(**db_params)
cur = conn.cursor()

# Fetch product descriptions
cur.execute("SELECT id, description FROM products")
products = cur.fetchall()

# Extract descriptions
descriptions = [product[1] for product in products]
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(descriptions)

# Update the database with embeddings
for i, product in enumerate(products):
    product_id = product[0]
    print(product)
    embedding = embeddings[i] # Convert to Python list
    print(embedding.shape)
    # Construct the vector string representation
    embedding_str = str(embedding.tolist())
    cur.execute("UPDATE products SET embedding = %s WHERE id = %s", (embedding_str, product_id))

# Commit changes and close connection
conn.commit()


# The query string
query_string = 'baby sleeping accessories'

# Generate embedding for the query string
query_embedding = model.encode(query_string).tolist()

# Construct the SQL query using the cosine similarity operator (<->)
# Assuming you have an index that supports cosine similarity (e.g., ivfflat with vector_cosine_ops)
sql_query = """
    SELECT id, description, (embedding <-> %s::vector) as similarity
    FROM products
    ORDER BY similarity
    LIMIT 5;
"""

# Execute the query
cur.execute(sql_query, (query_embedding,))

# Fetch and print the results
results = cur.fetchall()
for result in results:
    product_id, description, similarity = result
    print(f"ID: {product_id}, Description: {description}, Similarity: {similarity}")

cur.close()
conn.close()



print("Embeddings generated and updated successfully!")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
