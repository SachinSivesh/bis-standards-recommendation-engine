from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import ensure_index
from retriever import retrieve

app = Flask(__name__)
CORS(app)

print("[Backend] Loading RAG model and index... This happens only once.")
index, chunks, model = ensure_index()
print("[Backend] Initialization complete. Ready to serve.")

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'results': []})
            
        retrieved_standards = retrieve(
            query=query,
            index=index,
            chunks=chunks,
            model=model,
            top_k_chunks=200,
            top_n_standards=5,
            alpha=0.6
        )
        
        return jsonify({'results': retrieved_standards})
    except Exception as e:
        print(f"[Error] {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
