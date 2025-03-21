from pinecone import Pinecone, ServerlessSpec
from src.llm.factory import ModelFactoryProvider

class RAG:
    def __init__(self, index_name: str):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        if index_name not in self.pc.list_indexes():
            self.create_pinecone_index(index_name)
        self.index = self.pc.Index(index_name)

    def query(self, query: str, top_k: int = 5) -> list[dict]:
        model_factory = ModelFactoryProvider.get_instance()
        vector = model_factory.create_embeddings(query)[0]
        response = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
        return response.matches
    
    def create_pinecone_index(self, index_name: str, dimension: int = 1536):
        """Create Pinecone index if it doesn't exist"""
        if index_name not in self.pc.list_indexes():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
