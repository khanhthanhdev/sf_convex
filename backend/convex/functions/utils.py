from datetime import datetime
from typing import Any, Dict
from convex import ConvexClient

# Initialize Convex client
client = ConvexClient("your-convex-deployment-url")  # Replace with your actual Convex deployment URL

def get_current_timestamp() -> float:
    """Get current timestamp in seconds since epoch."""
    return datetime.now().timestamp()

def update_document(collection: str, id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to update a document with versioning and timestamp."""
    current_time = get_current_timestamp()
    
    # Add/update metadata
    updates["updatedAt"] = current_time
    if "version" in updates:
        updates["version"] += 1
    
    return client.mutation(
        "updateDocument",
        {"collection": collection, "id": id, "updates": updates}
    )

def create_document(collection: str, document: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to create a new document with timestamps."""
    current_time = get_current_timestamp()
    
    # Add metadata
    document["createdAt"] = current_time
    document["updatedAt"] = current_time
    if "version" not in document:
        document["version"] = 1
    
    return client.mutation("createDocument", {"collection": collection, "document": document})
