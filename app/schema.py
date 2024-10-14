"""
from pydantic import BaseModel, Field

class GradeDocuments(BaseModel):

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
    
    
    """
    #Model that may be used in some scenarios