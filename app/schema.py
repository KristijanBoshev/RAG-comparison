
from pydantic import BaseModel, Field

#Model that may be used in some scenarios
"""
class GradeDocuments(BaseModel):

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
    
    
    """
    
class Entities(BaseModel):
    """Identifying information about entities."""

    names: list[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )

        