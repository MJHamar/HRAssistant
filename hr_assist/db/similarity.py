"""
SQLAlchemy custom column element for pgvector similarity functions.
Provides sim_() factory function for vector similarity queries.
"""
from typing import List, Literal, Union, Any
from sqlalchemy.sql import ColumnElement
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.types import Float
from sqlalchemy import Column


class SimilarityFunction(ColumnElement):
    """
    Custom SQLAlchemy ColumnElement for pgvector similarity operations.

    Supports cosine, euclidean, and inner_product distance metrics
    compatible with pgvector extension operators.
    """
    inherit_cache = True

    def __init__(
        self,
        left: Column,
        right: Union[List[float], ColumnElement],
        metric: Literal["cosine", "euclidean", "inner_product"] = "cosine"
    ):
        """
        Initialize similarity function.

        Args:
            left: Database column containing vector (should be vector type)
            right: Query vector (list of floats) or another vector column
            metric: Distance metric - cosine, euclidean, or inner_product
        """
        self.left = left
        self.right = right
        self.metric = metric
        self.type = Float()

    def _copy_internals(self, clone=None, **kw):
        """Support for SQLAlchemy expression copying."""
        self.left = self.left._copy_internals(clone=clone, **kw)
        if hasattr(self.right, '_copy_internals'):
            self.right = self.right._copy_internals(clone=clone, **kw)


@compiles(SimilarityFunction)
def visit_similarity_function(element: SimilarityFunction, compiler, **kw):
    """
    Compile SimilarityFunction to PostgreSQL pgvector operators.

    Metric mapping:
    - cosine: <=> (cosine distance, 0 = identical, 2 = opposite)
    - euclidean: <-> (L2 distance)
    - inner_product: <#> (negative inner product, smaller = more similar)
    """
    left_expr = compiler.process(element.left, **kw)

    # Handle right side - could be parameter or another column
    if isinstance(element.right, list):
        # Convert list to PostgreSQL array literal with proper casting
        right_expr = f"ARRAY{element.right}::vector"
    else:
        right_expr = compiler.process(element.right, **kw)

    # Map metrics to pgvector operators
    operator_map = {
        "cosine": "<=>",
        "euclidean": "<->",
        "inner_product": "<#>"
    }

    operator = operator_map.get(element.metric)
    if not operator:
        raise ValueError(f"Unsupported similarity metric: {element.metric}")

    return f"({left_expr} {operator} {right_expr})"


def sim_(
    column: Column,
    vector: Union[List[float], ColumnElement],
    metric: Literal["cosine", "euclidean", "inner_product"] = "cosine"
) -> SimilarityFunction:
    """
    Factory function to create vector similarity expressions.

    Args:
        column: Database column containing vectors (should be vector type)
        vector: Query vector as list of floats or another vector column
        metric: Distance metric - "cosine", "euclidean", or "inner_product"

    Returns:
        SimilarityFunction that can be used in WHERE, ORDER BY, and SELECT clauses
    """
    return SimilarityFunction(column, vector, metric)


__all__ = ["SimilarityFunction", "sim_"]