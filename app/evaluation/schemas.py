from pydantic import BaseModel, Field


class EvaluationTestCase(BaseModel):
    # Defines the benchmark question, expected answer, keywords, and category.
    question: str
    keywords: list[str] = Field(default_factory=list)
    reference_answer: str
    category: str


class RetrievalMetrics(BaseModel):
    # Stores the existing deterministic keyword-based retrieval metrics.
    mrr: float = 0.0
    ndcg: float = 0.0
    keyword_coverage: float = 0.0
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0


class SemanticRetrievalMetrics(BaseModel):
    # Stores semantic retrieval metrics based on evidence relevance judgments.
    mrr: float = 0.0
    ndcg: float = 0.0
    recall_at_k: float = 0.0
    evidence_support_rate: float = 0.0


class AnswerMetrics(BaseModel):
    # Stores metrics used to evaluate the quality of the generated answer.
    accuracy: float | None = None
    completeness: float | None = None
    relevance: float | None = None


class RetrievedEvidence(BaseModel):
    # Represents a raw chunk returned by the retrieval and reranking pipeline.
    document: str
    source: str = ""
    distance: float | None = None


class EvidenceJudgment(BaseModel):
    # Stores whether one retrieved chunk supports the requested answer.
    rank: int
    source: str = ""
    support_probability: float = 0.0
    supports_answer: bool = False


class SubqueryEvaluation(BaseModel):
    # Stores retrieval evaluation results for one generated subquery.

    subquery: str

    target_keywords: list[str] = Field(
        default_factory=list
    )

    mrr: float = 0.0

    recall_at_k: float = 0.0

    evidence: list[RetrievedEvidence] = Field(
        default_factory=list
    )

    matched_keywords: list[str] = Field(
        default_factory=list
    )

    semantic: SemanticRetrievalMetrics = Field(
        default_factory=SemanticRetrievalMetrics
    )

    evidence_judgments: list[EvidenceJudgment] = Field(
        default_factory=list
    )


class EvaluationResult(BaseModel):
    # Stores the complete evaluation result for one benchmark question.

    question: str
    category: str

    reference_answer: str
    generated_answer: str = ""

    retrieval: RetrievalMetrics

    semantic_retrieval: SemanticRetrievalMetrics = Field(
        default_factory=SemanticRetrievalMetrics
    )

    answer: AnswerMetrics | None = None

    evidence: list[RetrievedEvidence] = Field(
        default_factory=list
    )

    sources: list[str] = Field(
        default_factory=list
    )

    subqueries: list[SubqueryEvaluation] = Field(
        default_factory=list
    )

    subquery_mrr: float = 0.0
    subquery_recall_at_k: float = 0.0
    subquery_coverage: float = 0.0

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0

    latency_ms: float = 0.0

    passed: bool = False

    error: str | None = None