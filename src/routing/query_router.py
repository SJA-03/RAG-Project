DOMAIN_KEYWORDS = {
    "os": {
        "operating system",
        "operating systems",
        "process",
        "thread",
        "cpu scheduling",
        "scheduler",
        "context switch",
        "pcb",
        "system call",
        "kernel",
        "deadlock",
        "paging",
        "virtual memory",
        "fork",
        "mutex",
        "semaphore",
    },
    "ml": {
        "machine learning",
        "regression",
        "classification",
        "svm",
        "decision tree",
        "random forest",
        "feature engineering",
        "overfitting",
        "underfitting",
        "cross validation",
        "supervised",
        "unsupervised",
    },
    "db": {
        "database",
        "sql",
        "join",
        "transaction",
        "normalization",
        "acid",
        "schema",
        "index",
        "query optimization",
        "er diagram",
        "relational",
        "dbms",
    },
    "ds": {
        "data structure",
        "array",
        "linked list",
        "stack",
        "queue",
        "tree",
        "graph",
        "hash table",
        "heap",
        "binary search tree",
        "sorting",
        "recursion",
    },
    "dl": {
        "deep learning",
        "neural network",
        "cnn",
        "rnn",
        "lstm",
        "transformer",
        "attention",
        "backpropagation",
        "gradient descent",
        "embedding",
        "dropout",
        "batch normalization",
    },
}


def route_query(query: str) -> str | None:
    normalized_query = query.lower().strip()
    if not normalized_query:
        return None

    best_domain: str | None = None
    best_score = 0

    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in normalized_query)
        if score > best_score:
            best_domain = domain
            best_score = score

    return best_domain if best_score > 0 else None
