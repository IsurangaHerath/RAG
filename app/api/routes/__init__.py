"""API route handlers (section 5.6, section 6.2, Stages 53-57).

These are the only files that import FastAPI; everything upstream
(retrieval, generation, domain) stays framework-free. The routes never
build prompts or call the model themselves — they delegate to
`GroundedAnswerPipeline`, so the pipeline stays testable independently
of the web framework.

No real LLM is called in tests (`tests/unit/test_api_routes.py` uses the
stubbed pipeline); the route layer proves transport and validation, not
model quality.
"""
