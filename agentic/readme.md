- agentic

  - good at tool calling and agentic loops,
    - decomposed & grounded complex user queries/tasks to tool/function
  - can call multiple tools in parallel and reliably,
    - llm  only to chose the proper tools,  no need to do any works by itself；
  - and knows when to stop (in the agentic loops)
    - max Iteration reached
    - mark_task_complete(decision made by LLM)
    - no more tool calls are needed(decision made by LLM)
