<!-- codebase-memory-mcp:start -->
# Codebase Knowledge Graph (codebase-memory-mcp)

This project uses codebase-memory-mcp to maintain a knowledge graph of the codebase.
Aider has no MCP support, so invoke the graph through the CLI (e.g. via /run).
ALWAYS prefer these commands over grep/glob/file-search for code discovery.

## Priority Order (CLI form)
1. Find functions/classes/routes:
   codebase-memory-mcp cli search_graph '{"project":"<name>","name_pattern":".*Foo.*"}'
2. Who calls X / what does X call:
   codebase-memory-mcp cli trace_path '{"project":"<name>","function_name":"Foo","direction":"both"}'
3. Read a specific function/class:
   codebase-memory-mcp cli get_code_snippet '{"project":"<name>","qualified_name":"<qn>"}'
4. Complex patterns (Cypher):
   codebase-memory-mcp cli query_graph '{"project":"<name>","query":"MATCH ..."}'
5. Project overview:
   codebase-memory-mcp cli get_architecture '{"project":"<name>"}'

First use in a repo: codebase-memory-mcp cli index_repository '{"repo_path":"<abs path>"}'
List indexed projects (for <name>): codebase-memory-mcp cli list_projects '{}'

## When to fall back to grep/glob
- Searching for string literals, error messages, config values
- Searching non-code files (Dockerfiles, shell scripts, configs)
- When the CLI returns insufficient results
<!-- codebase-memory-mcp:end -->
