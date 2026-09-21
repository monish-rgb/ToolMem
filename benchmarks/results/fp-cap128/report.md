# MCPMark file_property LLM A/B

Model: "gemini-2.5-flash", temperature 0.0, k=4

| Task | Arm | Passed | Avg provider calls | Avg total calls | Avg tokens |
|---|---|---:|---:|---:|---:|
| size_classification | baseline | 0/4 | 17.75 | 17.75 | 14499.0 |
| size_classification | toolatlas | 4/4 | 24.5 | 25.5 | 45383.5 |
| time_classification | baseline | 0/4 | 21.25 | 21.25 | 23821.8 |
| time_classification | toolatlas | 0/4 | 22.75 | 23.75 | 24826.5 |
