# MCPMark file_property LLM A/B

Model: "gemini-2.5-flash", temperature 0.0, k=4

| Task | Arm | Passed | Avg provider calls | Avg total calls | Avg tokens |
|---|---|---:|---:|---:|---:|
| size_classification | baseline | 0/4 | 21.5 | 21.5 | 27536.0 |
| size_classification | toolatlas | 4/4 | 24.5 | 25.5 | 60269.2 |
| time_classification | baseline | 0/4 | 24.25 | 24.25 | 27688.5 |
| time_classification | toolatlas | 0/4 | 20.25 | 21.25 | 23400.2 |
