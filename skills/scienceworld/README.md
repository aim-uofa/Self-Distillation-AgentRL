# ScienceWorld skill bank

Fixed English strategies, not dynamically generated plans or executable skills.
The teacher receives general skills plus one category selected by the dataset's
`scienceworld/<category>` data source. `SKILL_ALL=true` concatenates all eight
specializations; it is not recommended with a 4096-token budget. Ordinary student
rollouts and validation/final evaluation do not receive these skills.

The five test categories are measure, test_conductivity, find, chemistry_mix, and
lifespan. The exact 2120-task training pool also includes state changes, powering
components, and identifying life stages, so those three have their own files.
None of these files includes evaluation task IDs or task-specific answers.

## Provenance and adaptations

Summarized and adapted from [ZJUNLP SkillNet](https://github.com/zjunlp/SkillNet/tree/358ec824577e8a5aa6857c0eff268e492703606b/experiments/src/skills/scienceworld),
pinned at `358ec824577e8a5aa6857c0eff268e492703606b` (MIT; notice below).
The following source SKILL.md files were inspected for these summaries:

| Local file | SkillNet source directories (prefix `scienceworld-`) |
|---|---|
| general_skills | task-interpreter, task-focuser, room-navigator, room-scanner, ambiguous-action-resolution, container-inspector, container-transfer, inventory-manager, instruction-reader |
| measure | measurement-taker, temperature-measurer, threshold-evaluator, heating-apparatus-setup, substance-cooler |
| test_conductivity | conductivity-tester, circuit-builder, material-classifier |
| find | object-selector, living-entity-identifier, room-scanner, inventory-manager |
| chemistry_mix | mixture-creator, recipe-retriever, instruction-reader, liquid-filler, container-transfer |
| lifespan | animal-identifier, task-interpreter, object-selector |
| state_change | heating-apparatus-setup, substance-cooler, process-monitor |
| power | circuit-builder, conductivity-tester, process-monitor |
| life_stages | growth-focuser, living-entity-identifier, animal-identifier |

SkillNet's experiment uses `simplificationStr="easy"`; AgentGym does not. These
summaries therefore replace teleportation with normal navigation and remove
assumptions that all doors/containers are open. They generalize fixed object and
box colors, avoid copying instance answers, and distinguish observed physical
effects from focus acknowledgements. The conditional-box-placer source was also
inspected but its contradictory same-color branches were not copied. No SkillNet
generated-code execution, API calls, retrieval service, or automatic skill updates
are introduced. These are adaptations, not a claim to reproduce SkillNet's method.

## SkillNet license

MIT License

Copyright (c) 2026 ZJUNLP

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
