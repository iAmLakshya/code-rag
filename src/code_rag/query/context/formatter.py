from code_rag.query.context.models import EnrichedContext


def format_context_for_llm(enriched: EnrichedContext) -> str:
    sections = []

    sections.append("## Query Context\n")
    sections.append(f"**Intent**: {enriched.intent.value}\n")
    sections.append(f"**Summary**: {enriched.graph_summary}\n")

    if enriched.primary_contexts:
        sections.append("\n## Primary Entities\n")

        for i, ctx in enumerate(enriched.primary_contexts, 1):
            entity = ctx.entity
            sections.append(f"### {i}. {entity.name} ({entity.node_type})\n")
            sections.append(f"**File**: `{entity.file_path}`")

            if entity.start_line:
                sections.append(f" (lines {entity.start_line}-{entity.end_line})")
            sections.append("\n")

            if entity.signature:
                sections.append(f"**Signature**: `{entity.signature}`\n")

            if ctx.implementation_summary:
                sections.append(f"**Summary**: {ctx.implementation_summary}\n")

            if ctx.code_snippet:
                lang = ctx.code_snippet.language or ""
                sections.append(f"\n```{lang}\n{ctx.code_snippet.content}\n```\n")

            if ctx.caller_summaries:
                sections.append(f"\n**Called by**: {', '.join(ctx.caller_summaries[:3])}\n")

            if ctx.callee_summaries:
                sections.append(f"**Calls**: {', '.join(ctx.callee_summaries[:3])}\n")

            if ctx.related_entities:
                sections.append(f"**Related**: {', '.join(ctx.related_entities[:5])}\n")

    if enriched.call_chain_explanations:
        sections.append("\n## Call Chains\n")
        for explanation in enriched.call_chain_explanations:
            sections.append(f"- {explanation}\n")

    caller_info = []
    for ctx in enriched.primary_contexts:
        if ctx.caller_summaries:
            caller_info.extend(ctx.caller_summaries)
    if caller_info:
        sections.append("\n## Callers (Call Sites)\n")
        for caller in caller_info[:10]:
            sections.append(f"- {caller}\n")

    if enriched.hierarchy_explanations:
        sections.append("\n## Inheritance Hierarchy\n")
        for explanation in enriched.hierarchy_explanations:
            sections.append(f"- {explanation}\n")

    if enriched.code_snippets:
        sections.append("\n## Related Code\n")
        for snippet in enriched.code_snippets[:3]:
            sections.append(f"### {snippet.entity_name} ({snippet.entity_type})\n")
            sections.append(f"**File**: `{snippet.file_path}` (lines {snippet.start_line}-{snippet.end_line})\n")
            lang = snippet.language or ""
            sections.append(f"```{lang}\n{snippet.content}\n```\n")

    if enriched.file_summaries:
        sections.append("\n## Relevant Files\n")
        for file_path, summary in list(enriched.file_summaries.items())[:5]:
            sections.append(f"- {summary}\n")

    if enriched.reasoning_notes:
        sections.append("\n## Analysis Notes\n")
        for note in enriched.reasoning_notes:
            sections.append(f"- {note}\n")

    return "".join(sections)
