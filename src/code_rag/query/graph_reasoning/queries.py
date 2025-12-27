class MultiHopGraphQueries:
    FIND_TRANSITIVE_CALLERS = """
    MATCH path = (caller)-[:CALLS*1..{max_hops}]->(target)
    WHERE target.name = $name
       OR target.qualified_name = $name
       OR target.name CONTAINS $name
       OR toLower(target.name) CONTAINS toLower($name)
    WITH path, caller, target, length(path) as depth
    WHERE caller <> target
    RETURN DISTINCT
        labels(caller)[0] as node_type,
        caller.name as name,
        caller.qualified_name as qualified_name,
        caller.file_path as file_path,
        caller.signature as signature,
        caller.docstring as docstring,
        caller.summary as summary,
        caller.start_line as start_line,
        caller.end_line as end_line,
        caller.is_async as is_async,
        depth,
        target.name as target_name
    ORDER BY depth
    LIMIT $limit
    """

    FIND_TRANSITIVE_CALLEES = """
    MATCH path = (source)-[:CALLS*1..{max_hops}]->(callee)
    WHERE source.name = $name
       OR source.qualified_name = $name
       OR source.name CONTAINS $name
       OR toLower(source.name) CONTAINS toLower($name)
    WITH path, source, callee, length(path) as depth
    WHERE callee <> source
    RETURN DISTINCT
        labels(callee)[0] as node_type,
        callee.name as name,
        callee.qualified_name as qualified_name,
        callee.file_path as file_path,
        callee.signature as signature,
        callee.docstring as docstring,
        callee.summary as summary,
        callee.start_line as start_line,
        callee.end_line as end_line,
        callee.is_async as is_async,
        depth,
        source.name as source_name
    ORDER BY depth
    LIMIT $limit
    """

    FIND_CALL_CHAIN = """
    MATCH path = shortestPath((source)-[:CALLS*1..{max_hops}]->(target))
    WHERE (source.name = $source_name OR source.qualified_name = $source_name)
    AND (target.name = $target_name OR target.qualified_name = $target_name)
    RETURN
        [node in nodes(path) | {{
            node_type: labels(node)[0],
            name: node.name,
            qualified_name: node.qualified_name,
            file_path: node.file_path,
            signature: node.signature,
            summary: node.summary
        }}] as path_nodes,
        length(path) as path_length
    LIMIT 5
    """

    FIND_ALL_PATHS = """
    MATCH path = (source)-[:CALLS*1..{max_hops}]->(target)
    WHERE (source.name = $source_name OR source.qualified_name = $source_name)
    AND (target.name = $target_name OR target.qualified_name = $target_name)
    WITH path, length(path) as path_length
    ORDER BY path_length
    LIMIT 10
    RETURN
        [node in nodes(path) | {{
            node_type: labels(node)[0],
            name: node.name,
            qualified_name: node.qualified_name,
            file_path: node.file_path,
            signature: node.signature,
            summary: node.summary
        }}] as path_nodes,
        path_length
    """

    FIND_FULL_HIERARCHY = """
    MATCH (target:Class)
    WHERE target.name = $name OR target.qualified_name = $name
    OPTIONAL MATCH ancestor_path = (target)-[:EXTENDS*1..5]->(ancestor:Class)
    OPTIONAL MATCH descendant_path = (descendant:Class)-[:EXTENDS*1..5]->(target)
    WITH target,
         collect(DISTINCT {
             node: ancestor,
             depth: length(ancestor_path),
             direction: 'ancestor'
         }) as ancestors,
         collect(DISTINCT {
             node: descendant,
             depth: length(descendant_path),
             direction: 'descendant'
         }) as descendants
    RETURN
        {
            node_type: 'Class',
            name: target.name,
            qualified_name: target.qualified_name,
            file_path: target.file_path,
            signature: target.signature,
            summary: target.summary
        } as target_node,
        [a in ancestors WHERE a.node IS NOT NULL | {
            node_type: 'Class',
            name: a.node.name,
            qualified_name: a.node.qualified_name,
            file_path: a.node.file_path,
            depth: a.depth,
            direction: a.direction
        }] + [d in descendants WHERE d.node IS NOT NULL | {
            node_type: 'Class',
            name: d.node.name,
            qualified_name: d.node.qualified_name,
            file_path: d.node.file_path,
            depth: d.depth,
            direction: d.direction
        }] as hierarchy_nodes
    """

    FIND_CLASS_WITH_METHODS = """
    MATCH (c:Class)
    WHERE c.name = $name OR c.qualified_name = $name
    OPTIONAL MATCH (c)-[:DEFINES_METHOD]->(m:Method)
    RETURN
        {
            node_type: 'Class',
            name: c.name,
            qualified_name: c.qualified_name,
            file_path: c.file_path,
            signature: c.signature,
            docstring: c.docstring,
            summary: c.summary,
            start_line: c.start_line,
            end_line: c.end_line
        } as class_node,
        collect({
            node_type: 'Method',
            name: m.name,
            qualified_name: m.qualified_name,
            file_path: m.file_path,
            signature: m.signature,
            docstring: m.docstring,
            summary: m.summary,
            start_line: m.start_line,
            end_line: m.end_line,
            is_async: m.is_async,
            is_static: m.is_static,
            is_classmethod: m.is_classmethod
        }) as methods
    """

    FIND_FILE_CONTEXT = """
    MATCH (f:File {path: $file_path})-[:DEFINES]->(entity)
    OPTIONAL MATCH (entity)-[:CALLS]->(callee)
    OPTIONAL MATCH (caller)-[:CALLS]->(entity)
    OPTIONAL MATCH (entity)-[:EXTENDS]->(parent:Class)
    OPTIONAL MATCH (child:Class)-[:EXTENDS]->(entity)
    RETURN
        {
            node_type: labels(entity)[0],
            name: entity.name,
            qualified_name: entity.qualified_name,
            file_path: entity.file_path,
            signature: entity.signature,
            docstring: entity.docstring,
            summary: entity.summary,
            start_line: entity.start_line,
            end_line: entity.end_line
        } as entity_node,
        count(DISTINCT callee) as callee_count,
        count(DISTINCT caller) as caller_count,
        collect(DISTINCT parent.qualified_name)[0] as parent_class,
        count(DISTINCT child) as child_count
    ORDER BY entity.start_line
    """

    FIND_IMPLEMENTATION_CONTEXT = """
    MATCH (entity)
    WHERE entity.name = $name OR entity.qualified_name = $name

    // Get direct callers
    OPTIONAL MATCH (caller)-[:CALLS]->(entity)
    WITH entity, collect(DISTINCT {
        node_type: labels(caller)[0],
        name: caller.name,
        qualified_name: caller.qualified_name,
        file_path: caller.file_path,
        summary: caller.summary
    })[0..10] as callers

    // Get direct callees
    OPTIONAL MATCH (entity)-[:CALLS]->(callee)
    WITH entity, callers, collect(DISTINCT {
        node_type: labels(callee)[0],
        name: callee.name,
        qualified_name: callee.qualified_name,
        file_path: callee.file_path,
        summary: callee.summary
    })[0..10] as callees

    // Get containing file's other entities
    OPTIONAL MATCH (f:File {path: entity.file_path})-[:DEFINES]->(sibling)
    WHERE sibling <> entity
    WITH entity, callers, callees, collect(DISTINCT {
        node_type: labels(sibling)[0],
        name: sibling.name,
        qualified_name: sibling.qualified_name,
        summary: sibling.summary,
        start_line: sibling.start_line
    })[0..10] as siblings

    RETURN
        {
            node_type: labels(entity)[0],
            name: entity.name,
            qualified_name: entity.qualified_name,
            file_path: entity.file_path,
            signature: entity.signature,
            docstring: entity.docstring,
            summary: entity.summary,
            start_line: entity.start_line,
            end_line: entity.end_line,
            is_async: entity.is_async,
            parent_class: entity.parent_class
        } as entity_node,
        callers,
        callees,
        siblings
    """

    FIND_USAGE_PATTERNS = """
    MATCH (user)-[:CALLS]->(target)
    WHERE target.name = $name OR target.qualified_name = $name
    WITH user, target
    OPTIONAL MATCH (user)-[:CALLS]->(other_callee)
    WHERE other_callee <> target
    WITH user, target, collect(DISTINCT other_callee.name)[0..5] as context_calls
    RETURN
        {
            node_type: labels(user)[0],
            name: user.name,
            qualified_name: user.qualified_name,
            file_path: user.file_path,
            signature: user.signature,
            summary: user.summary,
            start_line: user.start_line,
            end_line: user.end_line
        } as user_node,
        context_calls
    ORDER BY user.file_path, user.start_line
    LIMIT $limit
    """

    FIND_MODULE_DEPENDENCIES = """
    MATCH (f:File {path: $file_path})-[:IMPORTS]->(i:Import)
    OPTIONAL MATCH (imported_entity)
    WHERE imported_entity.name = i.name OR imported_entity.qualified_name CONTAINS i.name
    RETURN
        i.name as import_name,
        i.source as import_source,
        i.is_external as is_external,
        {
            node_type: labels(imported_entity)[0],
            name: imported_entity.name,
            qualified_name: imported_entity.qualified_name,
            file_path: imported_entity.file_path,
            summary: imported_entity.summary
        } as resolved_entity
    """

    FIND_ENTITY_FUZZY = """
    MATCH (n)
    WHERE n.name CONTAINS $name
       OR n.qualified_name CONTAINS $name
       OR toLower(n.name) CONTAINS toLower($name)
       OR toLower(n.qualified_name) CONTAINS toLower($name)
    WITH n,
         CASE
             WHEN n.name = $name THEN 0
             WHEN n.qualified_name = $name THEN 1
             WHEN toLower(n.name) = toLower($name) THEN 2
             WHEN n.name STARTS WITH $name THEN 3
             WHEN n.name ENDS WITH $name THEN 4
             WHEN n.name CONTAINS $name THEN 5
             ELSE 6
         END as match_score
    ORDER BY match_score, n.name
    LIMIT $limit
    RETURN
        labels(n)[0] as node_type,
        n.name as name,
        n.qualified_name as qualified_name,
        n.file_path as file_path,
        n.signature as signature,
        n.docstring as docstring,
        n.summary as summary,
        n.start_line as start_line,
        n.end_line as end_line,
        n.is_async as is_async,
        n.parent_class as parent_class,
        match_score
    """

    GET_ENTITY_CENTRALITY = """
    MATCH (n)
    WHERE n.name = $name OR n.qualified_name = $name
    OPTIONAL MATCH (caller)-[:CALLS]->(n)
    WITH n, count(DISTINCT caller) as in_degree
    OPTIONAL MATCH (n)-[:CALLS]->(callee)
    WITH n, in_degree, count(DISTINCT callee) as out_degree
    OPTIONAL MATCH (n)-[r]-()
    RETURN
        n.name as name,
        n.qualified_name as qualified_name,
        in_degree,
        out_degree,
        in_degree + out_degree as total_degree,
        count(DISTINCT r) as relationship_count
    """
