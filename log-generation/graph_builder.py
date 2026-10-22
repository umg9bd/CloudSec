"""
Action Dependency Graph Builder: Constructs PROPOSED state graphs for Stage 4 GNN analysis.

Explicitly maps tasks, proposed actions, AWS services, target resources, and predicted events
into a structured multigraph with PROPOSED edge states.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:
    from .schemas import (
        ActionDependencyGraph,
        GraphEdge,
        GraphNode,
        ParsedIntent,
        PredictedCloudTrailEvent,
        ProposedAction,
    )
except (ImportError, ValueError):
    from schemas import (
        ActionDependencyGraph,
        GraphEdge,
        GraphNode,
        ParsedIntent,
        PredictedCloudTrailEvent,
        ProposedAction,
    )

logger = logging.getLogger(__name__)


class ActionGraphBuilder:
    def build_graph(
        self,
        request_id: str,
        intent: ParsedIntent,
        proposed_actions: List[ProposedAction],
        predicted_events: List[PredictedCloudTrailEvent],
    ) -> ActionDependencyGraph:
        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []
        seen_node_ids = set()

        def add_node(node: GraphNode):
            if node.id not in seen_node_ids:
                nodes.append(node)
                seen_node_ids.add(node.id)

        # 1. Root Task Node
        task_node_id = f"task:{request_id}"
        add_node(GraphNode(
            id=task_node_id,
            type="Task",
            label=f"Task: {intent.task_category}",
            state="PROPOSED",
            properties={
                "task_category": intent.task_category,
                "desired_outcome": intent.desired_outcome,
            }
        ))

        # Map prediction_id to event
        event_by_act_id: Dict[str, PredictedCloudTrailEvent] = {
            ev.action_id: ev for ev in predicted_events
        }

        # 2. Build nodes and edges for each proposed action
        for action in proposed_actions:
            act_node_id = f"action:{action.action_id}"
            add_node(GraphNode(
                id=act_node_id,
                type="ProposedAction",
                label=f"Action: {action.operation}",
                state="PROPOSED",
                properties={
                    "step": action.step,
                    "action_id": action.action_id,
                    "service": action.service,
                    "operation": action.operation,
                    "iam_permission": action.iam_permission,
                    "mapping_rule_id": action.mapping_rule_id,
                }
            ))

            # Edge: Task PROPOSES Action
            edges.append(GraphEdge(
                source=task_node_id,
                target=act_node_id,
                relation="PROPOSES",
                state="PROPOSED",
                properties={"step": action.step}
            ))

            # Service Node
            svc_node_id = f"service:{action.service}"
            add_node(GraphNode(
                id=svc_node_id,
                type="Service",
                label=f"Service: {action.service}.amazonaws.com",
                state="PROPOSED",
                properties={"service_name": action.service}
            ))

            # Edge: Action INVOKES Service
            edges.append(GraphEdge(
                source=act_node_id,
                target=svc_node_id,
                relation="INVOKES",
                state="PROPOSED"
            ))

            # Resource Node (derived canonical identity matching feature_engine9)
            res_id = self._canonical_resource_id(action, intent)
            if res_id:
                res_node_id = f"resource:{res_id}"
                add_node(GraphNode(
                    id=res_node_id,
                    type="Resource",
                    label=f"Resource: {res_id}",
                    state="PROPOSED",
                    properties={"resource_identifier": res_id}
                ))

                # Edge: Action TARGETS Resource
                edges.append(GraphEdge(
                    source=act_node_id,
                    target=res_node_id,
                    relation="TARGETS",
                    state="PROPOSED"
                ))

            # Predicted Event Node
            if action.action_id in event_by_act_id:
                ev = event_by_act_id[action.action_id]
                ev_node_id = f"event:{ev.prediction_id}"
                add_node(GraphNode(
                    id=ev_node_id,
                    type="PredictedEvent",
                    label=f"PredictedEvent: {ev.eventName}",
                    state="PREDICTED",
                    properties={
                        "prediction_id": ev.prediction_id,
                        "eventSource": ev.eventSource,
                        "eventName": ev.eventName,
                        "step": ev.step,
                        "mapping_rule_id": ev.mapping_rule_id,
                    }
                ))

                # Edge: Action MAPS_TO_EVENT PredictedEvent
                edges.append(GraphEdge(
                    source=act_node_id,
                    target=ev_node_id,
                    relation="MAPS_TO_EVENT",
                    state="PROPOSED"
                ))

            # Action dependencies
            for dep_act_id in action.depends_on:
                dep_node_id = f"action:{dep_act_id}"
                edges.append(GraphEdge(
                    source=act_node_id,
                    target=dep_node_id,
                    relation="DEPENDS_ON",
                    state="PROPOSED"
                ))

        return ActionDependencyGraph(nodes=nodes, edges=edges)

    def _canonical_resource_id(self, action: ProposedAction, intent: ParsedIntent) -> Optional[str]:
        params = intent.provided_parameters
        if action.service == "s3":
            bucket = params.get("bucketName")
            if bucket:
                return f"arn:aws:s3:::{bucket}"
        elif action.service == "iam":
            role = params.get("roleName")
            if role:
                return f"arn:aws:iam:::role/{role}"
        elif action.service == "kms":
            return "arn:aws:kms:::key/<PROPOSED_KEY_ID>"
        elif action.service == "ec2":
            sg = params.get("groupName")
            if sg:
                return f"arn:aws:ec2:::security-group/{sg}"
        elif action.service == "lambda":
            fn = params.get("functionName")
            if fn:
                return f"arn:aws:lambda:::function:{fn}"
        return None
