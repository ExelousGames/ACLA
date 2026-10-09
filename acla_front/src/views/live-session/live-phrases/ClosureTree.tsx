import React from 'react';
import type { ConditionSnapshot, NodeSnapshot } from './closure';

function ConditionList({ conditions, label }: { conditions: readonly ConditionSnapshot[]; label: string }) {
    return <ul className="live-phrases__conditions" aria-label={label}>
        {conditions.map((condition, index) => {
            const status = condition.inputMissing ? 'Missing input' : condition.conditionFit === null ? 'Not evaluated' : condition.conditionFit ? 'Met' : 'Not met';
            const content = <>
                <span className="live-phrases__condition-label">
                    {index > 0 && condition.connector && <span className="live-phrases__connector" data-connector={condition.connector}>{condition.connector.toUpperCase()}</span>}
                    <span>{condition.description}{condition.conditions && <span className="live-phrases__meta"> ({condition.conditions.length} conditions)</span>}</span>
                </span>
                <span className="live-phrases__condition-status">{status}</span>
            </>;
            return <li key={index} className="live-phrases__condition" data-condition-fit={condition.conditionFit} data-input-missing={condition.inputMissing}>
                {condition.conditions
                    ? <details className="live-phrases__disclosure live-phrases__condition-group" role="group" aria-label="Condition group">
                        <summary className="live-phrases__condition-content" aria-label={`${label} group ${index + 1}`}>{content}</summary>
                        <ConditionList conditions={condition.conditions} label="Grouped conditions" />
                    </details>
                    : <div className="live-phrases__condition-content">{content}</div>}
            </li>;
        })}
    </ul>;
}

/** Composite renderer: only the structural kind matters, never a node's name or rule ID. */
export default function ClosureTree({ node, parentName, index = 0 }: { node: NodeSnapshot; parentName?: string; index?: number }) {
    const isRoot = parentName === undefined;
    const isClosure = node.kind === 'closure';
    const actionLabel = `${parentName} action ${index + 1}: ${node.name}`;
    const conditions = <ConditionList conditions={node.conditions} label={`${isClosure ? node.name : actionLabel} conditions`} />;
    const conditionHeading = <>
        <strong>{isClosure ? 'Entry conditions' : 'Action conditions'}</strong>
        <span className="live-phrases__meta">{node.conditions.filter((condition) => condition.conditionFit).length}/{node.conditions.length} met</span>
    </>;
    return <details className={`live-phrases__disclosure live-phrases__closure${isRoot ? ' live-phrases__root' : ''}`}
        open={isRoot} data-node-id={node.id} data-current={node.current} data-on-path={node.onPath}>
        <summary aria-label={isClosure ? `${node.name} closure` : actionLabel}>
            {!isRoot && <span className="live-phrases__action-number" aria-hidden="true">{index + 1}</span>}
            <span className="live-phrases__summary-text">
                {isRoot
                    ? <h3>{node.name} closure <span className="live-phrases__meta">({node.children.length} children)</span></h3>
                    : <strong>{node.name}</strong>}
                <span className="live-phrases__meta">{isClosure ? `${node.conditions.length} entry conditions · ${node.children.length} children` : `Action · ${node.execution?.status ?? 'idle'}`}{node.current ? ' · Current' : ''}</span>
            </span>
            <span className="live-phrases__status" data-status={node.status} data-active={node.onPath}>{node.status}</span>
        </summary>
        <div className="live-phrases__closure-body">
            <p>{node.description}</p>
            {Object.keys(node.fields).length > 0 && <dl className="live-phrases__fields">
                {Object.entries(node.fields).map(([label, value]) => <React.Fragment key={label}>
                    <dt>{label}</dt><dd>{value === null ? '—' : String(value)}</dd>
                </React.Fragment>)}
            </dl>}
            {node.execution?.error && <p className="live-phrases__error">{node.execution.error}</p>}
            {isClosure
                ? <details className="live-phrases__disclosure live-phrases__condition-section">
                    <summary aria-label={`${node.name} entry conditions`}>{conditionHeading}</summary>
                    {conditions}
                </details>
                : <div className="live-phrases__condition-section">
                    <div className="live-phrases__condition-content">{conditionHeading}</div>
                    {conditions}
                </div>}
            {node.children.length > 0 && <>
                <h4 className="live-phrases__actions-heading">Children <span className="live-phrases__meta">· {node.children.length} in order</span></h4>
                <ol className="live-phrases__catalog" aria-label={`${node.name} children`}>
                    {node.children.map((child, childIndex) => <li key={child.id}>
                        <ClosureTree node={child} parentName={node.name} index={childIndex} />
                    </li>)}
                </ol>
            </>}
        </div>
    </details>;
}
