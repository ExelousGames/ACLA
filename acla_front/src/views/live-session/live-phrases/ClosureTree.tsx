import React, { useId, useRef, useState } from 'react';
import { ArrowLeftIcon, ChevronRightIcon, FileTextIcon } from '@radix-ui/react-icons';
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
                    ? <div className="live-phrases__condition-group" role="group" aria-label="Condition group">
                        <div className="live-phrases__condition-content">{content}</div>
                        <ConditionList conditions={condition.conditions} label="Grouped conditions" />
                    </div>
                    : <div className="live-phrases__condition-content">{content}</div>}
            </li>;
        })}
    </ul>;
}

function NodeIcon({ kind }: { kind: NodeSnapshot['kind'] }) {
    return kind === 'action' ? <FileTextIcon className="live-phrases__node-icon" aria-hidden="true" />
        : <svg className="live-phrases__node-icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
            <path d="M2.5 6V4.5h5l2 2h8v10h-15V6Z" strokeLinejoin="round" />
        </svg>;
}

const tabs = ['Contents', 'Conditions', 'Details'] as const;
type Tab = typeof tabs[number];

/** Browse snapshot IDs so live updates never change the directory the user is inspecting. */
export default function ClosureTree({ node }: { node: NodeSnapshot }) {
    const [path, setPath] = useState<string[]>([]);
    const [selectedTab, setSelectedTab] = useState<Tab | null>(null);
    const heading = useRef<HTMLHeadingElement>(null);
    const tabId = useId();
    const ancestors = [node];
    for (const id of path) {
        const child = ancestors[ancestors.length - 1].children.find((candidate) => candidate.id === id);
        if (!child) break;
        ancestors.push(child);
    }
    const current = ancestors[ancestors.length - 1];
    const parent = ancestors[ancestors.length - 2];
    const currentPath = ancestors.slice(1).map((ancestor) => ancestor.id);
    const isClosure = current.kind === 'closure';
    const activeTab = selectedTab ?? (isClosure ? 'Contents' : 'Details');
    const conditionLabel = isClosure ? current.name : `${parent?.name} action ${(parent?.children.findIndex((child) => child.id === current.id) ?? -1) + 1}: ${current.name}`;
    const navigate = (nextPath: string[]) => {
        setPath(nextPath);
        setSelectedTab(null);
        heading.current?.focus({ preventScroll: true });
    };

    return <div className="live-phrases__explorer" data-node-id={current.id} data-current={current.current} data-on-path={current.onPath}>
        <div className="live-phrases__directory-bar">
            <button type="button" className="live-phrases__back" aria-label="Go back one level" disabled={!parent}
                onClick={() => navigate(currentPath.slice(0, -1))}>
                <ArrowLeftIcon aria-hidden="true" /> Back
            </button>
            <nav className="live-phrases__breadcrumbs" aria-label="Phrase directory">
                <ol>{ancestors.map((ancestor, index) => <li key={ancestor.id}>
                    {index > 0 && <ChevronRightIcon aria-hidden="true" />}
                    {index === ancestors.length - 1
                        ? <span aria-current="page">{ancestor.name}</span>
                        : <button type="button" onClick={() => navigate(currentPath.slice(0, index))}>{ancestor.name}</button>}
                </li>)}</ol>
            </nav>
        </div>
        <div className="live-phrases__directory-heading">
            <NodeIcon kind={current.kind} />
            <div className="live-phrases__summary-text">
                <h3 ref={heading} tabIndex={-1}>{current.name}</h3>
                <span className="live-phrases__meta">{isClosure ? `Closure · ${current.children.length} items` : `Action · ${current.execution?.status ?? 'idle'}`}{current.current ? ' · Current' : ''}</span>
            </div>
            <span className="live-phrases__status" data-status={current.status} data-active={current.onPath}>{current.status}</span>
        </div>
        {current.execution?.error && <p className="live-phrases__error">{current.execution.error}</p>}
        <div className="live-phrases__tabs" role="tablist" aria-label="Directory information">
            {tabs.map((tab, index) => <button key={tab} type="button" role="tab" id={`${tabId}-${tab}`} aria-selected={activeTab === tab}
                aria-controls={`${tabId}-panel`} tabIndex={activeTab === tab ? 0 : -1}
                onClick={() => setSelectedTab(tab)} onKeyDown={(event) => {
                    const nextIndex = event.key === 'ArrowRight' ? (index + 1) % tabs.length
                        : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length
                            : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : null;
                    if (nextIndex === null) return;
                    event.preventDefault();
                    setSelectedTab(tabs[nextIndex]);
                    const buttons = event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>('[role="tab"]');
                    buttons?.[nextIndex].focus();
                }}>{tab}{tab === 'Contents' && <span className="live-phrases__meta">{current.children.length}</span>}</button>)}
        </div>
        <div className="live-phrases__tab-panel" role="tabpanel" id={`${tabId}-panel`} aria-labelledby={`${tabId}-${activeTab}`} tabIndex={0}>
            {activeTab === 'Contents' && (current.children.length > 0
                ? <ol className="live-phrases__catalog" aria-label={`${current.name} children`}>
                    {current.children.map((child, index) => <li key={child.id}>
                        <button type="button" className="live-phrases__item" data-current={child.current} data-on-path={child.onPath}
                            aria-label={child.kind === 'closure' ? `${child.name} closure` : `${current.name} action ${index + 1}: ${child.name}`}
                            onClick={() => navigate([...currentPath, child.id])}>
                            <span className="live-phrases__action-number" aria-hidden="true">{index + 1}</span>
                            <NodeIcon kind={child.kind} />
                            <span className="live-phrases__summary-text">
                                <strong>{child.name}</strong>
                                <span className="live-phrases__meta">{child.kind === 'closure' ? `${child.conditions.length} entry conditions · ${child.children.length} items` : `Action · ${child.execution?.status ?? 'idle'}`}{child.current ? ' · Current' : ''}</span>
                            </span>
                            <span className="live-phrases__status" data-status={child.status} data-active={child.onPath}>{child.status}</span>
                            <ChevronRightIcon className="live-phrases__item-chevron" aria-hidden="true" />
                        </button>
                    </li>)}
                </ol> : <p className="live-phrases__hint">No items in this {isClosure ? 'closure' : 'action'}.</p>)}
            {activeTab === 'Conditions' && <>
                <div className="live-phrases__condition-content">
                    <strong>{isClosure ? 'Entry conditions' : 'Action conditions'}</strong>
                    <span className="live-phrases__meta">{current.conditions.filter((condition) => condition.conditionFit).length}/{current.conditions.length} met</span>
                </div>
                {current.conditions.length > 0 ? <ConditionList conditions={current.conditions} label={`${conditionLabel} conditions`} />
                    : <p className="live-phrases__hint">No conditions.</p>}
            </>}
            {activeTab === 'Details' && <>
                <p>{current.description}</p>
                {Object.keys(current.fields).length > 0 && <dl className="live-phrases__fields">
                    {Object.entries(current.fields).map(([label, value]) => <React.Fragment key={label}>
                        <dt>{label}</dt><dd>{value === null ? '—' : String(value)}</dd>
                    </React.Fragment>)}
                </dl>}
            </>}
        </div>
    </div>;
}
