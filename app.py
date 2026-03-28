import itertools
import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Sequence, Set, Tuple

import streamlit as st


@dataclass(frozen=True)
class FunctionalDependency:
    lhs: FrozenSet[str]
    rhs: FrozenSet[str]


DEFAULT_SCHEMA = "R(FlightNo, PilotID, PilotName, GateNo, Terminal, PassengerID, SeatNo, TicketID)"
DEFAULT_FDS = (
    "FlightNo -> PilotID, GateNo\n"
    "PilotID -> PilotName\n"
    "GateNo -> Terminal\n"
    "TicketID -> FlightNo, PassengerID, SeatNo"
)


def powerset(items: Sequence[str]) -> Iterable[Tuple[str, ...]]:
    for r in range(len(items) + 1):
        yield from itertools.combinations(items, r)


def parse_relation_schema(schema_text: str) -> List[str]:
    text = schema_text.strip()
    if not text:
        return []

    match = re.search(r"\((.*?)\)", text)
    if match:
        inner = match.group(1)
    else:
        inner = text

    attrs = [a.strip() for a in inner.split(",") if a.strip()]
    return list(dict.fromkeys(attrs))


def parse_functional_dependencies(fd_text: str) -> List[FunctionalDependency]:
    lines = [line.strip() for line in fd_text.splitlines() if line.strip()]
    fds: List[FunctionalDependency] = []

    for line in lines:
        if "->" not in line:
            continue
        lhs_text, rhs_text = [part.strip() for part in line.split("->", 1)]
        lhs_attrs = frozenset([a.strip() for a in lhs_text.split(",") if a.strip()])
        rhs_attrs = [a.strip() for a in rhs_text.split(",") if a.strip()]
        if not lhs_attrs:
            continue
        for rhs in rhs_attrs:
            fds.append(FunctionalDependency(lhs=lhs_attrs, rhs=frozenset([rhs])))

    deduped = list(dict.fromkeys(fds))
    return deduped


def closure(attributes: Set[str], fds: Sequence[FunctionalDependency]) -> Set[str]:
    result = set(attributes)
    changed = True

    while changed:
        changed = False
        for fd in fds:
            if fd.lhs.issubset(result) and not fd.rhs.issubset(result):
                result |= set(fd.rhs)
                changed = True

    return result


def closure_map_for_single_attributes(attrs: Sequence[str], fds: Sequence[FunctionalDependency]) -> Dict[str, Set[str]]:
    return {a: closure({a}, fds) for a in attrs}


def minimal_cover(fds: Sequence[FunctionalDependency]) -> List[FunctionalDependency]:
    cover = [FunctionalDependency(fd.lhs, frozenset(fd.rhs)) for fd in fds]

    # Remove extraneous attributes from LHS.
    reduced: List[FunctionalDependency] = []
    for fd in cover:
        lhs = set(fd.lhs)
        rhs = set(fd.rhs)
        for attr in list(lhs):
            trial_lhs = frozenset(lhs - {attr})
            if not trial_lhs:
                continue
            other = [FunctionalDependency(f.lhs, f.rhs) for f in cover if f != fd]
            other.append(FunctionalDependency(trial_lhs, frozenset(rhs)))
            if rhs.issubset(closure(set(trial_lhs), other)):
                lhs.remove(attr)
        reduced.append(FunctionalDependency(frozenset(lhs), frozenset(rhs)))

    # Remove redundant dependencies.
    final_cover = reduced.copy()
    for fd in reduced:
        trial = [f for f in final_cover if f != fd]
        if set(fd.rhs).issubset(closure(set(fd.lhs), trial)):
            final_cover = trial

    return list(dict.fromkeys(final_cover))


def find_candidate_keys(attributes: Sequence[str], fds: Sequence[FunctionalDependency]) -> List[Set[str]]:
    all_attrs = set(attributes)
    if not all_attrs:
        return []

    rhs_attrs = set().union(*(set(fd.rhs) for fd in fds)) if fds else set()
    must_have = all_attrs - rhs_attrs
    remaining = sorted(list(all_attrs - must_have))

    keys: List[Set[str]] = []

    for r in range(len(remaining) + 1):
        for combo in itertools.combinations(remaining, r):
            trial = set(must_have) | set(combo)
            if any(existing.issubset(trial) for existing in keys):
                continue
            if closure(trial, fds) == all_attrs:
                keys.append(trial)

    return sorted(keys, key=lambda k: (len(k), sorted(k)))


def is_superkey(attrs: Set[str], relation_attrs: Set[str], fds: Sequence[FunctionalDependency]) -> bool:
    return closure(attrs, fds).issuperset(relation_attrs)


def fd_to_text(fd: FunctionalDependency) -> str:
    lhs = ", ".join(sorted(fd.lhs))
    rhs = ", ".join(sorted(fd.rhs))
    return f"{lhs} -> {rhs}"


def relation_to_text(name: str, attrs: Set[str]) -> str:
    return f"{name}({', '.join(sorted(attrs))})"


def non_prime_attributes(attributes: Set[str], candidate_keys: Sequence[Set[str]]) -> Set[str]:
    prime = set().union(*candidate_keys) if candidate_keys else set()
    return attributes - prime


def find_2nf_violations(
    attributes: Set[str],
    candidate_keys: Sequence[Set[str]],
    fds: Sequence[FunctionalDependency],
) -> List[Tuple[Set[str], str]]:
    if not candidate_keys:
        return []

    np_attrs = non_prime_attributes(attributes, candidate_keys)
    violations: List[Tuple[Set[str], str]] = []

    for key in candidate_keys:
        if len(key) <= 1:
            continue
        for subset in powerset(sorted(list(key))):
            subset_set = set(subset)
            if not subset_set or subset_set == key:
                continue
            subset_closure = closure(subset_set, fds)
            for attr in np_attrs:
                if attr in subset_closure:
                    violations.append((subset_set, attr))

    # Deduplicate results.
    unique = []
    seen = set()
    for lhs, rhs in violations:
        signature = (tuple(sorted(lhs)), rhs)
        if signature not in seen:
            seen.add(signature)
            unique.append((lhs, rhs))
    return unique


def find_3nf_violations(
    attributes: Set[str],
    candidate_keys: Sequence[Set[str]],
    fds: Sequence[FunctionalDependency],
) -> List[FunctionalDependency]:
    prime = set().union(*candidate_keys) if candidate_keys else set()
    violations: List[FunctionalDependency] = []

    for fd in fds:
        lhs = set(fd.lhs)
        rhs = set(fd.rhs)
        if rhs.issubset(lhs):
            continue
        if is_superkey(lhs, attributes, fds):
            continue
        if all(a in prime for a in rhs):
            continue
        violations.append(fd)

    return violations


def find_bcnf_violations(attributes: Set[str], fds: Sequence[FunctionalDependency]) -> List[FunctionalDependency]:
    violations = []
    for fd in fds:
        lhs = set(fd.lhs)
        rhs = set(fd.rhs)
        if rhs.issubset(lhs):
            continue
        if not is_superkey(lhs, attributes, fds):
            violations.append(fd)
    return violations


def synthesize_3nf(attributes: Set[str], fds: Sequence[FunctionalDependency], candidate_keys: Sequence[Set[str]]) -> Dict[str, Set[str]]:
    min_cov = minimal_cover(fds)
    relation_sets: List[Set[str]] = []

    for fd in min_cov:
        relation_sets.append(set(fd.lhs) | set(fd.rhs))

    # Remove subsets.
    cleaned: List[Set[str]] = []
    for rel in relation_sets:
        if not any(rel < other for other in relation_sets):
            cleaned.append(rel)

    if candidate_keys and not any(any(key.issubset(rel) for key in candidate_keys) for rel in cleaned):
        cleaned.append(set(candidate_keys[0]))

    result: Dict[str, Set[str]] = {}
    for idx, rel in enumerate(cleaned, start=1):
        result[f"T3_{idx}"] = rel

    if not result:
        result["T3_1"] = set(attributes)

    return result


def decompose_2nf(
    attributes: Set[str],
    candidate_keys: Sequence[Set[str]],
    fds: Sequence[FunctionalDependency],
) -> Dict[str, Set[str]]:
    violations = find_2nf_violations(attributes, candidate_keys, fds)
    if not violations:
        return {"T2_1": set(attributes)}

    main_attrs = set(attributes)
    relations: Dict[str, Set[str]] = {}

    for idx, (lhs, rhs) in enumerate(violations, start=1):
        rel_attrs = set(lhs) | {rhs}
        relations[f"T2_{idx}"] = rel_attrs
        if rhs in main_attrs:
            main_attrs.remove(rhs)

    relations[f"T2_{len(relations) + 1}"] = main_attrs
    return relations


def project_fds_to_relation(attrs: Set[str], fds: Sequence[FunctionalDependency]) -> List[FunctionalDependency]:
    return [fd for fd in fds if set(fd.lhs).issubset(attrs) and set(fd.rhs).issubset(attrs)]


def bcnf_decompose_recursive(
    relation_attrs: Set[str],
    fds: Sequence[FunctionalDependency],
    name_prefix: str,
    counter: List[int],
) -> Dict[str, Set[str]]:
    local_fds = project_fds_to_relation(relation_attrs, fds)
    violations = find_bcnf_violations(relation_attrs, local_fds)

    if not violations:
        counter[0] += 1
        return {f"{name_prefix}_{counter[0]}": relation_attrs}

    fd = violations[0]
    x = set(fd.lhs)
    y = set(fd.rhs)

    r1 = x | y
    r2 = relation_attrs - (y - x)

    out: Dict[str, Set[str]] = {}
    out.update(bcnf_decompose_recursive(r1, fds, name_prefix, counter))
    out.update(bcnf_decompose_recursive(r2, fds, name_prefix, counter))
    return out


def check_1nf(attributes: Set[str]) -> Tuple[bool, List[str]]:
    warnings = []
    for a in attributes:
        if any(token in a.lower() for token in ["list", "set", "array", "multi"]):
            warnings.append(
                f"Possible repeating-group signal in attribute name '{a}'. Consider atomic values only."
            )

    if warnings:
        return False, warnings
    return True, ["No obvious non-atomic attributes detected from naming; assumed 1NF."]


def style_app() -> None:
    st.markdown(
        """
        <style>
            :root {
                --aviation-blue: #0d3b66;
                --sky-blue: #5da9e9;
                --cloud-white: #f7fbff;
                --runway-gray: #dce6f2;
            }
            .stApp {
                background: linear-gradient(180deg, #eef6ff 0%, #ffffff 60%);
            }
            header[data-testid="stHeader"] {
                display: none;
            }
            [data-testid="stToolbar"] {
                display: none;
            }
            [data-testid="stDecoration"] {
                display: none;
            }
            #MainMenu {
                display: none;
            }
            h1, h2, h3, h4 {
                color: var(--aviation-blue) !important;
                letter-spacing: 0.2px;
            }
            .page-title {
                color: var(--aviation-blue);
                font-size: 2.1rem;
                font-weight: 700;
                line-height: 1.2;
                margin: 0.5rem 0 0.2rem 0;
            }
            .section-title {
                color: var(--aviation-blue);
                font-size: 1.55rem;
                font-weight: 700;
                margin: 0.8rem 0 0.35rem 0;
            }
            .subsection-title {
                color: var(--aviation-blue);
                font-size: 1.2rem;
                font-weight: 700;
                margin: 0.75rem 0 0.3rem 0;
            }
            .block-container {
                padding-top: 0.9rem;
            }
            .airport-card {
                background: var(--cloud-white);
                border: 1px solid #cfe1f5;
                border-left: 5px solid var(--sky-blue);
                border-radius: 10px;
                padding: 0.9rem 1rem;
                margin: 0.5rem 0 1rem 0;
            }
            .success-card {
                background: #ecfdf3;
                border: 1px solid #8fd2a8;
                border-left: 6px solid #1e7b34;
                border-radius: 10px;
                padding: 0.8rem 1rem;
                margin: 0.6rem 0 0.8rem 0;
                color: #135a27;
                font-weight: 700;
            }
            .key-summary-card {
                background: #eef6ff;
                border: 1px solid #b9d6f2;
                border-left: 6px solid #0d3b66;
                border-radius: 10px;
                padding: 0.8rem 1rem;
                margin: 0.5rem 0 1rem 0;
                color: #0b2f57;
            }
            .key-summary-card strong {
                color: #0d3b66;
            }
            .nf-ok {
                color: #1e7b34;
                font-weight: 600;
            }
            .nf-bad {
                color: #b42318;
                font-weight: 600;
            }
            code {
                color: #0b2f57 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_relation_set(title: str, relations: Dict[str, Set[str]]) -> None:
    st.markdown(f'<div class="subsection-title">{title}</div>', unsafe_allow_html=True)
    for name, attrs in relations.items():
        st.code(relation_to_text(name, attrs), language="text")


def infer_fk_attributes(relations: Dict[str, Set[str]], relation_keys: Dict[str, List[Set[str]]]) -> Dict[str, Set[str]]:
    fk_by_relation: Dict[str, Set[str]] = {name: set() for name in relations}
    names = list(relations.keys())

    for r1 in names:
        for r2 in names:
            if r1 == r2:
                continue
            r1_attrs = relations[r1]
            for key in relation_keys.get(r2, []):
                if key and key.issubset(r1_attrs):
                    fk_by_relation[r1] |= set(key)

    return fk_by_relation


def describe_table(attrs: Set[str]) -> str:
    top = sorted(attrs)[:2]
    if not top:
        return "Details"
    return f"{' & '.join(top)} Details"


def render_schema_grid(title: str, relations: Dict[str, Set[str]], fds: Sequence[FunctionalDependency]) -> None:
    st.markdown(f'<div class="subsection-title">{title}</div>', unsafe_allow_html=True)
    relation_keys: Dict[str, List[Set[str]]] = {}

    for name, attrs in relations.items():
        projected = project_fds_to_relation(attrs, fds)
        relation_keys[name] = find_candidate_keys(sorted(attrs), projected)

    fk_attrs = infer_fk_attributes(relations, relation_keys)
    items = list(relations.items())
    columns_per_row = 3 if len(items) >= 5 else 2

    for i in range(0, len(items), columns_per_row):
        row_items = items[i : i + columns_per_row]
        cols = st.columns(columns_per_row)

        for col, (name, attrs) in zip(cols, row_items):
            with col:
                st.markdown(f"**Table {name}: {describe_table(attrs)}**")
                primary_key = relation_keys.get(name, [set()])[0] if relation_keys.get(name) else set()
                rows = []

                for attr in sorted(attrs):
                    roles = []
                    if attr in primary_key:
                        roles.append("PK")
                    if attr in fk_attrs.get(name, set()):
                        roles.append("FK")
                    rows.append(
                        {
                            "Attribute": attr,
                            "Key Role": ", ".join(roles) if roles else "Non-key",
                        }
                    )

                st.dataframe(rows, use_container_width=True, hide_index=True)
                if primary_key:
                    st.caption(f"Primary Key: ({', '.join(sorted(primary_key))})")


def render_final_schema_answer(relations: Dict[str, Set[str]], fds: Sequence[FunctionalDependency]) -> None:
    st.markdown('<div class="subsection-title">Final Schema Answer</div>', unsafe_allow_html=True)
    st.success("BCNF final schema generated. Use this as the final normalized design.")

    relation_keys: Dict[str, List[Set[str]]] = {}
    for name, attrs in relations.items():
        projected = project_fds_to_relation(attrs, fds)
        relation_keys[name] = find_candidate_keys(sorted(attrs), projected)

    key_groups: Dict[Tuple[str, ...], List[str]] = {}
    for name, keys in relation_keys.items():
        if keys:
            key_sig = tuple(sorted(keys[0]))
            key_groups.setdefault(key_sig, []).append(name)

    rows = []
    for name, attrs in sorted(relations.items()):
        key = relation_keys.get(name, [])
        pk = tuple(sorted(key[0])) if key else tuple()
        pk_text = ", ".join(pk) if pk else "Not inferred"
        attrs_text = ", ".join(sorted(attrs))

        join_hint = "No explicit FK inferred"
        if pk and len(attrs) > len(pk):
            peers = [t for t in key_groups.get(pk, []) if t != name]
            if peers:
                join_hint = f"Join on ({pk_text}) with: {', '.join(sorted(peers))}"

        rows.append(
            {
                "Table": name,
                "Attributes": attrs_text,
                "Primary Key": pk_text,
                "Join Hint": join_hint,
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)


def run_normalization_analysis(attributes: List[str], fds: List[FunctionalDependency]) -> Dict[str, object]:
    attribute_set = set(attributes)
    candidate_keys = find_candidate_keys(attributes, fds)
    primary_key = candidate_keys[0] if candidate_keys else set()

    closure_map = closure_map_for_single_attributes(attributes, fds)
    rhs_attrs = set().union(*(set(fd.rhs) for fd in fds)) if fds else set()
    must_have = attribute_set - rhs_attrs
    maybe_part = attribute_set - must_have

    base_relation = {"T0": attribute_set}

    is_1nf, nf1_notes = check_1nf(attribute_set)
    violations_2nf = find_2nf_violations(attribute_set, candidate_keys, fds)
    relations_2nf = decompose_2nf(attribute_set, candidate_keys, fds)

    min_cov = minimal_cover(fds)
    violations_3nf = find_3nf_violations(attribute_set, candidate_keys, min_cov)
    relations_3nf = synthesize_3nf(attribute_set, fds, candidate_keys)

    violations_bcnf = find_bcnf_violations(attribute_set, min_cov)
    relations_bcnf = bcnf_decompose_recursive(attribute_set, min_cov, "TB", [0])

    return {
        "attributes": attributes,
        "attribute_set": attribute_set,
        "fds": fds,
        "candidate_keys": candidate_keys,
        "primary_key": primary_key,
        "closure_map": closure_map,
        "must_have": must_have,
        "maybe_part": maybe_part,
        "base_relation": base_relation,
        "is_1nf": is_1nf,
        "nf1_notes": nf1_notes,
        "violations_2nf": violations_2nf,
        "relations_2nf": relations_2nf,
        "violations_3nf": violations_3nf,
        "relations_3nf": relations_3nf,
        "violations_bcnf": violations_bcnf,
        "relations_bcnf": relations_bcnf,
    }


def filter_fds_to_schema(fds: Sequence[FunctionalDependency], attrs: Set[str]) -> List[FunctionalDependency]:
    return [fd for fd in fds if set(fd.lhs).issubset(attrs) and set(fd.rhs).issubset(attrs)]


def main() -> None:
    st.set_page_config(page_title="Airport DB Normalization", page_icon=":airplane_departure:", layout="wide")
    style_app()

    st.markdown('<div class="page-title">International Airport Database Normalization</div>', unsafe_allow_html=True)
    st.caption("Interactive, step-by-step normalization from closure discovery to BCNF decomposition.")

    defaults = {
        "relation_input": DEFAULT_SCHEMA,
        "fd_input": DEFAULT_FDS,
        "analysis_ready": False,
        "analysis": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    tabs = st.tabs(["Problem Input", "Key Discovery", "Normalization Steps"])

    with tabs[0]:
        left_col, right_col = st.columns(2)
        with left_col:
            relation_input = st.text_input(
                "Relation Schema",
                key="relation_input",
                help="Format: R(A, B, C) or A, B, C",
            )
        with right_col:
            fd_input = st.text_area(
                "Functional Dependencies",
                key="fd_input",
                height=185,
                help="One FD per line. Example: Flight -> Pilot, Gate",
            )

        st.caption("Use commas between attributes and one dependency per line.")
        start_clicked = st.button("Start Normalization Engine", type="primary", use_container_width=True)

        if start_clicked:
            with st.spinner("Running normalization engine..."):
                parsed_attributes = parse_relation_schema(relation_input)
                parsed_fds = parse_functional_dependencies(fd_input)

                if not parsed_attributes:
                    st.session_state.analysis_ready = False
                    st.session_state.analysis = {}
                    st.error("Please provide a valid relation schema with at least one attribute.")
                else:
                    attribute_set = set(parsed_attributes)
                    parsed_fds = filter_fds_to_schema(parsed_fds, attribute_set)

                    analysis = run_normalization_analysis(parsed_attributes, parsed_fds)

                    st.session_state.analysis = analysis
                    st.session_state.analysis_ready = True

            if st.session_state.analysis_ready:
                st.markdown(
                    '<div class="success-card">Normalization engine finished successfully.</div>',
                    unsafe_allow_html=True,
                )

        if st.session_state.analysis_ready:
            analysis = st.session_state.analysis
            st.markdown('<div class="subsection-title">Parsed Problem</div>', unsafe_allow_html=True)
            st.code(f"R({', '.join(analysis['attributes'])})", language="text")

            st.write("Functional Dependencies in canonical single-RHS form:")
            if analysis["fds"]:
                st.code("\n".join(fd_to_text(fd) for fd in analysis["fds"]), language="text")
            else:
                st.warning("No valid functional dependencies were parsed from your input.")
        else:
            st.markdown(
                '<div class="airport-card">Click <strong>Start Normalization Engine</strong> to parse your input and unlock analysis tabs.</div>',
                unsafe_allow_html=True,
            )

    with tabs[1]:
        if not st.session_state.analysis_ready:
            st.info("Key Discovery is locked. Run Start Normalization Engine in Problem Input first.")
        else:
            analysis = st.session_state.analysis
            attributes = analysis["attributes"]
            fds = analysis["fds"]
            attribute_set = analysis["attribute_set"]
            candidate_keys = analysis["candidate_keys"]
            primary_key = analysis["primary_key"]

            st.markdown('<div class="subsection-title">Attribute Closures</div>', unsafe_allow_html=True)
            closure_map = analysis["closure_map"]
            for attr, attr_closure in closure_map.items():
                st.code(f"{attr}+ = {{{', '.join(sorted(attr_closure))}}}", language="text")

            custom_seed = st.text_input(
                "Try closure of custom attribute set",
                value=", ".join(sorted(primary_key)) if primary_key else attributes[0],
            )
            custom_attrs = {a.strip() for a in custom_seed.split(",") if a.strip()}
            custom_attrs = {a for a in custom_attrs if a in attribute_set}
            if custom_attrs:
                c = closure(custom_attrs, fds)
                st.code(
                    f"({', '.join(sorted(custom_attrs))})+ = {{{', '.join(sorted(c))}}}",
                    language="text",
                )

            st.markdown('<div class="subsection-title">The LHS/RHS Logic</div>', unsafe_allow_html=True)
            must_have = analysis["must_have"]
            maybe_part = analysis["maybe_part"]
            st.code(
                "\n".join(
                    [
                        f"Attributes never on RHS (must appear in every key): {sorted(must_have)}",
                        f"Attributes to test in combinations: {sorted(maybe_part)}",
                    ]
                ),
                language="text",
            )

            st.markdown('<div class="subsection-title">Candidate Keys</div>', unsafe_allow_html=True)
            if not candidate_keys:
                st.warning("No candidate key found from given dependencies. Check if FDs are complete.")
            else:
                key_lines = []
                for i, key in enumerate(candidate_keys, start=1):
                    marker = " (Primary Key)" if i == 1 else ""
                    key_lines.append(f"K{i} = {{{', '.join(sorted(key))}}}{marker}")

                st.markdown(
                    '<div class="key-summary-card"><strong>Final Candidate Keys</strong><br>'
                    + "<br>".join(key_lines)
                    + "</div>",
                    unsafe_allow_html=True,
                )

    with tabs[2]:
        if not st.session_state.analysis_ready:
            st.info("Normalization Steps are locked. Run Start Normalization Engine in Problem Input first.")
        else:
            analysis = st.session_state.analysis

            st.markdown('<div class="section-title">Step-by-Step Normalization</div>', unsafe_allow_html=True)

            base_relation = analysis["base_relation"]

            # 1NF
            is_1nf = analysis["is_1nf"]
            nf1_notes = analysis["nf1_notes"]
            st.markdown('<div class="subsection-title">1NF</div>', unsafe_allow_html=True)
            render_relation_set("Current State", base_relation)
            if is_1nf:
                st.markdown('<p class="nf-ok">1NF satisfied.</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="nf-bad">1NF violation(s) suspected.</p>', unsafe_allow_html=True)
            for note in nf1_notes:
                st.write(f"- {note}")
            st.write("Educational note: 1NF requires every field to hold atomic, indivisible values.")

            # 2NF
            violations_2nf = analysis["violations_2nf"]
            relations_2nf = analysis["relations_2nf"]

            st.markdown('<div class="subsection-title">2NF</div>', unsafe_allow_html=True)
            render_relation_set("Current State", base_relation)
            if violations_2nf:
                st.markdown('<p class="nf-bad">2NF violations found.</p>', unsafe_allow_html=True)
                for lhs, rhs in violations_2nf:
                    st.write(
                        f"- Partial Dependency detected: {', '.join(sorted(lhs))} -> {rhs}"
                    )
                render_schema_grid("Decomposed Tables", relations_2nf, analysis["fds"])
                st.write(
                    "Educational note: We split tables to remove dependencies on part of a composite key,"
                    " reducing update anomalies."
                )
            else:
                st.markdown('<p class="nf-ok">No 2NF partial dependency violations detected.</p>', unsafe_allow_html=True)
                render_schema_grid("Decomposed Tables", relations_2nf, analysis["fds"])

            # 3NF
            violations_3nf = analysis["violations_3nf"]
            relations_3nf = analysis["relations_3nf"]

            st.markdown('<div class="subsection-title">3NF</div>', unsafe_allow_html=True)
            render_relation_set("Current State", relations_2nf)
            if violations_3nf:
                st.markdown('<p class="nf-bad">3NF violations found.</p>', unsafe_allow_html=True)
                for fd in violations_3nf:
                    st.write(f"- Transitive or non-key dependency: {fd_to_text(fd)}")
                render_schema_grid("Decomposed Tables", relations_3nf, analysis["fds"])
                st.write(
                    "Educational note: 3NF decomposition removes transitive dependencies so non-key attributes"
                    " depend only on candidate keys."
                )
            else:
                st.markdown('<p class="nf-ok">No 3NF violations detected.</p>', unsafe_allow_html=True)
                render_schema_grid("Decomposed Tables", relations_3nf, analysis["fds"])

            # BCNF
            violations_bcnf = analysis["violations_bcnf"]
            relations_bcnf = analysis["relations_bcnf"]

            st.markdown('<div class="subsection-title">BCNF</div>', unsafe_allow_html=True)
            render_relation_set("Current State", relations_3nf)
            if violations_bcnf:
                st.markdown('<p class="nf-bad">BCNF violations found.</p>', unsafe_allow_html=True)
                for fd in violations_bcnf:
                    st.write(f"- Determinant is not a superkey: {fd_to_text(fd)}")
                render_schema_grid("Decomposed Tables", relations_bcnf, analysis["fds"])
                st.write(
                    "Educational note: BCNF is stricter than 3NF; every determinant must be a superkey."
                )
            else:
                st.markdown('<p class="nf-ok">No BCNF violations detected.</p>', unsafe_allow_html=True)
                render_schema_grid("Decomposed Tables", relations_bcnf, analysis["fds"])

            render_final_schema_answer(relations_bcnf, analysis["fds"])


if __name__ == "__main__":
    main()
