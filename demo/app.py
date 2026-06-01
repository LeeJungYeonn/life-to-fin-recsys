import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


DEMO_DIR = Path(__file__).resolve().parent
if (DEMO_DIR.parent / "code").exists():
    RECSYS_DIR = DEMO_DIR.parent
else:
    RECSYS_DIR = DEMO_DIR.parent / "life-to-fin-recsys"
CODE_DIR = RECSYS_DIR / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from portfolio_schema import BUCKET_COLUMNS, CATEGORICAL_COLUMNS  # type: ignore
from run_end_to_end import load_end_to_end_resources, run_end_to_end  # type: ignore


@st.cache_resource
def load_system():
    return load_end_to_end_resources(
        processed_dir=RECSYS_DIR / "dataset" / "processed",
        checkpoint_dir=RECSYS_DIR / "checkpoints",
        checkpoint_prefix=None,
        anchor_csv=RECSYS_DIR / "dataset" / "train.csv",
        load_knn_anchors=True,
    )


end_to_end_resources = load_system()
products_catalog = end_to_end_resources.products


@st.cache_data
def load_test_profiles():
    path = RECSYS_DIR / "dataset" / "test.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, usecols=["CASEID", *CATEGORICAL_COLUMNS])


test_profiles = load_test_profiles()


def label_from_options(options, value):
    for label, option_value in options.items():
        if option_value == value:
            return label
    return str(value)


def savings_reason_label(user_profile, savres_opts):
    for label, column in savres_opts.items():
        if int(user_profile.get(column, 0)) == 1:
            return label
    return "Not specified"


def derive_life_family_fields(agecl, married, kids, lf):
    has_kids = int(kids) > 0
    is_married = int(married) == 1
    is_under_55 = int(agecl) <= 3

    if is_married:
        famstruct = 4 if has_kids else 5
    elif has_kids:
        famstruct = 1
    else:
        famstruct = 2 if is_under_55 else 3

    if is_under_55:
        if is_married and has_kids:
            lifecl = 3
        elif is_married:
            lifecl = 2
        elif has_kids:
            lifecl = 4
        else:
            lifecl = 1
    else:
        lifecl = 5 if int(lf) == 1 else 6

    return lifecl, famstruct


def describe_risk_level(display_risk_level):
    descriptions = {
        1: "very conservative",
        2: "conservative",
        3: "moderate",
        4: "aggressive",
        5: "very aggressive",
    }
    return descriptions.get(int(display_risk_level), "model-selected")


def describe_allocation_mix(weights):
    cash = weights.get("cash", 0.0)
    bond = weights.get("bond", 0.0)
    pension = weights.get("pension", 0.0)
    equity = weights.get("equity", 0.0)

    notes = []
    if cash >= 50.0:
        notes.append(
            "The portfolio is anchored in liquidity, which can fit users whose profile implies a preference for stability or near-term flexibility."
        )
    elif cash >= 25.0:
        notes.append(
            "A meaningful liquidity sleeve is retained, while the rest of the portfolio is spread across longer-term assets."
        )
    else:
        notes.append(
            "Liquidity is kept relatively lean, so more of the recommendation is allocated to investment-oriented buckets."
        )

    if pension >= 20.0:
        notes.append(
            "The pension bucket is material, suggesting that long-horizon retirement exposure is important for this profile."
        )
    if bond >= 15.0:
        notes.append("Bond exposure adds a stabilizing layer between cash and equity risk.")
    if equity >= 25.0:
        notes.append(
            "Equity exposure is a major growth driver, so short-term market movement may matter more for this recommendation."
        )
    elif equity <= 10.0:
        notes.append(
            "Equity exposure is intentionally limited, keeping the growth component modest relative to safer buckets."
        )

    return notes


def format_allocation(weights):
    allocations = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(
        f"{bucket} {weight:.1f}%" for bucket, weight in allocations if weight > 0.0
    )


def summarize_profile_signals(profile_labels):
    stability = [
        f"labor-force status: {profile_labels['labor']}",
        f"housing: {profile_labels['housing']}",
        f"saving behavior: {profile_labels['saving_behavior']}",
    ]
    liquidity = [
        f"dependent children: {profile_labels['kids']}",
        f"family structure: {profile_labels['family']}",
        f"expense level: {profile_labels['expense_level']}",
        f"savings goal: {profile_labels['savings_reason']}",
    ]
    horizon = [
        f"age group: {profile_labels['age']}",
        f"life cycle: {profile_labels['life_cycle']}",
    ]

    if "Retirement" in profile_labels["savings_reason"]:
        horizon.append("savings goal includes retirement")
    if "Investment" in profile_labels["savings_reason"]:
        horizon.append("savings goal includes investment")

    return {
        "Stability signals": stability,
        "Liquidity needs": liquidity,
        "Long-term horizon signals": horizon,
    }


def describe_product_buckets(recommendation):
    items = recommendation.get("optimized_basket", recommendation.get("ranked_products", []))
    buckets = []
    for item in items:
        bucket = str(item.get("bucket", item.get("category", ""))).lower()
        if bucket and bucket not in buckets:
            buckets.append(bucket)

    descriptions = {
        "cash": "Cash-oriented products are included to preserve liquidity and reduce short-term portfolio movement.",
        "bond": "Bond products can be interpreted as a volatility buffer between cash and growth assets.",
        "pension": "Pension products support long-horizon retirement exposure within the predicted allocation.",
        "equity": "Equity or ETF products provide growth exposure while staying within the selected risk label.",
    }
    return [descriptions[bucket] for bucket in buckets if bucket in descriptions]


def anchor_allocation_text(knn_smoothing):
    if not knn_smoothing or not knn_smoothing.get("enabled"):
        return None

    anchor_allocation = knn_smoothing.get("anchor_allocation")
    if not anchor_allocation:
        return None

    anchor_weights = {
        bucket: float(value) * 100.0 for bucket, value in anchor_allocation.items()
    }
    return format_allocation(anchor_weights)


def build_profile_labels(user_profile, option_maps):
    return {
        "age": label_from_options(option_maps["agecl"], user_profile["AGECL"]),
        "life_cycle": label_from_options(option_maps["lifecl"], user_profile["LIFECL"]),
        "family": label_from_options(option_maps["famstruct"], user_profile["FAMSTRUCT"]),
        "labor": label_from_options(option_maps["lf"], user_profile["LF"]),
        "occupation": label_from_options(option_maps["occat2"], user_profile["OCCAT2"]),
        "housing": label_from_options(option_maps["housecl"], user_profile["HOUSECL"]),
        "saving_behavior": label_from_options(option_maps["wsaved"], user_profile["WSAVED"]),
        "expense_level": label_from_options(option_maps["expenshilo"], user_profile["EXPENSHILO"]),
        "savings_reason": savings_reason_label(user_profile, option_maps["savres"]),
        "kids": int(user_profile.get("KIDS", 0)),
    }


def generate_personalized_report(
    profile_labels,
    weights,
    display_risk_level,
    recommendation,
    knn_smoothing=None,
    caseid=None,
):
    risk_description = describe_risk_level(display_risk_level)
    allocation_text = format_allocation(weights)
    source_text = f"CASEID {int(caseid)}" if caseid is not None else "the survey responses"
    profile_signals = summarize_profile_signals(profile_labels)
    product_notes = describe_product_buckets(recommendation)
    anchor_text = anchor_allocation_text(knn_smoothing)

    report_lines = [
        "### Personalized Report",
        "**User profile summary**",
        (
            f"Based on {source_text}, this user appears to be in the "
            f"**{profile_labels['age']}** age group with a **{profile_labels['life_cycle']}** "
            f"life-cycle profile. The household context is **{profile_labels['family']}**, "
            f"with **{profile_labels['kids']}** dependent children."
        ),
        (
            f"The financial behavior signals are **{profile_labels['saving_behavior']}**, "
            f"**{profile_labels['expense_level']}** expenses, and a primary savings goal of "
            f"**{profile_labels['savings_reason']}**. Work and housing signals are "
            f"**{profile_labels['labor']}**, **{profile_labels['occupation']}**, and "
            f"**{profile_labels['housing']}**."
        ),
        "**Model output**",
        (
            f"- Predicted risk label: **{display_risk_level}/5** "
            f"({risk_description})\n"
            f"- Predicted allocation: **{allocation_text}**"
        ),
        "**Evidence from input profile**",
    ]

    for title, signals in profile_signals.items():
        report_lines.append(f"- {title}: {', '.join(signals)}")

    report_lines.append("**Why this allocation?**")
    report_lines.extend(f"- {note}" for note in describe_allocation_mix(weights))

    if anchor_text:
        report_lines.append(
            f"- Similar lifestyle profiles from the anchor set averaged **{anchor_text}**, which provides neighborhood-based context for this recommendation."
        )

    if product_notes:
        report_lines.append("**Why these products?**")
        report_lines.extend(f"- {note}" for note in product_notes)

    report_lines.append(
        "⚠️ Evidence note: This is a post-hoc interpretive report generated from model outputs, user profile signals, and product bucket metadata. It is not a direct explanation of the model's internal reasoning and is not financial advice."
    )
    return "\n\n".join(report_lines)


st.set_page_config(page_title="Life-to-Fin RecSys Demo", page_icon="L", layout="wide")
st.title("Lifestyle-to-Finance")
st.subheader("Demographic Data-Based Portfolio Recommendation")
st.markdown(
    "**A zero-shot recommendation for thin-file users without transaction history.**"
)
st.divider()

st.header("Step 1. Tell us about your lifestyle")

st.sidebar.header("Batch Comparison")
use_exact_caseid = st.sidebar.checkbox("Use exact test.csv CASEID")
selected_caseid = None
if use_exact_caseid:
    if test_profiles.empty:
        st.sidebar.warning("dataset/test.csv was not found.")
    else:
        selected_caseid = st.sidebar.selectbox(
            "CASEID",
            test_profiles["CASEID"].astype(int).tolist(),
        )

tab1, tab2, tab3 = st.tabs(
    ["Demographics", "Education & Work", "Spending & Saving"]
)

with st.form("survey_form"):
    with tab1:
        st.subheader("Demographics & Family Structure")
        famstruct_opts = {
            "Unmarried/no partner + children": 1,
            "Unmarried/no partner + no children + under 55": 2,
            "Unmarried/no partner + no children + 55 or older": 3,
            "Married/partner + children": 4,
            "Married/partner + no children": 5,
        }

        lifecl_opts = {
            "Under 55 + unmarried/no partner + no children": 1,
            "Under 55 + married/partner + no children": 2,
            "Under 55 + married/partner + children": 3,
            "Under 55 + unmarried/no partner + children": 4,
            "55 or older + working": 5,
            "55 or older + not working": 6,
        }
        
        agecl_opts = {
            "<35": 1,
            "35-44": 2,
            "45-54": 3,
            "55-64": 4,
            "65-74": 5,
            ">=75": 6,
        }
        agecl_ui = st.selectbox("Age group (AGECL)", list(agecl_opts.keys()))

        married_opts = {
            "Married / living with partner": 1,
            "Not married / not living with partner": 2,
        }
        married_ui = st.radio("Marital status (MARRIED)", list(married_opts.keys()))

        kids_ui = st.number_input(
            "Number of dependent children (KIDS)",
            min_value=0,
            max_value=20,
            value=0,
        )

    with tab2:
        st.subheader("Education, Occupation & Housing")
        col3, col4 = st.columns(2)
        with col3:
            edcl_opts = {
                "No high school diploma": 1,
                "High school/GED": 2,
                "Some college": 3,
                "College degree or higher": 4,
            }
            edcl_ui = st.selectbox("Education summary (EDCL)", list(edcl_opts.keys()))

            educ_opts = {
                "Less than 1 year (-1)": -1,
                "1-4 years (1)": 1,
                "5-6 years (2)": 2,
                "7-8 years (3)": 3,
                "9 years (4)": 4,
                "10 years (5)": 5,
                "11 years (6)": 6,
                "12 years, no diploma (7)": 7,
                "High school diploma (8)": 8,
                "Some college, no degree (9)": 9,
                "Associate degree, vocational (10)": 10,
                "Associate degree, academic (11)": 11,
                "Bachelor's degree (12)": 12,
                "Master's degree (13)": 13,
                "Doctorate/professional degree (14)": 14,
            }
            educ_ui = st.selectbox(
                "Detailed education (EDUC)",
                list(educ_opts.keys()),
                index=list(educ_opts.values()).index(12),
            )

            lf_opts = {"Working": 1, "Not working": 0}
            lf_ui = st.radio("Labor-force status (LF)", list(lf_opts.keys()))

        with col4:
            occat1_opts = {
                "Work for someone else": 1,
                "Self-employed / partnership": 2,
                "Retired / student / homemaker": 3,
                "Other non-working household head under 65": 4,
            }
            occat1_ui = st.selectbox("Work arrangement (OCCAT1)", list(occat1_opts.keys()))

            occat2_opts = {
                "Managerial / professional": 1,
                "Technical / sales / services": 2,
                "Other occupation": 3,
                "Not working": 4,
            }
            occat2_ui = st.selectbox("Occupation category (OCCAT2)", list(occat2_opts.keys()))

            indcat_opts = {
                "Mining / construction / manufacturing": 1,
                "Transport / utilities / trade / finance / real estate": 2,
                "Services / public administration": 4,
            }
            indcat_ui = st.selectbox("Industry category (INDCAT)", list(indcat_opts.keys()))

            housecl_opts = {
                "Owns home or similar": 1,
                "Other housing arrangement": 2,
            }
            housecl_ui = st.radio("Housing status (HOUSECL)", list(housecl_opts.keys()))

    with tab3:
        st.subheader("Expenses & Savings Goals")

        wsaved_opts = {
            "Spent less than income": 3,
            "Spent about equal to income": 2,
            "Spent more than income": 1,
        }
        wsaved_ui = st.radio("Past-year spending/saving (WSAVED)", list(wsaved_opts.keys()))

        expenshilo_opts = {
            "Normal": 3,
            "Unusually high": 1,
            "Unusually low": 2,
        }
        expenshilo_ui = st.selectbox("Expense level (EXPENSHILO)", list(expenshilo_opts.keys()))

        savres_opts = {
            "Cannot save": "SAVRES1",
            "Education": "SAVRES2",
            "Family": "SAVRES3",
            "Home": "SAVRES4",
            "Purchases": "SAVRES5",
            "Retirement": "SAVRES6",
            "Liquidity / the future": "SAVRES7",
            "Investment": "SAVRES8",
            "No particular reason": "SAVRES9",
        }
        savres_ui = st.selectbox(
            "Primary savings reason (SAVRES 1-9)",
            list(savres_opts.keys()),
            index=6,
        )

    submitted = st.form_submit_button("Recommend Portfolio")

if submitted:
    st.divider()
    st.header("Step 2. Portfolio Analysis")

    with st.spinner("Matching financial products and estimating allocation..."):
        user_profile = {
            "OCCAT1": occat1_opts[occat1_ui],
            "OCCAT2": occat2_opts[occat2_ui],
            "INDCAT": indcat_opts[indcat_ui],
            "LF": lf_opts[lf_ui],
            "HOUSECL": housecl_opts[housecl_ui],
            "EDCL": edcl_opts[edcl_ui],
            "EDUC": educ_opts[educ_ui],
            "AGECL": agecl_opts[agecl_ui],
            "KIDS": kids_ui,
            "MARRIED": married_opts[married_ui],
            "EXPENSHILO": expenshilo_opts[expenshilo_ui],
            "WSAVED": wsaved_opts[wsaved_ui],
        }

        selected_savres_key = savres_opts[savres_ui]
        for i in range(1, 10):
            col_name = f"SAVRES{i}"
            user_profile[col_name] = 1 if selected_savres_key == col_name else 0

        if use_exact_caseid and selected_caseid is not None and not test_profiles.empty:
            test_row = test_profiles[test_profiles["CASEID"] == selected_caseid].iloc[0]
            user_profile = {column: int(test_row[column]) for column in CATEGORICAL_COLUMNS}

        derived_lifecl, derived_famstruct = derive_life_family_fields(
            user_profile["AGECL"],
            user_profile["MARRIED"],
            user_profile["KIDS"],
            user_profile["LF"],
        )
        user_profile["LIFECL"] = derived_lifecl
        user_profile["FAMSTRUCT"] = derived_famstruct

        option_maps = {
            "agecl": agecl_opts,
            "lifecl": lifecl_opts,
            "famstruct": famstruct_opts,
            "lf": lf_opts,
            "occat2": occat2_opts,
            "housecl": housecl_opts,
            "wsaved": wsaved_opts,
            "expenshilo": expenshilo_opts,
            "savres": savres_opts,
        }
        profile_labels = build_profile_labels(user_profile, option_maps)

        result = run_end_to_end(
            user_profile,
            resources=end_to_end_resources,
            coerce_input=not (use_exact_caseid and selected_caseid is not None),
        )
        weights = {
            bucket: float(result["predicted_allocation"][bucket] * 100)
            for bucket in BUCKET_COLUMNS
        }
        direct_risk_level = result["risk_level_used_for_recommendation"]
        display_risk_level = direct_risk_level + 1
        recommendation = result["recommendation"]

        time.sleep(0.5)

    col_res1, col_res2 = st.columns([1.2, 0.8])

    with col_res1:
        st.subheader("Estimated Asset Allocation")
        df_weights = pd.DataFrame(list(weights.items()), columns=["Asset Class", "Weight (%)"])
        df_weights = df_weights[df_weights["Weight (%)"] > 1.0]

        fig = px.pie(
            df_weights,
            values="Weight (%)",
            names="Asset Class",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Teal,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=520)
        st.plotly_chart(fig, width="stretch")

    with col_res2:
        st.subheader("Risk Profile")
        st.info(f"Predicted Risk Label: **{display_risk_level}**")

    st.subheader("Recommended Financial Products")

    product_lookup = {}
    for product in products_catalog:
        if isinstance(product, dict):
            product_id = str(product.get("product_id", product.get("Ticker", "")))
            product_name = product.get("name", product.get("Name", product_id))
            provider = product.get("provider", product.get("Provider", "-"))
            product_type = product.get("product_type", product.get("Theme", "-"))
        else:
            product_id = str(getattr(product, "product_id", getattr(product, "Ticker", "")))
            product_name = getattr(product, "name", getattr(product, "Name", product_id))
            provider = getattr(product, "provider", getattr(product, "Provider", "-"))
            product_type = getattr(product, "product_type", getattr(product, "Theme", "-"))

        if product_id:
            product_lookup[product_id] = {
                "name": product_name,
                "provider": provider,
                "type": product_type,
            }

    items_to_display = recommendation.get(
        "optimized_basket",
        recommendation.get("ranked_products", []),
    )

    rec_rows = []
    for item in items_to_display:
        product_id = str(item.get("product_id", ""))
        info = product_lookup.get(product_id, {})
        rec_rows.append(
            {
                "Bucket": item.get("bucket", "-"),
                "Provider": info.get("provider", "-"),
                "Type": info.get("type", item.get("category", "-")),
                "Product": item.get("name", info.get("name", product_id)),
                "Weight (%)": round(float(item.get("weight", 0.0)) * 100, 2),
                "Score": round(float(item.get("score", 0.0)), 4),
            }
        )

    df_recs = pd.DataFrame(rec_rows)
    if not df_recs.empty:
        if "Weight (%)" in df_recs.columns:
            df_recs = df_recs.sort_values(by="Weight (%)", ascending=False).reset_index(drop=True)
        df_recs.index = df_recs.index + 1
        st.dataframe(df_recs, width="stretch")
    else:
        st.warning("No recommendations were returned.")

    report_caseid = selected_caseid if use_exact_caseid and selected_caseid is not None else None
    st.markdown(
        generate_personalized_report(
            profile_labels,
            weights,
            display_risk_level,
            recommendation,
            knn_smoothing=result["knn_smoothing"],
            caseid=report_caseid,
        )
    )
