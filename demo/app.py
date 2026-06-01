import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


DEMO_DIR = Path(__file__).resolve().parent
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

st.set_page_config(page_title="Life-to-Fin RecSys Demo", page_icon="L", layout="wide")
st.title("Life-to-Fin: Lifestyle-Based Portfolio Recommendation")
st.markdown(
    "**A zero-shot recommendation demo for thin-file users without transaction history.**"
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
        col1, col2 = st.columns(2)
        with col1:
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

        with col2:
            famstruct_opts = {
                "Unmarried/no partner + children": 1,
                "Unmarried/no partner + no children + under 55": 2,
                "Unmarried/no partner + no children + 55 or older": 3,
                "Married/partner + children": 4,
                "Married/partner + no children": 5,
            }
            famstruct_ui = st.selectbox("Family structure (FAMSTRUCT)", list(famstruct_opts.keys()))

            lifecl_opts = {
                "Under 55 + unmarried/no partner + no children": 1,
                "Under 55 + married/partner + no children": 2,
                "Under 55 + married/partner + children": 3,
                "Under 55 + unmarried/no partner + children": 4,
                "55 or older + working": 5,
                "55 or older + not working": 6,
            }
            lifecl_ui = st.selectbox("Life cycle (LIFECL)", list(lifecl_opts.keys()))

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
        col5, col6 = st.columns(2)
        with col5:
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

        with col6:
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
            "LIFECL": lifecl_opts[lifecl_ui],
            "FAMSTRUCT": famstruct_opts[famstruct_ui],
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

    col_res1, col_res2 = st.columns([1, 1])

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
        st.plotly_chart(fig, width="stretch")

    with col_res2:
        st.subheader("Risk Profile")
        st.info(f"Risk label used for recommendation: **{display_risk_level}**")
        payload = {
            "checkpoint_prefix": result["checkpoint_prefix"],
            "risk_source": result["risk_source_used_for_recommendation"],
            "risk_level_used_for_recommendation_0_based": direct_risk_level,
            "risk_label_display_1_based": display_risk_level,
            "predicted_risk_level_model": result["predicted_risk_level_model"],
            "predicted_risk_level_allocation": result["predicted_risk_level_allocation"],
            "knn_smoothing": result["knn_smoothing"]["enabled"],
        }
        if use_exact_caseid and selected_caseid is not None:
            payload["CASEID"] = int(selected_caseid)
        st.json(payload)

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
