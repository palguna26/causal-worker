import os
import time
import logging
import uuid
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('causal_worker')

# Load env
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set")

# Supabase pooler fix
if ":6543" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace(":6543", ":5432")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)


def get_or_create_node(conn, org_id, label, node_type):
    """
    Look up a global discovery node for this org by label, or create it.
    Uses session_id='GLOBAL_DISCOVERY' to separate abstract rules from user sessions.
    """
    query = text("""
        SELECT id FROM causal_nodes
        WHERE org_id = :org_id
          AND session_id = 'GLOBAL_DISCOVERY'
          AND label = :label
          AND node_type = :node_type
        LIMIT 1
    """)
    result = conn.execute(query, {"org_id": org_id, "label": label, "node_type": node_type}).fetchone()

    if result:
        return result[0]

    new_id = uuid.uuid4()
    insert = text("""
        INSERT INTO causal_nodes (id, org_id, session_id, agent_id, project_id, label, node_type, confidence, is_verified)
        VALUES (:id, :org_id, 'GLOBAL_DISCOVERY', 'system_discovery', 'GLOBAL', :label, :node_type, 1.0, true)
    """)
    conn.execute(insert, {"id": new_id, "org_id": org_id, "label": label, "node_type": node_type})
    return new_id


def upsert_edge(conn, org_id, from_id, to_id, relation_type, weight, explanation):
    """
    Insert or update the causal edge with the discovered weight.
    """
    upsert = text("""
        INSERT INTO causal_edges (org_id, from_node_id, to_node_id, relation_type, weight, observation_count, explanation)
        VALUES (:org_id, :from_id, :to_id, :relation_type, :weight, 1, :explanation)
        ON CONFLICT (from_node_id, to_node_id, relation_type) DO UPDATE
            SET weight = :weight,
                observation_count = causal_edges.observation_count + 1,
                updated_at = now()
    """)
    conn.execute(upsert, {
        "org_id": org_id, "from_id": from_id, "to_id": to_id,
        "relation_type": relation_type, "weight": weight, "explanation": explanation
    })


def ols_causal_effect(df, treatment_col, outcome_col, covariate_cols):
    """
    Lightweight OLS-based causal effect estimation (backdoor linear regression).
    Replaces dowhy.CausalModel to avoid the heavy scipy/cvxpy/matplotlib stack.

    Returns the estimated coefficient for the treatment variable, or None if
    the regression cannot be fit (e.g., insufficient variance / samples).
    """
    try:
        # Build design matrix: [1, treatment, covariates...]
        X = df[[treatment_col] + covariate_cols].copy()

        # One-hot encode string covariates
        X = pd.get_dummies(X, columns=covariate_cols, drop_first=True)

        # Add intercept column
        X.insert(0, '_intercept', 1.0)

        X_arr = X.values.astype(float)
        y_arr = df[outcome_col].values.astype(float)

        # Need at least as many rows as columns + 1
        if X_arr.shape[0] < X_arr.shape[1] + 1:
            return None

        # OLS: beta = (X'X)^-1 X'y  via lstsq for numerical stability
        coeffs, _, rank, _ = np.linalg.lstsq(X_arr, y_arr, rcond=None)

        if rank < X_arr.shape[1]:
            # Under-determined — not enough signal
            return None

        # Coefficient index 1 is the treatment (after intercept)
        return float(coeffs[1])

    except Exception as e:
        logger.warning(f"OLS failed: {e}")
        return None


def process_org(org_id):
    logger.info(f"Processing causal discovery for org: {org_id}")
    try:
        query = "SELECT action_type, tool_name, verdict FROM causal_ledger WHERE org_id = %(org_id)s"
        df = pd.read_sql(query, engine, params={"org_id": org_id})

        if df.empty:
            logger.info("Causal ledger is empty for this org.")
            return

        # Convert verdict to binary outcome
        df['outcome_success'] = df['verdict'].apply(lambda x: 1.0 if x == 'ALLOW' else 0.0)

        action_types = df['action_type'].unique()

        with engine.begin() as conn:
            success_node_id = get_or_create_node(conn, org_id, "outcome:success", "outcome")

            for action in action_types:
                df['treatment_action'] = (df['action_type'] == action).astype(float)

                if df['treatment_action'].nunique() < 2:
                    continue

                causal_effect = ols_causal_effect(
                    df,
                    treatment_col='treatment_action',
                    outcome_col='outcome_success',
                    covariate_cols=['tool_name'],
                )

                if causal_effect is None:
                    logger.info(f"Org {org_id} | Skipping '{action}' — insufficient data for regression.")
                    continue

                logger.info(f"Org {org_id} | Effect of '{action}' on success: {causal_effect:.4f}")

                # Transform raw coefficient → weight in [0, 1]
                weight = max(0.0, min(1.0, 0.5 + (causal_effect / 2.0)))

                action_node_id = get_or_create_node(conn, org_id, f"action:{action}", "action")

                upsert_edge(
                    conn,
                    org_id,
                    from_id=action_node_id,
                    to_id=success_node_id,
                    relation_type='caused',
                    weight=weight,
                    explanation=f"Batch discovery: Causal effect {causal_effect:.4f}"
                )

    except Exception as e:
        logger.error(f"Error processing org {org_id}: {e}", exc_info=True)


def run_causal_discovery():
    logger.info("Starting causal discovery batch job...")
    try:
        with engine.connect() as conn:
            orgs = conn.execute(text("SELECT DISTINCT org_id FROM causal_ledger")).fetchall()

        for org in orgs:
            process_org(org[0])

        logger.info("Causal discovery batch job completed.")
    except Exception as e:
        logger.error(f"Error fetching organizations: {e}", exc_info=True)


if __name__ == "__main__":
    logger.info("Initializing Causal Worker...")
    while True:
        run_causal_discovery()
        logger.info("Sleeping for 1 hour before next run...")
        time.sleep(3600)
